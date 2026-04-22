import re
import unicodedata
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup, Comment
from json_repair import repair_json


class FaultTolerantJsonPreprocessor:
    """
    Preprocesses potentially malformed JSON coming from an LLM
    into a form suitable for the Python parser.

    """

    def process(self, raw_annotations: str) -> str:
        return_value = unicodedata.normalize("NFKC", raw_annotations)
        # remove control chars that can potentially break JSON parsing.
        return_value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F\u2028\u2029]", "", return_value)
        # zero-width space
        return_value = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", "", return_value)
        return_value = return_value.replace("\xa0", " ")

        # pretty-printing like ```python, ```jsonl, etc.
        def strip_code_fences(text: str) -> str:
            pattern = r"```[a-zA-Z0-9]*\n|\n```$|^```[a-zA-Z0-9]*\r?\n|\r?\n```$"
            return re.sub(pattern, "", text).strip()

        return_value = strip_code_fences(return_value)

        return_value = repair_json(return_value, ensure_ascii=False)

        return return_value.strip()


class HtmlRemover:
    """
    Strips HTML tags sometimes returned as part of annotation by an LLM.
    """

    def process(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")

        return soup.get_text(separator=" ", strip=True)


class HtmlDenoiserBase(ABC):
    """Abstract preprocessor base class"""

    @abstractmethod
    def process(self, html: str) -> str: ...


class PassthroughDenoiser(HtmlDenoiserBase):
    """Keeps HTML intact"""

    def process(self, html: str) -> str:
        return html


class LightHtmlDenoiser(HtmlDenoiserBase):
    """
    Removes noise (scripts, css, etc.) from HTML to cut annotation costs
    and potentially improve LLM annotation quality.

    """

    def process(self, html: str) -> str:
        remove_tags = ["script", "style", "nav", "noscript", "link", "meta", "iframe"]
        remove_attrs = ["style", "onclick", "onload", "onerror"]

        soup = BeautifulSoup(html, "lxml")

        for tag in soup.find_all(remove_tags):
            tag.decompose()

        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        for tag in soup.find_all(recursive=True):
            for attr in list(tag.attrs):
                if attr in remove_attrs or attr.startswith("on"):
                    del tag[attr]

        return str(soup)
