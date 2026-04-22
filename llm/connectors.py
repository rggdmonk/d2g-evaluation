import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from llm.preprocessors import (
    FaultTolerantJsonPreprocessor,
    HtmlDenoiserBase,
    HtmlRemover,
    LightHtmlDenoiser,
    PassthroughDenoiser,
)
from llm.storage import AnnotationRunStorage, RunContext

logger = logging.getLogger(__name__)


class RetryableAPIError(Exception):
    """Signals that API request may succeed if retried."""


class LLMConnector(ABC):
    """Abstract connector base class"""

    def __init__(self, base_url: str, api_key: str, config: dict[str, Any]) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.config = config
        self.denoiser = self._get_html_denoiser_from_config()
        self.annotation_processors = [FaultTolerantJsonPreprocessor(), HtmlRemover()]

    @abstractmethod
    def _create_payload_from(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Converts the doc into the JSON payload understood by remote API."""
        ...

    @abstractmethod
    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Extracts annotations from API response."""
        ...

    # fmt: off
    async def call_llm(
        self,
        session: aiohttp.ClientSession,
        doc: dict[str, Any],
        context: RunContext,
        storage: AnnotationRunStorage
    ) -> dict[str, Any]:
    # fmt: on
        payload = self._create_payload_from(doc)
        await storage.save_request(doc["task_id"], payload, context)

        async with session.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=180,  # type: ignore[arg-type]
        ) as r:
            response_text = await r.text()
            await storage.save_response(doc["task_id"], response_text, context)
            if r.status == 200:  # noqa PLR2004 No magic: HTTP status code 200 is a well-known number.
                return self._parse_response(response_text)

            if r.status in (429, 500, 502, 503, 504):
                raise RetryableAPIError(f"Retryable API error {r.status}: {response_text}")  # noqa
            raise Exception(f"Non-retryable API error {r.status}: {response_text}")  # noqa

    def _get_html_denoiser_from_config(self) -> HtmlDenoiserBase:
        if self.config.get("html_denoiser") is None:
            return PassthroughDenoiser()

        supported_providers = {"light": LightHtmlDenoiser}

        denoiser_name = self.config.get("html_denoiser")
        if denoiser_name and denoiser_name not in supported_providers:
            suggestions = ", ".join(supported_providers.keys())
            error_message = f"Unsupported html_denoiser: {denoiser_name}. Pick one of: {suggestions}"
            raise ValueError(error_message)

        denoiser_cls = supported_providers[denoiser_name]
        logger.info(f"Using {denoiser_cls.__name__}")  # noqa G004

        return denoiser_cls()

    def _get_html_from(self, doc: dict[str, Any]) -> str:
        return self.denoiser.process(doc["html"])

    def _preprocess_annotations(self, raw_annotations: str) -> str:
        "Remove LLM formatting that breaks JSON parsing if present"
        return_value = raw_annotations
        for p in self.annotation_processors:
            return_value = p.process(return_value)

        return return_value


class OpenRouterConnector(LLMConnector):
    """Gathers annotations from OpenRouter.com"""

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        super().__init__("https://openrouter.ai/api/v1/chat/completions", api_key, config)

    def _create_payload_from(self, doc: dict[str, Any]) -> dict[str, Any]:
        html = self._get_html_from(doc)
        return {
            "model": self.config["model"],
            "temperature": 0.0,
            "top_k": 1,
            "seed": 1337,
            "messages": [
                {"role": "system", "content": self.config["prompt"]["system"]},
                {"role": "user", "content": self.config["prompt"]["user"].format(html=html)},
            ],
            **({"provider": self.config["provider"]} if "provider" in self.config else {}),
            **({"reasoning": self.config["reasoning"]} if "reasoning" in self.config else {}),
        }

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        try:
            data = json.loads(response_text)
            model_response = data["choices"][0]["message"]
            response_content = self._preprocess_annotations(model_response["content"])
            annotations = json.loads(response_content)["annotations"]

        except Exception as e:
            raise ValueError(f"Couldn't process API server response:\n{response_text}") from e  # noqa

        return {
            "annotations": annotations,
            # keep raw response for manual inspection/detecting bugs in preprocessing.
            "annotations_raw": response_content,
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            "total_tokens": data["usage"]["total_tokens"],
            **({"reasoning": model_response["reasoning"]} if "reasoning" in model_response else {}),
            **({"provider": data["provider"]} if "provider" in data else {}),
        }


class OpenAIConnector(LLMConnector):
    """Gathers annotations from OpenAI API"""

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        super().__init__("https://api.openai.com/v1/responses", api_key, config)

    def _create_payload_from(self, doc: dict[str, Any]) -> dict[str, Any]:
        html = self._get_html_from(doc)
        # fmt: off
        return {
            "model": self.config["model"],
            # Some Open AI models don't support setting temperature.
            # But they do support setting top_p
            "top_p": 0,
            "stream": False,
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {"type": "input_text", "text": self.config["prompt"]["system"]}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self.config["prompt"]["user"].format(html=html)}
                    ]
                }
            ],
            **({"reasoning": self.config["reasoning"]} if "reasoning" in self.config else {}),
            **({"max_output_tokens": self.config["max_output_tokens"]} if "max_output_tokens" in self.config else {}),
        }
        # fmt: on

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        try:
            data = json.loads(response_text)

            # Find the assistant message
            message = next(
                item for item in data["output"] if item.get("type") == "message" and item.get("role") == "assistant"
            )

            # Extract text content (Responses API supports multiple content blocks)
            response_content = "".join(
                block["text"] for block in message.get("content", []) if block.get("type") == "output_text"
            )

            response_content = self._preprocess_annotations(response_content)
            annotations = json.loads(response_content)["annotations"]
        except Exception as e:
            raise ValueError(f"Couldn't process OpenAI API response:\n{response_text}") from e  # noqa
        usage = data.get("usage", {})

        return {
            "annotations": annotations,
            "annotations_raw": response_content,
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            **({"reasoning": message["reasoning"]} if "reasoning" in message else {}),
        }
