import logging
from pathlib import Path

import polars as pl
from datasets import Dataset

from d2g_evaluation.evaluation.evaluate_human_vs_tool import HumanVsToolEvaluation
from llm.storage import AnnotationRunStorage

logger = logging.getLogger(__name__)


class RunStateReport:
    """
    Generates an overview of all runs of a given experiment:
    - Configuration file and dataset hashes to detect any unintended contamination;
    - Total docs processed.
    """

    def generate(self, storage: AnnotationRunStorage) -> pl.DataFrame:
        runs = storage.list_all_runs()
        return_value = []
        for r in runs:
            metadata = storage.load_metadata(r)
            docs_processed = len(storage.load_processed_ids(r))
            row = {
                "run_id": r,
                "docs_processed": docs_processed,
                "config": metadata["config"]["name"],
                "input_source": metadata["input_source"],
                "config_hash": metadata["config_hash"],
                "input_hash": metadata["input_hash"],
            }
            return_value.append(row)

        df = pl.DataFrame(return_value)
        with pl.Config(fmt_str_lengths=10**5):
            print(df)

            return df


class HumanVsLlmExperimentMetricsReport:
    """Reports Human vs LLM comparison metrics
    for a single experiment, averaged over all runs.
    """

    def generate(self, storage: AnnotationRunStorage) -> pl.DataFrame:
        logger.debug(f"Generating report for {storage.run_config['name']}.config...")  # noqa

        runs = storage.list_all_runs()
        rows = []
        for run_id in runs:
            eval_stats, llm_annotations, dataset = self.evaluate_single_run(run_id, storage)
            precision = round(eval_stats["precision"].mean(), 3)
            recall = round(eval_stats["recall"].mean(), 3)
            f1 = round(eval_stats["f1"].mean(), 3)

            # Document length vs token cost
            tokens_consumed = round(llm_annotations["total_tokens"].mean())
            doclength_chars = round(dataset["html"].map_elements(lambda x: len(x)).mean())
            compression_ratio = round(doclength_chars / tokens_consumed, 2)

            rows.append(
                {
                    "run_id": run_id,
                    "docs": float(len(llm_annotations)),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_doclength_chars": float(doclength_chars),
                    "mean_tokens_consumed": float(tokens_consumed),
                    "compression_ratio": compression_ratio,
                }
            )

        report = pl.DataFrame(rows)
        mean_numeric = report.select(pl.selectors.numeric().mean())
        mean_row = mean_numeric.with_columns(pl.lit("**MEAN**").alias("run_id"))
        mean_row = mean_row.select(["run_id"] + [c for c in mean_row.columns if c != "run_id"])
        report = pl.concat([report, mean_row])
        with pl.Config(fmt_str_lengths=10**5):
            print(report)
            return report

    def evaluate_single_run(
        self,
        run_id: str,
        storage: AnnotationRunStorage,
    ) -> tuple[pl.DataFrame, pl.DataFrame, Dataset]:
        logger.debug(f"Processing run_id: {run_id}...")  # noqa
        context = storage.get_runcontext_for(run_id)
        metadata = storage.load_metadata(run_id)
        # merge all annotations into a single string for metric computation
        llm_annotations = (
            storage.load_annotations(context)
            .with_columns(
                pl.col("annotations")
                .map_elements(lambda x: self._merge_annotations(x), return_dtype=pl.Utf8)
                .alias("annotations_as_string")
            )
            .sort(["task_id"])
        )

        logger.debug(f"Loaded {len(llm_annotations)} annotations")  # noqa

        annotated_ids = set(llm_annotations["task_id"])
        # fmt: off
        annotations_as_list = (
            llm_annotations["annotations"]
            .list
            .eval(pl.element().struct.field("text"))
            .to_list()
        )
        # fmt: on
        dataset = (
            pl.DataFrame(storage.load_dataset(metadata["input_source"])[0])
            .filter(pl.col("task_id").is_in(annotated_ids))
            .sort("task_id")
            .with_columns(
                pl.Series("llm_annotations_as_string", llm_annotations["annotations_as_string"]),
                pl.Series("llm_annotations", annotations_as_list),
            )
        )

        metrics = {
            "metric_name": "lcs_token_matching",
            "string_preprocessing_method": "normalize_string",
            "tokenization_method": "char_ngrams",
            "n": 3,
            "is_symmetric_forced": False,
        }

        eval_tool = HumanVsToolEvaluation()
        eval_tool.INSUFFICIENT_ANNOTATIONS = 1

        eval_result = eval_tool.evaluate(
            dataset=Dataset.from_polars(dataset),
            tool_column="llm_annotations_as_string",
            tool_name=llm_annotations[0]["model"],
            **metrics,
        ).to_polars()

        eval_stats = eval_result.select(
            [
                pl.col("task_id"),
                pl.col("sample_result_human_vs_tool")
                .struct.field("precision")
                .struct.field("mean")
                .round(3)
                .alias("precision"),
                pl.col("sample_result_human_vs_tool")
                .struct.field("recall")
                .struct.field("mean")
                .round(3)
                .alias("recall"),
                pl.col("sample_result_human_vs_tool").struct.field("f1").struct.field("mean").round(3).alias("f1"),
                pl.col("llm_annotations"),
                pl.col("llm_annotations_as_string"),
                pl.col("annotations")
                .list.eval(pl.element().struct[8].list[0].struct["text"])
                .alias("human_annotations"),
            ],
        )

        return eval_stats, llm_annotations, dataset

    def _merge_annotations(self, x: list[dict[str, str]]) -> str:
        return " ".join([str(k.get("text", "")) for k in x])


class HumanVsLlmSummaryReport:
    """Reports Human vs LLM comparison metrics
    across all experiments, row per config/model.
    """

    def generate(self, experiment_base_path: str) -> pl.DataFrame:
        # fmt:off
        logger.debug(
            f"Generating overview report for all experiments in {experiment_base_path}" # noqa
            )
        # fmt: on
        root = Path(experiment_base_path)
        if not root.exists():
            raise FileNotFoundError(f"Path {experiment_base_path} not found.")  # noqa
        experiment_report = HumanVsLlmExperimentMetricsReport()
        run_path = root / "runs"
        experiment_names = [p.name for p in run_path.iterdir() if p.is_dir()]
        schema = None
        rows = []
        for n in experiment_names:
            storage = AnnotationRunStorage(base_path=experiment_base_path, config=root / "configs" / f"{n}.config")
            single_eval = experiment_report.generate(storage)
            mean_row = list(single_eval.tail(1).row(0))
            mean_row[0] = n
            rows.append(mean_row)
            if not schema:
                schema = single_eval.schema
        # fmt: off
        return_value = pl.DataFrame(rows, schema)\
            .rename({"run_id": "experiment"})\
            .sort("f1", descending=True)
        # fmt: on

        print(return_value)
        return return_value
