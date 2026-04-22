"""
Integrates with an external LLM API in a resilient async manner
to annotate main content text in a given HTML file collection.

Inputs:
- A `jsonl` or `parquet` file with HTML docs;
    Each entry is expected to have two columns: 'task_id' (the unique html doc id in our notation) and 'html'.
- A config file specifying annotation run settings for experiment reproducibility.
    Example .config is included in containing directory.

Outputs:
- `.{outputfolder}/configs/{experimentname}.config`: Copy of the experiment configuration settings;
- `.{outputfolder}/runs/{experimentname}/{run_id}/results.jsonl`: LLM-generated main page content annotations;
- `.{outputfolder}/runs/{experimentname}/{run_id}/failures.jsonl`: API responses for documents that failed;
- `.{outputfolder}/runs/{experimentname}/{run_id}/metadata.json`: used dataset, config names and hashes;
- `.{outputfolder}/runs/{experimentname}/{run_id}/requests.jsonl`: Raw API HTTP request payload;
- `.{outputfolder}/runs/{experimentname}/{run_id}/responses.jsonl`: Raw API HTTP responses;
- `.{outputfolder}/runs/{experimentname}/metrics.txt` - Evaluation metrics report
- `.{outputfolder}/runs/{experimentname}/run_state.txt` - Experiment progress summary - total docs annotated, configs, dataset, runs.

Notes:
- By default outputs are stored in llm/.scratch/runs/{config_name}/;
- Requests are asynchronous to reduce time spent waiting for results;
- Supports multi-run experiments for metric averaging over stochastic outputs;
- Raw requests and API responses are logged for quality control and in case document reprocessing is needed;
- Has built-in automatic retry logic with exponential backoff and max retry limit;
- Max concurrent requests are capped (configurable) to avoid API throttling;
- If restarted after an error, will process only yet unannotated HTML docs to save API credits and wall time.\
    This is achieved by checking the documents already present in `results.jsonl` via the `task_id`.
    Deleting this file will cause the entire HTML collection to be reprocessed from scratch.
- When restarting an experiment with pre-existing results,
    validates that config has not been tampered with to avoid result contamination.

"""

import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from typing import Any

import aiohttp
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm.connectors import LLMConnector, OpenAIConnector, OpenRouterConnector, RetryableAPIError
from llm.evaluators import HumanVsLlmExperimentMetricsReport, RunStateReport
from llm.storage import AnnotationRunStorage, InvalidRunStateError, RunContext

# ============ Configuration ============
API_KEY = str(os.getenv("LLM_API_KEY"))
MAX_RETRIES = 5

# ======================================


# ---------- Logging ----------


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------- Helpers ----------


def get_connector_from_config(config: dict[str, Any]) -> LLMConnector:
    connector_name = config.get("connector")
    if connector_name is None:
        raise ValueError("Missing 'connector' in config.")  # noqa

    supported_providers = {
        "openrouter": OpenRouterConnector,
        "openai": OpenAIConnector,
    }

    connector_cls = supported_providers.get(connector_name)
    if connector_cls is None:
        raise ValueError(f"Unsupported connector: {connector_name}")  # noqa

    return connector_cls(api_key=API_KEY, config=config)


# ---------- API Call ----------


@retry(
    wait=wait_exponential(multiplier=3, min=5, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(exception_types=(RetryableAPIError, TimeoutError)),
)
async def call_llm(
    connector: LLMConnector,
    session: aiohttp.ClientSession,
    doc: dict[str, Any],
    context: RunContext,
    storage: AnnotationRunStorage,
) -> dict[str, Any]:
    return await connector.call_llm(session, doc, context, storage)


# ---------- Processing Loop ----------


async def process_docs(
    docs: list[dict[str, Any]],
    ds_metadata: dict,
    storage: AnnotationRunStorage,
    context: RunContext,
) -> None:
    processed = storage.load_processed_ids(context.run_id)
    config = context.run_config
    processed_lock = asyncio.Lock()
    sem = asyncio.Semaphore(config["concurrent_requests"])
    connector = get_connector_from_config(config)
    async with aiohttp.ClientSession() as session:

        async def worker(doc: dict[str, Any]) -> None:
            if "task_id" not in doc or "html" not in doc:
                logger.warning(f"Skipping invalid document: {doc}: missing 'task_id' or 'html' fields.")  # noqa
                return
            if doc["task_id"] in processed:
                logger.debug(f"Skipping task_id {doc['task_id']} (already processed)")  # noqa
                return
            async with sem:
                start_time = time.perf_counter()
                try:
                    parsed_response = await call_llm(connector, session, doc, context, storage)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_ms = round(duration_ms)
                    timestamp = datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")
                    result = {
                        "timestamp": timestamp,
                        "config": config["name"],
                        "model": config["model"],
                        "task_id": doc["task_id"],
                        "duration_ms": duration_ms,
                        "total_tokens": parsed_response["total_tokens"],
                        "annotations": parsed_response["annotations"],
                        "annotations_raw": parsed_response["annotations_raw"],
                    }
                    if "reasoning" in parsed_response:
                        result["reasoning"] = parsed_response["reasoning"]
                    if "provider" in parsed_response:
                        result["provider"] = parsed_response["provider"]
                    await storage.save_annotation(result, context)
                    async with processed_lock:
                        processed.add(doc["task_id"])
                    logger.info(f"Processed {doc['task_id']}")  # noqa
                except Exception as e:  # noqa BLE001
                    timestamp = datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")
                    traceback_text = traceback.format_exc()
                    await storage.log_failed_doc(
                        {
                            "timestamp": timestamp,
                            "config": config["name"],
                            "model": config["model"],
                            "task_id": doc["task_id"],
                            "error": f"{e}\n{traceback_text}",
                        },
                        context,
                    )

                    logger.error(f"Failed {doc.get('task_id')}: {e}\n{traceback_text}")  # noqa

        await asyncio.gather(*(worker(doc) for doc in docs))

        run_metadata = ds_metadata.copy()
        run_metadata["config"] = config
        run_metadata["config_hash"] = config["hash"]
        storage.save_run_metadata(run_metadata, context)


# ---------- Entry Point ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate main content in a given HTML file collection using an LLM")

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input file (.jsonl or .parquet).",
    )

    parser.add_argument(
        "--out_dir",
        "-od",
        default="llm/.scratch",
        help="Experiment output root folder.",
    )

    parser.add_argument("--config", "-c", required=True, help="Path to config file.")

    parser.add_argument(
        "--max_docs",
        "-m",
        type=int,
        default=None,
        help="Caps the number of documents to be annotated. Useful for debugging.",
    )

    parser.add_argument(
        "--n_runs",
        "-n",
        type=int,
        default=1,
        help="Specify how many times annotation experiment should be repeated. "
        "Useful for computing averaged metrics over varying API responses for the same input docs.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Since APIs/LLMs may and will fail, subsequent launches
    # will resume all existing runs to cut costs/time.
    # Every existing experiment run is resumed until all input docs are processed
    # and n_runs is satisfied.
    n_runs = max(1, args.n_runs)
    try:
        storage = AnnotationRunStorage(args.out_dir, args.config)
    except InvalidRunStateError as e:
        logger.fatal(e)
        sys.exit(1)

    docs, ds_metadata = storage.load_dataset(args.input)
    docs = docs[: args.max_docs]
    logger.info(f"Loaded {len(docs)} documents")  # noqa G004
    previous_runs = storage.list_all_runs()
    run_counter = 0

    for run_id in previous_runs:
        logger.info(f"Resuming run: {run_id}")  # noqa G004
        context = storage.get_runcontext_for(run_id)
        asyncio.run(process_docs(docs, ds_metadata, storage, context))
        run_counter += 1
    while n_runs - run_counter > 0:
        context = storage.get_new_runcontext()
        logger.info(f"Starting new run: {context.run_id}")  # noqa G004
        asyncio.run(process_docs(docs, ds_metadata, storage, context))
        run_counter += 1

    with pl.Config(fmt_str_lengths=10**5):
        run_state_report = RunStateReport().generate(storage=storage).__repr__()
        storage.save_experiment_artefact(run_state_report, "run_state.txt")
        metric_report = HumanVsLlmExperimentMetricsReport().generate(storage=storage).__repr__()
        storage.save_experiment_artefact(metric_report, "metrics.txt")

    logger.info("Processing complete.")
