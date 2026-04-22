import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class InvalidRunStateError(RuntimeError):
    pass


class RunContext:
    def __init__(self, run_config: dict[str, Any], run_id: str) -> None:
        self.run_id = run_id
        self.run_config = run_config


class AnnotationRunStorage:
    """Manages annotation run config/result storage on the local file system."""

    CONCURRENT_REQUESTS_DEFAULT = 5

    def __init__(self, base_path: str, config: str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.run_config = self._load_config(config)
        self._annotation_lock = asyncio.Lock()
        self._failure_lock = asyncio.Lock()
        self._response_lock = asyncio.Lock()

    def get_runcontext_for(self, run_id: str) -> RunContext:
        return RunContext(self.run_config, run_id)

    def get_new_runcontext(self) -> RunContext:
        run_config = self.run_config
        run_id = self._generate_new_run_id()

        return RunContext(run_config, run_id)

    def load_dataset(self, path: str) -> list[dict[str, Any]] | dict[str, Any]:
        logger.info(f"Loading dataset from {path}")  # noqa G004
        p = Path(path)
        if p.suffix == ".parquet":
            df = pl.read_parquet(path).sort("task_id")
            return_value = df.to_dicts()
        # Assume JSONL
        else:
            with p.open(encoding="utf-8") as f:
                return_value = [json.loads(line) for line in f]

        metadata = {"input_source": path, "input_hash": self._compute_dataset_hash(return_value)}

        return return_value, metadata

    def load_processed_ids(self, run_id: str) -> set[str]:
        """Returns a list of task_ids already processed successfully"""
        p = Path(self.base_path) / "runs" / self.run_config["name"] / run_id / "results.jsonl"
        logger.info(f"Loading existing LLM annotations from {p}")  # noqa G004

        if not p.exists():
            return set()
        with p.open(encoding="utf-8") as f:
            processed = set()
            for line in f:
                try:
                    processed.add(json.loads(line)["task_id"])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON line in {p}")  # noqa G004
            return processed

    def load_annotations(self, context: RunContext) -> pl.DataFrame:
        result_path = self.base_path / "runs" / self.run_config["name"] / context.run_id / "results.jsonl"
        if not result_path.exists():
            return pl.DataFrame([])

        with result_path.open(encoding="utf-8") as f:
            llm_annotations = [json.loads(line) for line in f]
        return pl.DataFrame(llm_annotations)

    async def save_annotation(self, annotation: dict[str, Any], context: RunContext) -> None:
        annotation_path = self.base_path / "runs" / context.run_config["name"] / context.run_id / "results.jsonl"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._annotation_lock:
            self._save_jsonl_line(annotation, annotation_path)

    async def log_failed_doc(self, fail_event: dict[str, Any], context: RunContext) -> None:
        fail_path = self.base_path / "runs" / context.run_config["name"] / context.run_id / "failures.jsonl"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._failure_lock:
            self._save_jsonl_line(fail_event, fail_path)

    async def save_response(self, task_id: int, response: str, context: RunContext) -> None:
        response_path = self.base_path / "runs" / context.run_config["name"] / context.run_id / "responses.jsonl"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._response_lock:
            entry = {
                "task_id": task_id,
                "timestamp": datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S"),
                "api_response": response,
            }
            self._save_jsonl_line(entry, response_path)

    async def save_request(self, task_id: int, request_payload: dict[str, Any], context: RunContext) -> None:
        response_path = self.base_path / "runs" / context.run_config["name"] / context.run_id / "requests.jsonl"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._response_lock:
            entry = {
                "task_id": task_id,
                "timestamp": datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S"),
                "payload": request_payload,
            }
            self._save_jsonl_line(entry, response_path)

    def save_experiment_artefact(self, contents: str, filename: str) -> None:
        p = self.base_path / "runs" / self.run_config["name"] / filename
        with Path.open(p, "w", encoding="utf-8") as f:
            f.write(contents)

    def load_run_artefact(self, run_id: str, artefact_name: str) -> pl.DataFrame:
        p = Path(self.base_path) / "runs" / self.run_config["name"] / run_id / artefact_name
        if not p.exists():
            return pl.DataFrame([])

        with p.open(encoding="utf-8") as f:
            llm_annotations = [json.loads(line) for line in f]

        return pl.DataFrame(llm_annotations)

    def load_metadata(self, run_id: str) -> dict[str, Any]:
        p = Path(self.base_path) / "runs" / self.run_config["name"] / run_id / "metadata.json"
        with p.open(encoding="utf-8") as f:
            return json.load(f)

    def save_run_metadata(self, run_metadata: dict[str, Any], context: RunContext) -> None:
        metadata_path = self.base_path / "runs" / context.run_config["name"] / context.run_id / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with Path.open(metadata_path, "w") as f:
            json.dump(run_metadata, f, indent=2)

    def list_all_runs(self) -> list[str]:
        base = self.base_path / "runs" / self.run_config["name"]
        if not base.exists():
            return []

        return [f.name for f in base.iterdir() if f.is_dir()]

    def get_run_path_for(self, context: RunContext) -> Path:
        return Path(self.base_path) / "runs" / context.run_config["name"] / context.run_id

    def _save_jsonl_line(self, obj: dict, path: str) -> None:
        """Append a single JSON object as a line."""
        p = Path(path)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _generate_new_run_id(self) -> str:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

        return f"{timestamp}"

    def _load_config(self, config: str) -> dict[str, Any]:
        """
        Load reproducible run configuration from the provided string.
        Accepts either a dictionary or path to JSON configuration file.

        Expected config keys:
        - model (str)
        - connector (str), e.g. 'openrouter'
        - prompt (dict), including 'system' and 'user' prompts

        Optional config keys:
        - concurrent_requests (int). Used to prevent rate limiting. The default value is set in CONCURRENT_REQUESTS

        Args:
            config (str): Dictionary containing configuration values or path to the JSON config file.

        Raises:
            ValueError: If required keys are missing or have wrong types.
        """

        path = Path(config)
        try:
            if path.is_file():  # noqa SIM108 this is more readable
                return_value = self._load_config_from_path(path)
            else:
                return_value = self._load_config_from_json(config)
        except Exception as e:
            raise ValueError(f"Failed to load config from: {config}") from e

        # set defaults
        if "concurrent_requests" not in return_value:
            return_value["concurrent_requests"] = self.CONCURRENT_REQUESTS_DEFAULT
        return_value["hash"] = self._compute_config_hash(return_value)

        # store the updated hash value and/or ephemeral JSON.
        self._validate_config_schema(return_value)
        self._verify_config_hash(return_value)
        self._save_config(return_value)

        logger.info(f"Loaded config {return_value}")  # noqa G004

        return return_value

    def _load_config_from_path(self, config_path: Path) -> dict[str, Any]:
        if not Path.exists(config_path):
            raise ValueError("Path not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            return_value = json.load(f)
            config_autoname = config_path.name.replace(".config", "")
            return_value["name"] = config_autoname

            return return_value

    def _load_config_from_json(self, config: str) -> dict[str, Any]:
        return_value = json.loads(config)
        if "name" not in return_value or not return_value["name"]:
            # use last part of model name like 'deepseek/model-x1' if experiment config not named explicitly
            config_autoname = return_value["model"].split("/")[-1]
            logger.warning(f"'name' missing from config. Using {config_autoname} as experiment name instead. ")  # noqa G004
            return_value["name"] = config_autoname

        return return_value

    def _compute_config_hash(self, config: dict[str, Any]) -> str:
        exclude_keys = ["name", "concurrent_requests", "hash"]
        # Underscored properties store technical metadata with no impact on result.
        filtered = {k: v for k, v in config.items() if k not in exclude_keys and not k.startswith("_")}

        canonical = json.dumps(filtered, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def _compute_dataset_hash(self, data: list[dict[str, Any]]) -> str:
        # Runs at ~200 MB/s.
        # If that's too slow, switch to xxh64.
        hasher = hashlib.sha256()
        for item in data:
            hasher.update(json.dumps(item, sort_keys=True, default=str, separators=(",", ":")).encode())
        return hasher.hexdigest()[:16]

    def _validate_config_schema(self, config: dict[str, Any]) -> None:
        schema = {
            "model": str,
            "connector": str,
            "concurrent_requests": int,
            "prompt": dict,
        }

        for key, expected_type in schema.items():
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")  # noqa
            # Convert to the right data type in-place
            try:
                config[key] = expected_type(config[key])
            except (ValueError, TypeError) as err:
                raise ValueError(  # noqa
                    f"Config key {key} must be of type {expected_type.__name__}, got {config[key]}"  # noqa
                ) from err

    def _verify_config_hash(self, config: dict[str, Any]) -> None:
        run_config_path = self.base_path / "configs" / f"{config['name']}.config"
        # Is the config file itself copied to output directory yet?
        if run_config_path.exists():
            # Check whether we're overwriting
            # an existing config with different settings
            if not self.get_last_run_id(config):
                # no runs yet, it's fine
                logger.info(f"Updating config: {run_config_path}")  # noqa G004
            else:
                # Do hashes match?
                existing_config = self._load_config_from_path(config_path=run_config_path)
                if "hash" not in existing_config:
                    existing_config["hash"] = self._compute_config_hash(config=existing_config)
                if existing_config["hash"] != config["hash"]:
                    # fmt: off

                    logger.warning(
                        "Found config with the same name, different contents" \
                        f" and previous results: {run_config_path}"  # noqa
                    )
                    run_container = self.base_path / "runs" / config["name"]
                    # ruff: noqa
                    raise InvalidRunStateError(
                        "Executing this would cause existing result contamination"
                        " with annotations obtained using different config settings."
                        f" Either rename the config or delete previous annotation runs stored in {run_container}"
                    )
                    # fmt: on

    def _save_config(self, config: dict[str, Any]) -> None:
        run_config_path = self.base_path / "configs" / f"{config['name']}.config"
        run_config_path.parent.mkdir(parents=True, exist_ok=True)

        # Name is auto-generated based on execution context
        # when loading config, do not store it
        # to avoid confusion if renaming later
        writable_config = config.copy()
        writable_config.pop("name", None)

        with Path.open(run_config_path, "w") as f:
            json.dump(writable_config, f, indent=2)

    def get_last_run_id(self, config: dict[str, Any]) -> str | None:
        base = Path(self.base_path) / "runs" / config["name"]

        if not base.exists():
            return None

        folders = [f.name for f in base.iterdir() if f.is_dir()]
        if not folders:
            return None

        return max(folders)
