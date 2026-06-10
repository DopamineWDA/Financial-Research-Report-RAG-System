#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import ctypes
import json
import math
import os
import inspect
import importlib
import types
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.paths import DEFAULT_BGE_MODEL


DEFAULT_QWEN_PLUS_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_LOCAL_BGE_PATH = DEFAULT_BGE_MODEL


def preload_conda_libstdcpp() -> None:
    candidate = Path(sys.executable).resolve().parent.parent / "lib" / "libstdc++.so.6"
    if not candidate.exists():
        return
    try:
        ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return


def install_vertexai_shim() -> None:
    try:
        import langchain_community.chat_models.vertexai  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    module_name = "langchain_community.chat_models.vertexai"
    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: D401
        """Compatibility shim for ragas import-time isinstance checks."""

        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS on generated answers with faithfulness, answer relevancy, context precision/recall, and noise sensitivity."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to answers.json created by generate_ragas_answers.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "ragas_eval",
        help="Directory to save RAGAS evaluation results.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("XINFERENCE_ENDPOINT", "http://127.0.0.1:9997"),
        help="OpenAI-compatible endpoint base URL for the evaluator LLM. Ignored if --judge-api-base is set.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RAGAS_JUDGE_MODEL", "qwen-plus"),
        help="Evaluator model name / uid.",
    )
    parser.add_argument(
        "--judge",
        choices=["auto", "local", "qwen-plus"],
        default="qwen-plus",
        help="Quick preset for evaluator backend. 'qwen-plus' auto-fills DashScope-compatible base URL unless overridden.",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key for the local/openai-compatible evaluator client.")
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default=DEFAULT_QWEN_PLUS_BASE_URL,
        help="Optional OpenAI-compatible API base for remote judge models such as qwen-plus. Example: https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default="DASHSCOPE_API_KEY",
        help="Environment variable containing judge API key, or pass the key itself directly.",
    )
    parser.add_argument(
        "--embedding-model-path",
        default=DEFAULT_LOCAL_BGE_PATH,
        help="Embedding model path or model id for RAGAS metrics that require embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size passed to ragas.evaluate.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall", "noise_sensitivity"],
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        help="Metrics to run. Default excludes noise_sensitivity for faster local evaluation.",
    )
    parser.add_argument(
        "--answer-relevancy-strictness",
        type=int,
        default=1,
        help="How many reverse questions answer_relevancy asks the judge to generate. Use 1 for local/slow judges.",
    )
    parser.add_argument(
        "--llm-adapter",
        choices=["auto", "instructor", "litellm"],
        default="auto",
        help="Structured-output adapter passed to ragas.llm_factory.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=45,
        help="Per-request timeout in seconds passed to ragas RunConfig.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="How many times ragas retries a failed metric request.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent ragas workers.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional JSONL checkpoint path for resumable per-record evaluation.",
    )
    parser.add_argument(
        "--resume",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resume from checkpoint by skipping finished qids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N records. 0 means all records.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved judge configuration and preflight logs.",
    )
    parser.add_argument(
        "--probe-chat",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run one minimal chat completion before evaluation to verify the judge path is live.",
    )
    parser.add_argument(
        "--probe-ragas-structured",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run one minimal structured-output call through ragas.llm_factory before evaluation.",
    )
    return parser.parse_args()


def require_dependencies() -> None:
    preload_conda_libstdcpp()
    install_vertexai_shim()

    errors = {}
    for module_name in ("ragas", "datasets", "openai", "sentence_transformers", "langchain_community"):
        try:
            __import__(module_name)
        except Exception as exc:
            errors[module_name] = f"{type(exc).__name__}: {exc}"
    if errors:
        details = "\n".join(f"- {name}: {msg}" for name, msg in errors.items())
        raise SystemExit(
            "RAGAS dependency import check failed:\n"
            f"{details}\n"
            f"Current python: {sys.executable}\n"
            "Note: Xinference only provides the model service. This evaluation script still needs local Python deps.\n"
            "Suggested install:\n"
            "pip install ragas datasets openai sentence-transformers langchain-community"
        )


def normalize_openai_base_url(raw_endpoint: str) -> str:
    endpoint = raw_endpoint.rstrip("/")
    if endpoint.endswith("/v1/chat/completions"):
        endpoint = endpoint[: -len("/chat/completions")]
    elif not endpoint.endswith("/v1"):
        endpoint = endpoint + "/v1"
    return endpoint


def resolve_judge_base_url(args: argparse.Namespace) -> str:
    if args.judge == "qwen-plus" and not args.judge_api_base:
        return normalize_openai_base_url(DEFAULT_QWEN_PLUS_BASE_URL)
    if args.judge_api_base:
        return normalize_openai_base_url(args.judge_api_base)
    return normalize_openai_base_url(args.endpoint)


def resolve_judge_api_key(args: argparse.Namespace) -> str:
    if args.judge == "qwen-plus" and not args.judge_api_key_env and args.api_key == "EMPTY":
        raise RuntimeError(
            "judge=qwen-plus requires --judge-api-key-env (env name or direct key), unless you pass a usable --api-key explicitly."
        )
    if args.judge_api_key_env:
        env_value = os.environ.get(args.judge_api_key_env)
        if env_value:
            return env_value
        # If the value looks like an env var name but is unset, fail loudly instead of
        # treating the variable name itself as a secret.
        if args.judge_api_key_env.isupper() and "sk-" not in args.judge_api_key_env.lower():
            raise RuntimeError(
                f"Judge API key env var is not set: {args.judge_api_key_env}\n"
                "Please export it first, or pass the real key directly with --judge-api-key-env sk-..."
            )
        return args.judge_api_key_env
    return args.api_key


def validate_remote_judge_endpoint(base_url: str, model: str, api_key: str, timeout: float = 15.0) -> None:
    models_url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Remote judge endpoint validation failed.\n"
            f"base_url: {base_url}\n"
            f"models_url: {models_url}\n"
            f"http_status: {exc.code}\n"
            f"detail: {detail[:500]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Remote judge endpoint is unreachable.\n"
            f"base_url: {base_url}\n"
            f"models_url: {models_url}\n"
            f"reason: {exc}"
        ) from exc

    if response.status >= 400:
        raise RuntimeError(
            "Remote judge endpoint responded with an error.\n"
            f"base_url: {base_url}\n"
            f"models_url: {models_url}\n"
            f"status: {response.status}\n"
            f"body: {body[:500]}"
        )

    if model and model not in body:
        print(
            "Warning: remote endpoint is reachable, but the requested model name "
            f"'{model}' was not found in /models response. Please double-check --model."
        )


def validate_local_judge_endpoint(base_url: str, model: str, timeout: float = 5.0) -> None:
    models_url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Local judge endpoint is unreachable.\n"
            f"base_url: {base_url}\n"
            f"models_url: {models_url}\n"
            f"reason: {exc}\n"
            "Please start your local OpenAI-compatible model service first, "
            "or pass the correct --endpoint / --model."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Failed to validate local judge endpoint.\n"
            f"base_url: {base_url}\n"
            f"models_url: {models_url}\n"
            f"reason: {exc}"
        ) from exc

    if response.status >= 400:
        raise RuntimeError(
            "Local judge endpoint responded with an error.\n"
            f"base_url: {base_url}\n"
            f"models_url: {models_url}\n"
            f"status: {response.status}\n"
            f"body: {body[:500]}"
        )

    if model and model not in body:
        print(
            "Warning: local endpoint is reachable, but the requested model name "
            f"'{model}' was not found in /models response. Please double-check --model."
        )


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Invalid answers JSON, missing 'records': {path}")
    return records


def build_dataset(records: list[dict[str, Any]]):
    from datasets import Dataset

    rows = []
    for item in records:
        contexts = item.get("contexts", []) or item.get("retrieved_contexts", [])
        if not isinstance(contexts, list):
            contexts = [str(contexts)]
        contexts = [str(ctx).strip() for ctx in contexts if str(ctx).strip()]
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        reference = str(item.get("ground_truth", "") or item.get("reference", "")).strip()
        rows.append(
            {
                "qid": item.get("qid"),
                "query_type": item.get("query_type"),
                # Old RAGAS column names.
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference,
                # Newer RAGAS single-turn schema names.
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
            }
        )
    return Dataset.from_list(rows)


def _resolve_metric_factory(name: str) -> Any:
    from ragas import metrics as ragas_metrics

    legacy_aliases = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
        "noise_sensitivity": "NoiseSensitivity",
    }
    modern_classes = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "ResponseRelevancy",
        "context_precision": "LLMContextPrecisionWithReference",
        "context_recall": "LLMContextRecall",
        "noise_sensitivity": "NoiseSensitivity",
    }

    alias_name = legacy_aliases.get(name)
    if alias_name and hasattr(ragas_metrics, alias_name):
        candidate = getattr(ragas_metrics, alias_name)
        if not inspect.ismodule(candidate):
            return candidate

    class_name = modern_classes.get(name)
    if class_name and hasattr(ragas_metrics, class_name):
        candidate = getattr(ragas_metrics, class_name)
        if not inspect.ismodule(candidate):
            return candidate

    # Some ragas versions expose collections as submodules and place the usable
    # metric class/instance inside the module instead of on the parent namespace.
    collection_modules = {
        "faithfulness": "ragas.metrics.collections.faithfulness",
        "answer_relevancy": "ragas.metrics.collections.answer_relevancy",
        "context_precision": "ragas.metrics.collections.context_precision",
        "context_recall": "ragas.metrics.collections.context_recall",
        "noise_sensitivity": "ragas.metrics.collections.noise_sensitivity",
    }
    collection_exports = {
        "faithfulness": ["faithfulness", "Faithfulness"],
        "answer_relevancy": ["answer_relevancy", "AnswerRelevancy", "ResponseRelevancy"],
        "context_precision": ["context_precision", "ContextPrecision", "LLMContextPrecisionWithReference"],
        "context_recall": ["context_recall", "ContextRecall", "LLMContextRecall"],
        "noise_sensitivity": ["noise_sensitivity", "NoiseSensitivity"],
    }
    module_name = collection_modules.get(name)
    if module_name:
        try:
            collection_module = importlib.import_module(module_name)
        except Exception:
            collection_module = None
        if collection_module is not None:
            for export_name in collection_exports.get(name, []):
                if hasattr(collection_module, export_name):
                    candidate = getattr(collection_module, export_name)
                    if not inspect.ismodule(candidate):
                        return candidate

    raise AttributeError(f"Unsupported or missing RAGAS metric: {name}")


def _instantiate_metric(factory_or_metric: Any) -> Any:
    if inspect.isclass(factory_or_metric):
        return factory_or_metric()
    return copy.deepcopy(factory_or_metric)


def make_metrics(metric_names: list[str], answer_relevancy_strictness: int) -> list[Any]:
    metrics: list[Any] = []
    for metric_name in metric_names:
        metric = _instantiate_metric(_resolve_metric_factory(metric_name))
        # Normalize names so saved output remains stable across ragas versions.
        if hasattr(metric, "name"):
            metric.name = metric_name
        if metric_name == "answer_relevancy" and hasattr(metric, "strictness"):
            metric.strictness = answer_relevancy_strictness
        metrics.append(metric)
    return metrics


def evaluation_result_to_plain(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_dict"):
        try:
            return result.to_dict()
        except Exception:
            pass
    if hasattr(result, "_repr_dict") and isinstance(getattr(result, "_repr_dict"), dict):
        return dict(getattr(result, "_repr_dict"))
    if hasattr(result, "scores"):
        try:
            scores = getattr(result, "scores")
            if isinstance(scores, list) and scores:
                summary: dict[str, float] = {}
                metric_names = list(scores[0].keys())
                for metric_name in metric_names:
                    values = []
                    for row in scores:
                        try:
                            values.append(float(row[metric_name]))
                        except Exception:
                            continue
                    if values:
                        summary[metric_name] = sum(values) / len(values)
                if summary:
                    return summary
        except Exception:
            pass
    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            if hasattr(df, "mean"):
                series = df.mean(numeric_only=True)
                if hasattr(series, "to_dict"):
                    return series.to_dict()
        except Exception:
            pass
    if isinstance(result, dict):
        return result
    try:
        return dict(result)
    except Exception:
        pass
    if hasattr(result, "__iter__"):
        try:
            return {k: v for k, v in result}
        except Exception:
            pass
    raise TypeError(f"Unable to normalize evaluation result of type {type(result).__name__}")


def normalize_metrics(raw: dict[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in raw.items():
        try:
            numeric = float(value)
        except Exception:
            continue
        if math.isnan(numeric):
            continue
        normalized[str(key)] = numeric
    return normalized


def build_run_slug(args: argparse.Namespace) -> str:
    safe_metrics = "-".join(args.metrics)
    safe_model = str(args.model).replace("/", "_")
    safe_adapter = str(args.llm_adapter).replace("/", "_")
    return "__".join(
        [
            args.input_json.stem,
            safe_model,
            safe_metrics,
            f"adapter-{safe_adapter}",
            f"judge-{args.judge}",
        ]
    )


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return args.checkpoint_path
    return args.output_dir / f"ragas_eval_checkpoint__{build_run_slug(args)}.jsonl"


def load_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def append_checkpoint_row(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def aggregate_metric_rows(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in metric_names:
        values: list[float] = []
        for row in rows:
            try:
                numeric = float(row["metrics"][metric_name])
            except Exception:
                continue
            if math.isnan(numeric):
                continue
            values.append(numeric)
        if values:
            summary[metric_name] = sum(values) / len(values)
    return summary


def filter_records(records: Iterable[dict[str, Any]], query_type: str) -> list[dict[str, Any]]:
    return [item for item in records if item.get("query_type") == query_type]


def run_eval_on_records(
    *,
    records: list[dict[str, Any]],
    llm: Any,
    embeddings: Any,
    batch_size: int,
    metric_names: list[str],
    answer_relevancy_strictness: int,
    request_timeout: int,
    max_retries: int,
    max_workers: int,
) -> dict[str, float]:
    from ragas import evaluate
    from ragas.run_config import RunConfig

    dataset = build_dataset(records)
    metrics = make_metrics(metric_names, answer_relevancy_strictness)
    run_config = RunConfig(
        timeout=request_timeout,
        max_retries=max_retries,
        max_workers=max_workers,
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        batch_size=batch_size,
        raise_exceptions=False,
        show_progress=True,
    )
    return normalize_metrics(evaluation_result_to_plain(result))


def save_result(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"ragas_eval__{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def save_checkpoint_summary(output_dir: Path, run_slug: str, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"ragas_eval_rows__{run_slug}.json"
    file_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def build_hf_embeddings(model_path: str) -> Any:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def probe_chat_completion(client: Any, model: str) -> None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请只回复：ok"},
            ],
            temperature=0,
            max_tokens=16,
        )
    except Exception as exc:
        raise RuntimeError(
            "Judge probe chat failed before RAGAS evaluation started.\n"
            f"model: {model}\n"
            f"reason: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        content = response.choices[0].message.content
    except Exception:
        content = "<unreadable>"
    print(f"[probe-chat] model={model} response={content!r}")


class _StructuredProbePayload(BaseModel):
    status: str


def probe_ragas_structured_output(llm: Any) -> None:
    prompt = "Return JSON with one field named status and set it to ok."
    try:
        response = llm.generate(prompt, _StructuredProbePayload)
    except Exception as exc:
        raise RuntimeError(
            "RAGAS structured-output probe failed before evaluation started.\n"
            "This usually means the selected ragas adapter cannot talk to the configured OpenAI-compatible backend.\n"
            f"reason: {type(exc).__name__}: {exc}"
        ) from exc

    print(f"[probe-ragas] structured status={getattr(response, 'status', '<missing>')!r}")


def main() -> None:
    args = parse_args()
    require_dependencies()

    from openai import OpenAI
    from ragas.llms import llm_factory

    records = load_records(args.input_json)
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    run_slug = build_run_slug(args)
    checkpoint_path = resolve_checkpoint_path(args)
    base_url = resolve_judge_base_url(args)
    api_key = resolve_judge_api_key(args)
    if args.verbose:
        print(f"[config] judge={args.judge}")
        print(f"[config] model={args.model}")
        print(f"[config] base_url={base_url}")
        print(f"[config] input_records={len(records)}")
        print(f"[config] metrics={args.metrics}")
        print(f"[config] embedding_model_path={args.embedding_model_path}")
        print(f"[config] checkpoint_path={checkpoint_path}")
    if args.judge == "qwen-plus":
        validate_remote_judge_endpoint(base_url, args.model, api_key)
        if args.verbose:
            print("[preflight] remote /models validation passed")
    else:
        validate_local_judge_endpoint(base_url, args.model)
        if args.verbose:
            print("[preflight] local /models validation passed")
    client = OpenAI(api_key=api_key, base_url=base_url)
    if args.probe_chat:
        probe_chat_completion(client, args.model)
    llm = llm_factory(
        args.model,
        provider="openai",
        client=client,
        adapter=args.llm_adapter,
        system_prompt="You are a helpful assistant that evaluates RAG systems.",
    )
    if args.verbose:
        print("[preflight] llm_factory ok")
        print(f"[preflight] ragas_llm_adapter={args.llm_adapter}")
        print(f"[preflight] request_timeout={args.request_timeout}s max_retries={args.max_retries} max_workers={args.max_workers}")
    if args.probe_ragas_structured:
        probe_ragas_structured_output(llm)
    embeddings = build_hf_embeddings(args.embedding_model_path)
    if args.verbose:
        print("[preflight] embeddings ready")
    checkpoint_rows = load_checkpoint_rows(checkpoint_path) if args.resume else []
    checkpoint_by_qid = {
        str(row.get("qid")): row
        for row in checkpoint_rows
        if isinstance(row, dict) and row.get("qid") is not None and isinstance(row.get("metrics"), dict)
    }
    if args.verbose and checkpoint_by_qid:
        print(f"[resume] loaded_completed={len(checkpoint_by_qid)}")

    total_records = len(records)
    for idx, record in enumerate(records, start=1):
        qid = str(record.get("qid"))
        if args.resume and qid in checkpoint_by_qid:
            if args.verbose:
                print(f"[resume] skip {idx}/{total_records} qid={qid}")
            continue

        metrics = run_eval_on_records(
            records=[record],
            llm=llm,
            embeddings=embeddings,
            batch_size=1,
            metric_names=args.metrics,
            answer_relevancy_strictness=args.answer_relevancy_strictness,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            max_workers=1,
        )
        checkpoint_row = {
            "qid": qid,
            "query_type": record.get("query_type"),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        append_checkpoint_row(checkpoint_path, checkpoint_row)
        checkpoint_by_qid[qid] = checkpoint_row
        print(f"[checkpoint] {idx}/{total_records} qid={qid} metrics={json.dumps(metrics, ensure_ascii=False)}")

    completed_rows = []
    for record in records:
        qid = str(record.get("qid"))
        row = checkpoint_by_qid.get(qid)
        if row is not None:
            completed_rows.append(row)

    overall = aggregate_metric_rows(completed_rows, args.metrics)
    by_type = {}
    for query_type in ("fact", "compare", "summary"):
        subset_rows = [row for row in completed_rows if row.get("query_type") == query_type]
        if subset_rows:
            by_type[query_type] = aggregate_metric_rows(subset_rows, args.metrics)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_json": str(args.input_json),
        "evaluator_model": args.model,
        "endpoint": base_url,
        "judge_preset": args.judge,
        "judge_mode": "remote_openai_compatible" if (args.judge_api_base or args.judge == "qwen-plus") else "local_openai_compatible",
        "embedding_model_path": args.embedding_model_path,
        "metrics": args.metrics,
        "answer_relevancy_strictness": args.answer_relevancy_strictness,
        "checkpoint_path": str(checkpoint_path),
        "completed_records": len(completed_rows),
        "requested_records": len(records),
        "metric_notes": {
            "faithfulness": "Higher is better.",
            "answer_relevancy": "Higher is better.",
            "context_precision": "Higher is better.",
            "context_recall": "Higher is better.",
            "noise_sensitivity": "Lower is better.",
        },
        "overall": overall,
        "by_type": by_type,
    }
    saved = save_result(args.output_dir, payload)
    rows_saved = save_checkpoint_summary(args.output_dir, run_slug, completed_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved checkpoint rows: {rows_saved}")
    print(f"\nSaved RAGAS result: {saved}")


if __name__ == "__main__":
    main()
