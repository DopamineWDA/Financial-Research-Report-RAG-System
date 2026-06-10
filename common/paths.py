from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

APP_ROOT = REPO_ROOT / "app"
CLI_ROOT = REPO_ROOT / "cli"
COMMON_ROOT = REPO_ROOT / "common"
DOCS_ROOT = REPO_ROOT / "docs"
EVAL_ROOT = REPO_ROOT / "eval"
PREPROCESS_ROOT = REPO_ROOT / "preprocess"
RETRIEVAL_ROOT = REPO_ROOT / "retrieval"
SCRIPTS_ROOT = REPO_ROOT / "scripts"

DATA_ROOT = REPO_ROOT / "data"
PARSED_ROOT = REPO_ROOT / "data_parsed"
CHUNKED_ROOT = REPO_ROOT / "data_chunked"
INDEX_ROOT = REPO_ROOT / "indexes"
OUTPUT_ROOT = REPO_ROOT / "output"

QA_OUTPUT_ROOT = OUTPUT_ROOT / "qa"
RETRIEVAL_OUTPUT_ROOT = OUTPUT_ROOT / "retrieval"
DEEPDOC_OUTPUT_ROOT = OUTPUT_ROOT / "deepdoc_parse"
LEGACY_PARSE_OUTPUT_ROOT = OUTPUT_ROOT / "legacy_parse"

RAGFLOW_ROOT = WORKSPACE_ROOT / "ragflow"
DEFAULT_BGE_MODEL = os.environ.get("BGE_EMBED_MODEL", "BAAI/bge-large-zh-v1.5")
DEFAULT_RERANKER_MODEL = os.environ.get("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_VISION_MODEL = os.environ.get("RAG_VISION_MODEL", str(WORKSPACE_ROOT / "LLM" / "Qwen2.5-VL-3B-Instruct"))
DEFAULT_XINFERENCE_ENDPOINT = os.environ.get("XINFERENCE_ENDPOINT", "http://127.0.0.1:9997")
DEFAULT_XINFERENCE_MODEL = os.environ.get("XINFERENCE_MODEL", "qwen3-8b")


def resolve_repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)

