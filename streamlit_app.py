#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
PARSED_ROOT = REPO_ROOT / "data_parsed"
INDEX_ROOT = REPO_ROOT / "indexes"
DEFAULT_ENDPOINT = os.environ.get("XINFERENCE_ENDPOINT", "http://127.0.0.1:9997")
DEFAULT_MODEL = os.environ.get("XINFERENCE_MODEL", "qwen3-8b")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.answer_query import QAApp, ConversationTurn  # noqa: E402


def safe_stem(stem: str) -> str:
    stem = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", stem)
    return stem[:120] or "parsed_pdf"


def iter_index_dirs() -> list[Path]:
    return sorted(
        [
            path for path in INDEX_ROOT.iterdir()
            if path.is_dir()
            and (path / "build_meta.json").exists()
            and (path / "index.faiss").exists()
            and (path / "docstore.jsonl").exists()
        ]
    )


def preferred_default_index(index_dirs: list[Path]) -> Path:
    for path in index_dirs:
        if "hnsw" in path.name.lower():
            return path
    return index_dirs[0]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_chunk_config(input_dir: Path) -> tuple[int, int]:
    match = re.search(r"chunked_(\d+)_(\d+)$", input_dir.name)
    if not match:
        raise ValueError(f"无法从目录名推断 chunk 配置: {input_dir}")
    return int(match.group(1)), int(match.group(2))


def normalize_index_type(index_type: str) -> str:
    mapping = {
        "IndexFlatIP": "flat",
        "IndexIVFFlat": "ivf",
        "IndexHNSWFlat": "hnsw",
    }
    return mapping.get(index_type, index_type.lower())


def resolve_output_index_dir(input_dir: Path, index_type: str) -> Path:
    suffix = input_dir.name
    return INDEX_ROOT / f"faiss_{index_type}_{suffix}_bge-large-zh-v1.5"


def build_app_args(
    *,
    index_dir: Path,
    endpoint: str,
    model: str,
    history_rounds: int,
    rerank_top_k: int,
    first_stage_top_k: int,
    route: str,
    vector_weight: float,
    bm25_weight: float,
    rrf_k: int,
    rerank_threshold: float,
    query_decompose: bool,
    show_used_evidence_text: bool,
) -> Namespace:
    return Namespace(
        query=None,
        index_dir=index_dir,
        endpoint=endpoint,
        model=model,
        api_key=None,
        route=route,
        first_stage_top_k=first_stage_top_k,
        rerank_top_k=rerank_top_k,
        vector_candidate_k=first_stage_top_k,
        bm25_candidate_k=first_stage_top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        title_weight=1.0,
        rrf_k=rrf_k,
        rerank_model_path="BAAI/bge-reranker-v2-m3",
        reranker_use_fp16=True,
        rerank_threshold=rerank_threshold,
        min_valid_claims=1,
        max_evidence_chars=900,
        show_text_chars=9999,
        hide_retrieval_chunks=True,
        show_used_evidence_text=show_used_evidence_text,
        query_decompose=query_decompose,
        show_timings=False,
        interactive=False,
        history_rounds=history_rounds,
        output_json_dir=REPO_ROOT / "output" / "qa",
        json=False,
    )


def app_config_key(args: Namespace) -> str:
    payload = {
        "index_dir": str(args.index_dir),
        "endpoint": args.endpoint,
        "model": args.model,
        "history_rounds": args.history_rounds,
        "rerank_top_k": args.rerank_top_k,
        "first_stage_top_k": args.first_stage_top_k,
        "route": args.route,
        "vector_weight": args.vector_weight,
        "bm25_weight": args.bm25_weight,
        "rrf_k": args.rrf_k,
        "rerank_threshold": args.rerank_threshold,
        "query_decompose": args.query_decompose,
        "show_used_evidence_text": args.show_used_evidence_text,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@st.cache_resource(show_spinner=False)
def load_cached_app(config_key: str, args_dict: dict[str, Any]) -> QAApp:
    normalized = dict(args_dict)
    normalized["index_dir"] = Path(normalized["index_dir"])
    normalized["output_json_dir"] = Path(normalized["output_json_dir"])
    return QAApp(Namespace(**normalized))


def get_or_create_app(args: Namespace) -> QAApp:
    config_key = app_config_key(args)
    if st.session_state.get("qa_app_config_key") != config_key or "qa_app" not in st.session_state:
        args_dict = vars(args).copy()
        args_dict["index_dir"] = str(args.index_dir)
        args_dict["output_json_dir"] = str(args.output_json_dir)
        st.session_state.qa_app = load_cached_app(config_key, args_dict)
        st.session_state.qa_app_config_key = config_key
        turns: list[ConversationTurn] = []
        pending_user: str | None = None
        for message in st.session_state.get("chat_messages", []):
            if message.get("role") == "user":
                pending_user = message.get("content", "")
            elif message.get("role") == "assistant" and pending_user:
                turns.append(ConversationTurn(user=pending_user, assistant=message.get("content", "")))
                pending_user = None
        st.session_state.qa_app.history = turns[-args.history_rounds :] if args.history_rounds > 0 else []
    return st.session_state.qa_app


def invalidate_loaded_app() -> None:
    for key in ["qa_app", "qa_app_config_key"]:
        st.session_state.pop(key, None)


def list_pdf_roots() -> list[Path]:
    return sorted([path for path in DATA_ROOT.iterdir() if path.is_dir() and path.name.endswith("_pdfs")])


def list_index_documents(index_dir: Path) -> list[dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    with (index_dir / "docstore.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            source_pdf = record.get("source_pdf") or ""
            doc_id = record.get("doc_id") or Path(source_pdf).stem
            if source_pdf and source_pdf not in docs:
                docs[source_pdf] = {"source_pdf": source_pdf, "doc_id": doc_id}
    return sorted(docs.values(), key=lambda item: item["doc_id"])


def run_command(cmd: list[str], *, cwd: Path) -> str:
    env = os.environ.copy()
    env_prefix = Path(sys.executable).resolve().parent.parent
    env_lib = env_prefix / "lib"
    if env_lib.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{env_lib}:{current}" if current else str(env_lib)
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if completed.returncode != 0:
        output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}"
            + (f"\n\n{output}" if output else "")
        )
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    return output.strip()


def parse_uploaded_pdf(pdf_path: Path, relative_parent: Path) -> Path:
    if not pdf_path.exists():
        raise FileNotFoundError(f"上传文件未落盘成功: {pdf_path}")
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "preprocess" / "deepdoc_parser.py"),
            "--input",
            str(pdf_path),
            "--output-dir",
            str(PARSED_ROOT),
            "--preserve-root",
            str(DATA_ROOT),
            "--json-only",
        ],
        cwd=REPO_ROOT,
    )
    parsed_path = PARSED_ROOT / relative_parent / f"{safe_stem(pdf_path.stem)}.parsed.json"
    if not parsed_path.exists():
        raise FileNotFoundError(f"解析完成后未找到 parsed 文件: {parsed_path}")
    return parsed_path


def chunk_parsed_pdf(parsed_path: Path, chunk_output_dir: Path, chunk_size: int, overlap: int) -> Path:
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "preprocess" / "rechunk_parsed_pdf.py"),
            str(parsed_path),
            "--output-dir",
            str(chunk_output_dir),
            "--chunk-size",
            str(chunk_size),
            "--overlap",
            str(overlap),
        ],
        cwd=REPO_ROOT,
    )
    chunk_path = chunk_output_dir / parsed_path.name.replace(".parsed.json", f".chunks.cs{chunk_size}.ov{overlap}.json")
    if not chunk_path.exists():
        raise FileNotFoundError(f"切块完成后未找到 chunk 文件: {chunk_path}")
    return chunk_path


def rebuild_index(index_dir: Path, target_index_type: str) -> tuple[str, Path]:
    meta = load_json(index_dir / "build_meta.json")
    params = meta.get("index_build_params", {})
    input_dir = Path(meta["input_dir"])
    output_dir = resolve_output_index_dir(input_dir, target_index_type)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_faiss_index.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model-path",
        str(meta["model_path"]),
        "--batch-size",
        str(meta.get("batch_size", 32)),
        "--max-length",
        str(meta.get("max_length", 512)),
        "--index-type",
        target_index_type,
        "--include-chunk-types",
        *meta.get("included_chunk_types", ["text_chunk", "raw_table_chunk"]),
    ]
    if meta.get("use_fp16"):
        cmd.append("--use-fp16")
    if not meta.get("normalize_embeddings", True):
        cmd.append("--no-normalize")
    if target_index_type == "ivf":
        cmd.extend(["--ivf-nlist", str(params.get("nlist", 0))])
        cmd.extend(["--ivf-nprobe", str(params.get("nprobe", 16))])
    if target_index_type == "hnsw":
        cmd.extend(["--hnsw-m", str(params.get("hnsw_m", 32))])
        cmd.extend(["--hnsw-ef-construction", str(params.get("ef_construction", 100))])
        cmd.extend(["--hnsw-ef-search", str(params.get("ef_search", 64))])
    return run_command(cmd, cwd=REPO_ROOT), output_dir


def delete_pdf_related_files(index_dir: Path, pdf_paths: list[str]) -> list[str]:
    meta = load_json(index_dir / "build_meta.json")
    chunk_dir = Path(meta["input_dir"])
    removed: list[str] = []
    pdf_set = set(pdf_paths)
    for pdf_str in pdf_paths:
        pdf_path = Path(pdf_str)
        if pdf_path.exists():
            pdf_path.unlink()
            removed.append(str(pdf_path))
        try:
            relative_parent = pdf_path.parent.relative_to(DATA_ROOT)
        except ValueError:
            relative_parent = Path()
        parsed_path = PARSED_ROOT / relative_parent / f"{safe_stem(pdf_path.stem)}.parsed.json"
        if parsed_path.exists():
            parsed_path.unlink()
            removed.append(str(parsed_path))

    for chunk_file in chunk_dir.glob("*.json"):
        try:
            payload = load_json(chunk_file)
        except Exception:
            continue
        if payload.get("source") in pdf_set:
            chunk_file.unlink()
            removed.append(str(chunk_file))
    return removed


def handle_uploads(index_dir: Path, target_pdf_dir: Path, uploaded_files: list[Any], rebuild_index_type: str) -> tuple[str, Path]:
    meta = load_json(index_dir / "build_meta.json")
    chunk_dir = Path(meta["input_dir"])
    chunk_size, overlap = parse_chunk_config(chunk_dir)
    try:
        relative_parent = target_pdf_dir.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(f"上传目录必须位于 {DATA_ROOT}") from exc

    logs: list[str] = []
    target_pdf_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        target_path = target_pdf_dir / Path(uploaded.name).name
        if target_path.exists():
            raise FileExistsError(f"文件已存在，请先删除或改名后再上传: {target_path.name}")
        target_path.write_bytes(uploaded.getbuffer())
        logs.append(f"uploaded: {target_path}")
        parsed_path = parse_uploaded_pdf(target_path, relative_parent)
        logs.append(f"parsed: {parsed_path}")
        chunk_path = chunk_parsed_pdf(parsed_path, chunk_dir, chunk_size, overlap)
        logs.append(f"chunked: {chunk_path}")

    rebuild_log, rebuilt_index_dir = rebuild_index(index_dir, rebuild_index_type)
    logs.append(rebuild_log)
    return "\n".join(logs), rebuilt_index_dir


def handle_delete(index_dir: Path, selected_pdfs: list[str], rebuild_index_type: str) -> tuple[str, Path]:
    removed = delete_pdf_related_files(index_dir, selected_pdfs)
    rebuild_log, rebuilt_index_dir = rebuild_index(index_dir, rebuild_index_type)
    return "\n".join(removed + [rebuild_log]), rebuilt_index_dir


def render_sources(payload: dict[str, Any]) -> None:
    answer = payload.get("answer", {})
    evidences = payload.get("evidences", [])
    evidence_by_id = {item["evidence_id"]: item for item in evidences}
    used = answer.get("used_evidence", [])
    if not payload.get("retrieval_used"):
        return

    st.markdown("**引用来源**")
    claims = answer.get("claims", [])
    if claims:
        for idx, claim in enumerate(claims, start=1):
            citations = "、".join(claim.get("citations", [])) or "无"
            st.markdown(f"{idx}. {claim.get('claim', '')}  `[{citations}]`")
    else:
        st.caption("本次回答没有结构化 claims。")

    evidence_ids = used or [item.get("evidence_id") for item in evidences]
    for evidence_id in evidence_ids:
        item = evidence_by_id.get(evidence_id)
        if not item:
            continue
        filename = Path(item["source_pdf"]).name
        page_start = item.get("page_start") or "?"
        page_end = item.get("page_end") or "?"
        pages = f"{page_start}-{page_end}" if page_start != page_end else f"{page_start}"
        st.markdown(f"- `{evidence_id}` {filename}，第 {pages} 页")
        with st.expander(
            f"查看 {evidence_id} evidence chunk",
            expanded=False,
        ):
            st.write(f"chunk_type: {item.get('chunk_type')}")
            st.write(f"reranker_score: {item.get('reranker_score', 0):.4f}")
            st.write(item.get("text", ""))


def render_query_timings(payload: dict[str, Any]) -> None:
    timings = payload.get("timings") or []
    if not timings:
        return

    timing_map = {
        str(item.get("stage", "")): float(item.get("seconds", 0.0))
        for item in timings
        if item.get("stage")
    }
    retrieve_seconds = timing_map.get("retrieve_two_stage", 0.0)
    generation_seconds = timing_map.get("rag_generation", timing_map.get("direct_generation", 0.0))
    total_seconds = timing_map.get("wall_clock_total", 0.0)
    st.caption(
        f"耗时: 检索 {retrieve_seconds:.2f}s | 生成 {generation_seconds:.2f}s | 总计 {total_seconds:.2f}s"
    )


st.set_page_config(page_title="金融 RAG Demo", page_icon=":bar_chart:", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 980px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.75;
        font-size: 1.02rem;
    }
    .rag-toolbar {
        padding: 0.8rem 1rem;
        border: 1px solid rgba(49, 51, 63, 0.10);
        border-radius: 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .rag-caption {
        color: #667085;
        font-size: 0.94rem;
        margin-top: 0.2rem;
    }
    .new-chat-note {
        color: #667085;
        font-size: 0.88rem;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initial_chat_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "你好，我是你的金融 RAG 助手。你可以直接提问；如果需要检索知识库，我会同时展示引用文件、页码和使用到的 evidence chunk。",
            "payload": None,
        }
    ]


def default_applied_settings(default_index_dir: Path) -> dict[str, Any]:
    return {
        "selected_index_dir": str(default_index_dir),
        "endpoint": DEFAULT_ENDPOINT,
        "model": DEFAULT_MODEL,
        "route": "hybrid_weightsum",
        "history_rounds": 6,
        "rerank_top_k": 8,
        "first_stage_top_k": 50,
        "vector_weight": 0.7,
        "rrf_k": 60,
        "rerank_threshold": 0.5,
        "query_decompose": True,
        "show_used_evidence_text": True,
    }


def draft_key(name: str) -> str:
    return f"draft_{name}"


def sync_draft_settings(applied: dict[str, Any]) -> None:
    for key, value in applied.items():
        st.session_state[draft_key(key)] = value


st.title("📝金融研报RAG问答助手")

index_dirs = iter_index_dirs()
if not index_dirs:
    st.error("未找到可用索引目录。请先在 `RAG/indexes/` 下构建索引。")
    st.stop()
default_index_dir = preferred_default_index(index_dirs)
available_index_values = [str(path) for path in index_dirs]
applied_settings = st.session_state.get("applied_settings")
if applied_settings is None:
    applied_settings = default_applied_settings(default_index_dir)
    st.session_state.applied_settings = dict(applied_settings)
    sync_draft_settings(applied_settings)
else:
    applied_settings = dict(applied_settings)
    defaults = default_applied_settings(default_index_dir)
    for key, value in defaults.items():
        applied_settings.setdefault(key, value)
    if applied_settings["selected_index_dir"] not in available_index_values:
        applied_settings["selected_index_dir"] = str(default_index_dir)
    st.session_state.applied_settings = dict(applied_settings)
    if draft_key("selected_index_dir") not in st.session_state:
        sync_draft_settings(applied_settings)

pending_selected_index_dir = st.session_state.pop("pending_selected_index_dir", None)
if pending_selected_index_dir is not None:
    updated_settings = dict(st.session_state.applied_settings)
    updated_settings["selected_index_dir"] = pending_selected_index_dir
    st.session_state.applied_settings = updated_settings
    sync_draft_settings(updated_settings)

applied_settings = dict(st.session_state.applied_settings)
if applied_settings["selected_index_dir"] not in available_index_values:
    applied_settings["selected_index_dir"] = str(default_index_dir)
    st.session_state.applied_settings = dict(applied_settings)
    sync_draft_settings(applied_settings)

index_dir = Path(applied_settings["selected_index_dir"])

with st.sidebar:
    if st.button("＋ New Chat", use_container_width=True):
        for key in [
            "chat_messages",
            "admin_log",
            "last_result",
            "show_used_evidence_text",
            "engine_loaded",
            "engine_autoload_requested",
        ]:
            st.session_state.pop(key, None)
        invalidate_loaded_app()
        st.session_state.chat_messages = initial_chat_messages()
        st.rerun()
    st.markdown('<div class="new-chat-note">清空当前对话，但保留侧边栏配置。</div>', unsafe_allow_html=True)
    st.header("⚙️问答设置")
    with st.form("qa_settings_form", border=False):
        st.selectbox(
            "索引目录",
            available_index_values,
            key=draft_key("selected_index_dir"),
        )
        st.text_input("Xinference Endpoint", key=draft_key("endpoint"))
        st.text_input("模型 UID", key=draft_key("model"))
        st.selectbox("检索路由", ["hybrid_weightsum", "hybrid_rrf"], key=draft_key("route"))
        st.slider("历史对话轮数", min_value=0, max_value=10, step=1, key=draft_key("history_rounds"))
        st.slider("匹配知识条数", min_value=1, max_value=20, step=1, key=draft_key("rerank_top_k"))
        st.slider("first-stage-top-k", min_value=10, max_value=100, step=5, key=draft_key("first_stage_top_k"))
        st.slider("vector-weight", min_value=0.0, max_value=1.0, step=0.05, key=draft_key("vector_weight"))
        draft_bm25_weight = round(1.0 - float(st.session_state[draft_key("vector_weight")]), 2)
        st.caption(f"bm25-weight = {draft_bm25_weight:.2f}，当前两者和为 1")
        st.number_input("rrf-k", min_value=1, max_value=500, step=1, key=draft_key("rrf_k"))
        st.slider("拒答阈值", min_value=0.0, max_value=1.0, step=0.05, key=draft_key("rerank_threshold"))
        save_settings_clicked = st.form_submit_button("保存设置并重新加载", use_container_width=True)

    st.divider()
    st.header("向量库管理")
    rebuild_index_type = st.segmented_control(
        "重建索引类型",
        options=["hnsw", "flat"],
        default="hnsw",
        help="上传/删除文件后，按这里选择的索引类型重建。",
    )
    target_pdf_dir = Path(
        st.selectbox(
            "上传目标 PDF 目录",
            [str(path) for path in list_pdf_roots()],
        )
    )
    uploaded_files = st.file_uploader(
        "上传新 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    upload_clicked = st.button("上传并重建索引", use_container_width=True)

    docs = list_index_documents(index_dir)
    delete_options = {f"{item['doc_id']} | {Path(item['source_pdf']).name}": item["source_pdf"] for item in docs}
    selected_delete_labels = st.multiselect("删除已有文件", options=list(delete_options.keys()))
    delete_clicked = st.button("删除并重建索引", use_container_width=True)

if save_settings_clicked:
    updated_settings = {
        "selected_index_dir": st.session_state[draft_key("selected_index_dir")],
        "endpoint": st.session_state[draft_key("endpoint")],
        "model": st.session_state[draft_key("model")],
        "route": st.session_state[draft_key("route")],
        "history_rounds": int(st.session_state[draft_key("history_rounds")]),
        "rerank_top_k": int(st.session_state[draft_key("rerank_top_k")]),
        "first_stage_top_k": int(st.session_state[draft_key("first_stage_top_k")]),
        "vector_weight": float(st.session_state[draft_key("vector_weight")]),
        "rrf_k": int(st.session_state[draft_key("rrf_k")]),
        "rerank_threshold": float(st.session_state[draft_key("rerank_threshold")]),
        "query_decompose": applied_settings.get("query_decompose", True),
        "show_used_evidence_text": applied_settings.get("show_used_evidence_text", True),
    }
    st.session_state.applied_settings = updated_settings
    invalidate_loaded_app()
    st.session_state.engine_loaded = False
    st.session_state.engine_autoload_requested = False
    st.rerun()

applied_settings = dict(st.session_state.applied_settings)
index_dir = Path(applied_settings["selected_index_dir"])
endpoint = str(applied_settings["endpoint"])
model = str(applied_settings["model"])
route = str(applied_settings["route"])
history_rounds = int(applied_settings["history_rounds"])
rerank_top_k = int(applied_settings["rerank_top_k"])
first_stage_top_k = int(applied_settings["first_stage_top_k"])
vector_weight = float(applied_settings["vector_weight"])
bm25_weight = round(1.0 - vector_weight, 2)
rrf_k = int(applied_settings["rrf_k"])
rerank_threshold = float(applied_settings["rerank_threshold"])

args = build_app_args(
    index_dir=index_dir,
    endpoint=endpoint,
    model=model,
    history_rounds=history_rounds,
    rerank_top_k=rerank_top_k,
    first_stage_top_k=first_stage_top_k,
    route=route,
    vector_weight=vector_weight,
    bm25_weight=bm25_weight,
    rrf_k=int(rrf_k),
    rerank_threshold=rerank_threshold,
    query_decompose=bool(applied_settings.get("query_decompose", True)),
    show_used_evidence_text=bool(applied_settings.get("show_used_evidence_text", True)),
)
current_config_key = app_config_key(args)
if st.session_state.get("qa_app_config_key") not in {None, current_config_key}:
    invalidate_loaded_app()
    st.session_state.engine_loaded = False
    st.session_state.engine_autoload_requested = False

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = initial_chat_messages()
if "admin_log" not in st.session_state:
    st.session_state.admin_log = ""
if "engine_loaded" not in st.session_state:
    st.session_state.engine_loaded = False
if "engine_autoload_requested" not in st.session_state:
    st.session_state.engine_autoload_requested = False
st.session_state.show_used_evidence_text = True

if upload_clicked:
    if not uploaded_files:
        st.warning("请先选择至少一个 PDF 文件。")
    else:
        with st.spinner("正在解析、切块并重建索引，这一步可能需要一些时间..."):
            try:
                log_text, rebuilt_index_dir = handle_uploads(index_dir, target_pdf_dir, uploaded_files, rebuild_index_type)
            except Exception as exc:
                st.error(f"上传失败: {exc}")
            else:
                st.session_state.admin_log = log_text or "(no logs)"
                st.session_state.pending_selected_index_dir = str(rebuilt_index_dir)
                invalidate_loaded_app()
                st.session_state.engine_loaded = False
                st.session_state.engine_autoload_requested = False
                st.rerun()

if delete_clicked:
    selected_pdfs = [delete_options[label] for label in selected_delete_labels]
    if not selected_pdfs:
        st.warning("请先选择要删除的文件。")
    else:
        with st.spinner("正在删除文件并重建索引..."):
            try:
                log_text, rebuilt_index_dir = handle_delete(index_dir, selected_pdfs, rebuild_index_type)
            except Exception as exc:
                st.error(f"删除失败: {exc}")
            else:
                st.session_state.admin_log = log_text or "(no logs)"
                st.session_state.pending_selected_index_dir = str(rebuilt_index_dir)
                invalidate_loaded_app()
                st.session_state.engine_loaded = False
                st.session_state.engine_autoload_requested = False
                st.rerun()

if st.session_state.admin_log:
    st.success("最近一次向量库操作已完成。")
    with st.expander("查看操作日志", expanded=False):
        st.code(st.session_state.admin_log)

st.markdown('<div class="rag-toolbar">', unsafe_allow_html=True)
st.caption("侧边栏参数修改后，点击“保存设置并重新加载”才会生效。")
if not st.session_state.engine_loaded and not st.session_state.engine_autoload_requested:
    st.session_state.engine_autoload_requested = True
    st.rerun()

if st.session_state.engine_loaded and "qa_app" in st.session_state:
    app = st.session_state.qa_app
    startup_rows = [f"{item.stage}: {item.seconds:.3f}s" for item in app.startup_timings]
    st.markdown(
        """
        <div class="rag-caption">
        当前已保存配置对应的问答引擎已加载完成。修改侧边栏后需要先保存，随后会自动重新加载。
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("查看启动耗时", expanded=False):
        st.code("\n".join(startup_rows) if startup_rows else "(no timings)")
else:
    st.markdown(
        """
        <div class="rag-caption">
        正在为当前已保存配置自动加载问答引擎。
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.spinner("正在加载问答模型、检索缓存和应用配置..."):
        try:
            app = get_or_create_app(args)
            st.session_state.engine_loaded = True
            st.session_state.engine_autoload_requested = False
        except RuntimeError as exc:
            st.error(f"自动加载失败：{exc}")
            st.session_state.engine_autoload_requested = False
st.markdown("</div>", unsafe_allow_html=True)

st.caption("页面会在打开后自动完成当前已保存配置的引擎加载；后续提问不会再次重复加载。")

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        payload = message.get("payload")
        if message["role"] == "assistant" and payload:
            if payload.get("retrieval_used"):
                st.caption(
                    f"confidence={payload.get('answer', {}).get('confidence', 'low')} | "
                    f"should_refuse={payload.get('answer', {}).get('should_refuse', False)} | "
                    f"effective_query={payload.get('effective_query', '')}"
                )
            render_query_timings(payload)
            render_sources(payload)

prompt = st.chat_input("输入你的金融问题")
if prompt:
    if not st.session_state.engine_loaded or "qa_app" not in st.session_state:
        st.warning("问答引擎仍在自动加载中，请稍等片刻后再提问。")
        st.stop()
    st.session_state.chat_messages.append({"role": "user", "content": prompt, "payload": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("正在检索和生成回答..."):
            try:
                result = app.run_query(prompt)
            except Exception as exc:
                st.error(f"问答失败: {exc}")
            else:
                answer_text = result.final_answer or "我不确定"
                st.markdown(answer_text)
                if result.payload.get("retrieval_used"):
                    st.caption(
                        f"confidence={result.answer_confidence} | "
                        f"should_refuse={result.should_refuse} | "
                        f"effective_query={result.effective_query}"
                    )
                render_query_timings(result.payload)
                render_sources(result.payload)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": answer_text, "payload": result.payload}
                )
