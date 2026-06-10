#!/usr/bin/env python3
import argparse
import collections
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from common.paths import CHUNKED_ROOT, DEFAULT_BGE_MODEL, INDEX_ROOT


@dataclass
class ChunkRecord:
    faiss_id: Optional[int]
    docstore_id: str
    chunk_id: str
    doc_id: str
    text: str
    embedding_text: str
    chunk_type: str
    indexed: bool
    source_file: str
    source_pdf: str
    metadata: Dict


@dataclass
class LoadSummary:
    all_chunk_count_by_type: Dict[str, int]
    embedded_chunk_count_by_type: Dict[str, int]
    skip_count_by_reason: Dict[str, int]
    total_chunk_count: int
    embedded_chunk_count: int
    table_summary_with_table_id: int
    matched_raw_table_count: int
    raw_table_chunk_count: int


def parse_args() -> argparse.Namespace:
    default_input = CHUNKED_ROOT / "chunked_512_50"

    parser = argparse.ArgumentParser(
        description="Build a FAISS index from chunked RAG documents with FlagEmbedding bge-large-zh-v1.5."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input, help="Directory containing chunk JSON files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write the FAISS index and metadata. Defaults to indexes/faiss_<index-type>_<input-dir-name>_bge-large-zh-v1.5.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_BGE_MODEL,
        help="Local model path or Hugging Face model id for FlagEmbedding.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length passed to the embedding model.")
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivf", "hnsw"],
        default="hnsw",
        help="FAISS index type to build.",
    )
    parser.add_argument(
        "--ivf-nlist", #经验公式nlist = sqrt(num_vectors)，但不超过1000
        type=int,
        default=0,
        help="IVF 倒排列表的数量。如果为 0，则根据语料库大小选择启发式值。",
    )
    parser.add_argument(
        "--ivf-nprobe", #nprobe ≈ nlist 的 5% - 10%，但不超过 nlist
        type=int,
        default=16,
        help="搜索期间要探测的 IVF 列表的数量。",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW graph degree M.",
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=100,
        help="HNSW efConstruction parameter.",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=64,
        help="HNSW efSearch parameter.",
    )
    parser.add_argument(
        "--include-chunk-types",
        nargs="+",
        default=["text_chunk", "raw_table_chunk"],
        help="Chunk types to embed.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Enable fp16 inference in FlagEmbedding when the environment supports it.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before writing vectors into FAISS.",
    )
    return parser.parse_args()


def iter_chunk_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.json"))


def extract_doc_title(doc_id: str, source_pdf: str) -> str:
    raw = clean_value(doc_id)
    if not raw and source_pdf:
        raw = Path(source_pdf).stem

    if not raw:
        return ""

    raw = re.sub(r"^\d+_\d{4}-\d{2}-\d{2}_", "", raw)
    return raw.replace("_", " ").strip()


def extract_doc_title_from_chunks(chunks: Sequence[Dict], doc_id: str, source_pdf: str) -> str:
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("chunk_type") != "cover_summary_chunk":
            continue
        text = clean_value(chunk.get("text"))
        if not text:
            continue
        first_line = clean_value(text.splitlines()[0])
        if first_line.startswith("标题："):
            title = clean_value(first_line.removeprefix("标题："))
            if title:
                return title
    return extract_doc_title(doc_id, source_pdf)


def extract_cover_fields_from_chunks(chunks: Sequence[Dict], doc_id: str, source_pdf: str) -> tuple[str, str]:
    title = ""
    company = ""
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("chunk_type") != "cover_summary_chunk":
            continue
        text = clean_value(chunk.get("text"))
        if not text:
            continue
        for line in text.splitlines():
            line = clean_value(line)
            if line.startswith("标题：") and not title:
                title = clean_value(line.removeprefix("标题："))
            elif line.startswith("公司：") and not company:
                company = clean_value(line.removeprefix("公司："))
        if title and company:
            break
    return title or extract_doc_title(doc_id, source_pdf), company


def extract_table_summary_lookup(chunks: Sequence[Dict]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("chunk_type") != "table_summary_chunk":
            continue
        table_id = clean_value(metadata.get("table_id"))
        if not table_id:
            continue
        entry = {"caption": "", "context": ""}
        text = clean_value(chunk.get("text"))
        if text:
            for line in text.splitlines():
                line = clean_value(line)
                if line.startswith("表名：") and not entry["caption"]:
                    entry["caption"] = clean_value(line.removeprefix("表名："))
                elif line.startswith("前后文：") and not entry["context"]:
                    entry["context"] = clean_value(line.removeprefix("前后文："))
        lookup[table_id] = entry
    return lookup


def build_embedding_text(
    chunk_type: str,
    text: str,
    metadata: Dict,
    doc_title: str,
    doc_company: str,
    table_summary_lookup: Dict[str, Dict[str, str]],
) -> str:
    text = (text or "").strip()
    title = clean_value(doc_title)
    company = clean_value(doc_company)

    lines: List[str] = []
    if title:
        lines.append(f"标题：{title}")
    if company:
        lines.append(f"公司：{company}")

    if chunk_type == "raw_table_chunk":
        table_id = clean_value(metadata.get("table_id"))
        summary_info = table_summary_lookup.get(table_id, {})
        context = clean_value(summary_info.get("context"))
        caption = clean_value(summary_info.get("caption")) or clean_value(metadata.get("caption"))
        if context:
            lines.append(f"前后文：{context}")
        if caption:
            lines.append(f"表名：{caption}")

    lines.append(text)
    return "\n".join(line for line in lines if line).strip()


def clean_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_records(input_dir: Path, include_chunk_types: Sequence[str]) -> tuple[List[ChunkRecord], List[ChunkRecord], LoadSummary]:
    indexed_records: List[ChunkRecord] = []
    docstore_records: List[ChunkRecord] = []
    allowed_types = set(include_chunk_types)
    all_chunk_count_by_type: Counter[str] = collections.Counter()
    embedded_chunk_count_by_type: Counter[str] = collections.Counter()
    skip_count_by_reason: Counter[str] = collections.Counter()
    raw_table_ids_by_doc_table: Dict[tuple[str, str], str] = {}
    table_summary_doc_tables: List[tuple[str, str]] = []

    for source_file in iter_chunk_files(input_dir):
        with source_file.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        doc_id = payload.get("doc_id", source_file.stem)
        source_pdf = payload.get("source", "")
        chunks = payload.get("chunks", [])
        doc_title, doc_company = extract_cover_fields_from_chunks(chunks, doc_id, source_pdf)
        table_summary_lookup = extract_table_summary_lookup(chunks)
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chunk_type = metadata.get("chunk_type", "<missing>")
            all_chunk_count_by_type[chunk_type] += 1
            text = chunk.get("text", "")
            table_id = metadata.get("table_id")
            if chunk_type == "raw_table_chunk" and table_id:
                raw_table_ids_by_doc_table[(doc_id, table_id)] = chunk["id"]
            if chunk_type == "table_summary_chunk" and table_id:
                table_summary_doc_tables.append((doc_id, table_id))

            record = ChunkRecord(
                faiss_id=None,
                docstore_id=chunk["id"],
                chunk_id=chunk["id"],
                doc_id=doc_id,
                text=text,
                embedding_text=build_embedding_text(
                    chunk_type,
                    text,
                    metadata,
                    doc_title,
                    doc_company,
                    table_summary_lookup,
                ),
                chunk_type=chunk_type,
                indexed=False,
                source_file=str(source_file),
                source_pdf=source_pdf,
                metadata=metadata,
            )
            docstore_records.append(record)

            if chunk_type not in allowed_types:
                skip_count_by_reason[f"excluded_chunk_type:{chunk_type}"] += 1
                continue
            if not text or not text.strip():
                skip_count_by_reason["empty_text"] += 1
                continue
            record.faiss_id = len(indexed_records)
            record.indexed = True
            indexed_records.append(record)
            embedded_chunk_count_by_type[chunk_type] += 1

    summary = LoadSummary(
        all_chunk_count_by_type=dict(all_chunk_count_by_type),
        embedded_chunk_count_by_type=dict(embedded_chunk_count_by_type),
        skip_count_by_reason=dict(skip_count_by_reason),
        total_chunk_count=sum(all_chunk_count_by_type.values()),
        embedded_chunk_count=len(indexed_records),
        table_summary_with_table_id=len(table_summary_doc_tables),
        matched_raw_table_count=sum(1 for key in table_summary_doc_tables if key in raw_table_ids_by_doc_table),
        raw_table_chunk_count=all_chunk_count_by_type.get("raw_table_chunk", 0),
    )
    return indexed_records, docstore_records, summary


def resolve_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-large-zh-v1.5" / "snapshots"
    if model_path == "BAAI/bge-large-zh-v1.5" and cache_root.exists():
        snapshots = sorted(p for p in cache_root.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return model_path


def _flag_encode(model, texts: Sequence[str], batch_size: int, max_length: int) -> np.ndarray:
    try:
        embeddings = model.encode(texts, batch_size=batch_size, max_length=max_length)
    except TypeError:
        embeddings = model.encode(texts, batch_size=batch_size)
    return np.asarray(embeddings, dtype=np.float32)


def embed_with_flagembedding(
    texts: Sequence[str], model_path: str, batch_size: int, max_length: int, use_fp16: bool
) -> np.ndarray:
    try:
        from FlagEmbedding import FlagModel
    except ImportError as exc:
        raise RuntimeError("FlagEmbedding is not installed in the current environment.") from exc

    model = FlagModel(model_path, use_fp16=use_fp16)
    embeddings = _flag_encode(model, texts, batch_size=batch_size, max_length=max_length)
    return embeddings.astype(np.float32, copy=False)


def embed_with_sentence_transformers(texts: Sequence[str], model_path: str, batch_size: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed in the current environment.") from exc

    model = SentenceTransformer(model_path, trust_remote_code=False)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_texts(
    texts: Sequence[str], model_path: str, batch_size: int, max_length: int, use_fp16: bool
) -> tuple[np.ndarray, str]:
    errors = []

    try:
        return embed_with_flagembedding(texts, model_path, batch_size, max_length, use_fp16), "FlagEmbedding.FlagModel"
    except Exception as exc:  # pragma: no cover - fallback path depends on local env
        errors.append(f"FlagEmbedding failed: {exc!r}")

    try:
        return embed_with_sentence_transformers(texts, model_path, batch_size), "sentence_transformers.SentenceTransformer"
    except Exception as exc:  # pragma: no cover - fallback path depends on local env
        errors.append(f"sentence-transformers failed: {exc!r}")

    raise SystemExit("Unable to create embeddings.\n" + "\n".join(errors))


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _choose_ivf_nlist(num_vectors: int, requested_nlist: int) -> int:
    if requested_nlist > 0:
        return min(requested_nlist, num_vectors)
    heuristic = max(1, int(round(num_vectors ** 0.5)))
    return min(heuristic, num_vectors)


def build_faiss_index(vectors: np.ndarray, args: argparse.Namespace):
    try:
        import faiss
    except ImportError as exc:
        raise SystemExit(
            "faiss is not installed in the current environment. "
            "Install it first, for example: pip install faiss-cpu"
        ) from exc

    dim = vectors.shape[1]
    index_params: Dict[str, Union[int, str]] = {
        "metric": "inner_product",
    }

    if args.index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        index_params["index_type"] = "IndexFlatIP"
        return faiss, index, index_params

    if args.index_type == "ivf":
        nlist = _choose_ivf_nlist(len(vectors), args.ivf_nlist)
        nprobe = max(1, min(args.ivf_nprobe, nlist))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)
        index.nprobe = nprobe
        index_params.update(
            {
                "index_type": "IndexIVFFlat",
                "nlist": int(nlist),
                "nprobe": int(nprobe),
            }
        )
        return faiss, index, index_params

    index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = args.hnsw_ef_construction
    index.hnsw.efSearch = args.hnsw_ef_search
    index.add(vectors)
    index_params.update(
        {
            "index_type": "IndexHNSWFlat",
            "hnsw_m": int(args.hnsw_m),
            "ef_construction": int(args.hnsw_ef_construction),
            "ef_search": int(args.hnsw_ef_search),
        }
    )
    return faiss, index, index_params


def write_docstore(records: Sequence[ChunkRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_id_map(records: Sequence[ChunkRecord], output_path: Path) -> None:
    id_map = {str(record.faiss_id): record.docstore_id for record in records}
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(id_map, fh, ensure_ascii=False, indent=2)


def write_metadata(
    output_path: Path,
    *,
    args: argparse.Namespace,
    records: Sequence[ChunkRecord],
    vectors: np.ndarray,
    model_path: str,
    embedding_backend: str,
    load_summary: LoadSummary,
    index_params: Dict[str, Union[int, str]],
    build_seconds: float,
) -> None:
    doc_count = len({record.doc_id for record in records})
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "model_name": "bge-large-zh-v1.5",
        "model_path": model_path,
        "embedding_backend": embedding_backend,
        "build_seconds": round(build_seconds, 4),
        "embedding_text_template": {
            "text_chunk": "标题：{title}\\n公司：{company}\\n{text}",
            "raw_table_chunk": "标题：{title}\\n公司：{company}\\n前后文：{context}\\n表名：{caption}\\n{text}",
        },
        "bm25_body_template": {
            "text_chunk": "{text}",
            "raw_table_chunk": "前后文：{context}\\n表名：{caption}\\n{text}",
        },
        "index_family": "FAISS",
        "index_type": index_params["index_type"],
        "index_build_params": index_params,
        "similarity": "cosine_via_l2_normalized_inner_product",
        "included_chunk_types": list(args.include_chunk_types),
        "normalize_embeddings": not args.no_normalize,
        "vector_dim": int(vectors.shape[1]),
        "document_count": doc_count,
        "chunk_count": int(vectors.shape[0]),
        "all_chunk_count": load_summary.total_chunk_count,
        "all_chunk_count_by_type": load_summary.all_chunk_count_by_type,
        "chunk_count_by_type": load_summary.embedded_chunk_count_by_type,
        "skip_count_by_reason": load_summary.skip_count_by_reason,
        "raw_table_linking": {
            "strategy": "embed raw_table_chunk directly with title/company plus matched table_summary caption/context; keep table_summary_chunk in docstore as preprocessing artifact only",
            "table_summary_with_table_id": load_summary.table_summary_with_table_id,
            "matched_raw_table_count": load_summary.matched_raw_table_count,
            "raw_table_chunk_count_in_docstore": load_summary.raw_table_chunk_count,
        },
        "bm25_bonus": {
            "strategy": "bm25_body_score + title_weight * doc_title_company_bm25_score",
            "body_uses": "text_chunk body text; raw_table_chunk uses caption/context plus raw table body",
            "bonus_source": "cover_summary_chunk title/company extracted at document level",
        },
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "use_fp16": bool(args.use_fp16),
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def main() -> None:
    build_start = time.perf_counter()
    args = parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        input_suffix = args.input_dir.name
        args.output_dir = INDEX_ROOT / f"faiss_{args.index_type}_{input_suffix}_bge-large-zh-v1.5"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, docstore_records, load_summary = load_records(args.input_dir, args.include_chunk_types)
    if not records:
        raise SystemExit(f"No chunk records found in {args.input_dir}")

    texts = [record.embedding_text for record in records]
    model_path = resolve_model_path(args.model_path)
    vectors, embedding_backend = embed_texts(
        texts=texts,
        model_path=model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_fp16=args.use_fp16,
    )

    if vectors.ndim != 2:
        raise SystemExit(f"Expected 2D embeddings, got shape={vectors.shape}")

    if not args.no_normalize:
        vectors = normalize_embeddings(vectors)

    faiss, index, index_params = build_faiss_index(vectors, args)
    faiss.write_index(index, str(args.output_dir / "index.faiss"))

    np.save(args.output_dir / "embeddings.npy", vectors)
    write_docstore(docstore_records, args.output_dir / "docstore.jsonl")
    write_id_map(records, args.output_dir / "id_map.json")
    write_metadata(
        args.output_dir / "build_meta.json",
        args=args,
        records=records,
        vectors=vectors,
        model_path=model_path,
        embedding_backend=embedding_backend,
        load_summary=load_summary,
        index_params=index_params,
        build_seconds=time.perf_counter() - build_start,
    )

    print(f"index_path={args.output_dir / 'index.faiss'}")
    print(f"docstore_path={args.output_dir / 'docstore.jsonl'}")
    print(f"id_map_path={args.output_dir / 'id_map.json'}")
    print(f"build_meta_path={args.output_dir / 'build_meta.json'}")
    print(f"embedding_path={args.output_dir / 'embeddings.npy'}")
    print(f"index_type={index_params['index_type']}")
    print(f"vector_dim={vectors.shape[1]}")
    print(f"chunk_count={vectors.shape[0]}")
    print(f"document_count={len({record.doc_id for record in records})}")
    print(f"build_seconds={time.perf_counter() - build_start:.4f}")


if __name__ == "__main__":
    main()
