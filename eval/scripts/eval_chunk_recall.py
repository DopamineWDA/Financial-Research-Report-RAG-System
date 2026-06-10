#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9._%+-]+")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate strict dense recall across FAISS Flat chunked indexes."
    )
    parser.add_argument(
        "--indexes-root",
        type=Path,
        default=repo_root / "indexes",
        help="Directory containing faiss_flat_chunked_* index folders.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=repo_root / "eval" / "recall_eval.json",
        help="Strict recall evaluation set.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="faiss_flat_chunked_*_bge-large-zh-v1.5",
        help="Glob used to find candidate index directories.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k dense retrieval cutoff. Metrics are reported for @3/@5/@10 up to this value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size for evaluation queries.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to SentenceTransformer.",
    )
    parser.add_argument(
        "--route",
        choices=["vector", "hybrid_rrf"],
        default="vector",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument(
        "--vector-candidate-k",
        type=int,
        default=50,
        help="Vector candidate pool size for hybrid RRF.",
    )
    parser.add_argument(
        "--bm25-candidate-k",
        type=int,
        default=50,
        help="BM25 candidate pool size for hybrid RRF.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant.",
    )
    parser.add_argument(
        "--title-weight",
        type=float,
        default=1.0,
        help="Document-title BM25 bonus weight for hybrid RRF.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a markdown table.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().with_suffix(".md"),
        help="Path to write the markdown table.",
    )
    return parser.parse_args()


def load_samples(eval_path: Path) -> list[dict]:
    return json.loads(eval_path.read_text(encoding="utf-8"))


def load_model(index_dirs: list[Path], trust_remote_code: bool) -> tuple[SentenceTransformer, dict]:
    build_meta = json.loads((index_dirs[0] / "build_meta.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(build_meta["model_path"], trust_remote_code=trust_remote_code)
    return model, build_meta


def embed_queries(
    model: SentenceTransformer,
    build_meta: dict,
    queries: list[str],
    batch_size: int,
) -> np.ndarray:
    vectors = np.asarray(
        model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    if build_meta.get("normalize_embeddings", True):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, a_min=1e-12, a_max=None)
    return vectors


def load_indexed_records(index_dir: Path) -> list[dict]:
    records: list[dict] = []
    with (index_dir / "docstore.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("indexed") and record.get("faiss_id") is not None:
                records.append(record)
    records.sort(key=lambda item: int(item["faiss_id"]))
    return records


def tokenize(text: str) -> list[str]:
    import jieba

    tokens: list[str] = []
    for raw_token in _TOKEN_PATTERN.findall((text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw_token):
            tokens.extend(piece.strip() for piece in jieba.lcut(raw_token) if piece.strip())
        else:
            tokens.append(raw_token)
    return tokens


def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def fallback_title_from_doc_id(doc_id: str) -> str:
    raw = (doc_id or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^\d+_\d{4}-\d{2}-\d{2}_", "", raw)
    return raw.replace("_", " ").strip()


def extract_title_from_record(record: dict) -> str:
    if record.get("chunk_type") == "cover_summary_chunk":
        first_line = first_nonempty_line(record.get("text", ""))
        if first_line.startswith("标题："):
            return first_line.removeprefix("标题：").strip()

    first_line = first_nonempty_line(record.get("embedding_text", ""))
    if first_line.startswith("文章标题："):
        return first_line.removeprefix("文章标题：").strip()

    return fallback_title_from_doc_id(record.get("doc_id", ""))


def strip_title_prefix(text: str) -> str:
    lines = (text or "").splitlines()
    stripped_lines = list(lines)
    while stripped_lines and not stripped_lines[0].strip():
        stripped_lines.pop(0)
    if stripped_lines and stripped_lines[0].strip().startswith("文章标题："):
        stripped_lines.pop(0)
    return "\n".join(stripped_lines).strip()


def build_doc_title_map(records: list[dict]) -> dict[str, str]:
    title_map: dict[str, str] = {}
    for record in records:
        doc_id = record["doc_id"]
        if doc_id in title_map:
            continue
        title = extract_title_from_record(record)
        if title:
            title_map[doc_id] = title
    return title_map


def normalize_score_map(score_map: dict[int, float]) -> dict[int, float]:
    if not score_map:
        return {}
    values = np.asarray(list(score_map.values()), dtype=np.float32)
    max_value = float(values.max())
    min_value = float(values.min())
    if abs(max_value - min_value) < 1e-12:
        if max_value <= 0:
            return {key: 0.0 for key in score_map}
        return {key: 1.0 for key in score_map}
    return {key: (value - min_value) / (max_value - min_value) for key, value in score_map.items()}


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, scores.size)
    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
    sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    return [int(idx) for idx in sorted_indices]


def build_bm25_components(records: list[dict]) -> tuple[BM25Okapi, dict[str, float], list[str], list[list[str]]]:
    corpus_tokens = [tokenize(strip_title_prefix(record["embedding_text"])) for record in records]
    bm25 = BM25Okapi(corpus_tokens)

    title_map = build_doc_title_map(records)
    title_doc_ids: list[str] = []
    title_corpus_tokens: list[list[str]] = []
    for doc_id, title in title_map.items():
        tokens = tokenize(title)
        if not tokens:
            continue
        title_doc_ids.append(doc_id)
        title_corpus_tokens.append(tokens)
    title_bm25 = BM25Okapi(title_corpus_tokens) if title_corpus_tokens else None
    return bm25, title_bm25, title_doc_ids, title_corpus_tokens


def retrieve_ranked_ids(
    *,
    route: str,
    records: list[dict],
    query_vectors: np.ndarray,
    queries: list[str],
    top_k: int,
    vector_candidate_k: int,
    bm25_candidate_k: int,
    rrf_k: int,
    title_weight: float,
    index: faiss.Index,
) -> np.ndarray:
    if route == "vector":
        _, ranked_ids = index.search(query_vectors, top_k)
        return ranked_ids

    bm25, title_bm25, title_doc_ids, _ = build_bm25_components(records)
    ranked_rows: list[list[int]] = []
    vector_pool = max(top_k, vector_candidate_k)
    _, vector_ranked_ids = index.search(query_vectors, vector_pool)

    for qi, query in enumerate(queries):
        query_tokens = tokenize(query)
        body_scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
        combined_scores = body_scores.copy()

        if title_bm25 is not None and title_weight != 0.0:
            title_scores = np.asarray(title_bm25.get_scores(query_tokens), dtype=np.float32)
            title_score_by_doc = {
                doc_id: float(score)
                for doc_id, score in zip(title_doc_ids, title_scores)
                if float(score) > 0.0
            }
            for idx, record in enumerate(records):
                combined_scores[idx] += title_weight * float(title_score_by_doc.get(record["doc_id"], 0.0))

        bm25_top = top_indices(combined_scores, bm25_candidate_k)
        vector_top = [int(idx) for idx in vector_ranked_ids[qi][:vector_candidate_k] if int(idx) >= 0]

        merged_ids = []
        seen = set()
        for idx in vector_top + bm25_top:
            if idx not in seen:
                seen.add(idx)
                merged_ids.append(idx)

        vector_rank_map = {idx: rank for rank, idx in enumerate(vector_top, start=1)}
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_top, start=1)}
        fused_scores = []
        for idx in merged_ids:
            score = 0.0
            if idx in vector_rank_map:
                score += 1.0 / (rrf_k + vector_rank_map[idx])
            if idx in bm25_rank_map:
                score += 1.0 / (rrf_k + bm25_rank_map[idx])
            fused_scores.append((score, idx))
        fused_scores.sort(key=lambda item: item[0], reverse=True)
        ranked_rows.append([idx for _, idx in fused_scores[:top_k]])

    ranked_ids = np.full((len(ranked_rows), top_k), -1, dtype=np.int64)
    for row_idx, row in enumerate(ranked_rows):
        ranked_ids[row_idx, : len(row)] = row
    return ranked_ids


def strict_hit(sample: dict, records: list[dict], ranked_ids: np.ndarray, top_n: int) -> int:
    gold = {(item["doc_id"], item["block_id"]) for item in sample["gold_evidence"]}
    retrieved = set()
    for faiss_id in ranked_ids[:top_n]:
        if int(faiss_id) < 0:
            continue
        record = records[int(faiss_id)]
        metadata = record.get("metadata", {})
        doc_id = metadata.get("doc_id") or record.get("doc_id")
        for block_id in metadata.get("block_ids", []) or []:
            retrieved.add((doc_id, block_id))
    return int(gold.issubset(retrieved))


def evaluate_index(
    index_dir: Path,
    samples: list[dict],
    queries: list[str],
    query_vectors: np.ndarray,
    top_k: int,
    route: str,
    vector_candidate_k: int,
    bm25_candidate_k: int,
    rrf_k: int,
    title_weight: float,
) -> dict:
    build_meta = json.loads((index_dir / "build_meta.json").read_text(encoding="utf-8"))
    records = load_indexed_records(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    ranked_ids = retrieve_ranked_ids(
        route=route,
        records=records,
        query_vectors=query_vectors,
        queries=queries,
        top_k=top_k,
        vector_candidate_k=vector_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        rrf_k=rrf_k,
        title_weight=title_weight,
        index=index,
    )

    hits_by_k = {3: [], 5: [], 10: []}
    hits_at_10_by_type: dict[str, list[int]] = defaultdict(list)

    for sample, sample_ranked_ids in zip(samples, ranked_ids):
        for k in (3, 5, 10):
            hit = strict_hit(sample, records, sample_ranked_ids, top_n=k)
            hits_by_k[k].append(hit)
            if k == 10:
                hits_at_10_by_type[sample["query_type"]].append(hit)

    parts = index_dir.name.split("_")
    return {
        "index_dir": index_dir.name,
        "route": route,
        "chunk_size": int(parts[3]),
        "overlap": int(parts[4]),
        "chunk_count": int(build_meta["chunk_count"]),
        "strict_recall@3": float(np.mean(hits_by_k[3])),
        "strict_recall@5": float(np.mean(hits_by_k[5])),
        "strict_recall@10": float(np.mean(hits_by_k[10])),
        "fact@10": float(np.mean(hits_at_10_by_type["fact"])),
        "compare@10": float(np.mean(hits_at_10_by_type["compare"])),
        "summary@10": float(np.mean(hits_at_10_by_type["summary"])),
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "| chunk_size | overlap | chunk_count | Strict Recall@3 | Strict Recall@5 | Strict Recall@10 | "
        "Fact@10 | Compare@10 | Summary@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['chunk_size']} | {row['overlap']} | {row['chunk_count']} | "
            f"{row['strict_recall@3']:.4f} | {row['strict_recall@5']:.4f} | {row['strict_recall@10']:.4f} | "
            f"{row['fact@10']:.4f} | {row['compare@10']:.4f} | {row['summary@10']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    index_dirs = sorted(args.indexes_root.glob(args.pattern))
    if not index_dirs:
        raise SystemExit(f"No index directories matched pattern: {args.pattern}")
    if args.top_k < 10:
        raise SystemExit("--top-k must be at least 10 for this experiment.")

    samples = load_samples(args.eval_path)
    queries = [sample["query"] for sample in samples]
    model, build_meta = load_model(index_dirs, trust_remote_code=args.trust_remote_code)
    query_vectors = embed_queries(model, build_meta, queries, batch_size=args.batch_size)

    results = [
        evaluate_index(
            index_dir,
            samples=samples,
            queries=queries,
            query_vectors=query_vectors,
            top_k=args.top_k,
            route=args.route,
            vector_candidate_k=args.vector_candidate_k,
            bm25_candidate_k=args.bm25_candidate_k,
            rrf_k=args.rrf_k,
            title_weight=args.title_weight,
        )
        for index_dir in index_dirs
    ]
    results.sort(key=lambda item: (item["chunk_size"], item["overlap"]))

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    markdown = render_markdown(results)
    args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()


# 复跑命令：
# conda run -n xinference python RAG/scripts/eval_chunk_dense_recall.py --trust-remote-code
