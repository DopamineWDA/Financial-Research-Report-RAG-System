#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9._%+-]+")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare retrieval methods on fixed FAISS Flat chunked indexes."
    )
    parser.add_argument(
        "--index-dirs",
        type=Path,
        nargs="+",
        default=[
            repo_root / "indexes" / "faiss_flat_chunked_512_50_bge-large-zh-v1.5",
            repo_root / "indexes" / "faiss_flat_chunked_256_50_bge-large-zh-v1.5",
        ],
        help="Index directories to evaluate.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=repo_root / "eval" / "recall_eval.json",
        help="Strict recall evaluation set.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Retrieve top-k chunks, then score Recall@3/5/10 by prefix.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size for dense query encoding.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to SentenceTransformer.",
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.4,
        help="Normalized vector score weight for hybrid_weightsum.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.6,
        help="Normalized BM25 score weight for hybrid_weightsum.",
    )
    parser.add_argument(
        "--vector-candidate-k",
        type=int,
        default=50,
        help="Vector candidate pool for hybrids.",
    )
    parser.add_argument(
        "--bm25-candidate-k",
        type=int,
        default=50,
        help="BM25 candidate pool for hybrids.",
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
        help="Document-title BM25 bonus weight.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of markdown.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().with_suffix(".md"),
        help="Path to write the markdown table.",
    )
    return parser.parse_args()


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


def top_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, scores.size)
    candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
    sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    return [int(idx) for idx in sorted_indices]


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


def load_samples(eval_path: Path) -> list[dict]:
    return json.loads(eval_path.read_text(encoding="utf-8"))


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


def build_bm25(records: list[dict]) -> tuple[BM25Okapi, BM25Okapi | None, list[str]]:
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
    return bm25, title_bm25, title_doc_ids


def bm25_scores_for_query(
    query: str,
    records: list[dict],
    bm25: BM25Okapi,
    title_bm25: BM25Okapi | None,
    title_doc_ids: list[str],
    title_weight: float,
) -> np.ndarray:
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
    return combined_scores


def strict_hit(sample: dict, records: list[dict], ranked_ids: list[int], top_n: int) -> int:
    gold = {(item["doc_id"], item["block_id"]) for item in sample["gold_evidence"]}
    retrieved = set()
    for idx in ranked_ids[:top_n]:
        if idx < 0:
            continue
        record = records[idx]
        metadata = record.get("metadata", {})
        doc_id = metadata.get("doc_id") or record.get("doc_id")
        for block_id in metadata.get("block_ids", []) or []:
            retrieved.add((doc_id, block_id))
    return int(gold.issubset(retrieved))


def evaluate_route(samples: list[dict], records: list[dict], ranked_lists: list[list[int]]) -> dict[str, float]:
    hits3, hits5, hits10 = [], [], []
    for sample, ranked_ids in zip(samples, ranked_lists):
        hits3.append(strict_hit(sample, records, ranked_ids, 3))
        hits5.append(strict_hit(sample, records, ranked_ids, 5))
        hits10.append(strict_hit(sample, records, ranked_ids, 10))
    return {
        "Recall@3": float(np.mean(hits3)),
        "Recall@5": float(np.mean(hits5)),
        "Recall@10": float(np.mean(hits10)),
    }


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| index | retrieval | Recall@3 | Recall@5 | Recall@10 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['retrieval']} | "
            f"{row['Recall@3']:.4f} | {row['Recall@5']:.4f} | {row['Recall@10']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    samples = load_samples(args.eval_path)
    queries = [sample["query"] for sample in samples]

    build_meta = json.loads((args.index_dirs[0] / "build_meta.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(build_meta["model_path"], trust_remote_code=args.trust_remote_code)
    query_vectors = np.asarray(
        model.encode(
            queries,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    if build_meta.get("normalize_embeddings", True):
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        query_vectors = query_vectors / np.clip(norms, a_min=1e-12, a_max=None)

    rows: list[dict] = []
    for index_dir in args.index_dirs:
        records = load_indexed_records(index_dir)
        index = faiss.read_index(str(index_dir / "index.faiss"))
        bm25, title_bm25, title_doc_ids = build_bm25(records)

        _, dense_ids = index.search(query_vectors, args.top_k)
        dense_ranked = [[int(idx) for idx in row if int(idx) >= 0] for row in dense_ids]
        rows.append(
            {
                "index": index_dir.name,
                "retrieval": "dense",
                **evaluate_route(samples, records, dense_ranked),
            }
        )

        bm25_ranked: list[list[int]] = []
        hybrid_ws_ranked: list[list[int]] = []
        hybrid_rrf_ranked: list[list[int]] = []
        vector_pool = max(args.top_k, args.vector_candidate_k)
        vector_candidate_scores, vector_candidate_ids = index.search(query_vectors, vector_pool)

        for qi, query in enumerate(queries):
            scores = bm25_scores_for_query(
                query=query,
                records=records,
                bm25=bm25,
                title_bm25=title_bm25,
                title_doc_ids=title_doc_ids,
                title_weight=args.title_weight,
            )
            bm25_top = top_indices(scores, args.top_k)
            bm25_ranked.append(bm25_top)

            vector_top = [int(idx) for idx in vector_candidate_ids[qi][: args.vector_candidate_k] if int(idx) >= 0]
            bm25_candidate_top = top_indices(scores, args.bm25_candidate_k)

            merged_ids: list[int] = []
            seen = set()
            for idx in vector_top + bm25_candidate_top:
                if idx not in seen:
                    seen.add(idx)
                    merged_ids.append(idx)

            vector_score_map = normalize_score_map(
                {
                    int(idx): float(score)
                    for score, idx in zip(
                        vector_candidate_scores[qi][: args.vector_candidate_k],
                        vector_candidate_ids[qi][: args.vector_candidate_k],
                    )
                    if int(idx) >= 0
                }
            )
            bm25_score_map = normalize_score_map({idx: float(scores[idx]) for idx in bm25_candidate_top})

            ws_scored = []
            for idx in merged_ids:
                fused = (
                    args.vector_weight * vector_score_map.get(idx, 0.0)
                    + args.bm25_weight * bm25_score_map.get(idx, 0.0)
                )
                ws_scored.append((fused, idx))
            ws_scored.sort(key=lambda item: item[0], reverse=True)
            hybrid_ws_ranked.append([idx for _, idx in ws_scored[: args.top_k]])

            vector_rank_map = {idx: rank for rank, idx in enumerate(vector_top, start=1)}
            bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_candidate_top, start=1)}
            rrf_scored = []
            for idx in merged_ids:
                fused = 0.0
                if idx in vector_rank_map:
                    fused += 1.0 / (args.rrf_k + vector_rank_map[idx])
                if idx in bm25_rank_map:
                    fused += 1.0 / (args.rrf_k + bm25_rank_map[idx])
                rrf_scored.append((fused, idx))
            rrf_scored.sort(key=lambda item: item[0], reverse=True)
            hybrid_rrf_ranked.append([idx for _, idx in rrf_scored[: args.top_k]])

        rows.append(
            {
                "index": index_dir.name,
                "retrieval": "bm25",
                **evaluate_route(samples, records, bm25_ranked),
            }
        )
        rows.append(
            {
                "index": index_dir.name,
                "retrieval": "hybrid_weightsum",
                **evaluate_route(samples, records, hybrid_ws_ranked),
            }
        )
        rows.append(
            {
                "index": index_dir.name,
                "retrieval": "hybrid_rrf",
                **evaluate_route(samples, records, hybrid_rrf_ranked),
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    markdown = render_markdown(rows)
    args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
