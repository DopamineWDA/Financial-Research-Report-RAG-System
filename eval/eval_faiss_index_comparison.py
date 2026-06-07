#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare FAISS index types on strict Recall@10, search latency, build time, and index size."
    )
    parser.add_argument(
        "--index-dirs",
        type=Path,
        nargs="+",
        default=[
            repo_root / "indexes" / "faiss_flat_chunked_512_50_bge-large-zh-v1.5",
            repo_root / "indexes" / "faiss_hnsw_chunked_512_50_bge-large-zh-v1.5",
            repo_root / "indexes" / "faiss_ivf_chunked_512_50_bge-large-zh-v1.5",
        ],
        help="Index directories to compare.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=repo_root / "eval" / "recall_eval.json",
        help="Strict recall evaluation set.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k dense retrieval cutoff.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().with_suffix(".md"),
        help="Path to write the markdown table.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of markdown.")
    return parser.parse_args()


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


def compute_recall_at_10(samples: list[dict], records: list[dict], ranked_rows: np.ndarray) -> float:
    hits = []
    for sample, row in zip(samples, ranked_rows):
        ranked_ids = [int(idx) for idx in row if int(idx) >= 0]
        hits.append(strict_hit(sample, records, ranked_ids, 10))
    return float(np.mean(hits))


def measure_search_latency_ms(index: faiss.Index, query_vectors: np.ndarray, top_k: int) -> float:
    # Warm the index once to exclude initial load effects.
    index.search(query_vectors[:1], top_k)
    elapsed_ms = []
    for qv in query_vectors:
        start = time.perf_counter()
        index.search(qv.reshape(1, -1), top_k)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(elapsed_ms))


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| index | Recall@10 | 查询延迟(ms) | 构建时间(s) | 索引文件大小(MB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['Recall@10']:.4f} | {row['查询延迟(ms)']:.3f} | "
            f"{row['构建时间(s)']:.4f} | {row['索引文件大小(MB)']:.3f} |"
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
        meta = json.loads((index_dir / "build_meta.json").read_text(encoding="utf-8"))
        records = load_indexed_records(index_dir)
        index = faiss.read_index(str(index_dir / "index.faiss"))
        _, ranked_rows = index.search(query_vectors, args.top_k)
        rows.append(
            {
                "index": index_dir.name,
                "Recall@10": compute_recall_at_10(samples, records, ranked_rows),
                "查询延迟(ms)": measure_search_latency_ms(index, query_vectors, args.top_k),
                "构建时间(s)": float(meta["build_seconds"]),
                "索引文件大小(MB)": float((index_dir / "index.faiss").stat().st_size / 1024 / 1024),
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
