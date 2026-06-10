#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from common.paths import DEFAULT_RERANKER_MODEL, INDEX_ROOT, RETRIEVAL_OUTPUT_ROOT, REPO_ROOT
from retrieval import ThreeWayRetriever


OUTPUT_JSON_DIR = RETRIEVAL_OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the finance RAG index with first-stage retrieval and optional second-stage reranking.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_ROOT / "faiss_flat_chunked_512_50_bge-large-zh-v1.5",
        help="Directory containing index.faiss, docstore.jsonl, id_map.json, and build_meta.json.",
    )
    parser.add_argument("--query", type=str, required=True, help="User query text.")
    parser.add_argument(
        "--route",
        choices=["vector", "bm25", "hybrid", "hybrid_weightsum", "hybrid_rrf", "all"],
        default="hybrid_rrf",
        help="Which retrieval route to run.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Final number of returned chunks.")
    parser.add_argument(
        "--first-stage-top-k",
        type=int,
        default=50,
        help="Number of chunks returned by first-stage retrieval before reranking.",
    )
    parser.add_argument("--vector-candidate-k", type=int, default=50, help="Vector candidate pool size for hybrid.")
    parser.add_argument("--bm25-candidate-k", type=int, default=50, help="BM25 candidate pool size for hybrid.")
    parser.add_argument(
        "--fusion-mode",
        choices=["weighted_sum", "rrf"],
        default="rrf",
        help="Fusion algorithm for hybrid retrieval.",
    )
    parser.add_argument("--vector-weight", type=float, default=0.3, help="Vector score weight in hybrid fusion.")
    parser.add_argument("--bm25-weight", type=float, default=0.7, help="BM25 score weight in hybrid fusion.")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF smoothing constant used when --fusion-mode rrf.")
    parser.add_argument("--rrf-top-k", type=int, default=50, help="Final first-stage output size when using hybrid_rrf.")
    parser.add_argument("--title-weight", type=float, default=1.0, help="Document-title BM25 bonus weight.")
    parser.add_argument("--use-reranker", action="store_true", help="Enable second-stage reranking.")
    parser.add_argument(
        "--query-decompose",
        action="store_true",
        help="Enable heuristic query decomposition for compare/summary queries on hybrid routes.",
    )
    parser.add_argument(
        "--rerank-model-path",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help="Local path or Hugging Face repo id for the reranker model.",
    )
    parser.add_argument("--rerank-top-k", type=int, default=10, help="Top-k chunks kept after reranking.")
    parser.add_argument("--rerank-batch-size", type=int, default=16, help="Batch size for reranker scoring.")
    parser.add_argument("--reranker-use-fp16", action="store_true", help="Use fp16 when loading the reranker.")
    parser.add_argument("--show-text-chars", type=int, default=220, help="Preview length for chunk text.")
    parser.add_argument(
        "--output-json-dir",
        type=Path,
        default=OUTPUT_JSON_DIR,
        help="Directory for saving retrieval outputs as JSON files.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser.parse_args()

def format_hits(route: str, hits, preview_chars: int):
    payload = []
    for rank, hit in enumerate(hits, start=1):
        item = {
            "rank": rank,
            "route": route,
            "docstore_id": hit.docstore_id,
            "doc_id": hit.doc_id,
            "chunk_id": hit.chunk_id,
            "chunk_type": hit.chunk_type,
            "vector_score": hit.vector_score,
            "bm25_score": hit.bm25_score,
            "title_score": hit.title_score,
            "fused_score": hit.fused_score,
            "reranker_score": hit.reranker_score,
            "source_pdf": hit.source_pdf,
            "metadata": hit.metadata,
            "text_preview": hit.text[:preview_chars],
            "has_linked_raw_table": hit.linked_raw_table is not None,
        }
        if hit.linked_raw_table is not None:
            item["linked_raw_table_preview"] = hit.linked_raw_table.get("text", "")[:preview_chars]
        payload.append(item)
    return payload


def print_text(route: str, hits, preview_chars: int) -> None:
    print(f"\n=== {route.upper()} ===")
    if not hits:
        print("(no hits)")
        return
    for rank, hit in enumerate(hits, start=1):
        print(
            f"[{rank}] chunk_type={hit.chunk_type} fused={hit.fused_score:.4f} "
            f"vector={hit.vector_score:.4f} bm25={hit.bm25_score:.4f} "
            f"title={hit.title_score:.4f} rerank={hit.reranker_score:.4f}"
        )
        print(f"doc_id={hit.doc_id}")
        print(f"chunk_id={hit.chunk_id}")
        print(f"source_pdf={hit.source_pdf}")
        print(f"text={hit.text[:preview_chars].strip()}")
        if hit.linked_raw_table is not None:
            print(f"linked_raw_table={hit.linked_raw_table.get('text', '')[:preview_chars].strip()}")
        print()


def slugify_filename(text: str) -> str:
    value = (text or "").strip().replace(" ", "_")
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    slug = "".join(safe_chars).strip("_")
    return slug[:80] if slug else "query"


def build_run_payload(args: argparse.Namespace, route: str, hits) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": args.query,
        "route": route,
        "use_reranker": bool(args.use_reranker),
        "index_dir": str(args.index_dir),
        "params": {
            "top_k": args.top_k,
            "first_stage_top_k": args.first_stage_top_k,
            "vector_candidate_k": args.vector_candidate_k,
            "bm25_candidate_k": args.bm25_candidate_k,
            "fusion_mode": args.fusion_mode,
            "vector_weight": args.vector_weight,
            "bm25_weight": args.bm25_weight,
            "rrf_k": args.rrf_k,
            "rrf_top_k": args.rrf_top_k,
            "title_weight": args.title_weight,
            "rerank_model_path": args.rerank_model_path if args.use_reranker else None,
            "rerank_top_k": args.rerank_top_k if args.use_reranker else None,
            "rerank_batch_size": args.rerank_batch_size if args.use_reranker else None,
            "reranker_use_fp16": bool(args.reranker_use_fp16) if args.use_reranker else None,
            "query_decompose": bool(args.query_decompose),
        },
        "results": format_hits(route, hits, preview_chars=args.show_text_chars),
    }


def save_route_output(args: argparse.Namespace, route: str, hits) -> Path:
    args.output_json_dir.mkdir(parents=True, exist_ok=True)
    rerank_flag = "rerank_on" if args.use_reranker else "rerank_off"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = slugify_filename(args.query)
    file_path = args.output_json_dir / f"{route}__{rerank_flag}__{timestamp}__{query_slug}.json"
    payload = build_run_payload(args, route, hits)
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return file_path


def main() -> None:
    args = parse_args()
    try:
        retriever = ThreeWayRetriever(
            args.index_dir,
            reranker_model_path=args.rerank_model_path if args.use_reranker else None,
            reranker_use_fp16=args.reranker_use_fp16,
        )

        route_to_hits = {}
        hybrid_top_k = args.first_stage_top_k
        if args.route == "hybrid_rrf":
            hybrid_top_k = args.rrf_top_k

        if args.route in {"vector", "all"}:
            route_to_hits["vector"] = retriever.retrieve(query=args.query, route="vector", top_k=args.top_k)
        if args.route in {"bm25", "all"}:
            route_to_hits["bm25"] = retriever.retrieve(
                query=args.query,
                route="bm25",
                top_k=args.top_k,
                title_weight=args.title_weight,
                query_decompose=args.query_decompose,
            )
        routes_to_run = []
        if args.route == "all":
            routes_to_run.extend(["hybrid_weightsum", "hybrid_rrf"])
        elif args.route in {"hybrid", "hybrid_weightsum", "hybrid_rrf"}:
            routes_to_run.append(args.route)

        for route_name in routes_to_run:
            if route_name == "hybrid":
                route_name = "hybrid_weightsum" if args.fusion_mode == "weighted_sum" else "hybrid_rrf"

            first_stage_k = hybrid_top_k if route_name == "hybrid_rrf" else args.first_stage_top_k
            fusion_mode = "rrf" if route_name == "hybrid_rrf" else "weighted_sum"

            if args.use_reranker:
                route_to_hits[route_name] = retriever.retrieve_two_stage(
                    query=args.query,
                    route=route_name,
                    first_stage_top_k=first_stage_k,
                    rerank_top_k=args.rerank_top_k,
                    vector_candidate_k=args.vector_candidate_k,
                    bm25_candidate_k=args.bm25_candidate_k,
                    vector_weight=args.vector_weight,
                    bm25_weight=args.bm25_weight,
                    title_weight=args.title_weight,
                    fusion_mode=fusion_mode,
                    rrf_k=args.rrf_k,
                    rerank_batch_size=args.rerank_batch_size,
                    query_decompose=args.query_decompose,
                )
            else:
                route_to_hits[route_name] = retriever.retrieve(
                    query=args.query,
                    route=route_name,
                    top_k=first_stage_k,
                    vector_candidate_k=args.vector_candidate_k,
                    bm25_candidate_k=args.bm25_candidate_k,
                    fusion_mode=fusion_mode,
                    vector_weight=args.vector_weight,
                    bm25_weight=args.bm25_weight,
                    rrf_k=args.rrf_k,
                    title_weight=args.title_weight,
                    query_decompose=args.query_decompose,
                )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    saved_paths = {
        route: save_route_output(args, route, hits)
        for route, hits in route_to_hits.items()
    }

    if args.json:
        output = {
            route: {
                "saved_to": str(saved_paths[route]),
                "payload": build_run_payload(args, route, hits),
            }
            for route, hits in route_to_hits.items()
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    for route, hits in route_to_hits.items():
        print_text(route, hits, preview_chars=args.show_text_chars)
        print(f"saved_json={saved_paths[route]}")


if __name__ == "__main__":
    main()
