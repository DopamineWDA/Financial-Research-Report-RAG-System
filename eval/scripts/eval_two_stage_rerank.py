#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval import ThreeWayRetriever


ROUTE_CONFIGS = [
    {
        "route": "hybrid_weightsum",
        "first_stage_top_k": 50,
        "rerank_top_k": 10,
        "vector_candidate_k": 50,
        "bm25_candidate_k": 50,
        "vector_weight": 0.3,
        "bm25_weight": 0.7,
        "fusion_mode": "weighted_sum",
        "rrf_k": 60,
    },
    {
        "route": "hybrid_rrf",
        "first_stage_top_k": 50,
        "rerank_top_k": 10,
        "vector_candidate_k": 50,
        "bm25_candidate_k": 50,
        "vector_weight": 0.3,
        "bm25_weight": 0.7,
        "fusion_mode": "rrf",
        "rrf_k": 60,
    },
]

RECALL_KS = (3, 5, 10)
METRIC_KEYS = ("Recall@3", "Recall@5", "Recall@10", "MRR@10", "NDCG@10")
QUERY_TYPE_ORDER = ("fact", "compare", "summary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage hybrid retrieval + reranker with grouped evidence matching."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=REPO_ROOT / "indexes" / "faiss_hnsw_chunked_512_50_bge-large-zh-v1.5",
        help="Index directory used for both retrieval routes.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=REPO_ROOT / "eval" / "recall_eval_m.json",
        help="Evaluation set with grouped evidence annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "decompose_2",
        help="Output directory. The script rewrites it to keep exactly five files.",
    )
    parser.add_argument(
        "--rerank-model-path",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Local path or Hugging Face repo id for the reranker model.",
    )
    parser.add_argument("--reranker-use-fp16", action="store_true", help="Use fp16 when loading the reranker.")
    parser.add_argument("--rerank-batch-size", type=int, default=64, help="Batch size for reranker scoring.")
    parser.add_argument(
        "--title-weight",
        type=float,
        default=1.0,
        help="Document-level BM25 bonus weight sourced from cover-summary title/company fields.",
    )
    parser.add_argument(
        "--show-text-chars",
        type=int,
        default=0,
        help="Stored chunk text length. 0 or a negative value keeps the full chunk text.",
    )
    parser.add_argument(
        "--query-decompose",
        action="store_true",
        help="Enable heuristic query decomposition for compare/summary queries during retrieval.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional query limit for smoke testing. 0 means all queries.")
    return parser.parse_args()


def load_samples(eval_path: Path) -> list[dict[str, Any]]:
    return json.loads(eval_path.read_text(encoding="utf-8"))


def progress(iterable, *, desc: str, total: int):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def trim_text(text: str, show_text_chars: int) -> str:
    if show_text_chars <= 0:
        return text
    return text[:show_text_chars]


def clean_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def bm25_body_like_text(*, chunk_type: str, text: str, metadata: dict[str, Any] | None) -> str:
    if chunk_type != "raw_table_chunk":
        return clean_value(text)

    metadata = metadata or {}
    parts: list[str] = []
    context = clean_value(metadata.get("bm25_context"))
    caption = clean_value(metadata.get("caption"))
    raw_text = clean_value(text)
    if context:
        parts.append(f"前后文：{context}")
    if caption:
        parts.append(f"表名：{caption}")
    if raw_text:
        parts.append(raw_text)
    return "\n".join(parts).strip()


def infer_query_type_from_query(query: str) -> str | None:
    normalized = clean_value(query)
    compare_cues = ("谁更高", "谁更低", "谁更好", "谁更强", "谁更优", "哪个更高", "哪个更低", "哪个更好", "哪个更强", "哪个更优")
    summary_cues = ("归纳总结", "总结一下", "归纳一下", "总结", "归纳")
    if any(cue in normalized for cue in summary_cues):
        return "summary"
    if any(cue in normalized for cue in compare_cues):
        return "compare"
    return None


def get_component_fields(sample: dict[str, Any]) -> list[str]:
    if "gold_evidence_groups3" in sample:
        return ["gold_evidence_groups1", "gold_evidence_groups2", "gold_evidence_groups3"]
    if "gold_evidence_groups2" in sample:
        return ["gold_evidence_groups1", "gold_evidence_groups2"]
    if "gold_evidence_groups" in sample:
        return ["gold_evidence_groups"]
    raise ValueError(f"Unable to infer evidence components for sample: {sample.get('qid', '<unknown>')}")


def infer_query_type(sample: dict[str, Any]) -> str:
    query_type = infer_query_type_from_query(sample.get("query", ""))
    if query_type is not None:
        return query_type
    field_count = len(get_component_fields(sample))
    if field_count == 1:
        return "fact"
    if field_count == 2:
        return "compare"
    if field_count == 3:
        return "summary"
    raise ValueError(f"Unable to infer query type for sample: {sample.get('qid', '<unknown>')}")


def evidence_pair(item: dict[str, Any]) -> tuple[str, str]:
    return item["doc_id"], item["block_id"]


def get_components(sample: dict[str, Any]) -> list[list[set[tuple[str, str]]]]:
    components: list[list[set[tuple[str, str]]]] = []
    for field_name in get_component_fields(sample):
        groups = []
        for group in sample.get(field_name, []):
            groups.append({evidence_pair(item) for item in group})
        components.append(groups)
    return components


def get_retrieved_pairs(hit) -> set[tuple[str, str]]:
    metadata = hit.metadata or {}
    doc_id = metadata.get("doc_id") or hit.doc_id
    block_ids = metadata.get("block_ids") or []
    return {(doc_id, block_id) for block_id in block_ids}


def get_component_match_flags(sample: dict[str, Any], retrieved_pairs: set[tuple[str, str]]) -> list[bool]:
    flags = []
    for groups in get_components(sample):
        matched = False
        for group in groups:
            if retrieved_pairs & group:
                matched = True
                break
        flags.append(matched)
    return flags


def recall_at_k(sample: dict[str, Any], hits: list[Any], top_k: int) -> float:
    retrieved_pairs: set[tuple[str, str]] = set()
    for hit in hits[:top_k]:
        retrieved_pairs |= get_retrieved_pairs(hit)
    return float(all(get_component_match_flags(sample, retrieved_pairs)))


def reciprocal_rank_at_10(sample: dict[str, Any], hits: list[Any]) -> float:
    retrieved_pairs: set[tuple[str, str]] = set()
    for rank, hit in enumerate(hits[:10], start=1):
        retrieved_pairs |= get_retrieved_pairs(hit)
        if all(get_component_match_flags(sample, retrieved_pairs)):
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains: list[float], top_k: int) -> float:
    score = 0.0
    for rank, gain in enumerate(gains[:top_k], start=1):
        if gain <= 0.0:
            continue
        score += gain / math.log2(rank + 1)
    return score


def ndcg_at_10(sample: dict[str, Any], hits: list[Any]) -> float:
    gains = []
    for hit in hits[:10]:
        flags = get_component_match_flags(sample, get_retrieved_pairs(hit))
        gains.append(float(sum(flags)))
    if not gains or max(gains) <= 0.0:
        return 0.0
    dcg = dcg_at_k(gains, 10)
    idcg = dcg_at_k(sorted(gains, reverse=True), 10)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def group_to_serializable(group: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"doc_id": doc_id, "block_id": block_id} for doc_id, block_id in sorted(group)]


def matched_pairs_for_hit(sample: dict[str, Any], hit) -> list[dict[str, Any]]:
    retrieved_pairs = get_retrieved_pairs(hit)
    matched: list[dict[str, Any]] = []
    for component_index, groups in enumerate(get_components(sample), start=1):
        for group_index, group in enumerate(groups, start=1):
            overlap = sorted(retrieved_pairs & group)
            if overlap:
                matched.append(
                    {
                        "component_index": component_index,
                        "group_index": group_index,
                        "matched_pairs": [{"doc_id": doc_id, "block_id": block_id} for doc_id, block_id in overlap],
                    }
                )
    return matched


def component_summary(sample: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for component_index, groups in enumerate(get_components(sample), start=1):
        summary.append(
            {
                "component_index": component_index,
                "group_count": len(groups),
                "groups": [group_to_serializable(group) for group in groups],
            }
        )
    return summary


def format_hit(sample: dict[str, Any], rank: int, hit, show_text_chars: int) -> dict[str, Any]:
    metadata = hit.metadata or {}
    display_text = bm25_body_like_text(
        chunk_type=hit.chunk_type,
        text=hit.text,
        metadata=metadata,
    )
    payload = {
        "rank": rank,
        "doc_id": hit.doc_id,
        "chunk_id": hit.chunk_id,
        "chunk_type": hit.chunk_type,
        "source_pdf": hit.source_pdf,
        "block_ids": metadata.get("block_ids") or [],
        "matched_components": matched_pairs_for_hit(sample, hit),
        "text": trim_text(display_text, show_text_chars),
    }
    if hit.linked_raw_table is not None:
        linked_metadata = hit.linked_raw_table.get("metadata", {}) or {}
        payload["linked_raw_table"] = {
            "table_id": linked_metadata.get("table_id"),
            "block_ids": linked_metadata.get("block_ids") or [],
            "text": trim_text(
                bm25_body_like_text(
                    chunk_type="raw_table_chunk",
                    text=hit.linked_raw_table.get("text", ""),
                    metadata=linked_metadata,
                ),
                show_text_chars,
            ),
        }
    return payload


def evaluate_route(
    retriever: ThreeWayRetriever,
    samples: list[dict[str, Any]],
    *,
    route_config: dict[str, Any],
    rerank_batch_size: int,
    title_weight: float,
    show_text_chars: int,
    query_decompose: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    route_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    iterator = progress(samples, desc=f"Evaluating {route_config['route']}", total=len(samples))
    for sample in iterator:
        inferred_query_type = infer_query_type(sample)
        decompose_plan = None
        if query_decompose:
            decompose_plan = retriever.get_query_decompose_plan(
                sample["query"],
                final_top_k=route_config["rerank_top_k"],
            )
        hits = retriever.retrieve_two_stage(
            query=sample["query"],
            route=route_config["route"],
            first_stage_top_k=route_config["first_stage_top_k"],
            rerank_top_k=route_config["rerank_top_k"],
            vector_candidate_k=route_config["vector_candidate_k"],
            bm25_candidate_k=route_config["bm25_candidate_k"],
            vector_weight=route_config["vector_weight"],
            bm25_weight=route_config["bm25_weight"],
            title_weight=title_weight,
            fusion_mode=route_config["fusion_mode"],
            rrf_k=route_config["rrf_k"],
            rerank_batch_size=rerank_batch_size,
            query_decompose=query_decompose,
        )

        metrics = {
            "Recall@3": recall_at_k(sample, hits, 3),
            "Recall@5": recall_at_k(sample, hits, 5),
            "Recall@10": recall_at_k(sample, hits, 10),
            "MRR@10": reciprocal_rank_at_10(sample, hits),
            "NDCG@10": ndcg_at_10(sample, hits),
        }
        metric_rows.append({"query_type": inferred_query_type, **metrics})
        route_results.append(
            {
                "qid": sample["qid"],
                "query_type": inferred_query_type,
                "query": sample["query"],
                "answer": sample.get("answer"),
                "evidence_text": sample.get("evidence_text"),
                "evidence_components": component_summary(sample),
                "decompose_plan": decompose_plan,
                "metrics": metrics,
                "results": [format_hit(sample, rank, hit, show_text_chars) for rank, hit in enumerate(hits, start=1)],
            }
        )

    elapsed_seconds = time.perf_counter() - start_time
    return summarize_metrics(metric_rows), route_results, elapsed_seconds


def average_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not metric_rows:
        return {key: 0.0 for key in METRIC_KEYS}
    return {key: sum(float(row[key]) for row in metric_rows) / len(metric_rows) for key in METRIC_KEYS}


def summarize_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "overall": {"count": len(metric_rows), **average_metrics(metric_rows)},
        "by_type": {},
    }
    for query_type in QUERY_TYPE_ORDER:
        rows = [row for row in metric_rows if row["query_type"] == query_type]
        summary["by_type"][query_type] = {"count": len(rows), **average_metrics(rows)}
    return summary


def percent(value: float) -> str:
    return f"{value:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def query_decompose_tag(enabled: bool) -> str:
    return "qd_on" if enabled else "qd_off"


def route_result_filename(route: str, run_ts: str, *, query_decompose: bool) -> str:
    return f"eval_{route}__rerank_on__{query_decompose_tag(query_decompose)}__{run_ts}.json"


def route_readable_filename(route: str, run_ts: str, *, query_decompose: bool) -> str:
    return f"eval_{route}__rerank_on__{query_decompose_tag(query_decompose)}__{run_ts}__readable.md"


def summary_filename(run_ts: str, *, query_decompose: bool) -> str:
    return f"eval_rerank_summary__{query_decompose_tag(query_decompose)}__{run_ts}.md"


def matched_block_ids(hit: dict[str, Any]) -> set[str]:
    block_ids: set[str] = set()
    for component in hit.get("matched_components", []):
        for pair in component.get("matched_pairs", []):
            block_id = pair.get("block_id")
            if block_id:
                block_ids.add(block_id)
    return block_ids


def format_block_ids(block_ids: list[str], matched_ids: set[str]) -> str:
    formatted = [f"**{block_id}**" if block_id in matched_ids else block_id for block_id in block_ids]
    return "[" + ", ".join(formatted) + "]"


def format_matched_components(hit: dict[str, Any]) -> str:
    parts = []
    for component in hit.get("matched_components", []):
        pairs = ", ".join(
            f"({pair['doc_id']}, **{pair['block_id']}**)" for pair in component.get("matched_pairs", [])
        )
        parts.append(
            f"component {component['component_index']} / group {component['group_index']}: {pairs}"
        )
    return "；".join(parts)


def build_readable_markdown(route: str, route_results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        f"# {route} Eval Readable Output",
        "",
        "## Route Summary",
        "",
        f"- Query count: {summary['overall']['count']}",
        f"- Recall@3: {percent(summary['overall']['Recall@3'])}",
        f"- Recall@5: {percent(summary['overall']['Recall@5'])}",
        f"- Recall@10: {percent(summary['overall']['Recall@10'])}",
        f"- MRR@10: {percent(summary['overall']['MRR@10'])}",
        f"- NDCG@10: {percent(summary['overall']['NDCG@10'])}",
        "",
    ]

    for item in route_results:
        lines.extend(
            [
                f"## {item['qid']} [{item['query_type']}]",
                "",
                f"**Query**: {item['query']}",
                "",
            ]
        )
        decompose_plan = item.get("decompose_plan")
        if decompose_plan:
            lines.extend(
                [
                    "**Decomposed Queries**:",
                    "",
                    f"- query_type: {decompose_plan.get('query_type')}",
                    f"- branch_candidate_k: {decompose_plan.get('branch_candidate_k')}",
                    f"- branch_output_k: {decompose_plan.get('branch_output_k')}",
                    f"- final_top_k: {decompose_plan.get('final_top_k')}",
                ]
            )
            for idx, subquery in enumerate(decompose_plan.get("subqueries", []), start=1):
                lines.append(f"- subquery_{idx}: {subquery}")
            lines.extend(
                [
                    "",
                ]
            )
        lines.extend(
            [
                f"**Answer**: {item.get('answer') or ''}",
                "",
                "### Top 10 Chunks",
                "",
            ]
        )
        for hit in item["results"]:
            hit_matched_block_ids = matched_block_ids(hit)
            matched_title = " **[命中召回]**" if hit_matched_block_ids else ""
            lines.extend(
                [
                    f"#### Rank {hit['rank']}{matched_title}",
                    "",
                    f"- doc_id: {hit['doc_id']}",
                    f"- chunk_id: {hit['chunk_id']}",
                    f"- chunk_type: {hit['chunk_type']}",
                    f"- block_ids: {format_block_ids(hit['block_ids'], hit_matched_block_ids)}",
                ]
            )
            if hit_matched_block_ids:
                lines.extend(
                    [
                        f"- **matched_components**: {format_matched_components(hit)}",
                    ]
                )
            lines.extend(
                [
                    "",
                    "```text",
                    hit["text"],
                    "```",
                    "",
                ]
            )
            if "linked_raw_table" in hit:
                lines.extend(
                    [
                        "- linked_raw_table_block_ids: "
                        + format_block_ids(hit["linked_raw_table"]["block_ids"], hit_matched_block_ids),
                        "",
                        "```text",
                        hit["linked_raw_table"]["text"],
                        "```",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def build_overall_table(route_summaries: dict[str, dict[str, Any]]) -> str:
    rows = []
    for route in ("hybrid_weightsum", "hybrid_rrf"):
        metrics = route_summaries[route]["overall"]
        rows.append(
            [
                route,
                str(metrics["count"]),
                percent(metrics["Recall@3"]),
                percent(metrics["Recall@5"]),
                percent(metrics["Recall@10"]),
                percent(metrics["MRR@10"]),
                percent(metrics["NDCG@10"]),
            ]
        )
    return markdown_table(
        ["Route", "Count", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "NDCG@10"],
        rows,
    )


def build_type_table(route_summaries: dict[str, dict[str, Any]]) -> str:
    rows = []
    for route in ("hybrid_weightsum", "hybrid_rrf"):
        for query_type in QUERY_TYPE_ORDER:
            metrics = route_summaries[route]["by_type"][query_type]
            rows.append(
                [
                    route,
                    query_type,
                    str(metrics["count"]),
                    percent(metrics["Recall@3"]),
                    percent(metrics["Recall@5"]),
                    percent(metrics["Recall@10"]),
                    percent(metrics["MRR@10"]),
                    percent(metrics["NDCG@10"]),
                ]
            )
    return markdown_table(
        ["Route", "Type", "Count", "Recall@3", "Recall@5", "Recall@10", "MRR@10", "NDCG@10"],
        rows,
    )


def winner_text(lhs: float, rhs: float, higher_is_better: bool = True) -> str:
    if math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-12):
        return "两种方法持平"
    if (lhs > rhs) == higher_is_better:
        return "hybrid_weightsum 更好"
    return "hybrid_rrf 更好"


def build_analysis(route_summaries: dict[str, dict[str, Any]]) -> list[str]:
    analysis = []
    ws_overall = route_summaries["hybrid_weightsum"]["overall"]
    rrf_overall = route_summaries["hybrid_rrf"]["overall"]
    analysis.append(
        "总体上，"
        f"{winner_text(ws_overall['Recall@10'], rrf_overall['Recall@10'])} 在 Recall@10 上更占优，"
        f"weightsum={percent(ws_overall['Recall@10'])}，rrf={percent(rrf_overall['Recall@10'])}。"
    )
    analysis.append(
        "从排序质量看，"
        f"{winner_text(ws_overall['MRR@10'], rrf_overall['MRR@10'])} 在 MRR@10 上表现更强，"
        f"weightsum={percent(ws_overall['MRR@10'])}，rrf={percent(rrf_overall['MRR@10'])}。"
    )
    analysis.append(
        "按类型拆分时，建议重点看 compare 和 summary，"
        "因为这两类题必须满足跨组件联合命中，能更直接反映 rerank 后的多证据覆盖能力。"
    )
    for query_type in QUERY_TYPE_ORDER:
        ws = route_summaries["hybrid_weightsum"]["by_type"][query_type]
        rrf = route_summaries["hybrid_rrf"]["by_type"][query_type]
        analysis.append(
            f"{query_type} 类型上，"
            f"Recall@10 {winner_text(ws['Recall@10'], rrf['Recall@10'])}，"
            f"weightsum={percent(ws['Recall@10'])}，rrf={percent(rrf['Recall@10'])}；"
            f"MRR@10 {winner_text(ws['MRR@10'], rrf['MRR@10'])}，"
            f"weightsum={percent(ws['MRR@10'])}，rrf={percent(rrf['MRR@10'])}。"
        )
    return analysis


def build_summary_markdown(
    *,
    run_ts: str,
    args: argparse.Namespace,
    route_summaries: dict[str, dict[str, Any]],
    route_elapsed_seconds: dict[str, float],
    route_json_paths: dict[str, Path],
    route_readable_paths: dict[str, Path],
) -> str:
    experiment_record = [
        f"- 运行时间戳: {run_ts}",
        f"- Query decompose 标记: `{query_decompose_tag(args.query_decompose)}`",
        f"- 评测集: `{args.eval_path}`",
        f"- 索引: `{args.index_dir}`",
        "- 一阶段召回 1: `hybrid_weightsum`, vector/BM25 = `0.3/0.7`, top-50",
        "- 一阶段召回 2: `hybrid_rrf`, `rrf_k=60`, vector_candidate_k=`50`, bm25_candidate_k=`50`, rrf_top_k=`50`",
        f"- Query decompose: {'开启' if args.query_decompose else '关闭'}",
        "- 开启后仅对 compare/summary 生效；fact 仍走原始单 query 检索链路",
        f"- 二阶段重排: `{args.rerank_model_path}`，rerank top-10，batch_size=`{args.rerank_batch_size}`",
        f"- BM25 bonus: title/company 文档级加分，weight=`{args.title_weight}`",
        "- 当前索引设计: 只检索 `text_chunk` / `raw_table_chunk`；rerank 输入与向量检索增强文本保持一致",
        "- 评测规则: 严格执行“组内 OR、组间 AND”",
        "- fact: 命中 `gold_evidence_groups` 中任意一组即可",
        "- compare: 必须同时命中 `gold_evidence_groups1` 和 `gold_evidence_groups2`",
        "- summary: 必须同时命中 `gold_evidence_groups1`、`gold_evidence_groups2` 和 `gold_evidence_groups3`",
        f"- chunk 文本截断: {'不截断' if args.show_text_chars <= 0 else args.show_text_chars}",
        f"- 输出目录: `{args.output_dir}`",
        f"- hybrid_weightsum 耗时: {route_elapsed_seconds['hybrid_weightsum']:.2f}s",
        f"- hybrid_rrf 耗时: {route_elapsed_seconds['hybrid_rrf']:.2f}s",
        f"- hybrid_weightsum JSON: `{route_json_paths['hybrid_weightsum'].name}`",
        f"- hybrid_rrf JSON: `{route_json_paths['hybrid_rrf'].name}`",
        f"- hybrid_weightsum 阅读版: `{route_readable_paths['hybrid_weightsum'].name}`",
        f"- hybrid_rrf 阅读版: `{route_readable_paths['hybrid_rrf'].name}`",
    ]

    metric_notes = [
        "- Recall@3 / Recall@5 / Recall@10: 在 top-k 内是否满足该 query 类型要求的全部组件命中条件。",
        "- MRR@10: 从前往后看，prefix 首次满足完整组件条件时的倒数排名；10 名内无法满足则记 0。",
        "- NDCG@10: 每个 chunk 的 graded relevance 等于它命中的必需组件数，按 top-10 的理想重排做归一化。",
    ]

    analysis_lines = build_analysis(route_summaries)

    sections = [
        "# Two-Stage Rerank Eval Summary",
        "",
        "## Overall Metrics",
        "",
        build_overall_table(route_summaries),
        "",
        "## Metrics By Query Type",
        "",
        build_type_table(route_summaries),
        "",
        "## Analysis",
        "",
        *[f"- {line}" for line in analysis_lines],
        "",
        "## Experiment Record",
        "",
        *experiment_record,
        "",
        "## Metric Notes",
        "",
        *metric_notes,
        "",
    ]
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    samples = load_samples(args.eval_path)
    if args.limit > 0:
        samples = samples[: args.limit]

    clean_output_dir(args.output_dir)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    retriever = ThreeWayRetriever(
        args.index_dir,
        reranker_model_path=args.rerank_model_path,
        reranker_use_fp16=args.reranker_use_fp16,
    )

    route_summaries: dict[str, dict[str, Any]] = {}
    route_elapsed_seconds: dict[str, float] = {}
    route_json_paths: dict[str, Path] = {}
    route_readable_paths: dict[str, Path] = {}

    for route_config in ROUTE_CONFIGS:
        route = route_config["route"]
        summary, route_results, elapsed_seconds = evaluate_route(
            retriever,
            samples,
            route_config=route_config,
            rerank_batch_size=args.rerank_batch_size,
            title_weight=args.title_weight,
            show_text_chars=args.show_text_chars,
            query_decompose=args.query_decompose,
        )

        route_json_path = args.output_dir / route_result_filename(route, run_ts, query_decompose=args.query_decompose)
        route_readable_path = args.output_dir / route_readable_filename(
            route,
            run_ts,
            query_decompose=args.query_decompose,
        )

        write_json(
            route_json_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query_decompose_tag": query_decompose_tag(args.query_decompose),
                "eval_path": str(args.eval_path),
                "index_dir": str(args.index_dir),
                "route": route,
                "retrieval_config": {
                    **route_config,
                    "title_weight": args.title_weight,
                    "query_decompose": args.query_decompose,
                    "rerank_batch_size": args.rerank_batch_size,
                    "rerank_model_path": args.rerank_model_path,
                },
                "metrics": summary,
                "queries": route_results,
            },
        )
        write_text(route_readable_path, build_readable_markdown(route, route_results, summary))

        route_summaries[route] = summary
        route_elapsed_seconds[route] = elapsed_seconds
        route_json_paths[route] = route_json_path
        route_readable_paths[route] = route_readable_path

    summary_path = args.output_dir / summary_filename(run_ts, query_decompose=args.query_decompose)
    write_text(
        summary_path,
        build_summary_markdown(
            run_ts=run_ts,
            args=args,
            route_summaries=route_summaries,
            route_elapsed_seconds=route_elapsed_seconds,
            route_json_paths=route_json_paths,
            route_readable_paths=route_readable_paths,
        ),
    )

    output_manifest = {
        "output_dir": str(args.output_dir),
        "files": [
            route_json_paths["hybrid_weightsum"].name,
            route_json_paths["hybrid_rrf"].name,
            route_readable_paths["hybrid_weightsum"].name,
            route_readable_paths["hybrid_rrf"].name,
            summary_path.name,
        ],
    }
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
