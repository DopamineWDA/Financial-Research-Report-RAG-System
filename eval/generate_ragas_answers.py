#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.answer_query import ANSWER_SYSTEM_PROMPT, XinferenceChatClient  # noqa: E402


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model answers for a precomputed retrieval eval JSON and save RAGAS-ready outputs."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=REPO_ROOT / "output" / "decompose_1" / "eval_hybrid_weightsum__rerank_on__qd_on__20260601_233756.json",
        help="Precomputed retrieval evaluation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "ragas_answers",
        help="Directory to save per-query and merged outputs.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:9997",
        help="Xinference endpoint base URL or full chat completions URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen3-8b",
        help="Running Xinference model uid.",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many retrieved contexts from the eval JSON to send to the generator.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=900,
        help="Max chars kept for each retrieved context in the generation prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged output file if present.",
    )
    return parser.parse_args()


def load_eval_queries(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"Invalid eval JSON, missing 'queries': {path}")
    return payload, queries


def page_text(page_start: Any, page_end: Any) -> str:
    if page_start and page_end and page_start != page_end:
        return f"{page_start}-{page_end}"
    if page_start:
        return str(page_start)
    return "unknown"


def build_evidences(query_item: dict[str, Any], *, top_k: int, max_chars: int) -> list[dict[str, Any]]:
    evidences: list[dict[str, Any]] = []
    for idx, hit in enumerate(query_item.get("results", [])[:top_k], start=1):
        evidences.append(
            {
                "evidence_id": f"E{idx}",
                "rank": hit.get("rank", idx),
                "doc_id": hit.get("doc_id", ""),
                "chunk_id": hit.get("chunk_id", ""),
                "chunk_type": hit.get("chunk_type", ""),
                "source_pdf": hit.get("source_pdf", ""),
                "page_start": hit.get("page_start"),
                "page_end": hit.get("page_end"),
                "text": str(hit.get("text", ""))[:max_chars].strip(),
            }
        )
    return evidences


def format_history() -> str:
    return "无"


def format_evidence_prompt(query: str, evidences: list[dict[str, Any]]) -> str:
    sections = [f"用户问题：{query}", "", "可用证据如下："]
    for item in evidences:
        sections.extend(
            [
                f"[{item['evidence_id']}]",
                f"rank: {item['rank']}",
                f"doc_id: {item['doc_id']}",
                f"chunk_id: {item['chunk_id']}",
                f"page: {page_text(item.get('page_start'), item.get('page_end'))}",
                f"chunk_type: {item['chunk_type']}",
                f"text: {item['text']}",
                "",
            ]
        )
    return "\n".join(sections).strip()


def generate_answer(
    client: XinferenceChatClient,
    *,
    user_query: str,
    evidences: list[dict[str, Any]],
) -> dict[str, Any]:
    user_prompt = "\n".join(
        [
            f"最近对话历史：\n{format_history()}",
            "",
            f"用户原始问题：{user_query}",
            "",
            format_evidence_prompt(user_query, evidences),
        ]
    ).strip()
    try:
        return client.chat_json(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1400,
        )
    except Exception as exc:
        return {
            "answer": "现有证据不足，无法判断。",
            "claims": [],
            "used_evidence": [],
            "confidence": "low",
            "should_refuse": True,
            "generation_error": str(exc),
        }


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    claims = []
    for raw_claim in payload.get("claims", []) or []:
        claim_text = str(raw_claim.get("claim", "")).strip()
        citations = [str(item).strip() for item in raw_claim.get("citations", []) if str(item).strip()]
        if claim_text:
            claims.append({"claim": claim_text, "citations": citations})
    return {
        "answer": str(payload.get("answer", "")).strip(),
        "claims": claims,
        "used_evidence": [str(item).strip() for item in payload.get("used_evidence", []) if str(item).strip()],
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
        "should_refuse": bool(payload.get("should_refuse", False)),
        "generation_error": payload.get("generation_error"),
    }


def make_record(
    query_item: dict[str, Any],
    *,
    evidences: list[dict[str, Any]],
    model_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qid": query_item.get("qid"),
        "query_type": query_item.get("query_type"),
        "question": query_item.get("query"),
        "ground_truth": query_item.get("answer"),
        "reference": query_item.get("answer"),
        "reference_evidence_text": query_item.get("evidence_text"),
        "decompose_plan": query_item.get("decompose_plan"),
        "model_answer": model_payload["answer"],
        "answer": model_payload["answer"],
        "raw_answer_payload": model_payload,
        "retrieved_contexts": [item["text"] for item in evidences],
        "contexts": [item["text"] for item in evidences],
        "retrieved_context_records": evidences,
    }


def save_outputs(
    *,
    output_dir: Path,
    merged_payload: dict[str, Any],
    overwrite: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_json = output_dir / "answers.json"
    merged_jsonl = output_dir / "answers.jsonl"
    if (merged_json.exists() or merged_jsonl.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists in {output_dir}. Use --overwrite to replace existing files."
        )

    merged_json.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with merged_jsonl.open("w", encoding="utf-8") as fh:
        for item in merged_payload["records"]:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return merged_json, merged_jsonl


def main() -> None:
    args = parse_args()
    eval_payload, queries = load_eval_queries(args.input_json)
    run_name = f"{args.input_json.stem}__{args.model}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = args.output_dir / run_name

    client = XinferenceChatClient(endpoint=args.endpoint, model_uid=args.model, api_key=args.api_key)
    records: list[dict[str, Any]] = []

    for idx, query_item in enumerate(queries, start=1):
        question = str(query_item.get("query", "")).strip()
        evidences = build_evidences(query_item, top_k=args.top_k, max_chars=args.max_evidence_chars)
        raw_model_payload = generate_answer(client, user_query=question, evidences=evidences)
        model_payload = normalize_model_payload(raw_model_payload)
        record = make_record(query_item, evidences=evidences, model_payload=model_payload)
        records.append(record)
        print(f"[{idx:03d}/{len(queries)}] {record['qid']} done")

    merged_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_eval_json": str(args.input_json),
        "model": args.model,
        "endpoint": args.endpoint,
        "top_k": args.top_k,
        "max_evidence_chars": args.max_evidence_chars,
        "retrieval_metrics": eval_payload.get("metrics"),
        "retrieval_config": eval_payload.get("retrieval_config"),
        "records": records,
    }
    merged_json, merged_jsonl = save_outputs(
        output_dir=run_output_dir,
        merged_payload=merged_payload,
        overwrite=args.overwrite,
    )
    print(f"\nSaved merged JSON: {merged_json}")
    print(f"Saved merged JSONL: {merged_jsonl}")


if __name__ == "__main__":
    main()
