#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_DIR="${1:-${PROJECT_ROOT}/data_parsed}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/data_chunked}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENCODING="${ENCODING:-cl100k_base}"

SIZES_STR="${SIZES:-256 512 1024}"
OVERLAPS_STR="${OVERLAPS:-50 100 200}"

read -r -a SIZES <<< "${SIZES_STR}"
read -r -a OVERLAPS <<< "${OVERLAPS_STR}"

RECHUNK_SCRIPT="${PROJECT_ROOT}/preprocess/rechunk_parsed_pdf.py"
SUMMARY_CSV="${OUTPUT_BASE}/chunk_strategy_comparison.csv"
SUMMARY_MD="${OUTPUT_BASE}/chunk_strategy_comparison.md"

if [[ ! -f "${RECHUNK_SCRIPT}" ]]; then
  echo "Missing rechunk script: ${RECHUNK_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}"

echo "Input parsed data : ${INPUT_DIR}"
echo "Output base       : ${OUTPUT_BASE}"
echo "Chunk sizes       : ${SIZES[*]}"
echo "Overlaps          : ${OVERLAPS[*]}"
echo

for size in "${SIZES[@]}"; do
  for overlap in "${OVERLAPS[@]}"; do
    STRATEGY_DIR="${OUTPUT_BASE}/chunked_${size}_${overlap}"
    mkdir -p "${STRATEGY_DIR}"
    find "${STRATEGY_DIR}" -maxdepth 1 -type f -name '*.json' -delete

    echo "==> Running rechunk: chunk_size=${size}, overlap=${overlap}"
    "${PYTHON_BIN}" "${RECHUNK_SCRIPT}" "${INPUT_DIR}" \
      --output-dir "${STRATEGY_DIR}" \
      --chunk-size "${size}" \
      --overlap "${overlap}" \
      --encoding "${ENCODING}"
    echo
  done
done

SUMMARY_CSV="${SUMMARY_CSV}" SUMMARY_MD="${SUMMARY_MD}" OUTPUT_BASE="${OUTPUT_BASE}" \
SIZES_STR="${SIZES_STR}" OVERLAPS_STR="${OVERLAPS_STR}" "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

output_base = Path(os.environ["OUTPUT_BASE"])
summary_csv = Path(os.environ["SUMMARY_CSV"])
summary_md = Path(os.environ["SUMMARY_MD"])
sizes = [int(x) for x in os.environ["SIZES_STR"].split()]
overlaps = [int(x) for x in os.environ["OVERLAPS_STR"].split()]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?|[^\s]", text))


rows = []
for size in sizes:
    for overlap in overlaps:
        strategy_dir = output_base / f"chunked_{size}_{overlap}"
        files = sorted(strategy_dir.glob("*.json"))

        doc_count = len(files)
        total_chunks = 0
        total_chars = 0
        total_tokens = 0
        min_tokens = None
        max_tokens = None
        text_chunks = 0
        table_chunks = 0
        cover_chunks = 0
        figure_chunks = 0

        for file in files:
            data = json.loads(file.read_text(encoding="utf-8"))
            chunks = data.get("chunks", [])
            total_chunks += len(chunks)
            for chunk in chunks:
                text = chunk.get("text", "") or ""
                metadata = chunk.get("metadata", {}) or {}
                chunk_type = metadata.get("chunk_type", "") or ""
                if chunk_type == "text_chunk":
                    text_chunks += 1
                elif "table" in chunk_type:
                    table_chunks += 1
                elif chunk_type == "cover_summary_chunk":
                    cover_chunks += 1
                elif "figure" in chunk_type:
                    figure_chunks += 1

                total_chars += len(text)
                token_len = count_tokens(text)
                total_tokens += token_len
                min_tokens = token_len if min_tokens is None else min(min_tokens, token_len)
                max_tokens = token_len if max_tokens is None else max(max_tokens, token_len)

        row = {
            "chunk_size": size,
            "overlap": overlap,
            "docs": doc_count,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(safe_div(total_chunks, doc_count), 2),
            "avg_chars_per_chunk": round(safe_div(total_chars, total_chunks), 2),
            "avg_tokens_per_chunk": round(safe_div(total_tokens, total_chunks), 2),
            "min_tokens": min_tokens or 0,
            "max_tokens": max_tokens or 0,
            "text_chunks": text_chunks,
            "table_related_chunks": table_chunks,
            "cover_chunks": cover_chunks,
            "figure_chunks": figure_chunks,
            "recall_at_5": "",
        }
        rows.append(row)

fieldnames = [
    "chunk_size",
    "overlap",
    "docs",
    "total_chunks",
    "avg_chunks_per_doc",
    "avg_chars_per_chunk",
    "avg_tokens_per_chunk",
    "min_tokens",
    "max_tokens",
    "text_chunks",
    "table_related_chunks",
    "cover_chunks",
    "figure_chunks",
    "recall_at_5",
]

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with summary_md.open("w", encoding="utf-8") as f:
    f.write("# Chunk Strategy Comparison\n\n")
    f.write("| chunk_size | overlap | docs | total_chunks | avg_chunks/doc | avg_chars/chunk | avg_tokens/chunk | min_tokens | max_tokens | text_chunks | table_related_chunks | cover_chunks | figure_chunks | Recall@5 |\n")
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for row in rows:
        f.write(
            f"| {row['chunk_size']} | {row['overlap']} | {row['docs']} | {row['total_chunks']} | "
            f"{row['avg_chunks_per_doc']} | {row['avg_chars_per_chunk']} | {row['avg_tokens_per_chunk']} | "
            f"{row['min_tokens']} | {row['max_tokens']} | {row['text_chunks']} | {row['table_related_chunks']} | "
            f"{row['cover_chunks']} | {row['figure_chunks']} |  |\n"
        )

print(f"Summary CSV written to: {summary_csv}")
print(f"Summary MD written to : {summary_md}")
PY

echo
echo "Done. Comparison table:"
echo "  - ${SUMMARY_CSV}"
echo "  - ${SUMMARY_MD}"
