#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_BASE="${1:-${PROJECT_ROOT}/data_chunked}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/indexes}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/txs/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
INCLUDE_CHUNK_TYPES="${INCLUDE_CHUNK_TYPES:-text_chunk raw_table_chunk}"

IVF_NLIST="${IVF_NLIST:-0}"
IVF_NPROBE="${IVF_NPROBE:-16}"
HNSW_M="${HNSW_M:-32}"
HNSW_EF_CONSTRUCTION="${HNSW_EF_CONSTRUCTION:-100}"
HNSW_EF_SEARCH="${HNSW_EF_SEARCH:-64}"

BATCH_SCRIPT="${PROJECT_ROOT}/scripts/build_faiss_index.sh"

if [[ ! -f "${BATCH_SCRIPT}" ]]; then
  echo "Missing batch script: ${BATCH_SCRIPT}" >&2
  exit 1
fi

echo "Input base   : ${INPUT_BASE}"
echo "Output base  : ${OUTPUT_BASE}"
echo "Python bin   : ${PYTHON_BIN}"
echo "Model path   : ${MODEL_PATH}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Max length   : ${MAX_LENGTH}"
echo "Force rebuild: ${FORCE_REBUILD}"
echo "Chunk types  : ${INCLUDE_CHUNK_TYPES}"
echo "IVF nlist    : ${IVF_NLIST}"
echo "IVF nprobe   : ${IVF_NPROBE}"
echo "HNSW M       : ${HNSW_M}"
echo "HNSW efC     : ${HNSW_EF_CONSTRUCTION}"
echo "HNSW efS     : ${HNSW_EF_SEARCH}"
echo

for index_type in flat ivf hnsw; do
  echo "=============================="
  echo "Running index type: ${index_type}"
  echo "=============================="
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_BASE="${OUTPUT_BASE}" \
  MODEL_PATH="${MODEL_PATH}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  FORCE_REBUILD="${FORCE_REBUILD}" \
  INCLUDE_CHUNK_TYPES="${INCLUDE_CHUNK_TYPES}" \
  INDEX_TYPE="${index_type}" \
  IVF_NLIST="${IVF_NLIST}" \
  IVF_NPROBE="${IVF_NPROBE}" \
  HNSW_M="${HNSW_M}" \
  HNSW_EF_CONSTRUCTION="${HNSW_EF_CONSTRUCTION}" \
  HNSW_EF_SEARCH="${HNSW_EF_SEARCH}" \
  bash "${BATCH_SCRIPT}" "${INPUT_BASE}"
  echo
done

echo "Done. All index types completed."
