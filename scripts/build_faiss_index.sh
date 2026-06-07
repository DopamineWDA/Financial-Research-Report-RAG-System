#!/usr/bin/env bash

set -euo pipefail

SECONDS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_BASE="${1:-${PROJECT_ROOT}/data_chunked}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/indexes}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/txs/.cache/huggingface/hub/models--BAAI--bge-large-zh-v1.5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
INDEX_TYPE="${INDEX_TYPE:-flat}"
INCLUDE_CHUNK_TYPES_STR="${INCLUDE_CHUNK_TYPES:-text_chunk raw_table_chunk}"
IVF_NLIST="${IVF_NLIST:-0}"
IVF_NPROBE="${IVF_NPROBE:-16}"
HNSW_M="${HNSW_M:-32}"
HNSW_EF_CONSTRUCTION="${HNSW_EF_CONSTRUCTION:-100}"
HNSW_EF_SEARCH="${HNSW_EF_SEARCH:-64}"
BUILD_SCRIPT="${PROJECT_ROOT}/scripts/build_faiss_index.py"
read -r -a INCLUDE_CHUNK_TYPES <<< "${INCLUDE_CHUNK_TYPES_STR}"

if [[ ! -f "${BUILD_SCRIPT}" ]]; then
  echo "Missing build script: ${BUILD_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_BASE}" ]]; then
  echo "Input base directory does not exist: ${INPUT_BASE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}"

echo "Input base   : ${INPUT_BASE}"
echo "Output base  : ${OUTPUT_BASE}"
echo "Python bin   : ${PYTHON_BIN}"
echo "Model path   : ${MODEL_PATH}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Max length   : ${MAX_LENGTH}"
echo "Force rebuild: ${FORCE_REBUILD}"
echo "Index type   : ${INDEX_TYPE}"
echo "Chunk types  : ${INCLUDE_CHUNK_TYPES[*]}"
echo "IVF nlist    : ${IVF_NLIST}"
echo "IVF nprobe   : ${IVF_NPROBE}"
echo "HNSW M       : ${HNSW_M}"
echo "HNSW efC     : ${HNSW_EF_CONSTRUCTION}"
echo "HNSW efS     : ${HNSW_EF_SEARCH}"
echo

mapfile -t CHUNK_DIRS < <(find "${INPUT_BASE}" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ "${#CHUNK_DIRS[@]}" -eq 0 ]]; then
  echo "No chunk directories found under: ${INPUT_BASE}" >&2
  exit 1
fi

for chunk_dir in "${CHUNK_DIRS[@]}"; do
  job_start=${SECONDS}
  chunk_name="$(basename "${chunk_dir}")"
  output_dir="${OUTPUT_BASE}/faiss_${INDEX_TYPE}_${chunk_name}_bge-large-zh-v1.5"
  index_path="${output_dir}/index.faiss"
  docstore_path="${output_dir}/docstore.jsonl"
  id_map_path="${output_dir}/id_map.json"
  build_meta_path="${output_dir}/build_meta.json"
  embedding_path="${output_dir}/embeddings.npy"

  if [[ "${FORCE_REBUILD}" != "1" ]] \
    && [[ -f "${index_path}" ]] \
    && [[ -f "${docstore_path}" ]] \
    && [[ -f "${id_map_path}" ]] \
    && [[ -f "${build_meta_path}" ]] \
    && [[ -f "${embedding_path}" ]]; then
    echo "==> Skipping existing FAISS ${INDEX_TYPE} index for: ${chunk_name}"
    echo "    output_dir=${output_dir}"
    echo "elapsed_seconds=${SECONDS-job_start}"
    echo
    continue
  fi

  echo "==> Building FAISS ${INDEX_TYPE} index for: ${chunk_name}"
  "${PYTHON_BIN}" "${BUILD_SCRIPT}" \
    --input-dir "${chunk_dir}" \
    --output-dir "${output_dir}" \
    --model-path "${MODEL_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --index-type "${INDEX_TYPE}" \
    --include-chunk-types "${INCLUDE_CHUNK_TYPES[@]}" \
    --ivf-nlist "${IVF_NLIST}" \
    --ivf-nprobe "${IVF_NPROBE}" \
    --hnsw-m "${HNSW_M}" \
    --hnsw-ef-construction "${HNSW_EF_CONSTRUCTION}" \
    --hnsw-ef-search "${HNSW_EF_SEARCH}"
  echo "elapsed_seconds=${SECONDS-job_start}"
  echo
done

echo "Done. Indexes written under: ${OUTPUT_BASE}"
echo "total_elapsed_seconds=${SECONDS}"
