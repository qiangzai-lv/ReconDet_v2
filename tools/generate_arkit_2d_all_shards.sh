#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/shared-nvme/data/ARKitScenes_processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/data/arkit_coco}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
SHARD_START="${SHARD_START:-0}"
SHARD_END="${SHARD_END:-${MAX_PARALLEL}}"
SPLIT="${SPLIT:-all}"

if (( MAX_PARALLEL <= 0 )); then
  echo "MAX_PARALLEL must be positive" >&2
  exit 2
fi
if (( SHARD_START < 0 || SHARD_START >= MAX_PARALLEL )); then
  echo "SHARD_START must be in [0, MAX_PARALLEL)" >&2
  exit 2
fi
if (( SHARD_END <= SHARD_START || SHARD_END > MAX_PARALLEL )); then
  echo "SHARD_END must be in (SHARD_START, MAX_PARALLEL]" >&2
  exit 2
fi
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "test" && "${SPLIT}" != "all" ]]; then
  echo "SPLIT must be train, test, or all" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

running_pids=()
cleanup() {
  local pid
  for pid in "${running_pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

ensure_metadata() {
  local source_file="$1"
  local output_file="$2"
  if [[ ! -f "${output_file}" ]]; then
      --output "${output_file}"
  fi
}

run_split() {
  local split_name="$1"
  local worker_script="$2"
  local scene_dir="$3"
  local shard_id
  local pid

  mkdir -p "${scene_dir}"
  running_pids=()
  for ((shard_id = SHARD_START; shard_id < SHARD_END; shard_id++)); do
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      DATA_ROOT="${DATA_ROOT}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      NUM_SHARDS="${MAX_PARALLEL}" \
      SHARD_ID="${shard_id}" \
      SCENE_OUTPUT_DIR="${scene_dir}" \
      NUM_VIEWS="${NUM_VIEWS:-}" \
      VISUALIZATION_MAX_IMAGES="${VISUALIZATION_MAX_IMAGES:--1}" \
      bash "${worker_script}" &
    running_pids+=("$!")
  done
  for pid in "${running_pids[@]}"; do
    wait "${pid}"
  done
  running_pids=()
  echo "Completed ${split_name} shards [${SHARD_START}, ${SHARD_END})"
}

if [[ "${SPLIT}" == "train" || "${SPLIT}" == "all" ]]; then
  ensure_metadata \
    "${DATA_ROOT}/arkit_infos_train.pkl" \
    "${DATA_ROOT}/arkit_infos_train.pkl"
  run_split \
    train \
    tools/generate_arkit_2d_train.sh \
    "${OUTPUT_ROOT}/arkit_bbox_train_scenes"
fi

if [[ "${SPLIT}" == "test" || "${SPLIT}" == "all" ]]; then
  ensure_metadata \
    "${DATA_ROOT}/arkit_infos_val.pkl" \
    "${DATA_ROOT}/arkit_infos_val.pkl"
  run_split \
    test \
    tools/generate_arkit_2d_test.sh \
    "${OUTPUT_ROOT}/arkit_bbox_val_scenes"
fi
