#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/shared-nvme/data/ARKitScenes_processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/data/arkit_coco}"
NUM_VIEWS="${NUM_VIEWS:-600}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
ANN_FILE="${DATA_ROOT}/arkit_infos_train.pkl"
SCENE_OUTPUT_DIR="${SCENE_OUTPUT_DIR:-${OUTPUT_ROOT}/keypoints_bbox_train_scenes}"

mkdir -p "${OUTPUT_ROOT}" "${SCENE_OUTPUT_DIR}"

cd "${REPO_ROOT}"
if [[ ! -f "${ANN_FILE}" ]]; then
    --output "${ANN_FILE}"
fi

exec "${PYTHON_BIN}" utils/scannet_3d_to_coco_bbox.py \
  --data-root "${DATA_ROOT}" \
  --ann-file "${ANN_FILE}" \
  --output-dir "${SCENE_OUTPUT_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-id "${SHARD_ID}" \
  --num-views "${NUM_VIEWS}" \
  --sampling uniform \
  --depth-scale 1000 \
  --depth-window-radius 1 \
  --min-visible-points 3 \
  --min-visible-ratio 0.0 \
  --bbox-padding 2 \
  --center-min-samples 8 \
  --center-window-fraction 0.1 \
  --center-window-min-size 2 \
  --center-window-max-size 6 \
  --visualize \
  --visualization-dir "${OUTPUT_ROOT}/train_visualizations" \
  --visualization-max-images "${VISUALIZATION_MAX_IMAGES:--1}" \
  --log-level INFO
