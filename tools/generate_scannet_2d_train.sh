#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/root/shared-nvme/data/ScanNet_processed_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/data/scannet_coco_v2}"
NUM_VIEWS="${NUM_VIEWS:-600}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
ANN_FILE="${DATA_ROOT}/scannet_infos_train_mvod_with_ids.pkl"
SCENE_OUTPUT_DIR="${SCENE_OUTPUT_DIR:-${OUTPUT_ROOT}/keypoints_bbox_train_scenes}"

mkdir -p "${OUTPUT_ROOT}" "${SCENE_OUTPUT_DIR}"

cd "${REPO_ROOT}"
if [[ ! -f "${ANN_FILE}" ]]; then
  "${PYTHON_BIN}" utils/add_scannet_3d_instance_metadata.py \
    --input "${DATA_ROOT}/scannet_infos_train_mvod.pkl" \
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
  --min-visible-ratio 0.2 \
  --bbox-padding 2 \
  --center-min-samples 3 \
  --center-window-fraction 0.1 \
  --center-window-min-size 2 \
  --center-window-max-size 6 \
#  --visualize \
#  --visualization-dir "${OUTPUT_ROOT}/train_visualizations" \
#  --visualization-max-images "${VISUALIZATION_MAX_IMAGES:--1}" \
  --log-level INFO
