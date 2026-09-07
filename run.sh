bash tools/dist_train.sh configs/recondet/recondet_scannet.py 1


bash tools_mmdet/dist_train.sh configs/gdino/grounding_dino_swin-t_pretrain_obj365_ori.py 1

bash tools/dist_test.sh configs/recondet/recondet_scannet.py work_dirs/recondet_scannet/epoch_40.pth 1

python utils/scannet_3d_to_coco_bbox.py \
  --data-root /root/shared-nvme/data/ScanNet_processed_v2 \
  --ann-file /root/shared-nvme/data/ScanNet_processed_v2/scannet_infos_train_mvod_with_ids.pkl \
  --output /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_train.json \
  --rejections-output /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_train_rejections.json \
  --num-views 50 \
  --sampling uniform \
  --depth-scale 1000 \
  --depth-window-radius 1 \
  --min-visible-points 3 \
  --min-visible-ratio 0.2 \
  --bbox-padding 2 \
  --visualize \
  --visualization-dir /root/shared-nvme/data/scannet_coco_v2/train_visualizations \
  --log-level INFO


python utils/scannet_3d_to_coco_bbox.py \
  --data-root /root/shared-nvme/data/ScanNet_processed_v2 \
  --ann-file /root/shared-nvme/data/ScanNet_processed_v2/scannet_infos_val_mvod_with_ids.pkl \
  --output /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_val.json \
  --rejections-output /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_val_rejections.json \
  --num-views 50 \
  --sampling uniform \
  --depth-scale 1000 \
  --depth-window-radius 1 \
  --min-visible-points 3 \
  --min-visible-ratio 0.2 \
  --bbox-padding 2 \
  --visualize \
  --visualization-dir /root/shared-nvme/data/scannet_coco_v2/val_visualizations \
  --log-level INFO

python utils/add_scannet_3d_instance_metadata.py \
  --input /root/shared-nvme/data/ScanNet_processed_v2/scannet_infos_train_mvod.pkl \
  --output /root/shared-nvme/data/ScanNet_processed_v2/scannet_infos_train_mvod_with_ids.pkl


python utils/add_scannet_3d_instance_metadata.py \
  --input /root/shared-nvme/data/ScanNet_processed_v2/scannet_infos_val_mvod.pkl \
  --output /root/shared-nvme/data/ScanNet_processed_v2/scannet_infos_val_mvod_with_ids.pkl



PYTHON_BIN=/root/miniforge3/bin/python \
MAX_PARALLEL=8 \
SHARD_START=0 \
SHARD_END=4 \
SPLIT=train \
bash tools/generate_scannet_2d_all_shards.sh


PYTHON_BIN=/root/miniforge3/bin/python \
$PYTHON_BIN utils/sample_scannet_2d_coco.py \
  --input /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_train.json \
  --output /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_train_20views.json \
  --images-per-scene 20

PYTHON_BIN=/root/miniforge3/bin/python \
$PYTHON_BIN utils/sample_scannet_2d_coco.py \
  --input /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_val.json \
  --output /root/shared-nvme/data/scannet_coco_v2/keypoints_bbox_val_10views.json \
  --images-per-scene 10