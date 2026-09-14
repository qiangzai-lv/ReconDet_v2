# ReconDet

ReconDet 是一个面向室内场景的多视图 3D 目标检测项目。方法以 VGGT-Omega 从多张 RGB 图像中恢复的几何和重建 query 为基础，结合 GroundingDINO 的 2D 语义特征，通过跨视图 query 聚合和几何感知解码器预测场景级 3D bounding boxes。当前默认实验配置针对 ScanNet 18 类目标检测。

## 方法概览

```text
多视图 RGB
  ├─ VGGT-Omega：相机、稠密特征和重建点
  └─ GroundingDINO：2D 语义特征与类别分数
          ↓
每视图 reconstruction query
          ↓ 坐标对齐、前景筛选
跨视图 query 聚类与语义融合
          ↓
场景级 3D detection query
          ↓ 几何感知多视图 decoder
中心迭代优化 + 尺寸回归 + 类别预测
          ↓
3D box + NMS
```

训练时还使用深度、相机、重建点和 2D/3D instance 对齐监督。VGGT 主干冻结，并对指定 attention 模块使用 LoRA；GroundingDINO 的 2D 主干冻结，重建 decoder 和 reconstruction head 参与训练。

## 代码结构

```text
configs/recondet/          ReconDet 配置
recondet/recondet.py       主模型和训练/推理流程
recondet/grounding_dino_encoder.py
                            2D 语义编码与 reconstruction query
recondet/query_correspondence.py
                            query 筛选、跨视图聚类
recondet/geometry_attention.py
                            几何感知多视图 decoder
recondet/recondet_head.py  3D 中心、尺寸、类别和 NMS
recondet/multiview_pipeline.py
                            多视图采样、图像/深度/相机加载
recondet/scannet_multiview_dataset.py
                            ScanNet 多视图数据集
vggt_omega/                 VGGT-Omega 实现
tools/train.py              训练入口
tools/test.py               测试和评估入口
utils/                      ScanNet 标注及数据预处理工具
tests/                      单元和集成测试
```

## 环境

项目基于 PyTorch、MMEngine、MMCV、MMDetection 和 MMDetection3D。请先准备与本地 CUDA 匹配的 PyTorch，再安装对应版本的 OpenMMLab 依赖：

```bash
"mmcv>=2.0.0" 昇腾官方网站提供安装方案 算子也已经适配
"mmdet>=3.0.0" 没有特殊算子直接官网安装即可
"mmdet3d"  没有特殊算子
"vggt-omega"
```

## 预训练模型

默认配置 [configs/recondet/recondet_scannet.py](configs/recondet/recondet_scannet.py) 需要：

```text
VGGT-Omega checkpoint
GroundingDINO Swin-T 配置和 checkpoint
```

请在配置中修改 `vggt_omega_checkpoint` 和 `grounding_dino_checkpoint` 为本地文件路径。GroundingDINO 的类别文本与 ScanNet 18 类标签必须保持一致。

## ScanNet 数据准备

默认配置使用以下目录：

```text
/root/shared-nvme/data/ScanNet_processed_v2
/root/shared-nvme/data/scannet_coco_v2
```

如果数据位置不同，请同步修改配置中的 `data_root`、`scannet_ann_root` 和 `gt_points_dir`。

数据准备通常包括：

1. 准备 ScanNet 处理后的图像、深度、相机、点云和 3D 标注。
2. 为 train/val 标注补充 3D instance id：

```bash
python utils/add_scannet_3d_instance_metadata.py \
  --input /path/scannet_infos_train_mvod.pkl \
  --output /path/scannet_infos_train_mvod_with_ids.pkl
```

3. 生成用于 GroundingDINO 监督的多视图 COCO 标注：

```bash
python utils/scannet_3d_to_coco_bbox.py \
  --data-root /path/ScanNet_processed_v2 \
  --ann-file /path/scannet_infos_train_mvod_with_ids.pkl \
  --output /path/scannet_coco_v2/keypoints_bbox_train.json \
  --num-views 50 --sampling uniform
```

验证集使用对应的 val 标注重复执行。完整参数和分片生成脚本见 `run.sh`、`tools/generate_scannet_2d_*.sh` 及 `utils/`。

## 训练

单卡训练：

```bash
python tools/train.py configs/recondet/recondet_scannet.py
```

指定工作目录、启用 AMP 或从 checkpoint 恢复：

```bash
python tools/train.py configs/recondet/recondet_scannet.py \
  --work-dir work_dirs/recondet_scannet \
  --amp --resume
```

多卡训练：

```bash
bash tools/dist_train.sh configs/recondet/recondet_scannet.py 8
```

默认配置训练 200 个 epoch，训练阶段每个场景随机采样 40 张图像。

## 测试与评估

```bash
python tools/test.py \
  configs/recondet/recondet_scannet.py \
  work_dirs/recondet_scannet/best_mAP_0.25.pth
```

多卡测试：

```bash
bash tools/dist_test.sh \
  configs/recondet/recondet_scannet.py \
  work_dirs/recondet_scannet/epoch_40.pth 8
```

默认评估 Indoor 3D detection 的 IoU 0.25 和 0.5 mAP。测试阶段每个场景均匀采样 128 张图像。预测可视化可在配置中打开 `prediction_visualization`，或使用 MMDetection3D 的可视化参数。

## 配置要点

主要可调参数位于 `configs/recondet/recondet_scannet.py`：

- `n_images`：训练/测试使用的视图数量。
- `num_queries`：场景级 query 数量。
- `reconstruction_query_score_thr`：重建 query 前景筛选阈值。
- `query_clustering_cfg`：跨视图聚类的邻域数、类别代价和 query 代价。
- `scene_query_exchange_cfg`：跨视图语义 query 交互。
- `decoder_cfg`：几何感知 decoder 层数和通道数。
- `reconstruction_point_loss_weight`、`camera_loss_cfg`：重建和相机辅助监督。
- `bbox_head.loss_weights`：中心、尺寸、分类和 GIoU 损失权重。


