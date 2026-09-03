_base_ = ['../_base_/default_runtime.py']

import torch

try:
    import torch_npu  # noqa: F401
    _dist_backend_ = 'hccl' if torch.npu.is_available() else 'nccl'
except ImportError:
    _dist_backend_ = 'nccl'

resume = False

data_root = '/root/shared-nvme/data/ScanNet_processed'
vggt_omega_checkpoint = '/root/shared-nvme/data/vggt-omega/vggt_omega_1b_512.pt'

custom_imports = dict(imports=['recondet'], allow_failed_imports=False)

env_cfg = dict(dist_cfg=dict(backend=_dist_backend_))

_token_dim_ = 512
_decoder_layer_num = 8
model = dict(
    type='ReconDet',
    vggt_omega_checkpoint=vggt_omega_checkpoint,
    data_preprocessor=dict(
        type='VGGTDetDataPreprocessor',
        bgr_to_rgb=True,
        pad_size_divisor=16,
        pad_value=0),
    decoder_cfg=dict(  # the same with 3detr
        dec_dim=_token_dim_,
        dec_nhead=4,
        dec_ffn_dim=_token_dim_,
        dec_dropout=0.1,
        dec_nlayers=_decoder_layer_num
    ),
    deformable_num_points=4,
    geometry_source='gt',
    gt_points_dir=f'{data_root}/points',
    online_scale_point_stride=4,
    online_scale_max_depth=30.0,
    bbox_head=dict(
        type='ReconDetHead',
        n_classes=18,
        n_levels=_decoder_layer_num,
        n_channels=_token_dim_,
        n_reg_outs=6,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        mlp_dropout=0.3,
        matcher_cost_weights=dict(
            cls=1.0,
            center=0.0,
            obj_ness=0.0,
            giou=2.0
        ),
        loss_weights=dict(
            center_loss=5.0,
            size_loss=1.0,
            cls_loss=1.0,
            objness_loss=1.0,
            iou_loss=1.0,
            not_objness_loss=0.25
        ),
        learn_center_diff=True,
        if_v2_head=True,
        matcher='one2more',
        matcher_iou_thres=0.1,
        matcher_max_dynamic_samples=5,
        loss_layer_ids=list(range(_decoder_layer_num)),
        size_logit_range=(-10.0, 10.0)
    ),
    num_queries=256,
    token_dim=_token_dim_,
    test_only_last_layer=True,
    if_mix_precision=True,
    use_multi_layers=True,
    if_simpler_project=True,
    if_use_pred_pc_query=True,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.25, score_thr=.01)
)

# dataset
dataset_type = 'MultiViewScanNetDataset'

class_names = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
    'toilet', 'sink', 'bathtub', 'garbagebin'
]

train_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'pose_matrix', 'axis_align_matrix',
    'gt_camera_extrinsics', 'gt_camera_intrinsics'
]

test_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'pose_matrix', 'axis_align_matrix',
    'gt_camera_extrinsics', 'gt_camera_intrinsics'
]

input_modality = dict(
    use_camera=True,
    use_depth=False,
    use_lidar=False,
    use_neuralrecon_depth=False,
    use_ray=False)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline_Tgt',
        n_images=42,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],  # for what purpose?
        loading='gap',
        nerf_target_views=2,
        tgt_transforms=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ]
    ),
    dict(type='LoadFirstFramePose'),
    dict(type='PackNeRFDetInputs', keys=train_collect_keys)
]

test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline_Tgt',
        n_images=81,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ],
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        margin=10,
        depth_range=[0.5, 5.5],
        loading='random',
        nerf_target_views=1,
        tgt_transforms=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ]
    ),
    dict(type='LoadFirstFramePose'),
    dict(type='PackNeRFDetInputs', keys=test_collect_keys)
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='scannet_infos_train_pts.pkl',
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Depth',
            metainfo=dict(CLASSES=class_names))))

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val_pts.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d='Depth',
        metainfo=dict(CLASSES=class_names)))
test_dataloader = val_dataloader

val_evaluator = [dict(type='IndoorMetric')]
test_evaluator = val_evaluator

# train cfg
_warm_epoch = 0
_max_epoch = 200
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epoch, val_interval=2)
test_cfg = dict()
val_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2.5e-4,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=35., norm_type=2)
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=_max_epoch - 1,  # max_epochs - 1
        eta_min=1e-6,
        by_epoch=True,
        begin=_warm_epoch,
        end=_max_epoch
    )
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', save_best=['mAP_0.25'], rule="greater", interval=2, max_keep_ckpts=4),
    logger=dict(type='LoggerHook', interval=10)
)

find_unused_parameters = True
