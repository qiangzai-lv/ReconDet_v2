_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(imports=['recondet'], allow_failed_imports=False)

data_root = '/root/shared-nvme/data/ScanNet_processed_v2'
scannet_2d_ann_root = '/root/shared-nvme/data/scannet_coco_v2'
gt_points_dir = f'{data_root}/points'
vggt_omega_checkpoint = '/root/shared-nvme/data/vggt-omega/vggt_omega_1b_512.pt'

grounding_dino_config = 'configs/gdino/grounding_dino_swin-t_pretrain_obj365.py'
grounding_dino_checkpoint = '/root/shared-nvme/data/pretrain/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
grounding_dino_classes = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
    'toilet', 'sink', 'bathtub', 'garbage bin'
]

_token_dim_ = 512
_decoder_layer_num = 4
model = dict(
    type='ReconDet',
    vggt_omega_checkpoint=vggt_omega_checkpoint,
    g_dino_cfg=dict(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        semantic_classes=grounding_dino_classes),
    data_preprocessor=dict(
        type='ReconDetDataPreprocessor',
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
    query_clustering_cfg=dict(
        num_neighbors=4,
        class_cost_weight=0.5,
        query_cost_weight=0.25,
        min_cluster_size=0.05),
    gt_points_dir=gt_points_dir,
    supervise_2d_bbox=False,
    reconstruction_depth_loss_weight=5.0,
    reconstruction_point_loss_weight=2.0,
    supervise_camera_head=True,
    prediction_visualization=True,
    prediction_visualization_dir='work_dirs/recondet_visualizations',
    prediction_visualization_score_thr=0.1,
    camera_loss_cfg=dict(
        weight=5.0,
        loss_type='l1',
        gamma=0.6,
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=0.5,
        min_valid_points=100),
    supervise_confident_query_depth=True,
    confident_query_depth_cfg=dict(
        score_thr=0.05,
        loss_weight=5.0,
        window_fraction=0.1,
        min_window_size=2,
        max_window_size=4,
        abs_depth_tolerance=0.05,
        rel_depth_tolerance=0.01),
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
        loss_layer_ids=list(range(_decoder_layer_num))
    ),
    num_queries=256,
    token_dim=_token_dim_,
    test_only_last_layer=True,
    if_mix_precision=True,
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
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_instances_2d',
    'pose_matrix', 'axis_align_matrix', 'gt_depths_vggt',
    'gt_depth_valid_masks', 'gt_scene_points_vggt',
    'gt_extrinsics_vggt', 'gt_c2w_vggt', 'gt_intrinsics',
    'vggt_gt_scale'
]

test_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'pose_matrix',
    'axis_align_matrix'
]

train_input_modality = dict(
    use_camera=True,
    use_depth=True,
    use_lidar=False,
    use_neuralrecon_depth=False,
    use_ray=False)

test_input_modality = dict(
    use_camera=True,
    use_depth=False,
    use_lidar=False,
    use_neuralrecon_depth=False,
    use_ray=False)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=42,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ],
        loading='random',
        depth_scale=1000.0
    ),
    dict(
        type='BuildVGGTGroundTruth',
        points_root=gt_points_dir,
        num_point_features=6,
        max_depth=30.0),
    dict(type='LoadFirstFramePose'),
    dict(type='PackNeRFDetInputs', keys=train_collect_keys)
]

test_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=128,
        transforms=[
            dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(448, 448), keep_ratio=True, interpolation='bicubic'),
        ],
        loading='uniform'
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
            ann_file='scannet_infos_train_mvod.pkl',
            ann_file_2d=f'{scannet_2d_ann_root}/keypoints_bbox_train.json',
            pipeline=train_pipeline,
            modality=train_input_modality,
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
        ann_file='scannet_infos_val_mvod.pkl',
        pipeline=test_pipeline,
        modality=test_input_modality,
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
