_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(imports=['recondet'], allow_failed_imports=False)

data_root = '/root/shared-nvme/data/ARKitScenes_processed'
gt_points_dir = f'{data_root}/points'
train_ann_file = f'{data_root}/arkit_infos_train_20.pkl'
val_ann_file = f'{data_root}/arkit_infos_val_10.pkl'
train_2d_ann_file = '/root/shared-nvme/data/arkit_coco/arkit_bbox_train.json'
val_2d_ann_file = '/root/shared-nvme/data/arkit_coco/arkit_bbox_val.json'
vggt_omega_checkpoint = (
    '/root/shared-nvme/data/vggt-omega/vggt_omega_1b_512.pt')
grounding_dino_config = (
    'configs/gdino/grounding_dino_swin-t_pretrain_obj365.py')
grounding_dino_checkpoint = (
    '/root/shared-nvme/code/Recondet_Arkit/work_dirs/'
    'grounding_dino_swin-t_pretrain_obj365_ori/epoch_1.pth')

class_names = [
    'cabinet', 'refrigerator', 'shelf', 'stove', 'bed', 'sink', 'washer',
    'toilet', 'bathtub', 'oven', 'dishwasher', 'fireplace', 'stool', 'chair',
    'table', 'tv_monitor', 'sofa'
]

_token_dim_ = 512
_decoder_layer_num = 4
_fallback_query_range_ = [-8.0, -8.0, -4.0, 8.0, 8.0, 4.0]
model = dict(
    type='ReconDet',
    vggt_omega_checkpoint=vggt_omega_checkpoint,
    vggt_lora_cfg=dict(
        enabled=True,
        block_indices=[3, 4, 10, 11, 16, 17, 22, 23],
        branches=['frame_blocks', 'inter_frame_blocks'],
        target_modules=['attn.qkv', 'attn.proj'],
        rank=8,
        alpha=16,
        dropout=0.0,
        gradient_checkpointing=True,
        checkpoint_start_block=3),
    g_dino_cfg=dict(
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint,
        semantic_classes=class_names),
    data_preprocessor=dict(
        type='ReconDetDataPreprocessor',
        bgr_to_rgb=True,
        pad_size_divisor=16,
        pad_value=0),
    decoder_cfg=dict(
        dec_dim=_token_dim_,
        dec_nhead=4,
        dec_ffn_dim=_token_dim_,
        dec_dropout=0.1,
        dec_nlayers=_decoder_layer_num),
    deformable_num_points=4,
    query_clustering_cfg=dict(
        num_neighbors=4,
        class_cost_weight=0.5,
        query_cost_weight=0.25,
        min_cluster_size=0.05),
    query_xyz_range=_fallback_query_range_,
    gt_points_dir=gt_points_dir,
    supervise_2d_bbox=False,
    reconstruction_depth_loss_weight=5.0,
    reconstruction_point_loss_weight=2.0,
    supervise_camera_head=True,
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
        n_classes=17,
        n_levels=_decoder_layer_num,
        n_channels=_token_dim_,
        n_reg_outs=7,
        pts_assign_threshold=27,
        pts_center_threshold=18,
        mlp_dropout=0.3,
        cls_loss=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        matcher_cost_weights=dict(
            cls=2.0, center=1.0, size=1.0, yaw=1.0, iou=2.0),
        loss_weights=dict(
            center_loss=2.0,
            size_loss=1.0,
            yaw_loss=1.0,
            cls_loss=2.0,
            iou_loss=2.0),
        if_v2_head=True,
        matcher='repeated_hungarian',
        initial_size_anchor=(1.0, 1.0, 1.0),
        gt_repeat_num=5,
        center_range=_fallback_query_range_,
        size_logit_range=(-5.0, 5.0),
        loss_layer_ids=list(range(_decoder_layer_num))),
    num_queries=256,
    token_dim=_token_dim_,
    test_only_last_layer=True,
    if_mix_precision=True,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.25, score_thr=0.01))

dataset_type = 'MultiViewARKitDataset'
train_collect_keys = [
    'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_instances_2d',
    'pose_matrix', 'axis_align_matrix', 'gt_depths_vggt',
    'gt_depth_valid_masks', 'gt_scene_points_vggt', 'gt_extrinsics_vggt',
    'gt_c2w_vggt', 'gt_intrinsics', 'vggt_gt_scale', 'scene_bounds'
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

image_transforms = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='Resize',
        scale=(448, 448),
        keep_ratio=True,
        interpolation='bicubic')
]
train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=40,
        transforms=image_transforms,
        loading='random',
        depth_scale=1000.0),
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
        n_images=50,
        transforms=image_transforms,
        loading='uniform'),
    dict(type='LoadFirstFramePose'),
    dict(type='PackNeRFDetInputs', keys=test_collect_keys)
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_ann_file,
            ann_file_2d=train_2d_ann_file,
            restrict_to_2d_views=True,
            pipeline=train_pipeline,
            modality=train_input_modality,
            test_mode=False,
            filter_empty_gt=True,
            box_type_3d='Depth',
            metainfo=dict(CLASSES=class_names))))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        ann_file_2d=val_2d_ann_file,
        restrict_to_2d_views=False,
        pipeline=test_pipeline,
        modality=test_input_modality,
        test_mode=True,
        filter_empty_gt=True,
        box_type_3d='Depth',
        metainfo=dict(CLASSES=class_names)))
test_dataloader = val_dataloader

val_evaluator = [dict(type='IndoorMetric')]
test_evaluator = val_evaluator
_max_epoch = 200
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=_max_epoch, val_interval=2)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2.5e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=35.0, norm_type=2))
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=_max_epoch - 1,
        eta_min=1e-6,
        by_epoch=True,
        begin=0,
        end=_max_epoch)
]
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best=['mAP_0.25'],
        rule='greater',
        interval=2,
        max_keep_ckpts=4),
    logger=dict(type='LoggerHook', interval=10))
find_unused_parameters = True
