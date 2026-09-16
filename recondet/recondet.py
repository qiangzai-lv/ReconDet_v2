from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from recondet.camera_alignment import (
    denormalize_vggt_boxes, denormalize_vggt_gt_cameras,
    denormalize_vggt_gt_points,
    load_axis_aligned_points, normalize_query_points)
from recondet.detr3_models.helpers import GenericMLP
from recondet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from recondet.device import autocast, get_device
from recondet.feature_projection import VGGTFeatureProjector
from recondet.geometry_attention import GeometryAwareDeformableDecoder
from recondet.grounding_dino_encoder import GroundingDINOSemanticEncoder
from recondet.query_correspondence import select_reconstruction_boxes
from recondet.reconstruction_object_head import ReconstructionObjectHead
from recondet.vggt_camera_loss import (
    compute_vggt_camera_loss, VGGT_CAMERA_LOSS_DEFAULTS)
from recondet.vggt_ground_truth import mean_point_distance, transform_points
from recondet.vggt_lora import (
    configure_vggt_lora, enable_lora_parameters,
    log_vggt_lora_summary, vggt_feature_grad_context)
from recondet.prediction_visualization import (
    save_scene_reconstruction_visualizations)
from configs.recondet.visualization_colors import CLASS_COLORS
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

device = get_device()


def resolve_reconstruction_query_dims(semantic_encoder):
    query_dims = int(
        semantic_encoder.model.bbox_head.reconstruction_dims)
    if query_dims <= 0:
        raise ValueError('reconstruction query dimensions must be positive')
    return query_dims


@MODELS.register_module()
class ReconDet(Base3DDetector):
    def __init__(
            self,
            bbox_head: ConfigType,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            g_dino_cfg: OptConfigType = None,
            decoder_cfg: OptConfigType = None,
            num_queries=128,
            token_dim=1024,
            test_only_last_layer=True,
            position_embedding="fourier",
            if_mix_precision=False,
            vggt_omega_checkpoint=None,
            vggt_lora_cfg=None,
            deformable_num_points=4,
            reconstruction_nms_cfg=None,
            query_xyz_range=(-6.5, -9.0, -1.0, 6.5, 9.0, 4.5),
            gt_points_dir=None,
            supervise_2d_bbox=False,
            train_2d_only=False,
            reconstruction_depth_loss_weight=1.0,
            reconstruction_point_loss_weight=0.5,
            reconstruction_object_head_cfg=None,
            supervise_instance_consistency=False,
            instance_consistency_cfg=None,
            scene_query_exchange_cfg=None,
            supervise_camera_head=False,
            camera_loss_cfg=None,
            supervise_confident_query_depth=False,
            confident_query_depth_cfg=None,
            prediction_visualization=False,
            prediction_visualization_dir='work_dirs/recondet_visualizations',
            prediction_visualization_score_thr=0.1,
    ):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.train_2d_only = bool(train_2d_only)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.vggt_encoder = VGGTOmega()
        self.vggt_encoder.load_state_dict(
            torch.load(vggt_omega_checkpoint, map_location='cpu', weights_only=True)
        )
        dense_head = self.vggt_encoder.dense_head
        if self.vggt_encoder.camera_head is None or dense_head is None:
            raise ValueError('VGGT camera and dense heads are required')
        self.vggt_encoder.to(device)

        self.supervise_camera_head = bool(supervise_camera_head)
        self.camera_loss_cfg = dict(VGGT_CAMERA_LOSS_DEFAULTS)
        if camera_loss_cfg is not None:
            unknown_keys = set(camera_loss_cfg) - set(self.camera_loss_cfg)
            if unknown_keys:
                raise ValueError(
                    'Unknown camera_loss_cfg keys: '
                    f'{sorted(unknown_keys)}')
            self.camera_loss_cfg.update(camera_loss_cfg)
        self.vggt_lora_summary = configure_vggt_lora(
            self.vggt_encoder.aggregator, vggt_lora_cfg)
        log_vggt_lora_summary(self.vggt_lora_summary, vggt_lora_cfg)
        self.vggt_lora_enabled = bool(
            self.vggt_lora_summary.replaced_modules)
        self._configure_vggt_trainability(training=self.training)

        # gdino encoder
        self.semantic_encoder = GroundingDINOSemanticEncoder(
            config=g_dino_cfg['grounding_dino_config'],
            checkpoint=g_dino_cfg['grounding_dino_checkpoint'],
            classes=g_dino_cfg['semantic_classes'],
            supervise_2d_bbox=supervise_2d_bbox,
            pretrain_2d_only=self.train_2d_only,
            reconstruction_depth_loss_weight=(
                reconstruction_depth_loss_weight),
            reconstruction_point_loss_weight=(
                reconstruction_point_loss_weight),
            supervise_instance_consistency=(
                supervise_instance_consistency),
            instance_consistency_cfg=instance_consistency_cfg,
            scene_query_exchange_cfg=scene_query_exchange_cfg,
            supervise_confident_query_depth=(
                supervise_confident_query_depth),
            confident_query_depth_cfg=confident_query_depth_cfg)
        semantic_query_dims = self.semantic_encoder.model.embed_dims
        reconstruction_query_dims = resolve_reconstruction_query_dims(
            self.semantic_encoder)
        object_head_cfg = dict(reconstruction_object_head_cfg or {})
        object_head_cfg.setdefault('semantic_dims', semantic_query_dims)
        object_head_cfg.setdefault(
            'num_classes', len(g_dino_cfg['semantic_classes']))
        self.reconstruction_object_head = ReconstructionObjectHead(
            query_dims=reconstruction_query_dims, **object_head_cfg)
        if reconstruction_query_dims != token_dim:
            raise ValueError(
                'Fused reconstruction queries must match decoder dimensions')

        # detection decoder
        self.decoder = GeometryAwareDeformableDecoder(
            embed_dims=token_dim,
            num_layers=decoder_cfg['dec_nlayers'],
            num_heads=decoder_cfg['dec_nhead'],
            feedforward_channels=decoder_cfg['dec_ffn_dim'],
            num_feature_levels=4,
            num_points=deformable_num_points,
            dropout=decoder_cfg['dec_dropout'])

        self.feature_projector = VGGTFeatureProjector(
            dense_head=dense_head,
            dim_in=dense_head.norm.normalized_shape[0],
            out_channels=[token_dim] * len(
                dense_head.intermediate_layer_idx))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_queries = num_queries
        reconstruction_nms_cfg = dict(reconstruction_nms_cfg or {})
        unknown_nms_keys = set(reconstruction_nms_cfg) - {
            'iou_thr', 'foreground_score_thr'}
        if unknown_nms_keys:
            raise ValueError(
                f'Unknown reconstruction NMS keys: {sorted(unknown_nms_keys)}')
        self.reconstruction_nms_iou_thr = float(
            reconstruction_nms_cfg.get('iou_thr', 0.25))
        self.reconstruction_foreground_score_thr = float(
            reconstruction_nms_cfg.get('foreground_score_thr', 0.1))
        self.test_only_last_layer = test_only_last_layer

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=token_dim, pos_type=position_embedding, normalize=False
        )
        self.query_projection = GenericMLP(
            input_dim=token_dim,
            hidden_dims=[token_dim],
            output_dim=token_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.if_mix_precision = if_mix_precision
        if len(query_xyz_range) != 6:
            raise ValueError('query_xyz_range must contain 6 values')
        self.query_xyz_range = tuple(float(value) for value in query_xyz_range)
        if gt_points_dir is None:
            raise ValueError('VGGT GT inverse alignment requires gt_points_dir')
        self.gt_points_dir = Path(gt_points_dir)
        self.prediction_visualization = bool(prediction_visualization)
        self.prediction_visualization_dir = Path(prediction_visualization_dir)
        self.prediction_visualization_score_thr = float(
            prediction_visualization_score_thr)
        self._vggt_gt_scale_cache = {}
        if self.train_2d_only:
            self._configure_2d_pretrain_trainability()

    def _configure_2d_pretrain_trainability(self):
        frozen_modules = (
            self.vggt_encoder, self.feature_projector,
            self.reconstruction_object_head,
            self.decoder, self.bbox_head,
            self.pos_embedding, self.query_projection)
        for module in frozen_modules:
            module.requires_grad_(False)
            module.eval()

    def _configure_vggt_trainability(self, training):
        self.vggt_encoder.requires_grad_(False)
        self.vggt_encoder.eval()
        if self.train_2d_only:
            return
        if self.vggt_lora_enabled:
            enable_lora_parameters(self.vggt_encoder.aggregator)
            self.vggt_encoder.aggregator.train(bool(training))
        camera_head = self.vggt_encoder.camera_head
        camera_head.requires_grad_(self.supervise_camera_head)
        camera_head.train(bool(training and self.supervise_camera_head))

    def train(self, mode=True):
        super().train(mode)
        self._configure_vggt_trainability(training=mode)
        if self.train_2d_only:
            self._configure_2d_pretrain_trainability()
        return self

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList, mode):
        keep_graph = self.training and self.vggt_lora_enabled
        with vggt_feature_grad_context(keep_graph):
            # The data preprocessor converts raw BGR uint8 images to RGB without
            # normalization. VGGT-Omega expects RGB values in [0, 1].
            img = batch_inputs_dict['imgs'].float().div(255.0)
            with autocast(img.device):
                aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(
                    img)
                return aggregated_tokens_list, ps_idx, img

    def _load_axis_aligned_gt_points(self, metadata):
        lidar_path = Path(metadata['lidar_path'])
        if not lidar_path.is_absolute():
            lidar_path = self.gt_points_dir / lidar_path.name
        axis_align_matrix = metadata['axis_align_matrix']
        if isinstance(axis_align_matrix, torch.Tensor):
            axis_align_matrix = axis_align_matrix.detach().cpu().numpy()
        points = load_axis_aligned_points(
            lidar_path, int(metadata.get('num_pts_feats', 6)),
            axis_align_matrix)
        return lidar_path, points

    @staticmethod
    def _matrix_item(value, batch_index):
        if isinstance(value, torch.Tensor):
            matrix = value if value.ndim == 2 else value[batch_index]
            return matrix.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value if value.ndim == 2 else value[batch_index]
        matrix = value[batch_index]
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()
        return np.asarray(matrix)

    @torch.no_grad()
    def _resolve_vggt_gt_scale(self, batch_inputs_dict, batch_data_samples,
                               reference):
        provided = batch_inputs_dict.get('vggt_gt_scale')
        if provided is not None:
            scales = torch.as_tensor(
                provided, device=reference.device,
                dtype=torch.float32).reshape(-1)
            if scales.shape != (len(batch_data_samples),):
                raise ValueError('vggt_gt_scale must have shape [B]')
            if not torch.isfinite(scales).all() or (scales <= 0).any():
                raise ValueError('vggt_gt_scale must be finite and positive')
            return scales

        scales = []
        for batch_index, data_sample in enumerate(batch_data_samples):
            metadata = data_sample.metainfo
            lidar_path, points_aligned = self._load_axis_aligned_gt_points(
                metadata)
            first_frame_pose = self._matrix_item(
                batch_inputs_dict['pose_matrix'], batch_index).astype(np.float64)
            axis_align = self._matrix_item(
                batch_inputs_dict['axis_align_matrix'], batch_index).astype(
                    np.float64)
            first_c2w_aligned = axis_align @ first_frame_pose
            cache_key = (
                str(lidar_path), first_c2w_aligned.astype(np.float32).tobytes())
            if cache_key not in self._vggt_gt_scale_cache:
                points_first = transform_points(
                    points_aligned, np.linalg.inv(first_c2w_aligned))
                self._vggt_gt_scale_cache[cache_key] = mean_point_distance(
                    points_first)
            scales.append(self._vggt_gt_scale_cache[cache_key])
        return reference.new_tensor(scales, dtype=torch.float32)

    def _build_projection_cameras(self, vggt_token_list, ps_idx, images,
                                  batch_inputs_dict, batch_data_samples,
                                  return_pose_encoding=False):
        cached_tokens = [
            token.contiguous() if token is not None else None
            for token in vggt_token_list
        ]
        camera_grad_enabled = (
            self.supervise_camera_head and self.training
            and torch.is_grad_enabled())
        with torch.set_grad_enabled(camera_grad_enabled):
            with autocast(images.device, enabled=False):
                pose_encoding = self.vggt_encoder.camera_head(
                    cached_tokens, patch_token_start=ps_idx)

        with torch.no_grad(), autocast(images.device, enabled=False):
            extrinsics, intrinsics = encoding_to_camera(
                pose_encoding.detach(), images.shape[-2:])
        scene_scale = self._resolve_vggt_gt_scale(
            batch_inputs_dict, batch_data_samples, images)
        aligned_extrinsics = denormalize_vggt_gt_cameras(
            extrinsics,
            batch_inputs_dict['pose_matrix'],
            batch_inputs_dict['axis_align_matrix'],
            scene_scale)

        intrinsics = intrinsics.float()
        if not torch.isfinite(intrinsics).all():
            raise FloatingPointError('VGGT intrinsics contain non-finite values')
        batch_inputs_dict['vggt_gt_scale'] = scene_scale.detach()
        batch_inputs_dict['vggt_raw_extrinsics'] = extrinsics.detach()
        batch_inputs_dict['vggt_extrinsics'] = aligned_extrinsics.detach()
        batch_inputs_dict['vggt_intrinsics'] = intrinsics.detach()
        del cached_tokens
        cameras = (extrinsics.detach(), aligned_extrinsics.detach(),
                   intrinsics.detach())
        if return_pose_encoding:
            return cameras + (pose_encoding,)
        return cameras

    def _align_reconstruction_outputs(self, reconstruction_outputs,
                                      batch_inputs_dict, images):
        batch_size, num_views = images.shape[:2]
        points_vggt = reconstruction_outputs['points_vggt']
        query_count = points_vggt.shape[1]
        points_vggt = points_vggt.reshape(
            batch_size, num_views, query_count, 3)
        points_aligned = denormalize_vggt_gt_points(
            points_vggt,
            batch_inputs_dict['pose_matrix'],
            batch_inputs_dict['axis_align_matrix'],
            batch_inputs_dict['vggt_gt_scale'])
        outputs = dict(reconstruction_outputs)
        outputs['points_aligned'] = points_aligned.reshape(
            batch_size * num_views, query_count, 3)
        if ('bbox_centers_vggt' in outputs
                and 'bbox_sizes_vggt' in outputs):
            centers_vggt = outputs['bbox_centers_vggt'].reshape(
                batch_size, num_views, query_count, 3)
            sizes_vggt = outputs['bbox_sizes_vggt'].reshape_as(centers_vggt)
            centers_aligned, sizes_aligned = denormalize_vggt_boxes(
                centers_vggt, sizes_vggt,
                batch_inputs_dict['pose_matrix'],
                batch_inputs_dict['axis_align_matrix'],
                batch_inputs_dict['vggt_gt_scale'])
            outputs['bbox_centers_aligned'] = centers_aligned.reshape(
                batch_size * num_views, query_count, 3)
            outputs['bbox_sizes_aligned'] = sizes_aligned.reshape(
                batch_size * num_views, query_count, 3)
        return outputs

    def _select_reconstruction_boxes(self, reconstruction_outputs, images,
                                     batch_data_samples):
        batch_size, num_views = images.shape[:2]
        from mmengine.logging import MMLogger
        selected = select_reconstruction_boxes(
            reconstruction_outputs,
            batch_size=batch_size,
            num_views=num_views,
            num_queries=self.num_queries,
            query_xyz_range=self.query_xyz_range,
            nms_iou_thr=self.reconstruction_nms_iou_thr,
            logger=MMLogger.get_current_instance(),
            scene_ids=[sample.metainfo.get('scene_id', index)
                       for index, sample in enumerate(batch_data_samples)],
            foreground_score_thr=self.reconstruction_foreground_score_thr)
        query_count = reconstruction_outputs['bbox_scores'].shape[1]
        candidate_centers = reconstruction_outputs[
            'bbox_centers_aligned'].reshape(
                batch_size, num_views * query_count, 3)
        reconstruction_points = reconstruction_outputs[
            'points_aligned'].reshape(
                batch_size, num_views * query_count, 3)
        class_scores_2d = reconstruction_outputs['class_scores_2d'].reshape(
            batch_size, num_views * query_count, -1)
        foreground_scores_2d, foreground_labels_2d = (
            class_scores_2d.max(dim=-1))
        selected['diagnostics'] = []
        for index in range(batch_size):
            valid = selected['candidate_valid_mask'][index]
            fallback = selected['fallback_mask'][index]
            real = ~fallback
            selected['diagnostics'].append(dict(
            reconstruction_points=reconstruction_points[index].detach(),
            reconstruction_scores=foreground_scores_2d[index].detach(),
            reconstruction_labels=foreground_labels_2d[index].detach(),
            boxes_before_nms=torch.cat([
                reconstruction_outputs['bbox_centers_aligned'].reshape(
                    batch_size, num_views * query_count, 3)[index][valid],
                reconstruction_outputs['bbox_sizes_aligned'].reshape(
                    batch_size, num_views * query_count, 3)[index][valid]], dim=-1).detach(),
            labels_before_nms=reconstruction_outputs['bbox_labels'].reshape(
                batch_size, num_views * query_count)[index][valid].detach(),
            boxes_after_nms=torch.cat([
                selected['query_xyz'][index][real],
                selected['query_size'][index][real]], dim=-1).detach(),
            labels_after_nms=selected['labels'][index][real].detach(),
            fallback_boxes=torch.cat([
                selected['query_xyz'][index][fallback],
                selected['query_size'][index][fallback]], dim=-1).detach()))
        return selected

    def _predict_reconstruction_objects(self, reconstruction_outputs):
        predictions = self.reconstruction_object_head(
            reconstruction_outputs['reconstruction_query'],
            reconstruction_outputs['detection_query_2d'],
            reconstruction_outputs['points_vggt'])
        outputs = dict(reconstruction_outputs)
        outputs.update(predictions)
        return outputs

    def _compute_reconstruction_object_losses(
            self, reconstruction_outputs, batch_data_samples, num_views,
            batch_inputs_dict):
        matches = reconstruction_outputs.get('reconstruction_matches')
        if matches is None:
            raise RuntimeError(
                'Object reconstruction supervision requires 2D matches')
        losses, _ = self.reconstruction_object_head.loss(
            reconstruction_outputs, matches, batch_data_samples, num_views,
            first_frame_pose=batch_inputs_dict['pose_matrix'],
            axis_align_matrix=batch_inputs_dict['axis_align_matrix'],
            scene_scale=batch_inputs_dict['vggt_gt_scale'])
        return losses

    def get_box_features(self, feature_maps, batch_inputs_dict, images,
                         query_xyz, query, extrinsics, intrinsics):
        query_xyz = query_xyz.to(device=images.device, dtype=images.dtype)
        query = query.to(device=images.device, dtype=feature_maps[0].dtype)
        reference_points, reference_min, reference_max = normalize_query_points(
            query_xyz, self.query_xyz_range)
        batch_inputs_dict['query_xyz'] = query_xyz
        batch_inputs_dict['reference_min'] = reference_min
        batch_inputs_dict['reference_max'] = reference_max
        return self.decoder(
            query,
            feature_maps,
            reference_points,
            reference_min,
            reference_max,
            extrinsics,
            intrinsics,
            images.shape[-2:],
            self.pos_embedding,
            self.query_projection,
            self.bbox_head.center_heads)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        if self.train_2d_only:
            semantic_losses = self.semantic_encoder.loss(
                batch_inputs_dict['imgs'],
                batch_data_samples,
                return_reconstruction=False)
            return {f'gdino_{name}': value
                    for name, value in semantic_losses.items()}

        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')
        vggt_feature_maps = self.feature_projector(
            vggt_token_list, img, ps_idx)
        raw_extrinsics, extrinsics, intrinsics, pose_encoding = (
            self._build_projection_cameras(
                vggt_token_list, ps_idx, img, batch_inputs_dict,
                batch_data_samples, return_pose_encoding=True))
        semantic_losses, _, reconstruction_outputs = (
            self.semantic_encoder.loss(
                batch_inputs_dict['imgs'],
                batch_data_samples,
                vggt_feature_maps=vggt_feature_maps,
                vggt_extrinsics=raw_extrinsics,
                vggt_intrinsics=intrinsics,
                gt_depths_vggt=batch_inputs_dict['gt_depths_vggt'],
                gt_depth_valid_masks=(
                    batch_inputs_dict['gt_depth_valid_masks']),
                vggt_gt_scale=batch_inputs_dict['vggt_gt_scale'],
                return_reconstruction=True))
        losses = {f'gdino_{name}': value
                  for name, value in semantic_losses.items()}
        if self.supervise_camera_head:
            losses.update(compute_vggt_camera_loss(
                pose_encoding,
                batch_inputs_dict['gt_extrinsics_vggt'],
                batch_inputs_dict['gt_intrinsics'],
                batch_inputs_dict['gt_depth_valid_masks'],
                img.shape[-2:],
                **self.camera_loss_cfg))
        reconstruction_outputs = self._predict_reconstruction_objects(
            reconstruction_outputs)
        object_losses = self._compute_reconstruction_object_losses(
            reconstruction_outputs, batch_data_samples, img.shape[1],
            batch_inputs_dict)
        losses.update({f'gdino_{name}': value
                       for name, value in object_losses.items()})
        reconstruction_outputs = self._align_reconstruction_outputs(
            reconstruction_outputs, batch_inputs_dict, img)
        selected = self._select_reconstruction_boxes(
            reconstruction_outputs, img, batch_data_samples)
        query_xyz = selected['query_xyz']
        query_size = selected['query_size']
        query = selected['detection_query']
        box_features, refined_query_xyz = self.get_box_features(
            vggt_feature_maps, batch_inputs_dict, img, query_xyz, query,
            extrinsics, intrinsics)
        detection_losses = self.bbox_head.loss(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            initial_query_sizes=query_size,
            **kwargs)
        losses.update({f'recondet_{name}': value
                       for name, value in detection_losses.items()})

        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        if self.train_2d_only:
            return self.semantic_encoder.predict_scene_2d(
                batch_inputs_dict['imgs'], batch_data_samples)

        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'test')
        vggt_feature_maps = self.feature_projector(
            vggt_token_list, img, ps_idx)
        raw_extrinsics, extrinsics, intrinsics = self._build_projection_cameras(
            vggt_token_list, ps_idx, img, batch_inputs_dict,
            batch_data_samples)
        reconstruction_outputs, _ = (
            self.semantic_encoder.predict_reconstruction(
                batch_inputs_dict['imgs'], batch_data_samples,
                vggt_feature_maps, raw_extrinsics, intrinsics))
        reconstruction_outputs = self._predict_reconstruction_objects(
            reconstruction_outputs)
        reconstruction_outputs = self._align_reconstruction_outputs(
            reconstruction_outputs, batch_inputs_dict, img)
        selected = self._select_reconstruction_boxes(
            reconstruction_outputs, img, batch_data_samples)
        query_xyz = selected['query_xyz']
        query_size = selected['query_size']
        query = selected['detection_query']
        box_features, refined_query_xyz = self.get_box_features(
            vggt_feature_maps, batch_inputs_dict, img, query_xyz, query,
            extrinsics, intrinsics)
        results_list = self.bbox_head.predict(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            initial_query_sizes=query_size,
            **kwargs)
        if self.prediction_visualization:
            self._save_prediction_visualizations(
                batch_data_samples, results_list, reconstruction_outputs,
                num_views=img.shape[1],
                cluster_diagnostics=selected['diagnostics'])
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    @torch.no_grad()
    def _save_prediction_visualizations(self, batch_data_samples, results_list,
                                        reconstruction_outputs, num_views,
                                        cluster_diagnostics=None):
        points = reconstruction_outputs['points_aligned']
        scores = reconstruction_outputs['class_scores_2d'].amax(dim=-1)
        batch_size = len(batch_data_samples)
        points = points.reshape(batch_size, num_views, *points.shape[1:])
        scores = scores.reshape(batch_size, num_views, *scores.shape[1:])
        for batch_index, (sample, result) in enumerate(
                zip(batch_data_samples, results_list)):
            metadata = sample.metainfo
            _, gt_points = self._load_axis_aligned_gt_points(metadata)
            gt = sample.gt_instances_3d
            scene_id = metadata.get('scene_id', f'scene_{batch_index}')
            diagnostics = cluster_diagnostics[batch_index]
            save_scene_reconstruction_visualizations(
                    self.prediction_visualization_dir,
                    scene_id,
                    gt_points,
                    gt.bboxes_3d, gt.labels_3d,
                    diagnostics['reconstruction_points'],
                    diagnostics['reconstruction_labels'],
                    diagnostics['boxes_before_nms'],
                    diagnostics['labels_before_nms'],
                    diagnostics['boxes_after_nms'],
                    diagnostics['labels_after_nms'],
                    diagnostics['fallback_boxes'],
                    result.bboxes_3d, result.labels_3d,
                    class_colors=CLASS_COLORS)
            from mmengine.logging import MMLogger
            MMLogger.get_current_instance().info(
                'Prediction visualization box counts: '
                f'scene={scene_id} gt_boxes={len(gt.labels_3d)} '
                f'reconstruction_before_nms='
                f'{len(diagnostics["labels_before_nms"])} '
                f'reconstruction_after_nms='
                f'{len(diagnostics["labels_after_nms"])} '
                f'fallback_boxes={len(diagnostics["fallback_boxes"])} '
                f'final_detection_boxes={len(result.labels_3d)}')

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        if self.train_2d_only:
            raise RuntimeError(
                'Tensor mode is not supported during 2D pretraining')
        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')
        vggt_feature_maps = self.feature_projector(
            vggt_token_list, img, ps_idx)
        raw_extrinsics, extrinsics, intrinsics = self._build_projection_cameras(
            vggt_token_list, ps_idx, img, batch_inputs_dict,
            batch_data_samples)

        reconstruction_outputs, _ = (
            self.semantic_encoder.predict_reconstruction(
                batch_inputs_dict['imgs'], batch_data_samples,
                vggt_feature_maps, raw_extrinsics, intrinsics))
        reconstruction_outputs = self._predict_reconstruction_objects(
            reconstruction_outputs)
        reconstruction_outputs = self._align_reconstruction_outputs(
            reconstruction_outputs, batch_inputs_dict, img)
        selected = self._select_reconstruction_boxes(
            reconstruction_outputs, img, batch_data_samples)
        query_xyz = selected['query_xyz']
        query_size = selected['query_size']
        query = selected['detection_query']
        box_features, refined_query_xyz = self.get_box_features(
            vggt_feature_maps, batch_inputs_dict, img, query_xyz, query,
            extrinsics, intrinsics)

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, refined_query_xyz,
            initial_query_sizes=query_size)
        return results
