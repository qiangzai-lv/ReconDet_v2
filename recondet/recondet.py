from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from recondet.camera_alignment import (
    denormalize_vggt_gt_cameras, denormalize_vggt_gt_points,
    load_axis_aligned_points, normalize_query_points)
from recondet.detr3_models.helpers import GenericMLP
from recondet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from recondet.device import autocast, get_device
from recondet.feature_projection import VGGTFeatureProjector
from recondet.geometry_attention import GeometryAwareDeformableDecoder
from recondet.grounding_dino_encoder import GroundingDINOSemanticEncoder
from recondet.query_correspondence import (
    select_scene_reconstruction_queries, SemanticWeightedFPSClustering)
from recondet.reconstruction_object_head import (
    ReconstructionObjectHead, collect_matched_object_samples)
from recondet.vggt_camera_loss import (
    compute_vggt_camera_loss, VGGT_CAMERA_LOSS_DEFAULTS)
from recondet.vggt_ground_truth import mean_point_distance, transform_points
from recondet.vggt_lora import (
    configure_vggt_lora, enable_lora_parameters,
    log_vggt_lora_summary, vggt_feature_grad_context)
from recondet.prediction_visualization import (
    save_scene_cluster_visualization, save_scene_prediction_visualization)
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
            reconstruction_query_score_thr=0.1,
            query_clustering_cfg=None,
            query_xyz_range=(-6.5, -9.0, -1.0, 6.5, 9.0, 4.5),
            gt_points_dir=None,
            reconstruction_depth_loss_weight=1.0,
            reconstruction_point_loss_weight=0.5,
            reconstruction_object_head_cfg=None,
            scene_query_exchange_cfg=None,
            supervise_camera_head=False,
            camera_loss_cfg=None,
            supervise_confident_query_depth=False,
            confident_query_depth_cfg=None,
            prediction_visualization=False,
            prediction_visualization_dir='work_dirs/recondet_visualizations',
            prediction_visualization_score_thr=0.1,
            proposal_grouping='embedding',
            proposal_warmup_epochs=0,
            proposal_capacity=None,
    ):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

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
            reconstruction_depth_loss_weight=(
                reconstruction_depth_loss_weight),
            reconstruction_point_loss_weight=(
                reconstruction_point_loss_weight),
            scene_query_exchange_cfg=scene_query_exchange_cfg,
            supervise_confident_query_depth=(
                supervise_confident_query_depth),
            confident_query_depth_cfg=confident_query_depth_cfg)
        semantic_query_dims = self.semantic_encoder.model.embed_dims
        reconstruction_query_dims = resolve_reconstruction_query_dims(
            self.semantic_encoder)
        object_head_cfg = dict(reconstruction_object_head_cfg or {})
        self.reconstruction_object_head = ReconstructionObjectHead(
            query_dims=reconstruction_query_dims, **object_head_cfg)
        self.semantic_query_projection = torch.nn.Linear(
            semantic_query_dims, token_dim)
        self.detection_query_norm = torch.nn.LayerNorm(token_dim)

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
        if proposal_grouping not in ('embedding', 'gt'):
            raise ValueError("proposal_grouping must be 'embedding' or 'gt'")
        if proposal_warmup_epochs < 0:
            raise ValueError('proposal_warmup_epochs must be non-negative')
        self.proposal_grouping = proposal_grouping
        self.proposal_warmup_epochs = int(proposal_warmup_epochs)
        self.proposal_epoch = 0
        self.proposal_capacity = int(proposal_capacity or num_queries)
        if self.proposal_capacity <= 0:
            raise ValueError('proposal_capacity must be positive')
        query_clustering_cfg = dict(query_clustering_cfg or {})
        self.query_clustering_cfg = dict(query_clustering_cfg)
        self.scene_query_clustering = SemanticWeightedFPSClustering(
            num_clusters=num_queries, **query_clustering_cfg)
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
        self.reconstruction_query_score_thr = reconstruction_query_score_thr
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

    def _configure_vggt_trainability(self, training):
        self.vggt_encoder.requires_grad_(False)
        self.vggt_encoder.eval()
        if self.vggt_lora_enabled:
            enable_lora_parameters(self.vggt_encoder.aggregator)
            self.vggt_encoder.aggregator.train(bool(training))
        camera_head = self.vggt_encoder.camera_head
        camera_head.requires_grad_(self.supervise_camera_head)
        camera_head.train(bool(training and self.supervise_camera_head))

    def train(self, mode=True):
        super().train(mode)
        self._configure_vggt_trainability(training=mode)
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
        return outputs

    def _select_reconstruction_queries(self, reconstruction_outputs, images):
        batch_size, num_views = images.shape[:2]
        return select_scene_reconstruction_queries(
            reconstruction_outputs,
            batch_size=batch_size,
            num_views=num_views,
            score_threshold=self.reconstruction_query_score_thr,
            min_queries=self.proposal_capacity)

    def _compute_reconstruction_object_losses(
            self, reconstruction_outputs, batch_data_samples, num_views):
        matches = reconstruction_outputs.get('reconstruction_matches')
        if matches is None:
            raise RuntimeError(
                'Object reconstruction supervision requires 2D matches')
        samples = collect_matched_object_samples(
            reconstruction_outputs, matches, batch_data_samples, num_views)
        losses, _ = self.reconstruction_object_head.loss(samples)
        return losses

    def set_proposal_epoch(self, epoch):
        self.proposal_epoch = max(int(epoch), 0)

    def _use_gt_proposal_grouping(self):
        return (self.training and self.proposal_grouping == 'gt') or (
            self.training and self.proposal_epoch < self.proposal_warmup_epochs)

    def _fuse_detection_queries(self, reconstruction_query,
                                semantic_query_2d):
        projected_semantic = self.semantic_query_projection(
            semantic_query_2d.to(
                dtype=self.semantic_query_projection.weight.dtype))
        return self.detection_query_norm(
            reconstruction_query.to(projected_semantic.dtype) +
            projected_semantic)

    def _cluster_reconstruction_queries(self, selected_scenes,
                                        return_diagnostics=False):
        clustered = []
        diagnostics = []
        for scene in selected_scenes:
            selected_points = scene['points_aligned']
            selected_hidden = scene['reconstruction_query']
            selected_query_2d = scene['detection_query_2d']
            selected_class_scores = scene['class_scores_2d']
            selected_scores = scene['foreground_score']
            (cluster_xyz, cluster_size, cluster_query,
             cluster_semantic_query) = (
                self.scene_query_clustering(
                    selected_points[None], selected_hidden[None],
                    selected_query_2d[None], selected_class_scores[None],
                    selected_scores[None]))
            clustered.append(
                (cluster_xyz[0], cluster_size[0], cluster_query[0],
                 cluster_semantic_query[0]))
            diagnostics.append({
                'reconstruction_points': selected_points.detach(),
                'reconstruction_scores': selected_scores.detach(),
                'cluster_centers': cluster_xyz[0].detach(),
                'cluster_sizes': cluster_size[0].detach(),
            })
        (cluster_xyz, cluster_size, cluster_query,
         cluster_semantic_query) = [
            torch.stack(items).detach() for items in zip(*clustered)
        ]
        detection_query = self._fuse_detection_queries(
            cluster_query, cluster_semantic_query)
        if return_diagnostics:
            return (cluster_xyz, cluster_size, detection_query, diagnostics)
        return cluster_xyz, cluster_size, detection_query

    def _build_object_proposals(self, selected_scenes, batch_data_samples=None,
                                reconstruction_matches=None, num_views=None):
        """Build fixed-size direct object proposals plus legacy fillers."""
        proposal_batches = []
        for batch_index, scene in enumerate(selected_scenes):
            samples = {
                'queries': scene['reconstruction_query'],
                'points': scene['points_aligned'],
                'scene_indices': torch.zeros(
                    len(scene['reconstruction_query']), dtype=torch.long,
                    device=scene['reconstruction_query'].device),
                'view_indices': scene['source_view_id'].long(),
                'instance_embeddings': scene['instance_embeddings_2d'],
                'semantic_queries': scene['detection_query_2d'],
                'class_scores': scene['class_scores_2d'],
                'foreground_scores': scene['foreground_score'],
            }
            gt_group_indices = None
            if (self._use_gt_proposal_grouping()
                    and batch_data_samples is not None
                    and reconstruction_matches is not None
                    and num_views is not None):
                group_keys = []
                for candidate_index, (view_id, query_id) in enumerate(zip(
                        scene['source_view_id'].tolist(),
                        scene['source_query_id'].tolist())):
                    flat_view = batch_index * num_views + int(view_id)
                    query_indices, gt_indices = reconstruction_matches[flat_view]
                    matches = torch.nonzero(
                        query_indices.to(
                            device=scene['reconstruction_query'].device,
                            dtype=torch.long) == query_id,
                        as_tuple=False).flatten()
                    if len(matches) == 0:
                        group_keys.append(-(candidate_index + 1))
                    else:
                        if len(matches) > 1:
                            raise ValueError(
                                'reconstruction_matches contains duplicate '
                                f'query index {query_id} for view {flat_view}')
                        gt_index = int(gt_indices[matches[0]].item())
                        instance_ids = batch_data_samples[batch_index].gt_instances_2d[
                            int(view_id)].instance_ids_3d
                        if gt_index < 0 or gt_index >= len(instance_ids):
                            raise ValueError(
                                'reconstruction_matches contains an out-of-range '
                                f'GT index {gt_index} for view {flat_view}')
                        group_keys.append(int(instance_ids[gt_index]))
                gt_group_indices = torch.as_tensor(
                    group_keys, device=scene['reconstruction_query'].device,
                    dtype=torch.long)
            real = self.reconstruction_object_head.predict(
                samples, group_indices=gt_group_indices)[0]
            real_count = min(self.proposal_capacity, real['boxes'].shape[0])
            real_ids = torch.argsort(real['scores'], descending=True)[:real_count]
            centers = real['centers'][real_ids]
            sizes = real['sizes'][real_ids]
            queries = real['group_queries'][real_ids]
            semantic = real['semantic_queries']
            if semantic is None:
                semantic = queries.new_zeros((len(queries), self.semantic_encoder.model.embed_dims))
            else:
                semantic = semantic[real_ids]
            remaining_mask = torch.ones(
                len(scene['reconstruction_query']), dtype=torch.bool,
                device=centers.device)
            used = real['query_indices'][torch.isin(
                real['group_indices'], real_ids)]
            remaining_mask[used] = False
            fill_count = self.proposal_capacity - real_count
            filler = None
            if fill_count > 0 and remaining_mask.any():
                filler_source = {
                    key: value[remaining_mask]
                    for key, value in (
                        ('points_aligned', scene['points_aligned']),
                        ('reconstruction_query', scene['reconstruction_query']),
                        ('detection_query_2d', scene['detection_query_2d']),
                        ('class_scores_2d', scene['class_scores_2d']),
                        ('foreground_score', scene['foreground_score']))}
                clusterer = SemanticWeightedFPSClustering(
                    num_clusters=fill_count, **self.query_clustering_cfg)
                filler = clusterer(
                    filler_source['points_aligned'][None],
                    filler_source['reconstruction_query'][None],
                    filler_source['detection_query_2d'][None],
                    filler_source['class_scores_2d'][None],
                    filler_source['foreground_score'][None])
                filler = tuple(value[0] for value in filler)
            if fill_count > 0:
                if filler is None:
                    filler = (centers.new_zeros((0, 3)),
                              queries.new_zeros((0, 3)),
                              queries.new_zeros((0, queries.shape[-1])),
                              semantic.new_zeros((0, semantic.shape[-1])))
                filler_centers, _, filler_queries, filler_semantic = filler
                filler_count = filler_centers.shape[0]
                filler_sizes = centers.new_ones((filler_count, 3))
                filler_queries = self._fuse_detection_queries(
                    filler_queries, filler_semantic)
                centers = torch.cat([centers, filler_centers], dim=0)
                sizes = torch.cat([sizes, filler_sizes], dim=0)
                queries = torch.cat([self._fuse_detection_queries(
                    queries, semantic), filler_queries], dim=0)
                if centers.shape[0] < self.proposal_capacity:
                    pad = self.proposal_capacity - centers.shape[0]
                    if centers.shape[0] == 0:
                        raise RuntimeError('proposal packing produced no centers')
                    ids = torch.arange(pad, device=centers.device) % centers.shape[0]
                    centers = torch.cat([centers, centers[ids]], dim=0)
                    sizes = torch.cat([sizes, centers.new_ones((pad, 3))], dim=0)
                    queries = torch.cat([queries, queries[ids]], dim=0)
            else:
                queries = self._fuse_detection_queries(queries, semantic)
            proposal_batches.append((centers, sizes, queries))
        return tuple(torch.stack(values).detach() for values in zip(*proposal_batches))

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
        reconstruction_outputs = self._align_reconstruction_outputs(
            reconstruction_outputs, batch_inputs_dict, img)
        object_losses = self._compute_reconstruction_object_losses(
            reconstruction_outputs, batch_data_samples, img.shape[1])
        losses.update({f'gdino_{name}': value
                       for name, value in object_losses.items()})
        selected_reconstruction = self._select_reconstruction_queries(
            reconstruction_outputs, img)
        query_xyz, query_sizes, query = self._build_object_proposals(
            selected_reconstruction, batch_data_samples,
            reconstruction_outputs.get('reconstruction_matches'),
            img.shape[1])
        box_features, refined_query_xyz = self.get_box_features(
            vggt_feature_maps, batch_inputs_dict, img, query_xyz, query,
            extrinsics, intrinsics)
        detection_losses = self.bbox_head.loss(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            initial_sizes=query_sizes,
            **kwargs)
        losses.update({f'recondet_{name}': value
                       for name, value in detection_losses.items()})

        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'test')
        vggt_feature_maps = self.feature_projector(
            vggt_token_list, img, ps_idx)
        raw_extrinsics, extrinsics, intrinsics = self._build_projection_cameras(
            vggt_token_list, ps_idx, img, batch_inputs_dict,
            batch_data_samples)
        reconstruction_outputs, view_predictions = (
            self.semantic_encoder.predict_reconstruction(
                batch_inputs_dict['imgs'], batch_data_samples,
                vggt_feature_maps, raw_extrinsics, intrinsics))
        reconstruction_outputs = self._align_reconstruction_outputs(
            reconstruction_outputs, batch_inputs_dict, img)
        selected_reconstruction = self._select_reconstruction_queries(
            reconstruction_outputs, img)
        query_xyz, query_sizes, query = self._build_object_proposals(
            selected_reconstruction)
        box_features, refined_query_xyz = self.get_box_features(
            vggt_feature_maps, batch_inputs_dict, img, query_xyz, query,
            extrinsics, intrinsics)
        results_list = self.bbox_head.predict(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            initial_sizes=query_sizes,
            **kwargs)
        if self.prediction_visualization:
            self._save_prediction_visualizations(
                batch_data_samples, results_list, reconstruction_outputs,
                num_views=img.shape[1], cluster_diagnostics=None)
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
            save_scene_prediction_visualization(
                self.prediction_visualization_dir,
                scene_id,
                gt_points,
                gt.bboxes_3d,
                gt.labels_3d,
                result.bboxes_3d,
                result.scores_3d,
                result.labels_3d,
                points[batch_index].reshape(-1, 3),
                scores[batch_index].reshape(-1),
                self.semantic_encoder.classes,
                score_threshold=self.prediction_visualization_score_thr)
            if cluster_diagnostics is not None:
                diagnostics = cluster_diagnostics[batch_index]
                save_scene_cluster_visualization(
                    self.prediction_visualization_dir,
                    scene_id,
                    gt_points,
                    diagnostics['reconstruction_points'],
                    diagnostics['reconstruction_scores'],
                    diagnostics['cluster_centers'],
                    diagnostics['cluster_sizes'],
                    score_threshold=self.prediction_visualization_score_thr)

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
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
        reconstruction_outputs = self._align_reconstruction_outputs(
            reconstruction_outputs, batch_inputs_dict, img)
        selected_reconstruction = self._select_reconstruction_queries(
            reconstruction_outputs, img)
        query_xyz, query_sizes, query = self._build_object_proposals(
            selected_reconstruction)
        box_features, refined_query_xyz = self.get_box_features(
            vggt_feature_maps, batch_inputs_dict, img, query_xyz, query,
            extrinsics, intrinsics)

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, refined_query_xyz,
            initial_sizes=query_sizes)
        return results
