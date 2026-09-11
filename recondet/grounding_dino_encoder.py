from copy import deepcopy
from pathlib import Path
from typing import Sequence

import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.structures import InstanceData
from torch import nn

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores)
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules


def extract_grounding_dino_state_dict(state_dict):
    """Extract Grounding DINO weights from native or full ReconDet checkpoints."""
    cleaned = {
        name.removeprefix('module.'): value
        for name, value in state_dict.items()
    }
    prefix = 'semantic_encoder.model.'
    nested = {
        name[len(prefix):]: value
        for name, value in cleaned.items()
        if name.startswith(prefix)
    }
    if nested:
        return nested

    native_roots = (
        'backbone.', 'neck.', 'encoder.', 'decoder.', 'language_model.',
        'query_embedding.', 'bbox_head.', 'scene_query_exchange.',
        'reconstruction_decoder.')
    if any(name.startswith(native_roots) for name in cleaned):
        return cleaned
    raise ValueError(
        'Checkpoint does not contain native or nested Grounding DINO weights')


def require_pretrained_instance_projection(state_dict, projection):
    prefix = 'bbox_head.instance_projection.'
    missing = []
    mismatched = []
    for name, expected in projection.state_dict().items():
        checkpoint_name = prefix + name
        actual = state_dict.get(checkpoint_name)
        if actual is None:
            missing.append(checkpoint_name)
        elif actual.shape != expected.shape:
            mismatched.append(
                f'{checkpoint_name}: {tuple(actual.shape)} != '
                f'{tuple(expected.shape)}')
    if missing or mismatched:
        details = []
        if missing:
            details.append(f'missing keys {missing}')
        if mismatched:
            details.append(f'shape mismatches {mismatched}')
        raise ValueError(
            'Checkpoint has an incomplete pretrained instance_projection: '
            + '; '.join(details))


class GroundingDINOSemanticEncoder(nn.Module):
    """Frozen Grounding DINO front-end with trainable 3D reconstruction."""

    def __init__(self,
                 config: str,
                 checkpoint: str,
                 classes: Sequence[str],
                 reconstruction_point_loss_weight: float = 0.5,
                 scene_query_exchange_cfg=None,
                 supervise_confident_query_point: bool = False,
                 confident_query_point_cfg=None, camera_dims=2048) -> None:
        super().__init__()
        if not classes:
            raise ValueError('GroundingDINO classes must not be empty.')
        self.classes = tuple(classes)
        self.supervise_confident_query_point = bool(
            supervise_confident_query_point)
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        cfg = Config.fromfile(str(config_path))
        model_cfg = deepcopy(cfg.model)
        model_cfg['_scope_'] = 'mmdet'
        model_cfg.reconstruction_decoder.camera_dims = int(camera_dims)
        model_cfg.bbox_head.camera_dims = int(camera_dims)
        self.camera_dims = int(camera_dims)
        data_preprocessor_cfg = model_cfg['data_preprocessor']
        image_mean = data_preprocessor_cfg['mean']
        image_std = data_preprocessor_cfg['std']
        if model_cfg.get('backbone', {}).get('init_cfg') is not None:
            model_cfg.backbone.init_cfg = None
        model_cfg.bbox_head.supervise_confident_query_point = (
            self.supervise_confident_query_point)
        model_cfg.scene_query_exchange_cfg = scene_query_exchange_cfg
        model_cfg.bbox_head.reconstruction_point_loss_weight = float(
            reconstruction_point_loss_weight)
        model_cfg.bbox_head.confident_query_point_cfg = (
            confident_query_point_cfg)

        register_all_modules(init_default_scope=False)
        self.model = MMDET_MODELS.build(model_cfg)
        self.model.bbox_head.init_weights()
        checkpoint_data = CheckpointLoader.load_checkpoint(
            checkpoint, map_location='cpu')
        state_dict = checkpoint_data.get('state_dict', checkpoint_data)
        state_dict = extract_grounding_dino_state_dict(state_dict)
        state_dict = {
            name: value for name, value in state_dict.items()
            if not name.startswith('bbox_head.reconstruction_head.depth_head.')
            and not name.startswith(
                'reconstruction_decoder.query_position_embedding.norm.')
        }
        require_pretrained_instance_projection(
            state_dict, self.model.bbox_head.instance_projection)
        query_weight = state_dict.get('query_embedding.weight')
        if (query_weight is not None and
                query_weight.shape != self.model.query_embedding.weight.shape and
                query_weight.shape[1:] ==
                self.model.query_embedding.weight.shape[1:] and
                query_weight.shape[0] >= self.model.num_queries):
            state_dict['query_embedding.weight'] = query_weight[
                :self.model.num_queries].clone()
        load_state_dict(self.model, state_dict, strict=False)
        self.model._is_init = True

        self._configure_model_trainability()
        self.model.eval()

        self.token_positive_map = None
        self.register_buffer(
            'image_mean',
            torch.tensor(image_mean).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'image_std',
            torch.tensor(image_std).view(1, 3, 1, 1),
            persistent=False)

    def _configure_model_trainability(self):
        self.model.requires_grad_(False)
        reconstruction_decoder = self.model.reconstruction_decoder
        if reconstruction_decoder is not None:
            reconstruction_decoder.requires_grad_(True)
        self.model.bbox_head.reconstruction_head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The root training flag is required by Grounding DINO's query setup,
        # while every frozen 2D child must remain deterministic.
        for module in self.model.children():
            module.eval()
        reconstruction_decoder = self.model.reconstruction_decoder
        if reconstruction_decoder is not None:
            reconstruction_decoder.train(mode)
        self.model.bbox_head.reconstruction_head.train(mode)
        return self

    def _ensure_token_positive_map(self):
        if self.token_positive_map is not None:
            return
        tokenized, _, tokens_positive, _ = (
            self.model.get_tokens_and_prompts(self.classes, True))
        self.token_positive_map, _ = self.model.get_positive_map(
            tokenized, tokens_positive)

    def _attach_reconstruction_class_scores(self, reconstruction_outputs):
        if reconstruction_outputs is None:
            raise RuntimeError('Reconstruction outputs are not available')
        self._ensure_token_positive_map()
        query_count = reconstruction_outputs['reconstruction_query'].shape[1]
        token_scores = self.model.bbox_head._last_cls_scores[
            -1, :, -query_count:].sigmoid()
        positive_maps = [self.token_positive_map] * len(token_scores)
        class_scores = convert_grounding_to_cls_scores(
            token_scores, positive_maps)
        reconstruction_outputs['class_scores_2d'] = class_scores
        reconstruction_outputs['foreground_score'] = class_scores.amax(dim=-1)
        return reconstruction_outputs

    @staticmethod
    def _image_shape(data_sample, view_index: int, fallback):
        shape = data_sample.metainfo.get(
            'view_img_shapes', data_sample.metainfo.get('img_shape', fallback))
        if (isinstance(shape, (list, tuple)) and shape and
                isinstance(shape[0], (list, tuple))):
            shape = shape[view_index]
        return int(shape[0]), int(shape[1])

    def _make_data_sample(self, source_sample, view_index: int,
                          padded_shape) -> DetDataSample:
        image_shape = self._image_shape(
            source_sample, view_index, padded_shape)
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': image_shape,
            'ori_shape': image_shape,
            'batch_input_shape': tuple(padded_shape),
            'pad_shape': tuple(padded_shape),
            'scale_factor': (1.0, 1.0),
        })
        data_sample.text = self.classes
        data_sample.custom_entities = True
        return data_sample

    def _normalize_images(self, images, batch_data_samples):
        batch_size, num_views, channels, height, width = images.shape

        flattened = images.reshape(
            batch_size * num_views, channels, height, width).float()
        flattened = (flattened - self.image_mean) / self.image_std

        for batch_index, data_sample in enumerate(batch_data_samples):
            for view_index in range(num_views):
                image_height, image_width = self._image_shape(
                    data_sample, view_index, (height, width))
                flat_index = batch_index * num_views + view_index
                if image_height < height:
                    flattened[flat_index, :, image_height:, :] = 0
                if image_width < width:
                    flattened[flat_index, :, :, image_width:] = 0
        return flattened

    @staticmethod
    def _make_vggt_valid_ratios(image_shapes, padded_shape, num_levels):
        if image_shapes.ndim != 2 or image_shapes.shape[-1] != 2:
            raise ValueError('image_shapes must have shape [B*V, 2]')
        if num_levels <= 0:
            raise ValueError('num_levels must be positive')
        padded_height, padded_width = padded_shape
        padded_size = image_shapes.new_tensor(
            [padded_width, padded_height]).clamp_min(1)
        width_height = image_shapes[:, [1, 0]]
        ratios = (width_height / padded_size).clamp(0, 1)
        return ratios[:, None].expand(-1, num_levels, -1).contiguous()

    def _make_training_samples(self, batch_data_samples, num_views,
                               padded_shape, gt_depths_vggt=None,
                               gt_depth_valid_masks=None,
                               vggt_gt_scale=None, gt_extrinsics_vggt=None,
                               gt_intrinsics=None,
                               enable_confident_point=None,
                               grouped=False):
        if enable_confident_point is None:
            enable_confident_point = getattr(
                self, 'supervise_confident_query_point', False)
        supervise_confident_point = bool(enable_confident_point)
        if supervise_confident_point:
            if (gt_depths_vggt is None or gt_depth_valid_masks is None
                    or vggt_gt_scale is None or gt_extrinsics_vggt is None
                    or gt_intrinsics is None):
                raise RuntimeError(
                    'Confident query point supervision requires VGGT depth, '
                    'valid masks, scale, and GT cameras')
            expected_prefix = (len(batch_data_samples), num_views)
            if gt_depths_vggt.shape[:2] != expected_prefix:
                raise ValueError('GT depth shape must begin with [B, V]')
            if gt_depth_valid_masks.shape != gt_depths_vggt.shape:
                raise ValueError('GT depth masks must match GT depth maps')
            if (gt_extrinsics_vggt.shape != expected_prefix + (3, 4)
                    or gt_intrinsics.shape != expected_prefix + (3, 3)):
                raise ValueError(
                    'GT cameras must have shape [B,V,3,4] and [B,V,3,3]')
            if vggt_gt_scale.shape != (len(batch_data_samples),):
                raise ValueError('VGGT GT scale must have shape [B]')
        scene_samples = []
        for batch_index, source_sample in enumerate(batch_data_samples):
            source_instances_2d = getattr(
                source_sample, 'gt_instances_2d', None)
            if source_instances_2d is None:
                raise RuntimeError(
                    'ReConDet training requires per-view gt_instances_2d')
            if len(source_instances_2d) != num_views:
                raise ValueError(
                    'gt_instances_2d view count does not match input images: '
                    f'{len(source_instances_2d)} != {num_views}')
            samples = []
            for view_index in range(num_views):
                sample = self._make_data_sample(
                    source_sample, view_index, padded_shape)
                sample.scene_batch_index = batch_index
                sample.view_index = view_index
                if supervise_confident_point:
                    sample.gt_depth_vggt = gt_depths_vggt[
                        batch_index, view_index]
                    sample.gt_depth_valid_mask = gt_depth_valid_masks[
                        batch_index, view_index]
                    sample.vggt_gt_scale = vggt_gt_scale[batch_index]
                    sample.token_positive_map = self.token_positive_map
                    sample.gt_extrinsics_vggt = gt_extrinsics_vggt[
                        batch_index, view_index]
                    sample.gt_intrinsics = gt_intrinsics[batch_index, view_index]
                source_instances = source_instances_2d[view_index]
                instance_ids_3d = getattr(
                    source_instances, 'instance_ids_3d', None)
                instances = InstanceData(
                    bboxes=source_instances.bboxes,
                    labels=source_instances.labels,
                    centers_3d=source_instances.centers_3d,
                    centers_3d_vggt=source_instances.centers_3d_vggt,
                    center_3d_valid_mask=(
                        source_instances.center_3d_valid_mask))
                if instance_ids_3d is not None:
                    instances.instance_ids_3d = instance_ids_3d
                sample.gt_instances = instances
                samples.append(sample)
            scene_samples.append(samples)
        if grouped:
            return scene_samples
        return [sample for samples in scene_samples for sample in samples]

    def loss(self, images, batch_data_samples, vggt_feature_maps=None,
             camera_tokens=None,
             return_reconstruction=False, gt_depths_vggt=None,
             gt_depth_valid_masks=None, vggt_gt_scale=None,
             gt_extrinsics_vggt=None, gt_intrinsics=None):
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]

        normalized = self._normalize_images(images, batch_data_samples)
        self._ensure_token_positive_map()
        samples = self._make_training_samples(
            batch_data_samples, num_views, padded_shape,
            gt_depths_vggt=gt_depths_vggt,
            gt_depth_valid_masks=gt_depth_valid_masks,
            vggt_gt_scale=vggt_gt_scale,
            gt_extrinsics_vggt=gt_extrinsics_vggt, gt_intrinsics=gt_intrinsics,
            enable_confident_point=(
                self.supervise_confident_query_point
                and return_reconstruction))
        flattened_vggt_features = None
        if vggt_feature_maps is not None:
            flattened_vggt_features = [
                feature.reshape(
                    batch_size * num_views, *feature.shape[2:]).contiguous()
                for feature in vggt_feature_maps
            ]
        flattened_camera_tokens = None
        if vggt_feature_maps is not None:
            flattened_camera_tokens = self._flatten_camera_tokens(
                camera_tokens, batch_size, num_views)
        image_shapes = normalized.new_tensor([
            sample.metainfo['img_shape'] for sample in samples])
        vggt_valid_ratios = None
        if flattened_vggt_features is not None:
            vggt_valid_ratios = self._make_vggt_valid_ratios(
                image_shapes, padded_shape, len(flattened_vggt_features))
        result = self.model.loss(
            normalized,
            samples,
            vggt_feature_maps=flattened_vggt_features,
            camera_tokens=flattened_camera_tokens,
            image_shapes=image_shapes,
            vggt_valid_ratios=vggt_valid_ratios,
            num_views=num_views,
            return_reconstruction=return_reconstruction)
        if return_reconstruction:
            self._attach_reconstruction_class_scores(result[2])
        self._reshape_semantic_feature_maps(batch_size, num_views)
        return result

    def _flatten_camera_tokens(self, camera_tokens, batch_size, num_views):
        if camera_tokens is None or camera_tokens.shape != (
                batch_size, num_views, self.camera_dims):
            raise ValueError('camera_tokens must have shape [B, V, camera_dims]')
        return camera_tokens.reshape(batch_size * num_views, self.camera_dims)

    def _reshape_semantic_feature_maps(self, batch_size, num_views):
        flat_maps = self.model._last_semantic_feature_maps
        self.last_semantic_feature_maps = [
            feature.reshape(batch_size, num_views, *feature.shape[1:])
            for feature in flat_maps
        ]
        self.last_valid_ratios = self.model._last_valid_ratios.reshape(
            batch_size, num_views, *self.model._last_valid_ratios.shape[1:])

    @torch.no_grad()
    def predict_reconstruction(self, images, batch_data_samples,
                               vggt_feature_maps, camera_tokens):
        padded_shape = images.shape[-2:]
        batch_size, num_views = images.shape[:2]
        self._ensure_token_positive_map()
        flattened = self._normalize_images(images, batch_data_samples)
        samples = []
        for source_sample in batch_data_samples:
            for view_index in range(num_views):
                samples.append(self._make_data_sample(
                    source_sample, view_index, padded_shape))
        flattened_vggt_features = [
            feature.reshape(
                batch_size * num_views, *feature.shape[2:]).contiguous()
            for feature in vggt_feature_maps
        ]
        flattened_camera_tokens = self._flatten_camera_tokens(
            camera_tokens, batch_size, num_views)
        image_shapes = flattened.new_tensor([
            sample.metainfo['img_shape'] for sample in samples])
        vggt_valid_ratios = self._make_vggt_valid_ratios(
            image_shapes, padded_shape, len(flattened_vggt_features))
        was_training = self.model.training
        self.model.eval()
        view_predictions = self.model.predict(
            flattened, samples, rescale=False,
            vggt_feature_maps=flattened_vggt_features,
            camera_tokens=flattened_camera_tokens,
            image_shapes=image_shapes,
            vggt_valid_ratios=vggt_valid_ratios,
            num_views=num_views)
        self._reshape_semantic_feature_maps(batch_size, num_views)
        reconstruction_outputs = self.model._last_reconstruction_outputs
        self._attach_reconstruction_class_scores(reconstruction_outputs)
        if was_training:
            self.train()
        return reconstruction_outputs, view_predictions
