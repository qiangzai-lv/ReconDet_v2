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


class GroundingDINOSemanticEncoder(nn.Module):
    """GroundingDINO with frozen encoders and a trainable detection decoder."""

    def __init__(self,
                 config: str,
                 checkpoint: str,
                 classes: Sequence[str],
                 print_score_thr: float = 0.3,
                 supervise_2d_bbox: bool = True,
                 supervise_confident_query_depth: bool = False,
                 confident_query_depth_cfg=None) -> None:
        super().__init__()
        if not classes:
            raise ValueError('GroundingDINO classes must not be empty.')
        self.classes = tuple(classes)
        self.supervise_2d_bbox = bool(supervise_2d_bbox)
        self.supervise_confident_query_depth = bool(
            supervise_confident_query_depth)
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        cfg = Config.fromfile(str(config_path))
        model_cfg = deepcopy(cfg.model)
        model_cfg['_scope_'] = 'mmdet'
        data_preprocessor_cfg = model_cfg['data_preprocessor']
        image_mean = data_preprocessor_cfg['mean']
        image_std = data_preprocessor_cfg['std']
        if model_cfg.get('backbone', {}).get('init_cfg') is not None:
            model_cfg.backbone.init_cfg = None
        model_cfg.bbox_head.supervise_confident_query_depth = (
            self.supervise_confident_query_depth)
        model_cfg.bbox_head.supervise_2d_bbox = self.supervise_2d_bbox
        model_cfg.bbox_head.confident_query_depth_cfg = (
            confident_query_depth_cfg)

        register_all_modules(init_default_scope=False)
        self.model = MMDET_MODELS.build(model_cfg)
        self.model.bbox_head.init_weights()
        checkpoint_data = CheckpointLoader.load_checkpoint(
            checkpoint, map_location='cpu')
        state_dict = checkpoint_data.get('state_dict', checkpoint_data)
        state_dict = {
            name.removeprefix('module.'): value
            for name, value in state_dict.items()
        }
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
        self.print_score_thr = print_score_thr
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
        if self.supervise_2d_bbox:
            for module in (self.model.query_embedding, self.model.decoder,
                           self.model.bbox_head):
                module.requires_grad_(True)
        reconstruction_decoder = self.model.reconstruction_decoder
        if reconstruction_decoder is not None:
            reconstruction_decoder.requires_grad_(True)
        self.model.bbox_head.reconstruction_head.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.supervise_2d_bbox:
            # GroundingDINO checks its root ``training`` flag to construct
            # the encoder/denoising metadata consumed by bbox_head.loss.
            # Keep that control flow while all frozen detector children stay
            # in eval mode.
            for module in self.model.children():
                module.eval()
            reconstruction_decoder = self.model.reconstruction_decoder
            if reconstruction_decoder is not None:
                reconstruction_decoder.train(mode)
            self.model.bbox_head.reconstruction_head.train(mode)
            return self
        for module_name in ('backbone', 'neck', 'encoder', 'language_model'):
            module = getattr(self.model, module_name, None)
            if module is not None:
                module.eval()
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
        shape = data_sample.metainfo.get('img_shape', fallback)
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
            image_height, image_width = self._image_shape(
                data_sample, 0, (height, width))
            start = batch_index * num_views
            end = start + num_views
            if image_height < height:
                flattened[start:end, :, image_height:, :] = 0
            if image_width < width:
                flattened[start:end, :, :, image_width:] = 0
        return flattened

    def _make_training_samples(self, batch_data_samples, num_views,
                               padded_shape, gt_depths_vggt=None,
                               gt_depth_valid_masks=None,
                               vggt_gt_scale=None):
        supervise_confident_depth = getattr(
            self, 'supervise_confident_query_depth', False)
        if supervise_confident_depth:
            if (gt_depths_vggt is None or gt_depth_valid_masks is None
                    or vggt_gt_scale is None):
                raise RuntimeError(
                    'Confident query depth supervision requires VGGT depth, '
                    'valid masks, and scale')
            expected_prefix = (len(batch_data_samples), num_views)
            if gt_depths_vggt.shape[:2] != expected_prefix:
                raise ValueError('GT depth shape must begin with [B, V]')
            if gt_depth_valid_masks.shape != gt_depths_vggt.shape:
                raise ValueError('GT depth masks must match GT depth maps')
            if vggt_gt_scale.shape != (len(batch_data_samples),):
                raise ValueError('VGGT GT scale must have shape [B]')
        samples = []
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
            for view_index in range(num_views):
                sample = self._make_data_sample(
                    source_sample, view_index, padded_shape)
                if supervise_confident_depth:
                    sample.gt_depth_vggt = gt_depths_vggt[
                        batch_index, view_index]
                    sample.gt_depth_valid_mask = gt_depth_valid_masks[
                        batch_index, view_index]
                    sample.vggt_gt_scale = vggt_gt_scale[batch_index]
                    sample.token_positive_map = self.token_positive_map
                source_instances = source_instances_2d[view_index]
                instances = InstanceData(
                    bboxes=source_instances.bboxes,
                    labels=source_instances.labels,
                    centers_3d=source_instances.centers_3d,
                    centers_3d_vggt=source_instances.centers_3d_vggt,
                    center_depth_vggt=source_instances.center_depth_vggt,
                    center_3d_valid_mask=(
                        source_instances.center_3d_valid_mask),
                    center_depth_valid_mask=(
                        source_instances.center_depth_valid_mask),
                    instance_ids_3d=source_instances.instance_ids_3d)
                sample.gt_instances = instances
                samples.append(sample)
        return samples

    def _normalize_view(self, image, source_sample, view_index):
        normalized = (image[None].float() - self.image_mean) / self.image_std
        image_height, image_width = self._image_shape(
            source_sample, view_index, image.shape[-2:])
        if image_height < image.shape[-2]:
            normalized[:, :, image_height:, :] = 0
        if image_width < image.shape[-1]:
            normalized[:, :, :, image_width:] = 0
        return normalized

    def loss(self, images, batch_data_samples, vggt_feature_maps=None,
             vggt_extrinsics=None, vggt_intrinsics=None,
             return_reconstruction=False, gt_depths_vggt=None,
             gt_depth_valid_masks=None, vggt_gt_scale=None):
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]

        normalized = self._normalize_images(images, batch_data_samples)
        self._ensure_token_positive_map()
        samples = self._make_training_samples(
            batch_data_samples, num_views, padded_shape,
            gt_depths_vggt=gt_depths_vggt,
            gt_depth_valid_masks=gt_depth_valid_masks,
            vggt_gt_scale=vggt_gt_scale)
        flattened_vggt_features = None
        if vggt_feature_maps is not None:
            flattened_vggt_features = [
                feature.reshape(
                    batch_size * num_views, *feature.shape[2:]).contiguous()
                for feature in vggt_feature_maps
            ]
        flattened_extrinsics = None
        flattened_intrinsics = None
        if vggt_extrinsics is not None and vggt_intrinsics is not None:
            flattened_extrinsics = vggt_extrinsics.reshape(
                batch_size * num_views, *vggt_extrinsics.shape[2:]).contiguous()
            flattened_intrinsics = vggt_intrinsics.reshape(
                batch_size * num_views, *vggt_intrinsics.shape[2:]).contiguous()
        image_shapes = normalized.new_tensor([
            sample.metainfo['img_shape'] for sample in samples])
        result = self.model.loss(
            normalized,
            samples,
            vggt_feature_maps=flattened_vggt_features,
            vggt_extrinsics=flattened_extrinsics,
            vggt_intrinsics=flattened_intrinsics,
            image_shapes=image_shapes,
            return_reconstruction=return_reconstruction)
        if return_reconstruction:
            self._attach_reconstruction_class_scores(result[2])
        self._reshape_semantic_feature_maps(batch_size, num_views)
        return result

    def _reshape_semantic_feature_maps(self, batch_size, num_views):
        flat_maps = self.model._last_semantic_feature_maps
        self.last_semantic_feature_maps = [
            feature.reshape(batch_size, num_views, *feature.shape[1:])
            for feature in flat_maps
        ]
        self.last_valid_ratios = self.model._last_valid_ratios.reshape(
            batch_size, num_views, *self.model._last_valid_ratios.shape[1:])

    @torch.no_grad()
    def predict_and_print(self, images, batch_data_samples) -> None:
        was_training = self.model.training
        self.model.eval()
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]
        flattened = self._normalize_images(images, batch_data_samples)

        flat_samples = []
        view_indices = []
        for batch_index, source_sample in enumerate(batch_data_samples):
            for view_index in range(num_views):
                flat_samples.append(self._make_data_sample(
                    source_sample, view_index, padded_shape))
                view_indices.append((batch_index, view_index))

        predictions = self.model.predict(
            flattened, flat_samples, rescale=False)
        for prediction, (batch_index, view_index) in zip(
                predictions, view_indices):
            instances = prediction.pred_instances
            keep = instances.scores >= self.print_score_thr
            kept_indices = keep.nonzero(as_tuple=False).flatten().tolist()
            label_names = getattr(instances, 'label_names', [])
            output = {
                'bboxes': instances.bboxes[keep].detach().cpu(),
                'scores': instances.scores[keep].detach().cpu(),
                'labels': instances.labels[keep].detach().cpu(),
                'label_names': [label_names[index]
                                for index in kept_indices],
            }
            print(
                f'[GroundingDINO] batch={batch_index} view={view_index} '
                f'score_thr={self.print_score_thr}: {output}',
                flush=True)
        if was_training:
            self.model.train()
            for module_name in ('backbone', 'neck', 'encoder',
                                'language_model'):
                module = getattr(self.model, module_name, None)
                if module is not None:
                    module.eval()

    @torch.no_grad()
    def predict_reconstruction(self, images, batch_data_samples,
                               vggt_feature_maps, vggt_extrinsics,
                               vggt_intrinsics):
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
        flattened_extrinsics = vggt_extrinsics.reshape(
            batch_size * num_views, *vggt_extrinsics.shape[2:]).contiguous()
        flattened_intrinsics = vggt_intrinsics.reshape(
            batch_size * num_views, *vggt_intrinsics.shape[2:]).contiguous()
        image_shapes = flattened.new_tensor([
            sample.metainfo['img_shape'] for sample in samples])
        was_training = self.model.training
        self.model.eval()
        view_predictions = self.model.predict(
            flattened, samples, rescale=False,
            vggt_feature_maps=flattened_vggt_features,
            vggt_extrinsics=flattened_extrinsics,
            vggt_intrinsics=flattened_intrinsics,
            image_shapes=image_shapes)
        self._reshape_semantic_feature_maps(batch_size, num_views)
        reconstruction_outputs = self.model._last_reconstruction_outputs
        self._attach_reconstruction_class_scores(reconstruction_outputs)
        if was_training:
            self.train()
        return reconstruction_outputs, view_predictions
