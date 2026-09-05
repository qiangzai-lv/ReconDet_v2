from copy import deepcopy
from pathlib import Path
from typing import Sequence

import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.structures import InstanceData
from torch import nn

from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules


class GroundingDINOSemanticEncoder(nn.Module):
    """GroundingDINO with frozen encoders and a trainable detection decoder."""

    def __init__(self,
                 config: str,
                 checkpoint: str,
                 classes: Sequence[str],
                 print_score_thr: float = 0.3) -> None:
        super().__init__()
        if not classes:
            raise ValueError('GroundingDINO classes must not be empty.')
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

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for module in (self.model.query_embedding, self.model.decoder,
                       self.model.bbox_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if self.model.reconstruction_decoder is not None:
            for parameter in self.model.reconstruction_decoder.parameters():
                parameter.requires_grad = True
        self.model.eval()

        self.classes = tuple(classes)
        self.print_score_thr = print_score_thr
        self.register_buffer(
            'image_mean',
            torch.tensor(image_mean).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'image_std',
            torch.tensor(image_std).view(1, 3, 1, 1),
            persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module_name in ('backbone', 'neck', 'encoder', 'language_model'):
            module = getattr(self.model, module_name, None)
            if module is not None:
                module.eval()
        return self

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
                               padded_shape):
        image_height, image_width = padded_shape
        samples = []
        for source_sample in batch_data_samples:
            source_instances = source_sample.gt_instances_3d
            # 3D ScanNet annotations use ``labels_3d``. GroundingDINO's
            # per-view samples require the generic 2D ``labels`` field.
            source_labels = getattr(source_instances, 'labels_3d', None)
            if source_labels is None:
                source_labels = getattr(source_instances, 'labels', None)
            if source_labels is None:
                boxes_3d = getattr(source_instances, 'bboxes_3d', None)
                num_boxes = len(boxes_3d) if boxes_3d is not None else 0
                label_device = getattr(boxes_3d, 'device', None)
                if label_device is None:
                    label_device = getattr(boxes_3d, 'tensor', None)
                    label_device = getattr(label_device, 'device', None)
                source_labels = torch.zeros(
                    (num_boxes,), dtype=torch.long,
                    device=label_device)
            for view_index in range(num_views):
                sample = self._make_data_sample(
                    source_sample, view_index, padded_shape)
                instances = InstanceData()
                bboxes = source_instances.bboxes_2d
                if bboxes.ndim == 3:
                    bboxes = bboxes[:, view_index]
                bbox_visible = source_instances.bboxes_2d_visible
                if bbox_visible.ndim == 2:
                    bbox_visible = bbox_visible[:, view_index]
                bbox_size = bboxes[:, 2:] - bboxes[:, :2]
                valid = bbox_visible.bool()
                valid &= torch.isfinite(bboxes).all(dim=-1)
                valid &= (bbox_size > 0).all(dim=-1)

                instances.labels = source_labels[valid].clone()
                instances.bboxes_2d = bboxes[valid]
                instances.bboxes = bboxes[valid] * bboxes.new_tensor(
                    [image_width, image_height, image_width, image_height])
                instances.centers_3d = source_instances.centers_3d[valid]
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
             return_reconstruction=False):
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]

        normalized = self._normalize_images(images, batch_data_samples)
        samples = self._make_training_samples(
            batch_data_samples, num_views, padded_shape)
        flattened_vggt_features = None
        if vggt_feature_maps is not None:
            flattened_vggt_features = [
                feature.reshape(
                    batch_size * num_views, *feature.shape[2:]).contiguous()
                for feature in vggt_feature_maps
            ]
        result = self.model.loss(
            normalized,
            samples,
            vggt_feature_maps=flattened_vggt_features,
            return_reconstruction=return_reconstruction)
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
                               vggt_feature_maps):
        padded_shape = images.shape[-2:]
        batch_size, num_views = images.shape[:2]
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
        was_training = self.model.training
        self.model.eval()
        view_predictions = self.model.predict(
            flattened, samples, rescale=False,
            vggt_feature_maps=flattened_vggt_features)
        self._reshape_semantic_feature_maps(batch_size, num_views)
        reconstruction_outputs = self.model._last_reconstruction_outputs
        if was_training:
            self.model.train()
        return reconstruction_outputs, view_predictions
