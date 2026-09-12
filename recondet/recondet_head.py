# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from recondet.detr3_models.helpers import GenericMLP
from mmdet.utils import reduce_mean
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.structures.ops.iou3d_calculator import axis_aligned_bbox_overlaps_3d
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from recondet.matcher import RepeatedHungarianMatcher


def decode_size_residuals(size_residuals, initial_size_anchor,
                          size_logit_range):
    """Decode detached, multiplicative size refinement across layers."""
    if not size_residuals:
        return [], [], []
    if (len(size_logit_range) != 2
            or size_logit_range[0] >= size_logit_range[1]):
        raise ValueError('size_logit_range must be an increasing pair')

    first = size_residuals[0]
    if isinstance(initial_size_anchor, Tensor):
        if initial_size_anchor.shape != (first.shape[0], first.shape[2], 3):
            raise ValueError(
                'initial_size_anchor tensor must have shape [B, Q, 3]')
        anchor = initial_size_anchor.to(
            device=first.device, dtype=first.dtype).permute(0, 2, 1)
    else:
        if len(initial_size_anchor) != 3:
            raise ValueError('initial_size_anchor must contain three values')
        anchor = torch.as_tensor(
            initial_size_anchor, device=first.device, dtype=first.dtype
        ).view(1, 3, 1).expand_as(first)
    if not torch.isfinite(anchor).all() or (anchor <= 0).any():
        raise ValueError('initial_size_anchor must be finite and positive')
    reference_log = anchor.log()

    reference_logs = []
    predicted_logs = []
    predicted_sizes = []
    for residual in size_residuals:
        if residual.shape != first.shape:
            raise ValueError('all size residuals must have the same shape')
        reference_logs.append(reference_log)
        predicted_log = reference_log + residual.float()
        bounded_log = predicted_log.clamp(
            min=size_logit_range[0], max=size_logit_range[1])
        stable_log = predicted_log + (bounded_log - predicted_log).detach()
        predicted_logs.append(stable_log)
        predicted_sizes.append(stable_log.exp())
        reference_log = stable_log.detach()
    return reference_logs, predicted_logs, predicted_sizes


def matched_size_residual_loss(size_residuals, size_reference_logs, gt_sizes,
                               pred_indices, gt_indices, avg_factor):
    """L1 loss against the matched log ratio to the detached reference."""
    if pred_indices.numel() == 0:
        return size_residuals.sum() * 0.0
    target_residuals = (
        gt_sizes[gt_indices].clamp_min(1e-5).log()
        - size_reference_logs[pred_indices])
    loss = F.l1_loss(
        size_residuals[pred_indices], target_residuals, reduction='sum')
    return loss / max(float(avg_factor), 1.0)



@MODELS.register_module()
class ReconDetHead(BaseModule):

    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 cls_loss: ConfigType = dict(
                     type='mmdet.FocalLoss', use_sigmoid=True,
                     gamma=2.0, alpha=0.25, loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 mlp_dropout=0.3,
                 matcher_cost_weights={
                     'cls': 2.0, 'center': 1.0, 'size': 1.0, 'giou': 2.0},
                 loss_weights={'center_loss': 2.0, 'size_loss': 1.0,
                               'cls_loss': 2.0, 'iou_loss': 2.0},
                 if_v2_head=False,
                 matcher='repeated_hungarian',
                 loss_layer_ids=None,
                 initial_size_anchor=(1.0, 1.0, 1.0),
                 gt_repeat_num=5,
                 center_range=(-6.5, -9.0, -1.0, 6.5, 9.0, 4.5),
                 size_logit_range=(-10.0, 10.0),
                 ):
        super(ReconDetHead, self).__init__(init_cfg)
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        self.cls_loss = MODELS.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if if_v2_head:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels // 2, n_channels // 4, n_channels // 8],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        else:
            self.mlp_func = partial(
                GenericMLP,
                norm_fn_name="bn1d",
                activation="relu",
                use_conv=True,
                hidden_dims=[n_channels, n_channels],
                dropout=mlp_dropout,
                input_dim=n_channels,
            )
        self._init_layers(n_channels, n_reg_outs, n_classes, n_levels)
        if matcher != 'repeated_hungarian':
            raise ValueError('matcher must be repeated_hungarian')
        self.matcher = RepeatedHungarianMatcher(
            cost_weights=matcher_cost_weights,
            gt_repeat_num=gt_repeat_num,
            center_range=center_range)
        self.loss_weights = loss_weights
        if len(size_logit_range) != 2 or size_logit_range[0] >= size_logit_range[1]:
            raise ValueError('size_logit_range must be an increasing pair')
        self.size_logit_range = tuple(float(value) for value in size_logit_range)
        if len(initial_size_anchor) != 3:
            raise ValueError('initial_size_anchor must contain three values')
        initial_size_anchor = torch.tensor(
            initial_size_anchor, dtype=torch.float32)
        if (not torch.isfinite(initial_size_anchor).all()
                or (initial_size_anchor <= 0).any()):
            raise ValueError('initial_size_anchor must be finite and positive')
        self.register_buffer(
            'initial_size_anchor', initial_size_anchor, persistent=False)
        self.gt_repeat_num = int(gt_repeat_num)
        self.center_range = tuple(float(value) for value in center_range)
        if loss_layer_ids is None:
            loss_layer_ids = list(range(n_levels))
        self.loss_layer_ids = sorted(set(loss_layer_ids))
        if not self.loss_layer_ids:
            raise ValueError('loss_layer_ids must contain at least one layer')
        if self.loss_layer_ids[0] < 0 or self.loss_layer_ids[-1] >= n_levels:
            raise ValueError(
                f'loss_layer_ids must be within [0, {n_levels - 1}]')

    def _init_layers(self, n_channels, n_reg_outs, n_classes, n_levels):
        center_head = self.mlp_func(output_dim=3)
        size_mlp_func = partial(
            GenericMLP,
            norm_fn_name=None,
            activation='relu',
            use_conv=True,
            hidden_dims=[n_channels, n_channels],
            dropout=None,
            input_dim=n_channels)
        size_head = size_mlp_func(output_dim=3)
        semcls_head = self.mlp_func(output_dim=n_classes)
        self.center_heads = nn.ModuleList([
            copy.deepcopy(center_head) for _ in range(n_levels)
        ])
        self.size_heads = nn.ModuleList([
            copy.deepcopy(size_head) for _ in range(n_levels)
        ])
        self.semcls_heads = nn.ModuleList([
            copy.deepcopy(semcls_head) for _ in range(n_levels)
        ])
        for center_head in self.center_heads:
            nn.init.constant_(center_head.layers[-1].weight, 0.)
            nn.init.constant_(center_head.layers[-1].bias, 0.)
        for size_head in self.size_heads:
            nn.init.constant_(size_head.layers[-1].weight, 0.)
            nn.init.constant_(size_head.layers[-1].bias, 0.)
        prior_bias = -math.log((1.0 - 0.01) / 0.01)
        for semcls_head in self.semcls_heads:
            nn.init.constant_(semcls_head.layers[-1].bias, prior_bias)

    def forward(self, x, batch_inputs_dict, refined_query_xyz=None,
                layer_ids=None, initial_sizes=None):
        if layer_ids is None:
            layer_ids = list(range(len(x)))
        if len(layer_ids) != len(x):
            raise ValueError('Layer ids must match decoder outputs')
        if layer_ids != list(range(len(x))):
            raise ValueError(
                'Size refinement requires all decoder layers in order')
        if refined_query_xyz is None or len(refined_query_xyz) != len(x):
            raise ValueError('Refined references must match decoder outputs')

        center_preds = []
        size_residual_preds = []
        cls_preds = []
        for feature, center, layer_id in zip(
                x, refined_query_xyz, layer_ids):
            center_preds.append(center.permute(0, 2, 1))
            size_residual_preds.append(self.size_heads[layer_id](feature))
            cls_preds.append(self.semcls_heads[layer_id](feature))
        if initial_sizes is None:
            size_anchor = self.initial_size_anchor
        else:
            size_anchor = initial_sizes
        with torch.autocast(device_type=size_residual_preds[0].device.type,
                            enabled=False):
            size_reference_logs, size_log_preds, size_preds = (
                decode_size_residuals(
                    [residual.float() for residual in size_residual_preds],
                    size_anchor,
                    self.size_logit_range))
        return dict(
            center_preds=center_preds,
            size_preds=size_preds,
            size_residual_preds=size_residual_preds,
            size_reference_logs=size_reference_logs,
            size_log_preds=size_log_preds,
            cls_preds=cls_preds)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             batch_inputs_dict: dict, refined_query_xyz=None,
             initial_sizes=None, **kwargs) -> dict:
        if refined_query_xyz is None or len(refined_query_xyz) != len(x):
            raise ValueError('Loss requires one refined reference per layer')
        layer_ids = self.loss_layer_ids
        outputs = self(x, batch_inputs_dict, refined_query_xyz,
                       initial_sizes=initial_sizes)

        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            batch_input_points = [None for i in range(len(batch_data_samples))]

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = (
                       outputs['center_preds'], outputs['size_preds'],
                       outputs['size_residual_preds'],
                       outputs['size_reference_logs'],
                       outputs['size_log_preds'], outputs['cls_preds'], layer_ids,
                       batch_gt_instances_3d, batch_input_metas,
                       batch_input_points, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     size_residual_preds: List[List[Tensor]],
                     size_reference_logs: List[List[Tensor]],
                     size_log_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     layer_ids: List[int],
                     #  objness_preds: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_input_points,
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:

        if layer_ids[-1] >= len(center_preds):
            raise ValueError('Supervised layer id exceeds predictions')

        losses_by_layer = []
        for layer_id in layer_ids:
            center_losses = []
            size_losses = []
            cls_losses = []
            giou_losses = []
            matches = []
            for batch_id in range(len(batch_input_metas)):
                gt_bboxes = batch_gt_instances_3d[batch_id].bboxes_3d
                gt_sizes = gt_bboxes.tensor[:, 3:6].clamp_min(1e-5)
                matches.append(self.matcher._get_targets(
                    center_preds[layer_id][batch_id].t(),
                    size_preds[layer_id][batch_id].t(),
                    size_log_preds[layer_id][batch_id].t(),
                    cls_preds[layer_id][batch_id].t(),
                    gt_bboxes.gravity_center,
                    gt_sizes,
                    batch_gt_instances_3d[batch_id].labels_3d))
            local_num_pos = sum(match[0].numel() for match in matches)
            avg_factor = reduce_mean(center_preds[layer_id][0].new_tensor(
                [local_num_pos], dtype=torch.float32)).clamp_min(1.0).item()

            for batch_id in range(len(batch_input_metas)):
                center_loss, size_loss, cls_loss, giou_loss = \
                    self._loss_by_feat_single(
                        center_pred=center_preds[layer_id][batch_id],
                        size_pred=size_preds[layer_id][batch_id],
                        size_residual_pred=(
                            size_residual_preds[layer_id][batch_id]),
                        size_reference_log=(
                            size_reference_logs[layer_id][batch_id]),
                        size_log_pred=size_log_preds[layer_id][batch_id],
                        cls_pred=cls_preds[layer_id][batch_id],
                        input_meta=batch_input_metas[batch_id],
                        gt_bboxes=batch_gt_instances_3d[
                            batch_id].bboxes_3d,
                        gt_labels=batch_gt_instances_3d[
                            batch_id].labels_3d,
                        input_points=batch_input_points[batch_id],
                        match_indices=matches[batch_id],
                        avg_factor=avg_factor)
                center_losses.append(center_loss)
                size_losses.append(size_loss)
                cls_losses.append(cls_loss)
                giou_losses.append(giou_loss)
            losses_by_layer.append((layer_id, dict(
                center_loss=torch.sum(torch.stack(center_losses)),
                size_loss=torch.sum(torch.stack(size_losses)),
                cls_loss=torch.sum(torch.stack(cls_losses)),
                giou_loss=torch.sum(torch.stack(giou_losses)))))

        loss_dict = {}
        main_layer_id = layer_ids[-1]
        for layer_id, layer_losses in losses_by_layer:
            prefix = '' if layer_id == main_layer_id else f'd{layer_id}.'
            for name, value in layer_losses.items():
                loss_dict[f'{prefix}{name}'] = value
        return loss_dict

    def _loss_by_feat_single(self, center_pred, size_pred,
                             size_residual_pred, size_reference_log,
                             size_log_pred, cls_pred, input_meta,
                             gt_bboxes, gt_labels, input_points,
                             match_indices=None, avg_factor=None):
        del input_meta, input_points
        centers = center_pred.t()
        sizes = size_pred.t()
        size_residuals = size_residual_pred.t()
        size_reference_logs = size_reference_log.t()
        size_logs = size_log_pred.t()
        cls_scores = cls_pred.t()
        gt_centers = gt_bboxes.gravity_center
        gt_sizes = gt_bboxes.tensor[:, 3:6].clamp_min(1e-5)
        if match_indices is None:
            pred_indices, gt_indices = self.matcher._get_targets(
                centers, sizes, size_logs, cls_scores,
                gt_centers, gt_sizes, gt_labels)
        else:
            pred_indices, gt_indices = match_indices
        if avg_factor is None:
            num_pos = reduce_mean(centers.new_tensor(
                [pred_indices.numel()], dtype=torch.float32))
            avg_factor = num_pos.clamp_min(1.0).item()

        cls_target = torch.full(
            (centers.shape[0],), self.n_classes,
            dtype=torch.long, device=centers.device)
        cls_target[pred_indices] = gt_labels[gt_indices]
        cls_loss = self.cls_loss(
            cls_scores, cls_target, avg_factor=avg_factor)
        cls_loss = cls_loss * self.loss_weights['cls_loss']

        size_loss = matched_size_residual_loss(
            size_residuals, size_reference_logs, gt_sizes,
            pred_indices, gt_indices, avg_factor)
        size_loss = size_loss * self.loss_weights['size_loss']
        if pred_indices.numel() == 0:
            center_loss = centers.sum() * 0.0
            giou_loss = sizes.sum() * 0.0
        else:
            center_min = self.matcher.center_min.to(centers)
            center_extent = self.matcher.center_extent.to(centers)
            matched_centers = (
                centers[pred_indices] - center_min) / center_extent
            matched_gt_centers = (
                gt_centers[gt_indices] - center_min) / center_extent
            center_loss = F.l1_loss(
                matched_centers, matched_gt_centers,
                reduction='sum') / avg_factor

            pred_tp_bbox = self._center_size_pred_to_bbox(
                centers[pred_indices], sizes[pred_indices])
            gt_tp_bbox = self._center_size_pred_to_bbox(
                gt_centers[gt_indices], gt_sizes[gt_indices])
            giou = axis_aligned_bbox_overlaps_3d(
                pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0),
                mode='giou', is_aligned=True)
            giou_loss = (1.0 - giou).sum() / avg_factor
        center_loss = center_loss * self.loss_weights['center_loss']
        giou_loss = giou_loss * self.loss_weights['iou_loss']
        return center_loss, size_loss, cls_loss, giou_loss

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList, batch_inputs_dict,
                refined_query_xyz=None, layer_ids=None, initial_sizes=None,
                rescale: bool = False) -> InstanceList:

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outputs = self(x, batch_inputs_dict, refined_query_xyz, layer_ids,
                       initial_sizes=initial_sizes)
        predictions = self.predict_by_feat(
            [outputs['center_preds'][-1]],
            [outputs['size_preds'][-1]],
            [outputs['cls_preds'][-1]],
            batch_input_metas=batch_input_metas,
            rescale=rescale, batch_inputs_dict=batch_inputs_dict, batch_data_samples=batch_data_samples)
        return predictions

    def predict_by_feat(self, center_preds: List[List[Tensor]],
                        size_preds: List[List[Tensor]],
                        cls_preds: List[List[Tensor]],
                        batch_input_metas: List[dict], batch_inputs_dict: dict, batch_data_samples,
                        **kwargs) -> List[InstanceData]:

        results = []
        if 'points' in batch_inputs_dict.keys():
            batch_input_points = batch_inputs_dict['points']
        else:
            batch_input_points = [None for i in range(len(batch_input_metas))]
        for i in range(len(batch_input_metas)):
            results.append(
                self._predict_by_feat_single(
                    center_preds=[x[i] for x in center_preds],
                    size_preds=[x[i] for x in size_preds],
                    cls_preds=[x[i] for x in cls_preds],
                    input_meta=batch_input_metas[i],
                    input_points=batch_input_points[i],
                    data_samples=batch_data_samples[i]))
        return results

    def _predict_by_feat_single(self, center_preds, size_preds, cls_preds,
                                input_meta: dict, input_points, data_samples) -> InstanceData:

        mlvl_bboxes, mlvl_scores = [], []
        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[
                stage_idx].t()
            scores = cls_scores.sigmoid()

            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                centers = centers[ids]
                sizes = sizes[ids]
                scores = scores[ids]
            bboxes = self._center_size_pred_to_bbox(centers, sizes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes_after_nms, scores, labels = self._nms(bboxes, scores,
                                                     input_meta)  # bboxes(n_box, 6) (x_center, y_center, z_center, w, h, z)

        bboxes = input_meta['box_type_3d'](
            bboxes_after_nms, box_dim=6, with_yaw=False, origin=(.5, .5, .5))

        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        return results

    def find_max_iou_from_center_size_boxes(self, boxes1, boxes2):
        boxes1_tp = self._center_size_pred_to_bbox(boxes1[:, :3], boxes1[:, 3:6])
        boxes2_tp = self._center_size_pred_to_bbox(boxes2[:, :3], boxes2[:, 3:6])
        giou_2 = axis_aligned_bbox_overlaps_3d(boxes1_tp.unsqueeze(0), boxes2_tp.unsqueeze(0), mode='giou')  # giou
        giou_max_gt, max_gt_box_idx = torch.max(giou_2, axis=2)
        max_giou, max_pred_box_idx = torch.max(giou_max_gt, axis=1)
        assert max_giou <= 1 and max_giou >= -1
        return max_giou, max_gt_box_idx, max_pred_box_idx

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0] / 2.0, centers[:, 1] - sizes[:, 1] / 2.0,
            centers[:, 2] - sizes[:, 2] / 2.0, centers[:, 0] + sizes[:, 0] / 2.0,
            centers[:, 1] + sizes[:, 1] / 2.0, centers[:, 2] + sizes[:, 2] / 2.0
        ], -1)

    def _nms(self, bboxes, scores, img_meta):  # bbox is 6-dim. (x_min, y_min, z_min, x_max, y_max, z_max)
        scores, labels = scores.max(dim=1)
        ids = scores > self.test_cfg.score_thr
        bboxes = bboxes[ids]
        scores = scores[ids]
        labels = labels[ids]
        ids = self.aligned_3d_nms(bboxes, scores, labels,
                                  self.test_cfg.iou_thr)
        bboxes = bboxes[ids]
        bboxes = torch.stack(
            ((bboxes[:, 0] + bboxes[:, 3]) / 2.,
             (bboxes[:, 1] + bboxes[:, 4]) / 2.,
             (bboxes[:, 2] + bboxes[:, 5]) / 2., bboxes[:, 3] - bboxes[:, 0],
             bboxes[:, 4] - bboxes[:, 1], bboxes[:, 5] - bboxes[:, 2]),
            dim=1)  # (convert to (x_center, y_center, z_center, w, h, z))
        return bboxes, scores[ids], labels[ids]

    @staticmethod
    def aligned_3d_nms(boxes, scores, classes, thresh):

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        z1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]
        z2 = boxes[:, 5]
        area = (x2 - x1) * (y2 - y1) * (z2 - z1)
        zero = boxes.new_zeros(1, )

        score_sorted = torch.argsort(scores)
        pick = []
        while (score_sorted.shape[0] != 0):
            last = score_sorted.shape[0]
            i = score_sorted[-1]
            pick.append(i)

            xx1 = torch.max(x1[i], x1[score_sorted[:last - 1]])
            yy1 = torch.max(y1[i], y1[score_sorted[:last - 1]])
            zz1 = torch.max(z1[i], z1[score_sorted[:last - 1]])
            xx2 = torch.min(x2[i], x2[score_sorted[:last - 1]])
            yy2 = torch.min(y2[i], y2[score_sorted[:last - 1]])
            zz2 = torch.min(z2[i], z2[score_sorted[:last - 1]])
            classes1 = classes[i]
            classes2 = classes[score_sorted[:last - 1]]
            inter_l = torch.max(zero, xx2 - xx1)
            inter_w = torch.max(zero, yy2 - yy1)
            inter_h = torch.max(zero, zz2 - zz1)

            inter = inter_l * inter_w * inter_h
            iou = inter / (area[i] + area[score_sorted[:last - 1]] - inter)
            iou = iou * (classes1 == classes2).float()
            score_sorted = score_sorted[torch.nonzero(
                iou <= thresh, as_tuple=False).flatten()]

        indices = boxes.new_tensor(pick, dtype=torch.long)
        return indices
