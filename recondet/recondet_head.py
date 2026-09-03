# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor, nn

from recondet.detr3_models.helpers import GenericMLP
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.structures.ops.iou3d_calculator import axis_aligned_bbox_overlaps_3d
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from recondet.matcher import UnifiedMatcher, UnifiedMatcherMoreThanOne


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    # origin: point-cloud center.
    points = torch.stack(
        torch.meshgrid([
            torch.arange(n_voxels[0]),  # 40 W width, x
            torch.arange(n_voxels[1]),  # 40 D depth, y
            torch.arange(n_voxels[2])  # 16 H Height, z
        ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


@MODELS.register_module()
class ReconDetHead(BaseModule):

    def __init__(self,
                 n_classes: int,
                 n_levels: int,
                 n_channels: int,
                 n_reg_outs: int,
                 pts_assign_threshold: int,
                 pts_center_threshold: int,
                 objness_loss: ConfigType = dict(type='mmdet.FocalLoss', use_sigmoid=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 mlp_dropout=0.3,
                 matcher_cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0},
                 loss_weights={'center_loss': 5.0, 'size_loss': 1.0,
                               'cls_loss': 1.0,
                               'objness_loss': 1.0,
                               'iou_loss': 1.0,
                               'not_objness_loss': 0.25},
                 learn_center_diff=False,
                 if_v2_head=False,
                 matcher='one2one',
                 matcher_iou_thres=0.25,
                 matcher_max_dynamic_samples=10,
                 loss_layer_ids=None,
                 ):
        super(ReconDetHead, self).__init__(init_cfg)
        self.n_classes = n_classes
        self.n_levels = n_levels
        self.n_reg_outs = n_reg_outs
        self.pts_assign_threshold = pts_assign_threshold
        self.pts_center_threshold = pts_center_threshold
        class_weights = torch.ones((self.n_classes + 1))
        class_weights[-1] = loss_weights['not_objness_loss']
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)  # MODELS.build(cls_loss)
        self.objness_loss = MODELS.build(objness_loss)
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
        assert matcher in ['one2one', 'one2more']
        if matcher == 'one2one':
            self.matcher = UnifiedMatcher(cost_weights=matcher_cost_weights)
        elif matcher == 'one2more':
            self.matcher = UnifiedMatcherMoreThanOne(cost_weights=matcher_cost_weights,
                                                     matcher_iou_thres=matcher_iou_thres,
                                                     matcher_max_dynamic_samples=matcher_max_dynamic_samples)
        self.loss_weights = loss_weights
        self.learn_center_diff = learn_center_diff
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
        size_head = self.mlp_func(output_dim=3)
        semcls_head = self.mlp_func(output_dim=n_classes + 1)
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
        self.scales = nn.ModuleList([Scale(1.) for _ in range(n_levels)])

    def forward(self, x, batch_inputs_dict, refined_query_xyz=None,
                layer_ids=None):
        if layer_ids is None:
            layer_ids = list(range(len(x)))
        if len(layer_ids) != len(x):
            raise ValueError('Layer ids must match decoder outputs')
        if refined_query_xyz is None or len(refined_query_xyz) != len(x):
            raise ValueError('Refined references must match decoder outputs')

        center_preds = []
        size_preds = []
        cls_preds = []
        for feature, center, layer_id in zip(
                x, refined_query_xyz, layer_ids):
            center_preds.append(center.permute(0, 2, 1))
            size_preds.append(torch.exp(
                self.scales[layer_id](self.size_heads[layer_id](feature))))
            cls_preds.append(self.semcls_heads[layer_id](feature))
        return center_preds, size_preds, cls_preds

    @staticmethod
    def _batch_tensor(value, reference):
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (int, float)):
            tensor = torch.as_tensor([value])
        else:
            tensor = torch.stack([
                item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
                for item in value
            ], dim=0)
        return tensor.to(device=reference.device, dtype=torch.float32)

    def _transform_bbox_predictions(self, center_preds, size_preds,
                                    batch_inputs_dict):
        reference = center_preds[0]
        pose_matrix = self._batch_tensor(
            batch_inputs_dict['pose_matrix'], reference)
        axis_align_matrix = self._batch_tensor(
            batch_inputs_dict['axis_align_matrix'], reference)
        predicted_first_w2c = self._batch_tensor(
            batch_inputs_dict['predicted_first_w2c'], reference)
        scene_scale = self._batch_tensor(
            batch_inputs_dict['scene_scale'], reference).reshape(
                reference.shape[0], 1, 1)

        if pose_matrix.shape[-2:] != (4, 4):
            raise ValueError('pose_matrix must have shape [B, 4, 4]')
        if axis_align_matrix.shape[-2:] != (4, 4):
            raise ValueError('axis_align_matrix must have shape [B, 4, 4]')
        if predicted_first_w2c.shape[-2:] == (3, 4):
            bottom_row = predicted_first_w2c.new_zeros(
                predicted_first_w2c.shape[0], 1, 4)
            bottom_row[..., 0, 3] = 1
            predicted_first_w2c = torch.cat(
                [predicted_first_w2c, bottom_row], dim=1)
        if predicted_first_w2c.shape[-2:] != (4, 4):
            raise ValueError(
                'predicted_first_w2c must have shape [B, 3, 4] or [B, 4, 4]')

        transform = torch.bmm(
            axis_align_matrix,
            torch.bmm(pose_matrix, predicted_first_w2c))
        transformed_centers = []
        transformed_sizes = []
        for centers, sizes in zip(center_preds, size_preds):
            scaled_centers = centers.float() * scene_scale
            ones = scaled_centers.new_ones(
                scaled_centers.shape[0], 1, scaled_centers.shape[2])
            homogeneous_centers = torch.cat([scaled_centers, ones], dim=1)
            aligned_centers = torch.bmm(
                transform, homogeneous_centers)[:, :3, :]
            transformed_centers.append(aligned_centers)
            transformed_sizes.append(sizes.float() * scene_scale)
        return transformed_centers, transformed_sizes

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             batch_inputs_dict: dict, refined_query_xyz=None, **kwargs) -> dict:
        if refined_query_xyz is None or len(refined_query_xyz) != len(x):
            raise ValueError('Loss requires one refined reference per layer')
        layer_ids = self.loss_layer_ids
        supervised_features = [x[layer_id] for layer_id in layer_ids]
        supervised_references = [
            refined_query_xyz[layer_id] for layer_id in layer_ids
        ]
        center_preds, size_preds, cls_preds = self(
            supervised_features,
            batch_inputs_dict,
            supervised_references,
            layer_ids)
        center_preds, size_preds = self._transform_bbox_predictions(
            center_preds, size_preds, batch_inputs_dict)

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

        loss_inputs = (center_preds, size_preds, cls_preds, layer_ids,
                       batch_gt_instances_3d, batch_input_metas,
                       batch_input_points, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(self,
                     center_preds: List[List[Tensor]],
                     size_preds: List[List[Tensor]],
                     cls_preds: List[List[Tensor]],
                     layer_ids: List[int],
                     #  objness_preds: List[List[Tensor]],
                     batch_gt_instances_3d: InstanceList,
                     batch_input_metas: List[dict],
                     batch_input_points,
                     batch_gt_instances_ignore: OptInstanceList = None,
                     **kwargs) -> dict:

        if len(layer_ids) != len(center_preds):
            raise ValueError('Layer ids must match supervised predictions')

        losses_by_layer = []
        for prediction_id, layer_id in enumerate(layer_ids):
            center_losses = []
            size_losses = []
            cls_losses = []
            giou_losses = []
            for batch_id in range(len(batch_input_metas)):
                center_loss, size_loss, cls_loss, giou_loss = \
                    self._loss_by_feat_single(
                        center_preds=[center_preds[prediction_id][batch_id]],
                        size_preds=[size_preds[prediction_id][batch_id]],
                        cls_preds=[cls_preds[prediction_id][batch_id]],
                        input_meta=batch_input_metas[batch_id],
                        gt_bboxes=batch_gt_instances_3d[
                            batch_id].bboxes_3d,
                        gt_labels=batch_gt_instances_3d[
                            batch_id].labels_3d,
                        input_points=batch_input_points[batch_id])
                center_losses.append(center_loss)
                size_losses.append(size_loss)
                cls_losses.append(cls_loss)
                giou_losses.append(giou_loss)
            losses_by_layer.append((layer_id, dict(
                center_loss=torch.mean(torch.stack(center_losses)),
                size_loss=torch.mean(torch.stack(size_losses)),
                cls_loss=torch.mean(torch.stack(cls_losses)),
                giou_loss=torch.mean(torch.stack(giou_losses)))))

        loss_dict = {}
        main_layer_id = layer_ids[-1]
        for layer_id, layer_losses in losses_by_layer:
            prefix = '' if layer_id == main_layer_id else f'd{layer_id}.'
            for name, value in layer_losses.items():
                loss_dict[f'{prefix}{name}'] = value
        return loss_dict

    def _loss_by_feat_single(self, center_preds, size_preds, cls_preds,  # objness_preds,
                             input_meta, gt_bboxes, gt_labels, input_points):

        all_centers = torch.cat([c.t() for c in center_preds], dim=0)  # (Total_Pred, 3)
        all_sizes = torch.cat([s.t() for s in size_preds], dim=0)  # (Total_Pred, 3)
        all_cls = torch.cat([c.t() for c in cls_preds], dim=0)  # (Total_Pred, C)

        gt_centers = gt_bboxes.gravity_center
        gt_sizes = gt_bboxes.tensor[:, 3:6]

        all_pred_indices = []
        all_gt_indices = []
        offset = 0

        for stage_idx in range(len(center_preds)):
            centers, sizes, cls_scores = center_preds[stage_idx].t(), size_preds[stage_idx].t(), cls_preds[
                stage_idx].t()  # , objness_preds[stage_idx].t()
            cls_scores_softmax = F.softmax(cls_scores, dim=1)
            obj_scores = 1.0 - cls_scores_softmax[:, -1]
            n_predictions = centers.size(0)

            pred_indices, gt_indices = self.matcher._get_targets(
                centers, sizes, cls_scores, obj_scores,
                gt_centers, gt_sizes, gt_labels
            )

            all_pred_indices.append(pred_indices + offset)
            all_gt_indices.append(gt_indices)

            offset += n_predictions

        pred_indices, gt_indices = torch.cat(all_pred_indices), torch.cat(all_gt_indices)
        matched_centers = all_centers[pred_indices]
        matched_sizes = all_sizes[pred_indices]
        matched_gt_centers = gt_centers[gt_indices]
        matched_gt_sizes = gt_sizes[gt_indices]
        matched_gt_labels = gt_labels[gt_indices]
        center_loss = F.l1_loss(matched_centers, matched_gt_centers) * self.loss_weights['center_loss']
        size_loss = F.l1_loss(matched_sizes, matched_gt_sizes) * self.loss_weights['size_loss']
        cls_target = torch.ones((all_centers.shape[0]), device=all_centers.device) * self.n_classes
        cls_target = cls_target.long()
        cls_target[pred_indices] = matched_gt_labels
        cls_loss = self.cls_loss(all_cls, cls_target) * self.loss_weights['cls_loss']
        pred_tp_bbox = self._center_size_pred_to_bbox(matched_centers, matched_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(matched_gt_centers, matched_gt_sizes)
        giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou',
                                             is_aligned=True)
        giou_loss = (1.0 - giou).mean() * self.loss_weights['iou_loss']
        return center_loss, size_loss, cls_loss, giou_loss

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList, batch_inputs_dict,
                refined_query_xyz=None, layer_ids=None,
                rescale: bool = False) -> InstanceList:

        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        center_preds, size_preds, cls_preds = self(
            x, batch_inputs_dict, refined_query_xyz, layer_ids)
        center_preds, size_preds = self._transform_bbox_predictions(
            center_preds, size_preds, batch_inputs_dict)
        predictions = self.predict_by_feat(
            center_preds, size_preds, cls_preds,
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
            cls_scores = F.softmax(cls_scores, dim=1)
            objectness = 1 - cls_scores[:, -1]
            scores = cls_scores[:, :-1] * objectness.unsqueeze(-1)

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
