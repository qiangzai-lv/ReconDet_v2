import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from mmdet3d.structures.ops.iou3d_calculator import axis_aligned_bbox_overlaps_3d


def _ensure_finite(name, value):
    finite = torch.isfinite(value)
    if finite.all():
        return
    invalid_count = int((~finite).sum().item())
    finite_values = value[finite]
    finite_range = 'no finite values'
    if finite_values.numel() > 0:
        finite_range = (
            f'finite_min={finite_values.min().item():.6g}, '
            f'finite_max={finite_values.max().item():.6g}')
    raise FloatingPointError(
        f'{name} contains {invalid_count}/{value.numel()} non-finite values; '
        f'{finite_range}')


def _build_cost_matrix(all_centers, all_sizes, all_cls, all_objness,
                       gt_centers, gt_sizes, gt_labels, cost_weights):
    for name, value in (
            ('predicted centers', all_centers),
            ('predicted sizes', all_sizes),
            ('classification logits', all_cls),
            ('objectness scores', all_objness),
            ('ground-truth centers', gt_centers),
            ('ground-truth sizes', gt_sizes)):
        _ensure_finite(name, value)

    pred_boxes = UnifiedMatcher._center_size_pred_to_bbox(
        None, all_centers, all_sizes)
    gt_boxes = UnifiedMatcher._center_size_pred_to_bbox(
        None, gt_centers, gt_sizes)
    giou = axis_aligned_bbox_overlaps_3d(
        pred_boxes.unsqueeze(0), gt_boxes.unsqueeze(0), mode='giou').squeeze(0)
    _ensure_finite('GIoU matching cost', giou)

    cost_class = -all_cls.sigmoid()[:, gt_labels]
    cost_center = torch.cdist(all_centers, gt_centers, p=1)
    cost_objness = -all_objness.sigmoid()
    total_cost = (
        cost_weights['cls'] * cost_class +
        cost_weights['center'] * cost_center +
        cost_weights['obj_ness'] * cost_objness -
        cost_weights['giou'] * giou)
    _ensure_finite('Hungarian cost matrix', total_cost)
    return total_cost, giou


class RepeatedHungarianMatcher(nn.Module):
    """Hungarian matcher with repeated GTs and explicit log-size cost."""

    def __init__(self,
                 cost_weights=None,
                 gt_repeat_num=5,
                 center_range=(-6.5, -9.0, -1.0, 6.5, 9.0, 4.5),
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        if cost_weights is None:
            cost_weights = dict(cls=2.0, center=1.0, size=1.0, giou=2.0)
        required = {'cls', 'center', 'size', 'giou'}
        if set(cost_weights) != required:
            raise ValueError(f'cost_weights must contain exactly {required}')
        if gt_repeat_num < 1:
            raise ValueError('gt_repeat_num must be positive')
        if len(center_range) != 6:
            raise ValueError('center_range must contain six values')
        center_range = torch.tensor(center_range, dtype=torch.float32)
        center_extent = center_range[3:] - center_range[:3]
        if not torch.isfinite(center_range).all() or (center_extent <= 0).any():
            raise ValueError('center_range must be finite and increasing')
        self.cost_weights = dict(cost_weights)
        self.gt_repeat_num = int(gt_repeat_num)
        self.register_buffer('center_min', center_range[:3], persistent=False)
        self.register_buffer('center_extent', center_extent, persistent=False)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def _focal_cost(self, logits, gt_labels):
        probability = logits.sigmoid()
        eps = torch.finfo(probability.dtype).eps
        negative = ((1.0 - self.focal_alpha)
                    * probability.pow(self.focal_gamma)
                    * -(1.0 - probability + eps).log())
        positive = (self.focal_alpha
                    * (1.0 - probability).pow(self.focal_gamma)
                    * -(probability + eps).log())
        return positive[:, gt_labels] - negative[:, gt_labels]

    @staticmethod
    def _center_size_pred_to_bbox(centers, sizes):
        half_size = sizes / 2.0
        return torch.cat((centers - half_size, centers + half_size), dim=-1)

    @torch.no_grad()
    def _get_targets(self, pred_centers, pred_sizes, pred_size_logs,
                     pred_logits, gt_centers, gt_sizes, gt_labels):
        num_queries = pred_centers.shape[0]
        num_gt = gt_centers.shape[0]
        empty = torch.empty(0, dtype=torch.long, device=pred_centers.device)
        if num_queries == 0 or num_gt == 0:
            return empty, empty

        for name, value in (
                ('predicted centers', pred_centers),
                ('predicted sizes', pred_sizes),
                ('predicted log sizes', pred_size_logs),
                ('classification logits', pred_logits),
                ('ground-truth centers', gt_centers),
                ('ground-truth sizes', gt_sizes)):
            _ensure_finite(name, value)
        if (pred_sizes <= 0).any() or (gt_sizes <= 0).any():
            raise ValueError('predicted and ground-truth sizes must be positive')

        output_device = pred_centers.device
        pred_centers = pred_centers.float()
        pred_sizes = pred_sizes.float()
        pred_size_logs = pred_size_logs.float()
        pred_logits = pred_logits.float()
        gt_centers = gt_centers.float()
        gt_sizes = gt_sizes.float()

        repeat_num = self.gt_repeat_num
        repeated_gt_indices = torch.arange(
            num_gt, device=gt_centers.device).repeat(repeat_num)
        repeated_centers = gt_centers[repeated_gt_indices]
        repeated_sizes = gt_sizes[repeated_gt_indices]
        repeated_labels = gt_labels[repeated_gt_indices]

        center_min = self.center_min.to(pred_centers)
        center_extent = self.center_extent.to(pred_centers)
        pred_centers_normalized = (pred_centers - center_min) / center_extent
        gt_centers_normalized = (repeated_centers - center_min) / center_extent
        cost_center = torch.cdist(
            pred_centers_normalized, gt_centers_normalized, p=1)
        cost_size = torch.cdist(
            pred_size_logs, repeated_sizes.clamp_min(1e-5).log(), p=1)
        cost_class = self._focal_cost(pred_logits, repeated_labels)

        pred_boxes = self._center_size_pred_to_bbox(pred_centers, pred_sizes)
        gt_boxes = self._center_size_pred_to_bbox(
            repeated_centers, repeated_sizes)
        giou = axis_aligned_bbox_overlaps_3d(
            pred_boxes.unsqueeze(0), gt_boxes.unsqueeze(0),
            mode='giou').squeeze(0)
        total_cost = (
            self.cost_weights['cls'] * cost_class
            + self.cost_weights['center'] * cost_center
            + self.cost_weights['size'] * cost_size
            - self.cost_weights['giou'] * giou)
        _ensure_finite('Hungarian cost matrix', total_cost)

        pred_indices, repeated_indices = linear_sum_assignment(
            total_cost.cpu().numpy())
        pred_indices = torch.from_numpy(pred_indices).long().to(
            output_device)
        repeated_indices = torch.from_numpy(repeated_indices).long().to(
            output_device)
        return pred_indices, repeated_gt_indices.to(output_device)[
            repeated_indices]


class UnifiedMatcher(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}):
        super().__init__()
        self.cost_weights = cost_weights

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1)

        total_cost, _ = _build_cost_matrix(
            all_centers, all_sizes, all_cls, all_objness,
            gt_centers, gt_sizes, gt_labels, self.cost_weights)

        pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().numpy())
        return torch.from_numpy(pred_indices).long().to(all_centers.device), torch.from_numpy(gt_indices).long().to(
            all_centers.device)

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0] / 2.0, centers[:, 1] - sizes[:, 1] / 2.0,
            centers[:, 2] - sizes[:, 2] / 2.0, centers[:, 0] + sizes[:, 0] / 2.0,
            centers[:, 1] + sizes[:, 1] / 2.0, centers[:, 2] + sizes[:, 2] / 2.0
        ], -1)


class UnifiedMatcherMoreThanOne(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}, matcher_iou_thres=0.25,
                 matcher_max_dynamic_samples=10):
        super().__init__()
        self.cost_weights = cost_weights
        self.iou_threshold = matcher_iou_thres,
        self.matcher_max_dynamic_samples = matcher_max_dynamic_samples

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1)

        total_cost, giou = _build_cost_matrix(
            all_centers, all_sizes, all_cls, all_objness,
            gt_centers, gt_sizes, gt_labels, self.cost_weights)

        pred_indices, gt_indices = linear_sum_assignment(total_cost.cpu().numpy())
        pred_indices = torch.from_numpy(pred_indices).long().to(all_centers.device)
        gt_indices = torch.from_numpy(gt_indices).long().to(all_centers.device)

        used_pred_mask = torch.zeros(giou.size(0), dtype=torch.bool, device=giou.device)
        used_pred_mask[pred_indices] = True

        iou_mask = giou > self.iou_threshold[0]

        dynamic_preds = []
        dynamic_gts = []

        max_iou_per_gt = giou.max(dim=0).values
        sorted_gt_indices = torch.argsort(max_iou_per_gt)

        for gt_idx in sorted_gt_indices:
            candidate_mask = iou_mask[:, gt_idx] & ~used_pred_mask
            candidate_preds = torch.nonzero(candidate_mask, as_tuple=True)[0]

            if candidate_preds.numel() == 0:
                continue

            giou_values = giou[candidate_preds, gt_idx]

            if self.matcher_max_dynamic_samples < len(giou_values):
                _, topk_indices = torch.topk(giou_values, k=self.matcher_max_dynamic_samples)
                selected_preds = candidate_preds[topk_indices]
            else:
                selected_preds = candidate_preds

            dynamic_preds.append(selected_preds)
            dynamic_gts.append(torch.full_like(selected_preds, gt_idx))

            used_pred_mask[selected_preds] = True

        if dynamic_preds:
            dynamic_preds = torch.cat(dynamic_preds)
            dynamic_gts = torch.cat(dynamic_gts)
        else:
            dynamic_preds = torch.empty(0, dtype=torch.long, device=giou.device)
            dynamic_gts = torch.empty(0, dtype=torch.long, device=giou.device)

        combined_preds = torch.cat([pred_indices, dynamic_preds])
        combined_gts = torch.cat([gt_indices, dynamic_gts])

        return combined_preds, combined_gts

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0] / 2.0, centers[:, 1] - sizes[:, 1] / 2.0,
            centers[:, 2] - sizes[:, 2] / 2.0, centers[:, 0] + sizes[:, 0] / 2.0,
            centers[:, 1] + sizes[:, 1] / 2.0, centers[:, 2] + sizes[:, 2] / 2.0
        ], -1)
