import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from mmdet3d.structures.ops.iou3d_calculator import axis_aligned_bbox_overlaps_3d


class UnifiedMatcher(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}):
        super().__init__()
        self.cost_weights = cost_weights

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1)

        pred_tp_bbox = self._center_size_pred_to_bbox(all_centers, all_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(gt_centers, gt_sizes)

        with torch.no_grad():
            giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou')
            assert giou.shape[0] == 1
            giou = giou.squeeze(0)

        cost_class = -all_cls.sigmoid()[:, gt_labels]  # (Total_Pred, M)
        cost_center = torch.cdist(all_centers, gt_centers, p=1)  # (Total_Pred, M)
        cost_objness = -all_objness.sigmoid()  # (Total_Pred, M)
        # giou = generalized_box3d_iou(pred_corners, gt_corners, torch.tensor([gt_centers.size(0)]))[0]  # (Total_Pred, M)
        cost_giou = -giou

        total_cost = (
                self.cost_weights['cls'] * cost_class +
                self.cost_weights['center'] * cost_center +
                self.cost_weights['obj_ness'] * cost_objness +
                self.cost_weights['giou'] * cost_giou
        )

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

        pred_tp_bbox = self._center_size_pred_to_bbox(all_centers, all_sizes)
        gt_tp_bbox = self._center_size_pred_to_bbox(gt_centers, gt_sizes)

        with torch.no_grad():
            giou = axis_aligned_bbox_overlaps_3d(pred_tp_bbox.unsqueeze(0), gt_tp_bbox.unsqueeze(0), mode='giou')
            assert giou.shape[0] == 1
            giou = giou.squeeze(0)  # (Total_Pred, M)

        cost_class = -all_cls.sigmoid()[:, gt_labels]
        cost_center = torch.cdist(all_centers, gt_centers, p=1)
        cost_objness = -all_objness.sigmoid()
        cost_giou = -giou

        total_cost = (
                self.cost_weights['cls'] * cost_class +
                self.cost_weights['center'] * cost_center +
                self.cost_weights['obj_ness'] * cost_objness +
                self.cost_weights['giou'] * cost_giou
        )

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
