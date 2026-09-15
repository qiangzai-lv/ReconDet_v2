import torch
from mmdet3d.models.layers import aligned_3d_nms


def _center_size_to_minmax(centers, sizes):
    half_sizes = sizes / 2.0
    return torch.cat([centers - half_sizes, centers + half_sizes], dim=-1)


def _weighted_fps(points, scores, selected_points, count):
    """Select distant, high-score candidates, deterministically with repeats."""
    if count <= 0:
        return points.new_empty((0,), dtype=torch.long)
    if points.shape[0] == 0:
        raise ValueError('weighted FPS requires at least one candidate')
    weights = scores.float().clamp_min(0)
    selected = selected_points.float()
    remaining = torch.ones(
        points.shape[0], dtype=torch.bool, device=points.device)
    chosen = []
    min_distance = torch.full((points.shape[0],), float('inf'),
                              device=points.device, dtype=torch.float32)
    if selected.numel():
        min_distance = torch.cdist(
            points.float(), selected).square().amin(dim=1)
    else:
        first = weights.argmax()
        chosen.append(first)
        remaining[first] = False
        min_distance = torch.cdist(
            points.float(), points[first:first + 1]).square().squeeze(1)
    for _ in range(min(count, points.shape[0])):
        if len(chosen) >= min(count, points.shape[0]):
            break
        priority = min_distance * weights
        priority[~remaining] = -torch.inf
        index = priority.argmax()
        chosen.append(index)
        remaining[index] = False
        min_distance = torch.minimum(
            min_distance,
            torch.cdist(
                points.float(), points[index:index + 1]).square().squeeze(1))
    if len(chosen) < count:
        repeats = torch.tensor(chosen, device=points.device, dtype=torch.long)
        chosen.extend(repeats[torch.arange(
            count - len(chosen), device=points.device) % len(repeats)].tolist())
    return torch.tensor(chosen, device=points.device, dtype=torch.long)


def select_reconstruction_boxes(
        reconstruction_outputs, batch_size, num_views, num_queries,
        query_xyz_range, nms_iou_thr, logger, scene_ids=None,
        foreground_score_thr=0.1):
    """Select fixed-count scene proposals from reconstruction AABBs."""
    required = (
        'bbox_centers_aligned', 'bbox_sizes_aligned', 'bbox_scores',
        'bbox_labels', 'class_scores_2d', 'fused_detection_query',
        'points_aligned', 'valid_mask')
    missing = [key for key in required if key not in reconstruction_outputs]
    if missing:
        raise KeyError(f'Missing reconstruction outputs: {missing}')
    if min(batch_size, num_views, num_queries) <= 0:
        raise ValueError('batch_size, num_views and num_queries must be positive')
    if len(query_xyz_range) != 6:
        raise ValueError('query range must have 6 values')
    if not 0 <= nms_iou_thr <= 1:
        raise ValueError('NMS IoU threshold must be within [0, 1]')
    if not 0 <= foreground_score_thr <= 1:
        raise ValueError('foreground score threshold must be within [0, 1]')

    valid_mask = reconstruction_outputs['valid_mask']
    expected_views = batch_size * num_views
    if valid_mask.ndim != 2 or valid_mask.shape[0] != expected_views:
        raise ValueError('valid_mask must have shape [B * V, Q]')
    query_count = valid_mask.shape[1]
    query_dims = reconstruction_outputs['fused_detection_query'].shape[-1]
    expected_prefix = (expected_views, query_count)
    for key in required:
        if reconstruction_outputs[key].shape[:2] != expected_prefix:
            raise ValueError(f'{key} must begin with [B * V, Q]')

    centers = reconstruction_outputs['bbox_centers_aligned'].reshape(
        batch_size, num_views * query_count, 3)
    sizes = reconstruction_outputs['bbox_sizes_aligned'].reshape_as(centers)
    scores = reconstruction_outputs['bbox_scores'].reshape(
        batch_size, num_views * query_count)
    labels = reconstruction_outputs['bbox_labels'].reshape_as(scores).long()
    queries = reconstruction_outputs['fused_detection_query'].reshape(
        batch_size, num_views * query_count, query_dims)
    reconstruction_points = reconstruction_outputs['points_aligned'].reshape(
        batch_size, num_views * query_count, 3)
    class_scores_2d = reconstruction_outputs['class_scores_2d'].reshape(
        batch_size, num_views * query_count, -1)
    if class_scores_2d.shape[-1] == 0:
        raise ValueError('class_scores_2d must contain at least one class')
    foreground_scores_2d = class_scores_2d.amax(dim=-1)
    valid = valid_mask.reshape(
        batch_size, num_views * query_count).bool().clone()

    xyz_range = centers.new_tensor(query_xyz_range)
    xyz_min, xyz_max = xyz_range[:3], xyz_range[3:]
    valid &= torch.isfinite(centers).all(dim=-1)
    valid &= torch.isfinite(sizes).all(dim=-1)
    valid &= torch.isfinite(scores)
    valid &= torch.isfinite(queries).all(dim=-1)
    valid &= torch.isfinite(reconstruction_points).all(dim=-1)
    valid &= (sizes > 0).all(dim=-1)
    valid &= (centers >= xyz_min).all(dim=-1)
    valid &= (centers <= xyz_max).all(dim=-1)
    geometry_valid = valid & torch.isfinite(foreground_scores_2d)
    valid &= torch.isfinite(foreground_scores_2d)
    valid &= foreground_scores_2d >= foreground_score_thr

    output_centers = []
    output_sizes = []
    output_queries = []
    output_scores = []
    output_labels = []
    output_source_indices = []
    output_fallback_masks = []
    for batch_id in range(batch_size):
        candidate_indices = torch.nonzero(
            valid[batch_id], as_tuple=False).flatten()
        candidate_scores = scores[batch_id, candidate_indices].detach()
        if candidate_indices.numel():
            candidate_boxes = _center_size_to_minmax(
                centers[batch_id, candidate_indices].detach(),
                sizes[batch_id, candidate_indices].detach())
            local_keep = aligned_3d_nms(
                candidate_boxes, candidate_scores,
                labels[batch_id, candidate_indices].detach(), nms_iou_thr)
            local_keep = local_keep[candidate_scores[local_keep].argsort(
                descending=True)]
            selected_indices = candidate_indices[local_keep[:num_queries]]
        else:
            local_keep = candidate_indices
            selected_indices = candidate_indices

        real_count = len(selected_indices)
        fallback_count = num_queries - real_count
        selected_centers = centers[batch_id, selected_indices].detach()
        selected_sizes = sizes[batch_id, selected_indices].detach()
        selected_queries = queries[batch_id, selected_indices].detach()
        selected_scores = scores[batch_id, selected_indices].detach()
        selected_labels = labels[batch_id, selected_indices].detach()
        source_indices = selected_indices.detach()
        fallback_mask = torch.zeros(real_count, dtype=torch.bool,
                                    device=centers.device)

        if fallback_count:
            fallback_pool = candidate_indices
            if fallback_pool.numel() == 0:
                fallback_pool = torch.nonzero(
                    geometry_valid[batch_id], as_tuple=False).flatten()
            remaining = fallback_pool[~torch.isin(fallback_pool,
                                                   selected_indices)]
            if remaining.numel() == 0:
                remaining = fallback_pool
            fallback_indices = _weighted_fps(
                reconstruction_points[batch_id, remaining],
                foreground_scores_2d[batch_id, remaining],
                reconstruction_points[batch_id, selected_indices],
                fallback_count)
            fallback_source_indices = remaining[fallback_indices]
            selected_centers = torch.cat(
                [selected_centers,
                 centers[batch_id, fallback_source_indices].detach()], dim=0)
            selected_sizes = torch.cat(
                [selected_sizes,
                 sizes[batch_id, fallback_source_indices].detach()], dim=0)
            selected_queries = torch.cat([
                selected_queries,
                queries[batch_id, fallback_source_indices].detach()
            ], dim=0)
            selected_scores = torch.cat([
                selected_scores,
                scores[batch_id, fallback_source_indices].detach()
            ])
            selected_labels = torch.cat([
                selected_labels,
                labels[batch_id, fallback_source_indices].detach()
            ])
            source_indices = torch.cat([
                source_indices, fallback_source_indices.detach()
            ])
            fallback_mask = torch.cat([fallback_mask, torch.ones(
                fallback_count, dtype=torch.bool, device=centers.device)])
            scene_id = batch_id if scene_ids is None else scene_ids[batch_id]
            logger.warning(
                'Reconstruction proposals required weighted FPS fallback: '
                f'scene={scene_id} valid_candidates={len(candidate_indices)} '
                f'nms_keep={len(local_keep)} '
                f'weighted_fps_fallback={fallback_count} '
                f'random_fallback=0 num_queries={num_queries}')

        output_centers.append(selected_centers)
        output_sizes.append(selected_sizes)
        output_queries.append(selected_queries)
        output_scores.append(selected_scores)
        output_labels.append(selected_labels)
        output_source_indices.append(source_indices)
        output_fallback_masks.append(fallback_mask)

    return {
        'query_xyz': torch.stack(output_centers),
        'query_size': torch.stack(output_sizes),
        'detection_query': torch.stack(output_queries),
        'scores': torch.stack(output_scores),
        'labels': torch.stack(output_labels),
        'source_indices': torch.stack(output_source_indices),
        'candidate_valid_mask': valid,
        'fallback_mask': torch.stack(output_fallback_masks),
    }
