import torch
from mmdet3d.models.layers import aligned_3d_nms


def _center_size_to_minmax(centers, sizes):
    half_sizes = sizes / 2.0
    return torch.cat([centers - half_sizes, centers + half_sizes], dim=-1)


def select_reconstruction_boxes(
        reconstruction_outputs, batch_size, num_views, num_queries,
        query_xyz_range, fallback_bbox_size, nms_iou_thr,
        fallback_queries, logger, scene_ids=None):
    """Select fixed-count scene proposals from reconstruction AABBs."""
    required = (
        'bbox_centers_aligned', 'bbox_sizes_aligned', 'bbox_scores',
        'bbox_labels', 'fused_detection_query', 'valid_mask')
    missing = [key for key in required if key not in reconstruction_outputs]
    if missing:
        raise KeyError(f'Missing reconstruction outputs: {missing}')
    if min(batch_size, num_views, num_queries) <= 0:
        raise ValueError('batch_size, num_views and num_queries must be positive')
    if len(query_xyz_range) != 6 or len(fallback_bbox_size) != 3:
        raise ValueError('query range and fallback size must have 6 and 3 values')
    if not 0 <= nms_iou_thr <= 1:
        raise ValueError('NMS IoU threshold must be within [0, 1]')

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
    if fallback_queries.ndim != 2 or fallback_queries.shape[0] < num_queries:
        raise ValueError('fallback_queries must contain at least num_queries rows')
    if fallback_queries.shape[1] != query_dims:
        raise ValueError('fallback query dimensions do not match reconstruction')

    centers = reconstruction_outputs['bbox_centers_aligned'].reshape(
        batch_size, num_views * query_count, 3)
    sizes = reconstruction_outputs['bbox_sizes_aligned'].reshape_as(centers)
    scores = reconstruction_outputs['bbox_scores'].reshape(
        batch_size, num_views * query_count)
    labels = reconstruction_outputs['bbox_labels'].reshape_as(scores).long()
    queries = reconstruction_outputs['fused_detection_query'].reshape(
        batch_size, num_views * query_count, query_dims)
    valid = valid_mask.reshape(batch_size, num_views * query_count).bool()

    xyz_range = centers.new_tensor(query_xyz_range)
    xyz_min, xyz_max = xyz_range[:3], xyz_range[3:]
    fallback_size = centers.new_tensor(fallback_bbox_size)
    if (fallback_size <= 0).any() or (fallback_size > xyz_max - xyz_min).any():
        raise ValueError('fallback bbox size must be positive and fit the range')
    fallback_min = xyz_min + fallback_size / 2.0
    fallback_max = xyz_max - fallback_size / 2.0

    valid &= torch.isfinite(centers).all(dim=-1)
    valid &= torch.isfinite(sizes).all(dim=-1)
    valid &= torch.isfinite(scores)
    valid &= torch.isfinite(queries).all(dim=-1)
    valid &= (sizes > 0).all(dim=-1)
    valid &= (centers >= xyz_min).all(dim=-1)
    valid &= (centers <= xyz_max).all(dim=-1)

    output_centers = []
    output_sizes = []
    output_queries = []
    output_scores = []
    output_labels = []
    output_source_indices = []
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

        if fallback_count:
            random_centers = fallback_min + torch.rand(
                fallback_count, 3, device=centers.device,
                dtype=centers.dtype) * (fallback_max - fallback_min)
            selected_centers = torch.cat(
                [selected_centers, random_centers], dim=0)
            selected_sizes = torch.cat([
                selected_sizes,
                fallback_size.expand(fallback_count, -1)
            ], dim=0)
            selected_queries = torch.cat([
                selected_queries, fallback_queries[:fallback_count]
            ], dim=0)
            selected_scores = torch.cat([
                selected_scores, selected_scores.new_zeros(fallback_count)
            ])
            selected_labels = torch.cat([
                selected_labels,
                selected_labels.new_full((fallback_count,), -1)
            ])
            source_indices = torch.cat([
                source_indices,
                source_indices.new_full((fallback_count,), -1)
            ])
            scene_id = batch_id if scene_ids is None else scene_ids[batch_id]
            logger.warning(
                'Reconstruction proposals required random fallback: '
                f'scene={scene_id} valid_candidates={len(candidate_indices)} '
                f'nms_keep={len(local_keep)} random_fallback={fallback_count} '
                f'num_queries={num_queries} range={tuple(query_xyz_range)} '
                f'fallback_size={tuple(fallback_bbox_size)}')

        output_centers.append(selected_centers)
        output_sizes.append(selected_sizes)
        output_queries.append(selected_queries)
        output_scores.append(selected_scores)
        output_labels.append(selected_labels)
        output_source_indices.append(source_indices)

    return {
        'query_xyz': torch.stack(output_centers),
        'query_size': torch.stack(output_sizes),
        'detection_query': torch.stack(output_queries),
        'scores': torch.stack(output_scores),
        'labels': torch.stack(output_labels),
        'source_indices': torch.stack(output_source_indices),
        'candidate_valid_mask': valid,
    }
