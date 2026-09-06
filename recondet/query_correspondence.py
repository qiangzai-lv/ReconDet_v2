import torch
import torch.nn.functional as F


def assign_points_to_clusters(points, cluster_points):
    """Assign every candidate point to its nearest final cluster center."""
    return torch.cdist(points.float(), cluster_points.float()).argmin(dim=-1)


def select_candidate_indices(scores, threshold, minimum):
    """Select thresholded candidates, falling back to the top minimum."""
    finite_indices = torch.nonzero(
        torch.isfinite(scores), as_tuple=False).flatten()
    if finite_indices.numel() == 0:
        raise ValueError('Scene has no valid reconstruction queries')

    finite_scores = scores[finite_indices]
    indices = finite_indices[finite_scores > threshold]
    if indices.numel() < minimum:
        count = min(minimum, finite_indices.numel())
        indices = finite_indices[finite_scores.topk(count).indices]
    return indices


def select_scene_reconstruction_queries(
        reconstruction_outputs, batch_size, num_views,
        score_threshold=0.1, min_queries=256):
    """Select paired reconstruction outputs over every view of each scene."""
    if batch_size <= 0 or num_views <= 0:
        raise ValueError('batch_size and num_views must be positive')
    if min_queries <= 0:
        raise ValueError('min_queries must be positive')

    required = (
        'reconstruction_query', 'detection_query_2d', 'points_vggt',
        'points_aligned', 'class_scores_2d', 'valid_mask')
    missing = [key for key in required if key not in reconstruction_outputs]
    if missing:
        raise KeyError(f'Missing reconstruction outputs: {missing}')

    expected_views = batch_size * num_views
    query_count = reconstruction_outputs['valid_mask'].shape[-1]
    for key in required:
        value = reconstruction_outputs[key]
        if value.shape[:2] != (expected_views, query_count):
            raise ValueError(
                f'{key} must begin with [B*V, Q], got {tuple(value.shape)}')

    class_scores = reconstruction_outputs['class_scores_2d']
    if class_scores.ndim != 3 or class_scores.shape[-1] == 0:
        raise ValueError('class_scores_2d must have shape [B*V, Q, C]')
    foreground_scores = class_scores.amax(dim=-1)
    valid = reconstruction_outputs['valid_mask'].bool()
    valid &= torch.isfinite(foreground_scores)
    for key in ('reconstruction_query', 'detection_query_2d',
                'points_vggt', 'points_aligned'):
        valid &= torch.isfinite(reconstruction_outputs[key]).all(dim=-1)

    paired_keys = [
        key for key, value in reconstruction_outputs.items()
        if isinstance(value, torch.Tensor)
        and value.shape[:2] == (expected_views, query_count)
    ]
    selected_scenes = []
    for batch_id in range(batch_size):
        scene_scores = foreground_scores.reshape(
            batch_size, num_views * query_count)[batch_id]
        scene_valid = valid.reshape(
            batch_size, num_views * query_count)[batch_id]
        valid_count = int(scene_valid.sum().item())
        if valid_count < min_queries:
            raise ValueError(
                'Scene has fewer valid reconstruction queries than required: '
                f'{valid_count} < {min_queries}')

        selection_scores = scene_scores.masked_fill(~scene_valid, -torch.inf)
        indices = select_candidate_indices(
            selection_scores, score_threshold, min_queries)
        scene = {}
        for key in paired_keys:
            value = reconstruction_outputs[key].reshape(
                batch_size, num_views * query_count,
                *reconstruction_outputs[key].shape[2:])
            scene[key] = value[batch_id, indices]
        scene['foreground_score'] = scene_scores[indices]
        scene['source_view_id'] = torch.div(
            indices, query_count, rounding_mode='floor')
        scene['source_query_id'] = indices.remainder(query_count)
        selected_scenes.append(scene)
    return selected_scenes


def aggregate_cluster_view_references(points, bbox_centers, scores, view_ids,
                                      cluster_points, num_views,
                                      assignments=None):
    """Aggregate candidate bbox centers for each 3D cluster and view."""
    batch_size, num_clusters = cluster_points.shape[:2]
    if assignments is None:
        assignments = assign_points_to_clusters(points, cluster_points)

    group_ids = assignments * num_views + view_ids
    num_groups = num_clusters * num_views
    weights = scores.clamp_min(1e-6)
    weighted_centers = bbox_centers * weights[..., None]
    center_sums = bbox_centers.new_zeros((batch_size, num_groups, 2))
    center_sums.scatter_add_(
        1, group_ids[..., None].expand(-1, -1, 2), weighted_centers)
    weight_sums = scores.new_zeros((batch_size, num_groups))
    weight_sums.scatter_add_(1, group_ids, weights)

    references = center_sums / weight_sums.clamp_min(1e-6)[..., None]
    references = references.reshape(batch_size, num_clusters, num_views, 2)
    view_mask = weight_sums.reshape(
        batch_size, num_clusters, num_views).gt(0)
    return references, view_mask


class SemanticWeightedFPSClustering:

    def __init__(self, num_clusters=256, num_neighbors=4,
                 class_cost_weight=0.5, query_cost_weight=0.25,
                 min_cluster_size=0.05):
        if num_clusters <= 0:
            raise ValueError('num_clusters must be positive')
        if num_neighbors <= 0:
            raise ValueError('num_neighbors must be positive')
        if class_cost_weight < 0 or query_cost_weight < 0:
            raise ValueError('clustering cost weights must be non-negative')
        if min_cluster_size <= 0:
            raise ValueError('min_cluster_size must be positive')
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.class_cost_weight = float(class_cost_weight)
        self.query_cost_weight = float(query_cost_weight)
        self.min_cluster_size = float(min_cluster_size)

    def _weighted_fps(self, points, weights):
        num_candidates = points.shape[0]
        num_centers = min(self.num_clusters, num_candidates)
        centers = points.new_empty(num_centers, dtype=torch.long)
        centers[0] = weights.argmax()
        min_distance = torch.full(
            (num_candidates,), float('inf'), device=points.device,
            dtype=points.dtype)
        for center_id in range(1, num_centers):
            distance = (points - points[centers[center_id - 1]]).square().sum(-1)
            min_distance = torch.minimum(min_distance, distance)
            priority = min_distance * weights
            priority[centers[:center_id]] = -torch.inf
            centers[center_id] = priority.argmax()
        return centers

    @staticmethod
    def _validate_inputs(points, queries, queries_2d, class_scores, scores):
        batch_size, num_candidates, _ = points.shape
        if scores.shape != (batch_size, num_candidates):
            raise ValueError('scores must have shape [B, N]')
        for name, value in (('queries', queries), ('queries_2d', queries_2d),
                            ('class_scores', class_scores)):
            if value.ndim != 3 or value.shape[:2] != (batch_size,
                                                       num_candidates):
                raise ValueError(f'{name} must have shape [B, N, D]')
        if num_candidates == 0:
            raise ValueError('clustering requires at least one candidate')

    @staticmethod
    def _aggregate(values, assignment, weights, num_clusters, fallback):
        feature_dim = values.shape[-1]
        weighted_values = values.float() * weights[:, None]
        sums = values.new_zeros((num_clusters, feature_dim), dtype=torch.float32)
        sums.scatter_add_(
            0, assignment[:, None].expand(-1, feature_dim), weighted_values)
        weight_sums = weights.new_zeros(num_clusters, dtype=torch.float32)
        weight_sums.scatter_add_(0, assignment, weights.float())
        aggregated = sums / weight_sums.clamp_min(1e-6)[:, None]
        return torch.where(
            weight_sums[:, None] > 0, aggregated, fallback.float())

    def _cluster_sizes(self, points, assignment, num_clusters):
        indices = assignment[:, None].expand(-1, 3)
        cluster_min = points.new_full((num_clusters, 3), torch.inf)
        cluster_max = points.new_full((num_clusters, 3), -torch.inf)
        cluster_min.scatter_reduce_(
            0, indices, points, reduce='amin', include_self=True)
        cluster_max.scatter_reduce_(
            0, indices, points, reduce='amax', include_self=True)
        valid = torch.isfinite(cluster_min) & torch.isfinite(cluster_max)
        sizes = (cluster_max - cluster_min).clamp_min(self.min_cluster_size)
        return torch.where(
            valid, sizes, sizes.new_full((), self.min_cluster_size))

    @torch.no_grad()
    def __call__(self, points, queries, queries_2d, class_scores, scores):
        self._validate_inputs(
            points, queries, queries_2d, class_scores, scores)
        batch_size, num_candidates, _ = points.shape
        weights = scores.float().clamp_min(1e-6)
        outputs_points = []
        outputs_sizes = []
        outputs_queries = []
        outputs_semantic_queries = []
        for batch_id in range(batch_size):
            sample_points = points[batch_id].float()
            sample_queries = queries[batch_id].float()
            sample_semantic_queries = queries_2d[batch_id].float()
            sample_queries_2d = F.normalize(
                sample_semantic_queries, dim=-1)
            sample_classes = F.normalize(
                class_scores[batch_id].float().clamp_min(0), dim=-1)
            sample_weights = weights[batch_id]
            init_ids = self._weighted_fps(sample_points, sample_weights)
            centers = sample_points[init_ids].clone()
            seed_queries = sample_queries[init_ids]
            seed_semantic_queries = sample_semantic_queries[init_ids]
            seed_queries_2d = sample_queries_2d[init_ids]
            seed_classes = sample_classes[init_ids]

            distances = torch.cdist(sample_points, centers)
            neighbor_count = min(self.num_neighbors, len(centers))
            neighbor_distances, neighbor_ids = distances.topk(
                neighbor_count, dim=-1, largest=False, sorted=False)
            distance_scale = neighbor_distances.amax(
                dim=-1, keepdim=True).clamp_min(1e-6)
            normalized_distances = neighbor_distances / distance_scale

            local_classes = seed_classes[neighbor_ids]
            class_similarity = (
                sample_classes[:, None] * local_classes).sum(dim=-1)
            local_queries_2d = seed_queries_2d[neighbor_ids]
            query_similarity = (
                sample_queries_2d[:, None] * local_queries_2d).sum(dim=-1)
            assignment_cost = normalized_distances
            assignment_cost = assignment_cost + self.class_cost_weight * (
                1 - class_similarity)
            assignment_cost = assignment_cost + self.query_cost_weight * (
                1 - query_similarity)
            local_assignment = assignment_cost.argmin(dim=-1)
            assignment = neighbor_ids.gather(
                1, local_assignment[:, None]).squeeze(1)

            centers = self._aggregate(
                sample_points, assignment, sample_weights,
                len(centers), centers)
            cluster_sizes = self._cluster_sizes(
                sample_points, assignment, len(centers))
            aggregated_queries = self._aggregate(
                sample_queries, assignment, sample_weights,
                len(centers), seed_queries)
            aggregated_semantic_queries = self._aggregate(
                sample_semantic_queries, assignment, sample_weights,
                len(centers), seed_semantic_queries)
            center_queries = F.layer_norm(
                seed_queries + aggregated_queries,
                (sample_queries.shape[-1],))

            if len(centers) < self.num_clusters:
                pad_count = self.num_clusters - len(centers)
                pad_ids = torch.arange(
                    pad_count, device=points.device) % num_candidates
                centers = torch.cat([centers, sample_points[pad_ids]], dim=0)
                cluster_sizes = torch.cat([
                    cluster_sizes,
                    cluster_sizes.new_full(
                        (pad_count, 3), self.min_cluster_size)
                ], dim=0)
                center_queries = torch.cat(
                    [center_queries, sample_queries[pad_ids]], dim=0)
                aggregated_semantic_queries = torch.cat([
                    aggregated_semantic_queries,
                    sample_semantic_queries[pad_ids]
                ], dim=0)
            outputs_points.append(
                centers[:self.num_clusters].to(dtype=points.dtype))
            outputs_sizes.append(
                cluster_sizes[:self.num_clusters].to(dtype=points.dtype))
            outputs_queries.append(
                center_queries[:self.num_clusters].to(dtype=queries.dtype))
            outputs_semantic_queries.append(
                aggregated_semantic_queries[:self.num_clusters].to(
                    dtype=queries_2d.dtype))

        return (torch.stack(outputs_points), torch.stack(outputs_sizes),
                torch.stack(outputs_queries),
                torch.stack(outputs_semantic_queries))
