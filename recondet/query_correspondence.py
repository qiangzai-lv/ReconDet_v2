import logging

import torch
import torch.nn.functional as F


def _scatter_weighted_mean(values, assignments, weights, num_clusters):
    feature_dims = values.shape[1:]
    flat_values = values.float().reshape(values.shape[0], -1)
    weighted = flat_values * weights.float()[:, None]
    sums = flat_values.new_zeros(num_clusters, flat_values.shape[-1])
    sums.scatter_add_(
        0, assignments[:, None].expand_as(weighted), weighted)
    weight_sums = weights.float().new_zeros(num_clusters)
    weight_sums.scatter_add_(0, assignments, weights.float())
    means = sums / weight_sums.clamp_min(1e-6)[:, None]
    return means.reshape(num_clusters, *feature_dims), weight_sums


class IterativeInstanceClustering:
    """Cluster scene reconstruction points using geometry and 2D identity."""

    def __init__(self, num_clusters=256, foreground_score_thr=0.1,
                 num_iterations=4, fps_score_weight=0.5,
                 spatial_cost_weight=1.0, embedding_cost_weight=1.0,
                 assignment_chunk_size=4096, logger=None):
        if num_clusters <= 0 or num_iterations <= 0:
            raise ValueError('num_clusters and num_iterations must be positive')
        if not 0 <= foreground_score_thr <= 1:
            raise ValueError('foreground_score_thr must be within [0, 1]')
        if min(fps_score_weight, spatial_cost_weight,
               embedding_cost_weight) < 0:
            raise ValueError('clustering cost weights must be non-negative')
        if assignment_chunk_size <= 0:
            raise ValueError('assignment_chunk_size must be positive')
        self.num_clusters = int(num_clusters)
        self.foreground_score_thr = float(foreground_score_thr)
        self.num_iterations = int(num_iterations)
        self.fps_score_weight = float(fps_score_weight)
        self.spatial_cost_weight = float(spatial_cost_weight)
        self.embedding_cost_weight = float(embedding_cost_weight)
        self.assignment_chunk_size = int(assignment_chunk_size)
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _scene_scale(points):
        extent = points.amax(dim=0) - points.amin(dim=0)
        return extent.square().sum().sqrt().clamp_min(1e-6)

    def _weighted_fps(self, points, scores, scene_scale):
        count = points.shape[0]
        if count < self.num_clusters:
            raise ValueError(
                f'Weighted FPS requires at least {self.num_clusters} '
                f'candidates, got {count}')
        selected = torch.empty(
            self.num_clusters, dtype=torch.long, device=points.device)
        selected[0] = scores.argmax()
        available = torch.ones(count, dtype=torch.bool, device=points.device)
        available[selected[0]] = False
        min_distance = (points - points[selected[0]]).square().sum(dim=-1)
        squared_scale = scene_scale.square().clamp_min(1e-6)
        for index in range(1, self.num_clusters):
            spatial_priority = min_distance / squared_scale
            priority = spatial_priority + self.fps_score_weight * scores
            priority = priority.masked_fill(~available, -torch.inf)
            selected[index] = priority.argmax()
            available[selected[index]] = False
            distance = (points - points[selected[index]]).square().sum(dim=-1)
            min_distance = torch.minimum(min_distance, distance)
        return selected

    def _assign(self, points, embeddings, centers, prototypes, scene_scale):
        assignments = []
        assigned_costs = []
        for start in range(0, len(points), self.assignment_chunk_size):
            stop = min(start + self.assignment_chunk_size, len(points))
            spatial = torch.cdist(points[start:stop], centers) / scene_scale
            similarity = embeddings[start:stop] @ prototypes.t()
            cost = (self.spatial_cost_weight * spatial
                    + self.embedding_cost_weight * (1.0 - similarity))
            values, indices = cost.min(dim=-1)
            assignments.append(indices)
            assigned_costs.append(values)
        return torch.cat(assignments), torch.cat(assigned_costs)

    def _repair_empty_clusters(self, assignments, assigned_costs):
        counts = torch.bincount(assignments, minlength=self.num_clusters)
        empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
        if empty.numel() == 0:
            return assignments, 0
        order = assigned_costs.argsort(descending=True)
        repaired = assignments.clone()
        cursor = 0
        for cluster_id in empty:
            while cursor < len(order):
                point_id = order[cursor]
                cursor += 1
                donor = repaired[point_id]
                if counts[donor] > 1:
                    counts[donor] -= 1
                    repaired[point_id] = cluster_id
                    counts[cluster_id] = 1
                    break
            else:
                raise RuntimeError('Unable to deterministically repair empty cluster')
        return repaired, int(empty.numel())

    @torch.no_grad()
    def _cluster_assignments(self, points, embeddings, scores):
        points = points.float()
        embeddings = F.normalize(embeddings.float(), dim=-1)
        scores = scores.float().clamp_min(1e-6)
        scene_scale = self._scene_scale(points)
        seed_ids = self._weighted_fps(points, scores, scene_scale)
        centers = points[seed_ids].clone()
        prototypes = embeddings[seed_ids].clone()
        reset_count = 0
        iterations = 0
        for iteration in range(self.num_iterations):
            assignments, costs = self._assign(
                points, embeddings, centers, prototypes, scene_scale)
            assignments, resets = self._repair_empty_clusters(
                assignments, costs)
            reset_count += resets
            iterations = iteration + 1
            centers, _ = _scatter_weighted_mean(
                points, assignments, scores, self.num_clusters)
            embedding_means, _ = _scatter_weighted_mean(
                embeddings, assignments, scores, self.num_clusters)
            prototypes = F.normalize(embedding_means, dim=-1)
        return assignments, seed_ids, iterations, reset_count

    def __call__(self, reconstruction_outputs, batch_size, num_views):
        required = (
            'points_aligned', 'reconstruction_query', 'detection_query_2d',
            'instance_embeddings_2d', 'class_scores_2d', 'valid_mask')
        missing = [key for key in required if key not in reconstruction_outputs]
        if missing:
            raise KeyError(f'Missing reconstruction outputs: {missing}')
        if batch_size <= 0 or num_views <= 0:
            raise ValueError('batch_size and num_views must be positive')

        expected_views = batch_size * num_views
        valid_mask = reconstruction_outputs['valid_mask']
        if valid_mask.ndim != 2 or valid_mask.shape[0] != expected_views:
            raise ValueError('valid_mask must have shape [B * V, Q]')
        query_count = valid_mask.shape[1]
        for key in required:
            if reconstruction_outputs[key].shape[:2] != (
                    expected_views, query_count):
                raise ValueError(f'{key} must begin with [B * V, Q]')

        scene_values = {}
        for key in required:
            value = reconstruction_outputs[key]
            scene_values[key] = value.reshape(
                batch_size, num_views * query_count, *value.shape[2:])
        foreground_scores = scene_values['class_scores_2d'].amax(dim=-1)

        output_centers = []
        output_queries_3d = []
        output_queries_2d = []
        output_scores = []
        diagnostics = []
        for batch_id in range(batch_size):
            valid = scene_values['valid_mask'][batch_id].bool().clone()
            valid &= torch.isfinite(foreground_scores[batch_id])
            for key in ('points_aligned', 'reconstruction_query',
                        'detection_query_2d', 'instance_embeddings_2d'):
                valid &= torch.isfinite(
                    scene_values[key][batch_id]).all(dim=-1)
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            if valid_indices.numel() < self.num_clusters:
                raise ValueError(
                    'Scene has fewer valid reconstruction points than '
                    f'clusters: scene={batch_id} valid={len(valid_indices)} '
                    f'clusters={self.num_clusters}')
            threshold_mask = (
                foreground_scores[batch_id, valid_indices]
                >= self.foreground_score_thr)
            threshold_indices = valid_indices[threshold_mask]
            restored = threshold_indices.numel() < self.num_clusters
            candidate_indices = valid_indices if restored else threshold_indices
            if restored:
                self.logger.warning(
                    'Foreground threshold produced too few reconstruction '
                    f'points; restored all valid points: scene={batch_id} '
                    f'threshold={len(threshold_indices)} '
                    f'valid={len(valid_indices)} clusters={self.num_clusters}')

            points = scene_values['points_aligned'][
                batch_id, candidate_indices]
            queries_3d = scene_values['reconstruction_query'][
                batch_id, candidate_indices]
            queries_2d = scene_values['detection_query_2d'][
                batch_id, candidate_indices]
            embeddings = scene_values['instance_embeddings_2d'][
                batch_id, candidate_indices]
            scores = foreground_scores[batch_id, candidate_indices]
            assignments, seed_ids, iterations, reset_count = (
                self._cluster_assignments(points, embeddings, scores))
            weights = scores.detach().float().clamp_min(1e-6)
            centers, cluster_weights = _scatter_weighted_mean(
                points, assignments, weights, self.num_clusters)
            cluster_queries_3d, _ = _scatter_weighted_mean(
                queries_3d, assignments, weights, self.num_clusters)
            cluster_queries_2d, _ = _scatter_weighted_mean(
                queries_2d, assignments, weights, self.num_clusters)

            output_centers.append(centers.to(points.dtype))
            output_queries_3d.append(cluster_queries_3d.to(queries_3d.dtype))
            output_queries_2d.append(cluster_queries_2d.to(queries_2d.dtype))
            output_scores.append(cluster_weights)
            diagnostics.append({
                'valid_count': int(valid_indices.numel()),
                'threshold_candidate_count': int(threshold_indices.numel()),
                'candidate_count': int(candidate_indices.numel()),
                'restored_all_valid': restored,
                'iterations': iterations,
                'empty_cluster_resets': reset_count,
                'assignments': assignments.detach(),
                'candidate_indices': candidate_indices.detach(),
                'seed_indices': candidate_indices[seed_ids].detach(),
                'cluster_member_counts': torch.bincount(
                    assignments, minlength=self.num_clusters).detach(),
            })

        return {
            'cluster_centers': torch.stack(output_centers),
            'cluster_queries_3d': torch.stack(output_queries_3d),
            'cluster_queries_2d': torch.stack(output_queries_2d),
            'cluster_weight_sums': torch.stack(output_scores),
            'diagnostics': diagnostics,
        }
