from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from recondet.detr3_models.helpers import GenericMLP
from recondet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from recondet.device import autocast, get_device
from recondet.geometry_attention import GeometryAwareDeformableDecoder
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.geometry import unproject_depth_map_to_point_map_torch
from vggt_omega.utils.pose_enc import encoding_to_camera

device = get_device()


class ChannelProjecter(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.GroupNorm(num_groups=1, num_channels=in_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        self.res = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.proj(x) + self.res(x)
        del x
        return res  # [B, D, N, T]


@MODELS.register_module()
class ReconDet(Base3DDetector):
    def __init__(
            self,
            bbox_head: ConfigType,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            decoder_cfg: OptConfigType = None,
            num_queries=128,
            token_dim=1024,
            test_only_last_layer=True,
            position_embedding="fourier",
            if_mix_precision=False,
            if_save_vggt_feature=False,
            use_multi_layers=False,
            if_simpler_project=False,
            if_use_pred_pc_query=False,
            depth_thres=1000,
            vggt_omega_checkpoint=None,
            deformable_num_points=4,
            geometry_source='vggt',
            gt_points_dir=None,
            online_scale_point_stride=4,
            online_scale_max_depth=30.0,
    ):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.vggt_encoder = VGGTOmega()
        self.vggt_encoder.load_state_dict(
            torch.load(vggt_omega_checkpoint, map_location='cpu', weights_only=True)
        )
        self.vggt_encoder.to(device)

        for param in self.vggt_encoder.parameters():
            param.requires_grad = False

        self.vggt_encoder.eval()

        self.decoder = GeometryAwareDeformableDecoder(
            embed_dims=token_dim,
            num_layers=decoder_cfg['dec_nlayers'],
            num_heads=decoder_cfg['dec_nhead'],
            feedforward_channels=decoder_cfg['dec_ffn_dim'],
            num_feature_levels=4 if use_multi_layers else 1,
            num_points=deformable_num_points,
            dropout=decoder_cfg['dec_dropout'])

        if if_simpler_project:
            if use_multi_layers:
                self.proj_feat_dim0 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim1 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim2 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
                self.proj_feat_dim3 = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            else:
                self.proj_feat_dim = nn.Conv2d(
                    in_channels=2048,
                    out_channels=token_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
        else:
            if use_multi_layers:
                self.proj_feat_dim0 = ChannelProjecter(in_channels=2048, out_channels=token_dim)  # for _ in range(4)]
                self.proj_feat_dim1 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
                self.proj_feat_dim2 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
                self.proj_feat_dim3 = ChannelProjecter(in_channels=2048, out_channels=token_dim)
            else:
                self.proj_feat_dim = ChannelProjecter(in_channels=2048, out_channels=token_dim)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_queries = num_queries

        self.test_only_last_layer = test_only_last_layer

        self.if_use_pred_pc_query = if_use_pred_pc_query

        if self.if_use_pred_pc_query:
            self.pos_embedding = PositionEmbeddingCoordsSine(
                d_pos=token_dim, pos_type=position_embedding, normalize=False
            )
            self.query_projection = GenericMLP(
                input_dim=token_dim,
                hidden_dims=[token_dim],
                output_dim=token_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        self.if_mix_precision = if_mix_precision
        self.if_save_vggt_feature = if_save_vggt_feature

        self.use_multi_layers = use_multi_layers
        self.depth_thres = depth_thres
        if geometry_source not in {'vggt', 'gt'}:
            raise ValueError(
                "geometry_source must be either 'vggt' or 'gt'")
        self.geometry_source = geometry_source
        if gt_points_dir is None:
            raise ValueError('GT point-cloud loading requires gt_points_dir')
        if online_scale_point_stride <= 0:
            raise ValueError('online_scale_point_stride must be positive')
        if online_scale_max_depth <= 1e-4:
            raise ValueError('online_scale_max_depth must be greater than 1e-4')
        self.gt_points_dir = Path(gt_points_dir)
        self.online_scale_point_stride = online_scale_point_stride
        self.online_scale_max_depth = online_scale_max_depth
        self._gt_scene_diagonal_cache = {}

    @torch.no_grad()
    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList, mode):

        if self.vggt_encoder.training:
            for param in self.vggt_encoder.parameters():
                param.requires_grad = False

            self.vggt_encoder.eval()

        with torch.no_grad():
            # The data preprocessor converts raw BGR uint8 images to RGB without
            # normalization. VGGT-Omega expects RGB values in [0, 1].
            img = batch_inputs_dict['imgs'].float().div(255.0)
            with autocast(img.device):
                aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(
                    img)
                return aggregated_tokens_list, ps_idx, img

    @torch.no_grad()
    def batch_random_sample(self, points, k=100000, depth_mask=None):
        B, N, _ = points.shape
        device = points.device

        rand_values = torch.rand(B, N, device=device)
        if depth_mask is not None:
            rand_values[depth_mask] = 0

        perm = torch.argsort(rand_values, dim=-1, descending=True)

        indices = perm[:, :k]

        batch_indices = torch.arange(B, device=device)[:, None]

        return points[batch_indices, indices]

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

    @staticmethod
    @torch.no_grad()
    def _align_vggt_reconstruction(point_map, extrinsics, pose_matrix,
                                   axis_align_matrix, scene_scale):
        batch_size = point_map.shape[0]
        pose_matrix = ReconDet._batch_tensor(
            pose_matrix, point_map).reshape(batch_size, 4, 4)
        axis_align_matrix = ReconDet._batch_tensor(
            axis_align_matrix, point_map).reshape(batch_size, 4, 4)
        scene_scale = ReconDet._batch_tensor(
            scene_scale, point_map).reshape(batch_size)

        if extrinsics.shape[-2:] == (3, 4):
            extrinsics_h = extrinsics.new_zeros(
                *extrinsics.shape[:-2], 4, 4)
            extrinsics_h[..., :3, :] = extrinsics
            extrinsics_h[..., 3, 3] = 1
        elif extrinsics.shape[-2:] == (4, 4):
            extrinsics_h = extrinsics
        else:
            raise ValueError(
                'VGGT extrinsics must have shape [B, V, 3, 4] or '
                '[B, V, 4, 4]')
        extrinsics_h = extrinsics_h.float()

        with torch.autocast(device_type=point_map.device.type, enabled=False):
            first_w2c = extrinsics_h[:, 0]
            alignment = torch.bmm(
                axis_align_matrix, torch.bmm(pose_matrix, first_w2c))

            scaled_points = point_map.float() * scene_scale.view(
                batch_size, 1, 1, 1, 1)
            aligned_points = torch.einsum(
                'bij,bvhwj->bvhwi', alignment[:, :3, :3], scaled_points)
            aligned_points = aligned_points + alignment[
                :, None, None, None, :3, 3]

            scale_matrix = torch.eye(
                4, device=point_map.device, dtype=torch.float32).repeat(
                    batch_size, 1, 1)
            scale_matrix[:, :3, :3] *= scene_scale[:, None, None]
            vggt_to_aligned = torch.bmm(alignment, scale_matrix)
            scaled_camera_extrinsics = torch.matmul(
                scale_matrix[:, None], extrinsics_h)
            aligned_extrinsics = torch.matmul(
                scaled_camera_extrinsics,
                torch.linalg.inv(vggt_to_aligned)[:, None])

        return aligned_points, aligned_extrinsics[..., :3, :]

    @staticmethod
    def _robust_scene_diagonal(points):
        lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
        return float(np.linalg.norm(upper - lower))

    def _load_axis_aligned_gt_points(self, metadata):
        lidar_path = Path(metadata['lidar_path'])
        if not lidar_path.is_absolute():
            lidar_path = self.gt_points_dir / lidar_path.name
        point_dim = int(metadata.get('num_pts_feats', 6))
        raw_gt_points = np.fromfile(lidar_path, dtype=np.float32)
        if raw_gt_points.size % point_dim:
            raise ValueError(f'Unexpected GT point-cloud shape in {lidar_path}')
        gt_points = raw_gt_points.reshape(-1, point_dim)[:, :3]
        axis_align_matrix = metadata['axis_align_matrix']
        if isinstance(axis_align_matrix, torch.Tensor):
            axis_align_matrix = axis_align_matrix.cpu().numpy()
        axis_align_matrix = np.asarray(axis_align_matrix, dtype=np.float32)
        if axis_align_matrix.shape != (4, 4):
            raise ValueError('axis_align_matrix must have shape [4, 4]')
        gt_points = (
            gt_points @ axis_align_matrix[:3, :3].T +
            axis_align_matrix[:3, 3])
        return lidar_path, gt_points

    @torch.no_grad()
    def _estimate_online_scene_scale(
            self, point_map, depth_map, batch_data_samples):
        stride = self.online_scale_point_stride
        scales = []
        for batch_index, data_sample in enumerate(batch_data_samples):
            metadata = data_sample.metainfo
            lidar_path = Path(metadata['lidar_path'])
            cache_key = str(lidar_path)
            if cache_key not in self._gt_scene_diagonal_cache:
                _, gt_points = self._load_axis_aligned_gt_points(metadata)
                gt_diagonal = self._robust_scene_diagonal(gt_points)
                self._gt_scene_diagonal_cache[cache_key] = gt_diagonal
            else:
                gt_diagonal = self._gt_scene_diagonal_cache[cache_key]

            sampled_points = point_map[
                batch_index, :, ::stride, ::stride].reshape(-1, 3)
            sampled_depth = depth_map[
                batch_index, :, ::stride, ::stride].reshape(-1)
            valid = torch.isfinite(sampled_points).all(dim=-1)
            valid &= torch.isfinite(sampled_depth)
            valid &= sampled_depth > 1e-4
            valid &= sampled_depth < self.online_scale_max_depth
            sampled_points = sampled_points[valid]
            if len(sampled_points) == 0:
                raise ValueError('VGGT reconstruction has no valid scale points')
            vggt_points = sampled_points.float().cpu().numpy()
            vggt_diagonal = self._robust_scene_diagonal(vggt_points)
            if not np.isfinite(vggt_diagonal) or vggt_diagonal <= 1e-6:
                raise ValueError(
                    f'VGGT point-cloud range is too small: {vggt_diagonal}')
            scene_scale = gt_diagonal / vggt_diagonal
            if not np.isfinite(scene_scale) or scene_scale <= 0:
                raise ValueError(f'Invalid online scene scale: {scene_scale}')
            scales.append(scene_scale)
        return point_map.new_tensor(scales, dtype=torch.float32)

    @torch.no_grad()
    def pred_pc_from_vggt(self, aggregated_tokens_list_ori, ps_idx, images,
                          batch_inputs_dict, batch_data_samples):

        with torch.no_grad():
            with autocast(images.device):
                aggregated_tokens_list = [
                    token.contiguous() if token is not None else None
                    for token in aggregated_tokens_list_ori
                ]

            with autocast(images.device, enabled=False):

                pose_enc = self.vggt_encoder.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=ps_idx,
                )
                # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
                extrinsic, intrinsic = encoding_to_camera(pose_enc, images.shape[-2:])

                depth_map, depth_conf = self.vggt_encoder.dense_head(
                    aggregated_tokens_list,
                    images,
                    patch_token_start=ps_idx,
                )
                del aggregated_tokens_list

                assert depth_map.shape[-1] == 1
                depth_map = depth_map.squeeze(-1)

                point_map_by_unprojection_tensor = \
                    unproject_depth_map_to_point_map_torch(
                        depth_map, extrinsic, intrinsic)
                scene_scale = self._estimate_online_scene_scale(
                    point_map_by_unprojection_tensor, depth_map,
                    batch_data_samples)
                batch_inputs_dict['scene_scale'] = scene_scale.detach()
                point_map_by_unprojection_tensor, aligned_extrinsic = \
                    self._align_vggt_reconstruction(
                        point_map_by_unprojection_tensor,
                        extrinsic,
                        batch_inputs_dict['pose_matrix'],
                        batch_inputs_dict['axis_align_matrix'],
                        scene_scale)
                point_map_by_unprojection_tensor = \
                    point_map_by_unprojection_tensor.reshape(
                        point_map_by_unprojection_tensor.shape[0], -1,
                        point_map_by_unprojection_tensor.shape[-1])
                depth_mask = depth_map > self.depth_thres
                depth_mask = depth_mask.reshape(
                    point_map_by_unprojection_tensor.shape[0], -1)

                del depth_map, depth_conf, pose_enc

                sampled_point_map_by_unprojection_tensor = \
                    self.batch_random_sample(
                        point_map_by_unprojection_tensor, 100000, depth_mask)

                del point_map_by_unprojection_tensor

                return (sampled_point_map_by_unprojection_tensor.detach(),
                        aligned_extrinsic.detach(),
                        intrinsic.detach())

    def _build_patch_feature_maps(self, vggt_token_list, ps_idx,
                                  image_shape):
        cached_tokens = [
            tokens for tokens in vggt_token_list if tokens is not None
        ]
        if self.use_multi_layers:
            if len(cached_tokens) != 4:
                raise ValueError(
                    'VGGT-Omega must provide 4 cached feature layers, '
                    f'got {len(cached_tokens)}')
        else:
            cached_tokens = cached_tokens[-1:]

        patch_size = self.vggt_encoder.aggregator.patch_size
        patch_height = image_shape[0] // patch_size
        patch_width = image_shape[1] // patch_size
        feature_maps = []
        for level, tokens in enumerate(cached_tokens):
            patch_tokens = tokens[:, :, ps_idx:, :].permute(
                0, 3, 1, 2).contiguous()
            projector = getattr(self, f'proj_feat_dim{level}', None)
            if projector is None:
                projector = self.proj_feat_dim
            projected = projector(patch_tokens)
            batch_size, channels, num_views, num_tokens = projected.shape
            if num_tokens != patch_height * patch_width:
                raise ValueError(
                    'VGGT patch tokens do not match the input image grid: '
                    f'{num_tokens} != {patch_height} * {patch_width}')
            feature_maps.append(
                projected.permute(0, 2, 1, 3).reshape(
                    batch_size, num_views, channels,
                    patch_height, patch_width).contiguous())
        return feature_maps

    @torch.no_grad()
    def _get_gt_geometry(self, images, batch_inputs_dict,
                         batch_data_samples):
        query_points = []
        point_mins = []
        point_maxs = []
        for data_sample in batch_data_samples:
            _, gt_points = self._load_axis_aligned_gt_points(
                data_sample.metainfo)
            points = torch.as_tensor(
                gt_points, device=images.device, dtype=torch.float32)
            points = points[torch.isfinite(points).all(dim=-1)]
            if len(points) == 0:
                raise ValueError('Axis-aligned GT point cloud is empty')
            query_points.append(
                self._farthest_point_sample(points, self.num_queries))
            point_mins.append(points.amin(dim=0))
            point_maxs.append(points.amax(dim=0))

        query_xyz = torch.stack(query_points)
        point_min = torch.stack(point_mins)
        point_max = torch.stack(point_maxs)
        extrinsics = self._batch_tensor(
            batch_inputs_dict['gt_camera_extrinsics'], images)
        intrinsics = self._batch_tensor(
            batch_inputs_dict['gt_camera_intrinsics'], images)

        batch_size, num_views = images.shape[:2]
        if extrinsics.shape != (batch_size, num_views, 3, 4):
            raise ValueError(
                'GT camera extrinsics must have shape [B, V, 3, 4], got '
                f'{tuple(extrinsics.shape)}')
        if intrinsics.shape != (batch_size, num_views, 3, 3):
            raise ValueError(
                'GT camera intrinsics must have shape [B, V, 3, 3], got '
                f'{tuple(intrinsics.shape)}')
        return query_xyz, point_min, point_max, extrinsics, intrinsics

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, batch_data_samples):
        if not self.if_use_pred_pc_query:
            raise ValueError(
                'Projected deformable attention requires point queries')

        feature_maps = self._build_patch_feature_maps(
            vggt_token_list, ps_idx, images.shape[-2:])
        if self.geometry_source == 'gt':
            (query_xyz, point_min, point_max,
             camera_extrinsics, camera_intrinsics) = self._get_gt_geometry(
                 images, batch_inputs_dict, batch_data_samples)
        else:
            pred_pc, camera_extrinsics, camera_intrinsics = \
                self.pred_pc_from_vggt(
                    vggt_token_list, ps_idx, images, batch_inputs_dict,
                    batch_data_samples)
            batch_inputs_dict['vggt_extrinsics'] = camera_extrinsics
            batch_inputs_dict['vggt_intrinsics'] = camera_intrinsics
            query_xyz, _ = self.get_query_embeddings(
                pred_pc, point_cloud_dims=None)
            point_min = pred_pc.amin(dim=1)
            point_max = pred_pc.amax(dim=1)

        point_extent = (point_max - point_min).clamp_min(1e-3)
        range_padding = point_extent * 0.05
        reference_min = point_min - range_padding
        reference_max = point_max + range_padding
        reference_points = (
            (query_xyz - reference_min[:, None]) /
            (reference_max - reference_min)[:, None]).clamp(1e-5, 1 - 1e-5)

        batch_inputs_dict['query_xyz'] = query_xyz
        batch_inputs_dict['reference_min'] = reference_min
        batch_inputs_dict['reference_max'] = reference_max
        batch_size, num_queries = query_xyz.shape[:2]
        query = torch.zeros(
            batch_size, num_queries, feature_maps[0].shape[2],
            device=query_xyz.device, dtype=feature_maps[0].dtype)
        return self.decoder(
            query,
            feature_maps,
            reference_points,
            reference_min,
            reference_max,
            camera_extrinsics,
            camera_intrinsics,
            images.shape[-2:],
            self.pos_embedding,
            self.query_projection,
            self.bbox_head.center_heads)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:

        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with autocast(img.device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    batch_data_samples)
        else:
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                batch_data_samples)

        losses = self.bbox_head.loss(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with autocast(img.device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    batch_data_samples)
        else:
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                batch_data_samples)

        layer_ids = list(range(len(box_features)))
        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]]
            layer_ids = [layer_ids[-1]]

        results_list = self.bbox_head.predict(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            layer_ids=layer_ids,
            **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')

        if self.if_mix_precision:
            with autocast(img.device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    batch_data_samples)
        else:
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                batch_data_samples)

        layer_ids = list(range(len(box_features)))
        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]]
            layer_ids = [layer_ids[-1]]

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, refined_query_xyz, layer_ids)
        return results

    @staticmethod
    @torch.no_grad()
    def _farthest_point_sample(points, num_samples):
        if len(points) == 0:
            return points.new_zeros((num_samples, 3))

        sample_count = min(num_samples, len(points))
        indices = torch.empty(
            sample_count, dtype=torch.long, device=points.device)
        min_distances = torch.full(
            (len(points),), float('inf'), device=points.device)
        current = points.square().sum(dim=-1).argmax()
        for sample_index in range(sample_count):
            indices[sample_index] = current
            distances = (points - points[current]).square().sum(dim=-1)
            min_distances = torch.minimum(min_distances, distances)
            current = min_distances.argmax()
        sampled_points = points[indices]
        if sample_count < num_samples:
            sampled_points = torch.cat([
                sampled_points,
                sampled_points[-1:].expand(num_samples - sample_count, -1)
            ])
        return sampled_points

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_xyz = torch.stack([
            self._farthest_point_sample(points, self.num_queries)
            for points in encoder_xyz
        ])
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed
