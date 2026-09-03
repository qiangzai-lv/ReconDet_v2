from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from recondet.detr3_models.helpers import GenericMLP
from recondet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from recondet.device import autocast, get_device, get_amp_dtype
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
            if_use_atten_sample=False,
            atten_sample_ratio=10,
            depth_thres=1000,
            if_use_atten_fps=False,
            lambda_dist=1.0,
            if_task_query=False,
            if_add_noises=False,
            noise_level=None,
            vggt_omega_checkpoint=None,
            deformable_num_points=4,
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

        if if_task_query:
            raise ValueError(
                'task_query has no projectable 3D reference point and is not '
                'supported by projected deformable attention')
        self.if_task_query = False
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
        self.if_use_atten_sample = if_use_atten_sample
        self.atten_sample_ratio = atten_sample_ratio
        self.depth_thres = depth_thres
        self.if_use_atten_fps = if_use_atten_fps
        self.lambda_dist = lambda_dist
        self.if_add_noises = if_add_noises
        self.noise_level = noise_level

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
                if self.if_use_atten_sample or self.if_use_atten_fps:
                    aggregated_tokens_list, ps_idx, images_patch_attn = self.vggt_encoder.aggregator(
                        img,
                        return_patch_attention=True,
                    )
                    return aggregated_tokens_list, ps_idx, img, images_patch_attn
                else:
                    aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(img)
                    return aggregated_tokens_list, ps_idx, img, None

    @torch.no_grad()
    def batch_random_sample(self, points, k=100000, depth_mask=None, weights=None):
        B, N, _ = points.shape
        device = points.device

        rand_values = torch.rand(B, N, device=device)
        if depth_mask is not None:
            rand_values[depth_mask] = 0

        perm = torch.argsort(rand_values, dim=-1, descending=True)

        indices = perm[:, :k]

        batch_indices = torch.arange(B, device=device)[:, None]

        if weights is not None:
            return points[batch_indices, indices], weights[batch_indices, indices]
        else:
            return points[batch_indices, indices]

    @torch.no_grad()
    def pred_pc_from_vggt(self, aggregated_tokens_list_ori, ps_idx, images,
                          images_patch_attn):

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
                predicted_first_w2c = extrinsic[:, 0].detach()

                depth_map, depth_conf = self.vggt_encoder.dense_head(
                    aggregated_tokens_list,
                    images,
                    patch_token_start=ps_idx,
                )
                del aggregated_tokens_list

                assert depth_map.shape[-1] == 1
                depth_map = depth_map.squeeze(-1)

                if self.if_use_atten_sample:
                    images_patch_attn = images_patch_attn.float()

                    point_map_by_unprojection_tensor = unproject_depth_map_to_point_map_torch(depth_map, extrinsic,
                                                                                              intrinsic)
                    point_map_by_unprojection_tensor = point_map_by_unprojection_tensor.reshape(
                        point_map_by_unprojection_tensor.shape[0], point_map_by_unprojection_tensor.shape[1], -1,
                        point_map_by_unprojection_tensor.shape[-1])  # shape:(bs, view_num,  h * w, 3)

                    bs, num_frame, h, w = depth_map.shape

                    # # use depth 
                    depth_mask = depth_map > self.depth_thres  # shape [10, 40, 336, 448]
                    patch_size = self.vggt_encoder.aggregator.patch_size
                    attn_reshape = images_patch_attn.view(bs, num_frame, h // patch_size, w // patch_size)

                    attn_img_up = F.interpolate(attn_reshape,
                                                size=(h, w),  # (H, W)
                                                mode='bicubic',
                                                align_corners=False)

                    attn_img_up = attn_img_up.view(bs * num_frame, -1)  # (bs*num_frame, h*w)

                    min_val = torch.min(attn_img_up, -1, keepdim=True).values  # (bs*num_frame, 1)
                    max_val = torch.max(attn_img_up, -1, keepdim=True).values  # (bs*num_frame, 1)

                    denominator = max_val - min_val
                    norm_attn_img_up = torch.where(  # (bs*num_frame, h*w)
                        denominator != 0,
                        (attn_img_up - min_val) / denominator,
                        torch.tensor(1.0, device=attn_img_up.device)
                    )
                    attn_depth_mask = depth_mask.view(bs * num_frame, -1)
                    norm_attn_img_up[attn_depth_mask] = 0.0  # (bs*num_frame, h*w)
                    prob_dist_pre = norm_attn_img_up
                    num_point = prob_dist_pre.shape[-1]  # (bs*num_frame, h*w)
                    prob_dist = prob_dist_pre / prob_dist_pre.sum(dim=-1, keepdim=True)
                    num_samples = int(num_point / self.atten_sample_ratio)

                    # sampled_indices = torch.multinomial(prob_dist, num_samples, replacement=False)

                    topk_values, sampled_indices = torch.topk(prob_dist, num_samples, dim=1)

                    sampled_indices = sampled_indices.view(bs, num_frame, num_samples)
                    expanded_indices = sampled_indices.unsqueeze(-1).expand(-1, -1, -1, 3)

                    del norm_attn_img_up, attn_img_up, attn_reshape, prob_dist_pre, prob_dist
                    sampled_point_map_by_unprojection_tensor = torch.gather(point_map_by_unprojection_tensor, dim=2,
                                                                            index=expanded_indices)
                    sampled_point_map_by_unprojection_tensor = sampled_point_map_by_unprojection_tensor.reshape(
                        sampled_point_map_by_unprojection_tensor.shape[0], -1,
                        sampled_point_map_by_unprojection_tensor.shape[-1])
                    # print(1)
                    sampled_point_map_by_unprojection_tensor = self.batch_random_sample(
                        sampled_point_map_by_unprojection_tensor, 100000)

                    del depth_map, depth_conf, pose_enc

                elif self.if_use_atten_fps:
                    images_patch_attn = images_patch_attn.float()

                    point_map_by_unprojection_tensor = unproject_depth_map_to_point_map_torch(depth_map, extrinsic,
                                                                                              intrinsic)
                    point_map_by_unprojection_tensor = point_map_by_unprojection_tensor.reshape(
                        point_map_by_unprojection_tensor.shape[0], -1,
                        point_map_by_unprojection_tensor.shape[-1])  # shape:(bs, view_num,  h * w, 3)

                    bs, num_frame, h, w = depth_map.shape

                    # # use depth 
                    depth_mask = depth_map > self.depth_thres  # shape [10, 40, 336, 448]

                    patch_size = self.vggt_encoder.aggregator.patch_size

                    attn_reshape = images_patch_attn.view(bs, num_frame, h // patch_size, w // patch_size)

                    attn_img_up = F.interpolate(attn_reshape,
                                                size=(h, w),  # (H, W)
                                                mode='bicubic',
                                                align_corners=False)

                    attn_img_up = attn_img_up.view(bs, -1)  # (bs, h*w*num_frame)

                    min_val = torch.min(attn_img_up, -1, keepdim=True).values  # (bs*num_frame, 1)
                    max_val = torch.max(attn_img_up, -1, keepdim=True).values  # (bs*num_frame, 1)

                    denominator = max_val - min_val
                    norm_attn_img_up = torch.where(  # (bs*num_frame, h*w)
                        denominator != 0,
                        (attn_img_up - min_val) / denominator,
                        torch.tensor(1.0, device=attn_img_up.device)
                    )
                    attn_depth_mask = depth_mask.view(bs, -1)
                    norm_attn_img_up[attn_depth_mask] = 0.0  # (bs*num_frame, h*w)
                    prob_dist_pre = norm_attn_img_up
                    num_point = prob_dist_pre.shape[-1]  # (bs*num_frame, h*w)
                    prob_dist = prob_dist_pre / prob_dist_pre.sum(dim=-1, keepdim=True)
                    # num_samples = int(num_point / self.atten_sample_ratio)

                    prob_dist = prob_dist.view(bs, -1)

                    del norm_attn_img_up, attn_img_up, attn_reshape, prob_dist_pre

                    sampled_point_map_by_unprojection_tensor, atten_weights = self.batch_random_sample(
                        point_map_by_unprojection_tensor, 100000, weights=prob_dist)

                    del depth_map, depth_conf, pose_enc, prob_dist

                else:
                    point_map_by_unprojection_tensor = unproject_depth_map_to_point_map_torch(depth_map, extrinsic,
                                                                                              intrinsic)
                    point_map_by_unprojection_tensor = point_map_by_unprojection_tensor.reshape(
                        point_map_by_unprojection_tensor.shape[0], -1, point_map_by_unprojection_tensor.shape[-1])
                    depth_mask = depth_map > self.depth_thres
                    depth_mask = depth_mask.reshape(point_map_by_unprojection_tensor.shape[0], -1)

                    del depth_map, depth_conf, pose_enc

                    sampled_point_map_by_unprojection_tensor = self.batch_random_sample(
                        point_map_by_unprojection_tensor, 100000, depth_mask)

                del point_map_by_unprojection_tensor

                if self.if_use_atten_fps:
                    return (sampled_point_map_by_unprojection_tensor.detach(),
                            atten_weights, predicted_first_w2c,
                            extrinsic.detach(), intrinsic.detach())
                else:
                    return (sampled_point_map_by_unprojection_tensor.detach(),
                            None, predicted_first_w2c,
                            extrinsic.detach(), intrinsic.detach())

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

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, images_patch_attn):
        if not self.if_use_pred_pc_query:
            raise ValueError(
                'Projected deformable attention requires VGGT point queries')

        feature_maps = self._build_patch_feature_maps(
            vggt_token_list, ps_idx, images.shape[-2:])
        (pred_pc, atten_weights, predicted_first_w2c,
         vggt_extrinsics, vggt_intrinsics) = self.pred_pc_from_vggt(
             vggt_token_list, ps_idx, images, images_patch_attn)
        batch_inputs_dict['predicted_first_w2c'] = predicted_first_w2c
        batch_inputs_dict['vggt_extrinsics'] = vggt_extrinsics
        batch_inputs_dict['vggt_intrinsics'] = vggt_intrinsics

        if self.if_add_noises:
            pred_pc = self.add_normalized_noise_to_point_cloud(
                pred_pc, self.noise_level)
        if self.if_use_atten_fps:
            query_xyz, _ = self.get_query_embeddings_atten_fps(
                pred_pc, point_cloud_dims=None,
                atten_weights=atten_weights)
        else:
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
            vggt_extrinsics,
            vggt_intrinsics,
            images.shape[-2:],
            self.pos_embedding,
            self.query_projection,
            self.bbox_head.center_heads)

    def add_normalized_noise_to_point_cloud(self, pred_pc, noise_level):

        assert len(pred_pc.shape) == 3 and pred_pc.shape[2] == 3, "the shape of pred_pc should be [1, N, 3]"
        assert 0.0 <= noise_level <= 1.0, "the noise_level must be in the range [0, 1]"

        max_coords = torch.max(pred_pc, dim=1, keepdim=True)[0]  # [1, 1, 3]
        min_coords = torch.min(pred_pc, dim=1, keepdim=True)[0]  # [1, 1, 3]
        range_coords = max_coords - min_coords  # [1, 1, 3]

        actual_std = range_coords * noise_level

        noise = torch.randn_like(pred_pc) * actual_std

        noisy_pc = pred_pc + noise

        return noisy_pc

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:

        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples,
                                                                            'train')

        if self.if_mix_precision:
            with autocast(img.device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    images_patch_attn)
        else:
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                images_patch_attn)

        losses = self.bbox_head.loss(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            **kwargs)
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples,
                                                                            'train')

        if self.if_mix_precision:
            with autocast(img.device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    images_patch_attn)
        else:
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                images_patch_attn)

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
        vggt_token_list, ps_idx, img, images_patch_attn = self.extract_feat(batch_inputs_dict, batch_data_samples,
                                                                            'train')

        if self.if_mix_precision:
            with autocast(img.device):
                box_features, refined_query_xyz = self.get_box_features(
                    vggt_token_list, ps_idx, batch_inputs_dict, img,
                    images_patch_attn)
        else:
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                images_patch_attn)

        layer_ids = list(range(len(box_features)))
        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]]
            layer_ids = [layer_ids[-1]]

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, refined_query_xyz, layer_ids)
        return results

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def get_query_embeddings_atten_fps(self, encoder_xyz, point_cloud_dims, atten_weights=None):
        query_inds = self.attention_guided_prob_fps(encoder_xyz, atten_weights, self.num_queries,
                                                    lambda_dist=self.lambda_dist)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    @torch.no_grad()
    def attention_guided_prob_fps(self,
                                  points: torch.Tensor,
                                  attention_weights: torch.Tensor,
                                  num_samples: int,
                                  lambda_dist: float = 0.1,
                                  chunk_size: int = 16384,
                                  use_amp: bool = True,
                                  verbose: bool = False
                                  ) -> torch.Tensor:

        assert points.dim() == 3, "the shape of points should be [B, N, 3]"
        assert attention_weights.shape == points.shape[:2], "the shape of attention_weights should be [B, N]"

        with autocast(points.device, enabled=use_amp):
            B, N, _ = points.shape
            device = points.device
            batch_idx = torch.arange(B, device=device)[:, None]

            indices = torch.zeros((B, num_samples), dtype=torch.long, device=device)
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

            weights_min = attention_weights.min(1, keepdim=True).values
            weights_max = attention_weights.max(1, keepdim=True).values
            weights_norm = (attention_weights - weights_min) / (weights_max - weights_min + 1e-8)
            amp_dtype = get_amp_dtype(points.device)
            weights_norm = weights_norm.to(amp_dtype)

            first_idx = torch.argmax(weights_norm, dim=1)
            indices[:, 0] = first_idx
            mask[batch_idx, first_idx.unsqueeze(1)] = False

            min_dists = torch.full((B, N), float('inf'), dtype=amp_dtype, device=device)

            for k in range(1, num_samples):
                current_point = points.gather(1, indices[:, k - 1].view(-1, 1, 1).expand(-1, -1, 3))
                for i in range(0, N, chunk_size):
                    chunk = points[:, i:i + chunk_size]
                    dist_chunk = torch.norm(chunk - current_point, dim=-1)
                    min_dists[:, i:i + chunk_size] = torch.min(
                        min_dists[:, i:i + chunk_size],
                        dist_chunk.to(amp_dtype)
                    )

                dist_min = min_dists.min(1, keepdim=True).values
                dist_max = min_dists.max(1, keepdim=True).values
                dists_norm = (min_dists - dist_min) / (dist_max - dist_min + 1e-8)

                priority = weights_norm + lambda_dist * dists_norm
                priority[~mask] = -torch.inf

                next_idx = torch.argmax(priority, dim=1)
                indices[:, k] = next_idx
                mask[batch_idx, next_idx.unsqueeze(1)] = False

                if verbose and k % 10 == 0:
                    backend = getattr(torch, points.device.type, None)
                    mem = backend.memory_allocated() / 1024 ** 3 if backend is not None else 0.0
                    print(f"Step {k}: Mem {mem:.2f}GB | Min Dist {min_dists.min().item():.4f}")

        return indices
