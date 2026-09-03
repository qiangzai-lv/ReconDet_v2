from typing import List, Tuple, Union

import torch
from recondet.feature_projection import VGGTFeatureProjector

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from recondet.detr3_models.helpers import GenericMLP
from recondet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from recondet.device import autocast, get_device
from recondet.geometry_attention import GeometryAwareDeformableDecoder
from vggt_omega.models import VGGTOmega

device = get_device()


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
            vggt_omega_checkpoint=None,
            deformable_num_points=4,
            query_xyz_range=(-6.5, -9.0, -1.0, 6.5, 9.0, 4.5),
    ):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.vggt_encoder = VGGTOmega()
        self.vggt_encoder.load_state_dict(
            torch.load(vggt_omega_checkpoint, map_location='cpu', weights_only=True)
        )
        self.vggt_encoder.camera_head = None
        dense_head = self.vggt_encoder.dense_head
        self.vggt_encoder.dense_head = None
        self.vggt_encoder.to(device)

        for param in self.vggt_encoder.parameters():
            param.requires_grad = False

        self.vggt_encoder.eval()

        self.decoder = GeometryAwareDeformableDecoder(
            embed_dims=token_dim,
            num_layers=decoder_cfg['dec_nlayers'],
            num_heads=decoder_cfg['dec_nhead'],
            feedforward_channels=decoder_cfg['dec_ffn_dim'],
            num_feature_levels=4,
            num_points=deformable_num_points,
            dropout=decoder_cfg['dec_dropout'])

        self.feature_projector = VGGTFeatureProjector(
            dense_head=dense_head,
            dim_in=dense_head.norm.normalized_shape[0],
            out_channels=[token_dim] * len(
                dense_head.intermediate_layer_idx))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_queries = num_queries

        self.test_only_last_layer = test_only_last_layer

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

        query_xyz_range = torch.as_tensor(query_xyz_range, dtype=torch.float32)
        query_xyz_range = query_xyz_range.reshape(2, 3)
        self.register_buffer(
            'query_xyz_range', query_xyz_range, persistent=False)

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

    def _get_projection_cameras(self, batch_data_samples, images):
        extrinsics = []
        intrinsics = []
        image_height, image_width = images.shape[-2:]
        num_views = images.shape[1]
        for data_sample in batch_data_samples:
            metadata = data_sample.metainfo
            camera_info = metadata['lidar2img']
            sample_extrinsics = torch.as_tensor(
                camera_info['extrinsic'], device=images.device,
                dtype=torch.float32)
            sample_intrinsic = torch.as_tensor(
                camera_info['intrinsic'], device=images.device,
                dtype=torch.float32)[:3, :3].clone()
            ori_height, ori_width = metadata['ori_shape'][:2]
            image_scale = min(
                image_height / ori_height, image_width / ori_width)
            sample_intrinsic[:2] *= image_scale

            extrinsics.append(sample_extrinsics[:, :3])
            intrinsics.append(sample_intrinsic.expand(num_views, -1, -1))
        return torch.stack(extrinsics), torch.stack(intrinsics)

    def _set_identity_bbox_transform(self, batch_inputs_dict, reference):
        pose_matrix = self._batch_tensor(
            batch_inputs_dict['pose_matrix'], reference)
        axis_align_matrix = self._batch_tensor(
            batch_inputs_dict['axis_align_matrix'], reference)
        batch_inputs_dict['predicted_first_w2c'] = torch.bmm(
            torch.linalg.inv(pose_matrix),
            torch.linalg.inv(axis_align_matrix))
        batch_inputs_dict['scene_scale'] = reference.new_ones(
            reference.shape[0])

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, batch_data_samples):
        feature_maps = self.feature_projector(
            vggt_token_list,
            images,
            ps_idx)
        vggt_extrinsics, vggt_intrinsics = self._get_projection_cameras(
            batch_data_samples, images)
        batch_inputs_dict['vggt_extrinsics'] = vggt_extrinsics
        batch_inputs_dict['vggt_intrinsics'] = vggt_intrinsics

        batch_size = images.shape[0]
        reference_min = self.query_xyz_range[0].to(images).expand(
            batch_size, -1)
        reference_max = self.query_xyz_range[1].to(images).expand(
            batch_size, -1)
        point_cloud_dims = (reference_min, reference_max)
        query_xyz, _ = self.get_query_embeddings(
            reference_min[:, None], point_cloud_dims)
        reference_points = (
            (query_xyz - reference_min[:, None]) /
            (reference_max - reference_min)[:, None]).clamp(1e-5, 1 - 1e-5)

        batch_inputs_dict['query_xyz'] = query_xyz
        batch_inputs_dict['reference_min'] = reference_min
        batch_inputs_dict['reference_max'] = reference_max
        self._set_identity_bbox_transform(batch_inputs_dict, query_xyz)
        query = torch.zeros(
            batch_size, self.num_queries, feature_maps[0].shape[2],
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

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        reference_min, reference_max = point_cloud_dims
        batch_size = encoder_xyz.shape[0]
        reference_points = torch.rand(
            batch_size, self.num_queries, 3,
            device=encoder_xyz.device,
            dtype=encoder_xyz.dtype).clamp(1e-5, 1 - 1e-5)
        query_xyz = reference_min[:, None] + reference_points * (
            reference_max - reference_min)[:, None]
        pos_embed = self.pos_embedding(
            query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed
