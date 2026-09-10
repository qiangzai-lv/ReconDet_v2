from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.models.layers.transformer.utils import (
    coordinate_to_encoding, inverse_sigmoid)
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList

from recondet.grounding_dino_3d_decoder import (
    GroundingDINO3DDecoder, flatten_feature_maps, recover_feature_maps)
from recondet.scene_query_exchange import SceneQueryExchange


@MODELS.register_module()
class ReconGroundingDINO(GroundingDINO):

    def __init__(self, *args, reconstruction_decoder=None,
                 scene_query_exchange_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruction_decoder = None
        if reconstruction_decoder is not None:
            self.reconstruction_decoder = GroundingDINO3DDecoder(
                num_queries=self.num_queries, **reconstruction_decoder)
        self._active_vggt_feature_maps = None
        self._active_vggt_extrinsics = None
        self._active_vggt_intrinsics = None
        self._active_image_shapes = None
        self._active_vggt_valid_ratios = None
        self._active_num_views = 1
        self._last_reconstruction_hidden_states = None
        self._last_reconstruction_outputs = None
        self.scene_query_exchange = None
        self.scene_query_exchange_single_view_dropout = 0.0
        if scene_query_exchange_cfg is not None:
            cfg = dict(scene_query_exchange_cfg)
            enabled = bool(cfg.pop('enabled', True))
            self.scene_query_exchange_single_view_dropout = float(
                cfg.pop('single_view_dropout', 0.0))
            if not 0 <= self.scene_query_exchange_single_view_dropout <= 1:
                raise ValueError('single_view_dropout must be between 0 and 1')
            if enabled:
                self.scene_query_exchange = nn.ModuleList([
                    SceneQueryExchange(embed_dims=self.embed_dims, **cfg)
                    for _ in range(self.decoder.num_layers)
                ])

    def _exchange_detection_queries(self, query: Tensor, layer_id: int,
                                    bypass: bool = False,
                                    num_views: int = None) -> Tensor:
        """Apply one scene-local block to the trailing detection queries."""
        if self.scene_query_exchange is None or bypass:
            return query
        if query.ndim != 3 or query.shape[1] < self.num_queries:
            raise ValueError('decoder query must have shape [V, Q_total, D]')
        if num_views is None:
            num_views = query.shape[0]
        if num_views <= 0 or query.shape[0] % num_views != 0:
            raise ValueError(
                'flattened decoder batch must be divisible by num_views')
        detection_query = query[:, -self.num_queries:, :]
        batch_size = query.shape[0] // num_views
        exchanged = self.scene_query_exchange[layer_id](
            detection_query.reshape(
                batch_size, num_views, self.num_queries, -1)).reshape_as(
                    detection_query)
        return torch.cat([query[:, :-self.num_queries, :], exchanged], dim=1)

    @contextmanager
    def _using_vggt_features(self, feature_maps, extrinsics=None,
                             intrinsics=None, image_shapes=None,
                             vggt_valid_ratios=None, num_views=1):
        previous = (
            self._active_vggt_feature_maps,
            self._active_vggt_extrinsics,
            self._active_vggt_intrinsics,
            self._active_image_shapes,
            self._active_vggt_valid_ratios,
            self._active_num_views,
        )
        self._active_vggt_feature_maps = feature_maps
        self._active_vggt_extrinsics = extrinsics
        self._active_vggt_intrinsics = intrinsics
        self._active_image_shapes = image_shapes
        self._active_vggt_valid_ratios = vggt_valid_ratios
        self._active_num_views = int(num_views)
        self._last_reconstruction_hidden_states = None
        self._last_reconstruction_outputs = None
        try:
            yield
        finally:
            (self._active_vggt_feature_maps,
             self._active_vggt_extrinsics,
             self._active_vggt_intrinsics,
             self._active_image_shapes,
             self._active_vggt_valid_ratios,
             self._active_num_views) = previous

    def forward_transformer(
            self,
            img_feats: Tuple[Tensor],
            text_dict: Dict,
            batch_data_samples: OptSampleList = None,
            vggt_feature_maps=None,
            vggt_extrinsics=None,
            vggt_intrinsics=None,
            image_shapes=None,
            vggt_valid_ratios=None,
            num_views=None) -> Dict:
        if vggt_feature_maps is None:
            vggt_feature_maps = self._active_vggt_feature_maps
        if vggt_extrinsics is None:
            vggt_extrinsics = self._active_vggt_extrinsics
        if vggt_intrinsics is None:
            vggt_intrinsics = self._active_vggt_intrinsics
        if image_shapes is None:
            image_shapes = self._active_image_shapes
        if vggt_valid_ratios is None:
            vggt_valid_ratios = self._active_vggt_valid_ratios
        if num_views is None:
            num_views = self._active_num_views

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)
        self._last_valid_ratios = decoder_inputs_dict['valid_ratios']

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        self._last_semantic_feature_maps = recover_feature_maps(
            encoder_outputs_dict['memory'],
            encoder_outputs_dict['spatial_shapes'])

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(
            **decoder_inputs_dict,
            vggt_feature_maps=vggt_feature_maps,
            vggt_extrinsics=vggt_extrinsics,
            vggt_intrinsics=vggt_intrinsics,
            image_shapes=image_shapes,
            vggt_valid_ratios=vggt_valid_ratios,
            num_views=num_views)
        if not self.training:
            decoder_outputs_dict.pop('reconstruction_hidden_states', None)
            decoder_outputs_dict.pop('reconstruction_outputs', None)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        memory_text: Tensor = None,
                        text_attention_mask: Tensor = None,
                        vggt_feature_maps=None,
                        vggt_extrinsics=None,
                        vggt_intrinsics=None,
                        image_shapes=None,
                        vggt_valid_ratios=None,
                        num_views=1,
                        **kwargs) -> Dict:
        use_scene_exchange = self.scene_query_exchange is not None
        use_reconstruction = not (
            self.reconstruction_decoder is None or vggt_feature_maps is None)
        if not use_scene_exchange and not use_reconstruction:
            return super().forward_decoder(
                query=query,
                memory=memory,
                memory_mask=memory_mask,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                dn_mask=dn_mask,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                **kwargs)

        spatial_features = (
            flatten_feature_maps(vggt_feature_maps)
            if use_reconstruction else None)
        intermediate = []
        intermediate_reference_points = [reference_points]
        bypass_scene_exchange = (
            self.training
            and torch.rand((), device=query.device)
            < self.scene_query_exchange_single_view_dropout)

        for layer_id, layer in enumerate(self.decoder.layers):
            reference_points_input = reference_points[:, :, None] * torch.cat(
                [valid_ratios, valid_ratios], dim=-1)[:, None]
            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.decoder.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=memory,
                key_padding_mask=memory_mask,
                self_attn_mask=dn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                **kwargs)
            query = self._exchange_detection_queries(
                query, layer_id, bypass=bool(bypass_scene_exchange),
                num_views=num_views)

            bbox_delta = self.bbox_head.reg_branches[layer_id](query)
            new_reference_points = (
                bbox_delta + inverse_sigmoid(
                    reference_points, eps=1e-3)).sigmoid()
            reference_points = new_reference_points.detach()

            if self.decoder.return_intermediate:
                intermediate.append(self.decoder.norm(query))
                intermediate_reference_points.append(new_reference_points)

        if self.decoder.return_intermediate:
            inter_states = torch.stack(intermediate)
            references = torch.stack(intermediate_reference_points)
        else:
            inter_states = query
            references = reference_points

        if (query.shape[1] == self.num_queries and
                self.dn_query_generator is not None):
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0)

        if not use_reconstruction:
            # Store final decoder queries for embedding extraction
            if not self.training and hasattr(self, '_capture_queries'):
                # Extract final layer detection queries
                # inter_states: [L, B*V, Q, 256] or [B*V, Q, 256]
                if inter_states.ndim == 4:
                    final_queries = inter_states[-1]  # [B*V, Q, 256]
                else:
                    final_queries = inter_states  # [B*V, Q, 256]

                # Store queries - they will be reshaped in predict()
                self._last_decoder_queries = final_queries

            return {
                'hidden_states': inter_states,
                'references': list(references),
            }
        if (vggt_extrinsics is None or vggt_intrinsics is None or
                image_shapes is None):
            raise RuntimeError(
                'Query depth reconstruction requires raw VGGT cameras and '
                'per-view image shapes')
        semantic_query = self.decoder.norm(query)[
            :, -self.num_queries:, :]
        matching_reference_points = reference_points[
            :, -self.num_queries:, :2]
        with torch.no_grad():
            instance_embeddings = self.bbox_head.instance_projection(
                semantic_query).detach()
        reconstruction_hidden_states = self.reconstruction_decoder(
            semantic_query,
            spatial_features,
            matching_reference_points,
            (vggt_valid_ratios
             if vggt_valid_ratios is not None else valid_ratios),
            instance_embeddings=instance_embeddings,
            num_views=num_views)
        reconstruction_outputs = self.bbox_head.predict_reconstruction(
            reconstruction_hidden_states,
            matching_reference_points,
            vggt_extrinsics,
            vggt_intrinsics,
            image_shapes,
            instance_embeddings=instance_embeddings)
        reconstruction_outputs['detection_query_2d'] = semantic_query
        self._last_reconstruction_hidden_states = (
            reconstruction_hidden_states)
        self._last_reconstruction_outputs = reconstruction_outputs
        return {
            'hidden_states': inter_states,
            'references': list(references),
            'reconstruction_hidden_states': reconstruction_hidden_states,
            'reconstruction_outputs': reconstruction_outputs,
        }

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList, vggt_feature_maps=None,
             vggt_extrinsics=None, vggt_intrinsics=None,
             image_shapes=None,
             vggt_valid_ratios=None,
             num_views=1,
             return_reconstruction=False) -> Union[dict, list]:
        with self._using_vggt_features(
                vggt_feature_maps, vggt_extrinsics, vggt_intrinsics,
                image_shapes, vggt_valid_ratios=vggt_valid_ratios,
                num_views=num_views):
            losses = super().loss(batch_inputs, batch_data_samples)
        if not return_reconstruction:
            return losses
        if self._last_reconstruction_hidden_states is None:
            raise RuntimeError(
                'Reconstruction output requires VGGT feature maps')
        return (
            losses,
            self._last_reconstruction_hidden_states,
            self._last_reconstruction_outputs,
        )

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True,
                vggt_feature_maps=None, vggt_extrinsics=None,
                vggt_intrinsics=None, image_shapes=None,
                vggt_valid_ratios=None, num_views=1):
        with self._using_vggt_features(
                vggt_feature_maps, vggt_extrinsics, vggt_intrinsics,
                image_shapes, vggt_valid_ratios=vggt_valid_ratios,
                num_views=num_views):
            return super().predict(
                batch_inputs, batch_data_samples, rescale=rescale)
