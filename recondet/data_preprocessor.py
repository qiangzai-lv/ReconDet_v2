# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
from mmdet.models import DetDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmengine.model import stack_batch
from mmengine.utils import is_seq_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models.data_preprocessors.utils import multiview_img_stack_batch
from mmdet3d.registry import MODELS


@MODELS.register_module()
class VGGTDetDataPreprocessor(DetDataPreprocessor):

    def __init__(self,
                 batch_first: bool = True,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False,
                 batch_augments: Optional[List[dict]] = None) -> None:
        super(VGGTDetDataPreprocessor, self).__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)
        self.batch_first = batch_first

    def forward(self,
                data: Union[dict, List[dict]],
                training: bool = False) -> Union[dict, List[dict]]:

        if isinstance(data, list):
            num_augs = len(data)
            aug_batch_data = []
            for aug_id in range(num_augs):
                single_aug_batch_data = self.simple_process(
                    data[aug_id], training)
                aug_batch_data.append(single_aug_batch_data)
            return aug_batch_data

        else:
            return self.simple_process(data, training)

    def simple_process(self, data: dict, training: bool = False) -> dict:

        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)  # [(240, 320)]

        data = self.collate_data(data)  # after collate. inputs['imgs]=(bs, 40, 3, h, w). bgr2rgb, normalized
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

        if 'imgs' in inputs:
            imgs = inputs['imgs']

            if data_samples is not None:
                batch_input_shape = tuple(imgs[0].size()[-2:])  # (240, 320)
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })  # didn't use this info

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)
                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:  # no?
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['imgs'] = imgs  #

        if 'depth' in inputs.keys():
            batch_inputs['depth'] = inputs['depth']

        keys_to_check = ['intrinsic', 'pose_matrix', 'axis_align_matrix', 'avg_distance']

        for key in keys_to_check:
            if key in inputs.keys():
                batch_inputs[key] = inputs[key]

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def preprocess_img(self, _batch_img: Tensor) -> Tensor:
        if self._channel_conversion:
            _batch_img = _batch_img[[2, 1, 0], ...]

        _batch_img = _batch_img.float()
        if self._enable_normalize:
            if self.mean.shape[0] == 3:
                assert _batch_img.dim() == 3 and _batch_img.shape[0] == 3, (
                    'If the mean has 3 values, the input tensor '
                    'should in shape of (3, H, W), but got the '
                    f'tensor with shape {_batch_img.shape}')
            _batch_img = (_batch_img - self.mean) / self.std
        return _batch_img

    def collate_data(self, data: dict) -> dict:

        data = self.cast_data(data)  # type: ignore put on gpu

        if 'img' in data['inputs']:
            _batch_imgs = data['inputs']['img']
            # Process data with `pseudo_collate`.
            if is_seq_of(_batch_imgs, torch.Tensor):
                batch_imgs = []
                img_dim = _batch_imgs[0].dim()
                for _batch_img in _batch_imgs:
                    if img_dim == 3:  # standard img
                        _batch_img = self.preprocess_img(_batch_img)
                    elif img_dim == 4:
                        _batch_img = [
                            self.preprocess_img(_img) for _img in _batch_img
                        ]  # bgr_2_rgb, normalize.

                        _batch_img = torch.stack(_batch_img, dim=0)

                    batch_imgs.append(_batch_img)

                if img_dim == 3:
                    batch_imgs = stack_batch(batch_imgs, self.pad_size_divisor,
                                             self.pad_value)
                elif img_dim == 4:
                    batch_imgs = multiview_img_stack_batch(
                        batch_imgs, self.pad_size_divisor,
                        self.pad_value)  # stack a batch and pad to make sure size // pad_size_divisor. pad_value = 0

            elif isinstance(_batch_imgs, torch.Tensor):
                assert _batch_imgs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW '
                    'tensor or a list of tensor, but got a tensor with '
                    f'shape: {_batch_imgs.shape}')
                if self._channel_conversion:
                    _batch_imgs = _batch_imgs[:, [2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_imgs = _batch_imgs.float()
                if self._enable_normalize:
                    _batch_imgs = (_batch_imgs - self.mean) / self.std
                h, w = _batch_imgs.shape[2:]
                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                batch_imgs = F.pad(_batch_imgs, (0, pad_w, 0, pad_h),
                                   'constant', self.pad_value)
            else:
                raise TypeError(
                    'Output of `cast_data` should be a list of dict '
                    'or a tuple with inputs and data_samples, but got '
                    f'{type(data)}: {data}')

            data['inputs']['imgs'] = batch_imgs  # (bs, 40, 3, 240, 320). bgr2rgb, normalized
        if 'raydirs' in data['inputs']:
            _batch_dirs = data['inputs']['raydirs']
            batch_dirs = stack_batch(_batch_dirs)
            data['inputs']['raydirs'] = batch_dirs

        data.setdefault('data_samples', None)

        return data

    def _get_pad_shape(self, data: dict) -> List[Tuple[int, int]]:

        _batch_inputs = data['inputs']['img']
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                if ori_input.dim() == 4:
                    # mean multiview input, select one of the
                    # image to calculate the pad shape
                    ori_input = ori_input[0]
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got '
                            f'{type(data)}: {data}')
        return batch_pad_shape
