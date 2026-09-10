# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
import json
import warnings
from os import path as osp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import DepthInstance3DBoxes


def _normalise_image_key(value: Union[str, Path], data_root: Optional[Path] = None) -> str:
    path = Path(str(value))
    if path.is_absolute() and data_root is not None:
        try:
            path = path.relative_to(data_root)
        except ValueError:
            pass
    return path.as_posix().lstrip('./')


@dataclass(frozen=True)
class ScanNet2DAnnotationIndex:
    by_image_key: Dict[Tuple[str, str], list]
    images_by_view: Dict[Tuple[str, int], dict]
    annotations_by_image_id: Dict[int, list]
    category_id_to_name: Dict[int, str]
    images_by_path: Dict[Tuple[str, str], dict]


def load_scannet_2d_annotation_index(
        ann_file: Union[str, Path],
        data_root: Optional[Union[str, Path]] = None
        ) -> ScanNet2DAnnotationIndex:
    """Load generated COCO boxes and index them by scene and image path."""
    root = Path(data_root).resolve() if data_root is not None else None
    with Path(ann_file).open('r', encoding='utf-8') as handle:
        payload = json.load(handle)

    by_image_key: Dict[Tuple[str, str], list] = {}
    images_by_view: Dict[Tuple[str, int], dict] = {}
    image_records = {}
    category_id_to_name = {
        int(category['id']): str(category['name'])
        for category in payload.get('categories', [])
        if 'id' in category and 'name' in category
    }
    for image in payload.get('images', []):
        if 'id' not in image or 'file_name' not in image:
            raise ValueError('2D image records require id and file_name')
        scene_id = str(image.get(
            'scene_id', Path(image['file_name']).parent.name))
        image_id = int(image['id'])
        record = dict(image)
        record['scene_id'] = scene_id
        record['image_id'] = image_id
        record['file_name'] = _normalise_image_key(
            image['file_name'], root)
        image_records[image_id] = record
        key = (scene_id, record['file_name'])
        if key in by_image_key:
            raise ValueError(f'duplicate 2D image key: {key}')
        by_image_key[key] = []
        if 'view_index' in image:
            view_key = (scene_id, int(image['view_index']))
            if view_key in images_by_view:
                raise ValueError(f'duplicate 2D view key: {view_key}')
            images_by_view[view_key] = record

    annotations_by_image_id: Dict[int, list] = {
        image_id: [] for image_id in image_records}
    seen_instance_keys = set()
    for annotation in payload.get('annotations', []):
        if 'image_id' not in annotation or 'instance_id_3d' not in annotation:
            raise ValueError(
                '2D annotations require image_id and instance_id_3d')
        image_id = int(annotation['image_id'])
        if image_id not in image_records:
            raise ValueError(f'annotation references unknown image_id {image_id}')
        instance_id = int(annotation['instance_id_3d'])
        duplicate_key = (image_id, instance_id)
        if duplicate_key in seen_instance_keys:
            raise ValueError(
                f'duplicate 2D annotation for image/instance {duplicate_key}')
        seen_instance_keys.add(duplicate_key)
        if 'bbox' not in annotation or len(annotation['bbox']) != 4:
            raise ValueError(f'invalid bbox for annotation {annotation.get("id")}')
        record = dict(annotation)
        record['image_id'] = image_id
        record['instance_id_3d'] = instance_id
        record['image_width'] = image_records[image_id].get('width')
        record['image_height'] = image_records[image_id].get('height')
        annotations_by_image_id[image_id].append(record)

    for image_id, records in annotations_by_image_id.items():
        image = image_records[image_id]
        by_image_key[(image['scene_id'], image['file_name'])].extend(records)
    return ScanNet2DAnnotationIndex(
        by_image_key, images_by_view, annotations_by_image_id,
        category_id_to_name,
        {(r['scene_id'], r['file_name']): r for r in image_records.values()})


def build_view_2d_instances(
        records: list,
        scale_factor: Tuple[float, float],
        img_shape: Tuple[int, int],
        min_bbox_wh: Tuple[float, float] = (1e-2, 1e-2),
        category_id_to_label: Optional[Dict[int, int]] = None) -> dict:
    """Apply GroundingDINO's absolute xyxy resize/filter contract."""
    height, width = map(int, img_shape)
    x_scale, y_scale = map(float, scale_factor)
    if height <= 0 or width <= 0 or x_scale <= 0 or y_scale <= 0:
        raise ValueError('invalid image shape or scale factor for 2D boxes')
    boxes = []
    labels = []
    instance_ids = []
    centers = []
    depths = []
    visible_ratios = []
    for record in records:
        x, y, box_width, box_height = map(float, record['bbox'])
        if box_width < 1.0 or box_height < 1.0:
            continue
        if float(record.get('area', box_width * box_height)) <= 0:
            continue
        image_width = record.get('image_width')
        image_height = record.get('image_height')
        if image_width is not None and image_height is not None:
            inter_width = max(
                0.0, min(x + box_width, float(image_width)) - max(x, 0.0))
            inter_height = max(
                0.0, min(y + box_height, float(image_height)) - max(y, 0.0))
            if inter_width * inter_height == 0:
                continue
        box = np.asarray(
            [x, y, x + box_width, y + box_height], dtype=np.float32)
        box *= np.asarray([x_scale, y_scale, x_scale, y_scale], dtype=np.float32)
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(width))
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(height))
        if (box[2] - box[0] <= min_bbox_wh[0]
                or box[3] - box[1] <= min_bbox_wh[1]):
            continue
        category_id = int(record['category_id'])
        if 'bbox_label' in record:
            label = int(record['bbox_label'])
        elif category_id_to_label is None:
            label = category_id - 1
        else:
            if category_id not in category_id_to_label:
                raise ValueError(f'unknown 2D category_id {category_id}')
            label = category_id_to_label[category_id]
        center = record.get('center_3d')
        if center is None:
            center = [np.nan, np.nan, np.nan]
        if len(center) != 3:
            raise ValueError(
                f'center_3d must have three values for instance '
                f'{record["instance_id_3d"]}')
        boxes.append(box)
        labels.append(label)
        instance_ids.append(int(record['instance_id_3d']))
        centers.append(center)
        center_depth = record.get('center_depth')
        visible_ratio = record.get('visible_point_ratio')
        depths.append(float(
            np.nan if center_depth is None else center_depth))
        visible_ratios.append(float(
            np.nan if visible_ratio is None else visible_ratio))
    return {
        'bboxes': np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        'labels': np.asarray(labels, dtype=np.int64),
        'instance_ids_3d': np.asarray(instance_ids, dtype=np.int64),
        'centers_3d': np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        'center_depth': np.asarray(depths, dtype=np.float32),
        'visible_ratios': np.asarray(visible_ratios, dtype=np.float32),
    }


@DATASETS.register_module()
class MultiViewScanNetDataset(Det3DDataset):

    METAINFO = {
        'classes':
        ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
         'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator',
         'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin')
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 ann_file_2d: Optional[str] = None,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 **kwargs) -> None:

        self.remove_dontcare = remove_dontcare
        self._ann_file_2d = ann_file_2d
        if ann_file_2d is None:
            self._2d_annotation_index = None
        else:
            annotation_path = Path(ann_file_2d)
            if not annotation_path.is_absolute():
                annotation_path = Path(data_root) / annotation_path
            self._2d_annotation_index = load_scannet_2d_annotation_index(
                annotation_path, data_root)

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert 'use_camera' in self.modality and \
               'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:

        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def parse_data_info(self, info: dict) -> dict:

        scene_id = str(info.get(
            'scene_id', Path(info['img_paths'][0]).parent.name))
        info['scene_id'] = scene_id
        info['lidar_path'] = str(
            Path(self.data_root) / 'points' / f'{scene_id}.bin')
        info['num_pts_feats'] = int(info.get('num_pts_feats', 6))

        if self.modality['use_depth']:
            info['depth_info'] = []
        if self.modality['use_neuralrecon_depth']:
            info['depth_info'] = []

        if self.modality['use_lidar']:
            # implement lidar processing in the future
            raise NotImplementedError(
                'Please modified '
                '`MultiViewPipeline` to support lidar processing')

        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        if self._2d_annotation_index is not None:
            info['ann_info_2d'] = self._parse_2d_annotations(info)
            image_ids_2d = []
            for view_index in range(len(info['img_paths'])):
                image = self._2d_annotation_index.images_by_path.get(
                    (scene_id, _normalise_image_key(
                        info['img_paths'][view_index],
                        Path(self.data_root).resolve())))
                if image is None:
                    if self.test_mode:
                        # Test mode: use placeholder image_id for missing views
                        # These views will be skipped in metric computation
                        image_ids_2d.append(-1)
                    else:
                        raise ValueError(
                            '2D annotation image is missing for '
                            f'{scene_id} view {view_index}')
                else:
                    image_ids_2d.append(int(image['image_id']))
            info['image_ids_2d'] = image_ids_2d
        info['img_info'] = []
        info['lidar2img'] = []
        info['c2w'] = []
        info['camrotc2w'] = []
        info['lightpos'] = []
        # load img and depth_img
        for i in range(len(info['img_paths'])):
            img_filename = osp.join(self.data_root, info['img_paths'][i])

            info['img_info'].append(dict(filename=img_filename))
            if 'depth_info' in info.keys():
                if self.modality['use_neuralrecon_depth']:
                    info['depth_info'].append(
                        dict(filename=img_filename[:-4] + '.npy'))
                else:
                    image_path = Path(img_filename)
                    info['depth_info'].append(
                        dict(filename=str(
                            image_path.parent / 'depth' /
                            f'{image_path.stem}.png')))
            # implement lidar_info in input.keys() in the future.
            extrinsic = np.linalg.inv(
                info['axis_align_matrix'] @ info['lidar2cam'][i])
            info['lidar2img'].append(extrinsic.astype(np.float32))
            if self.modality['use_ray']:
                c2w = (
                    info['axis_align_matrix'] @ info['lidar2cam'][i]).astype(
                        np.float32)  # noqa
                info['c2w'].append(c2w)
                info['camrotc2w'].append(c2w[0:3, 0:3])
                info['lightpos'].append(c2w[0:3, 3])
        origin = np.array([.0, .0, .5])
        info['lidar2img'] = dict(
            extrinsic=info['lidar2img'],
            intrinsic=info['cam2img'].astype(np.float32), # every scene save an intrinsic
            origin=origin.astype(np.float32))

        if self.modality['use_ray']:
            info['ray_info'] = []

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = self._remove_dontcare(info['ann_info'])

        return info

    def _parse_2d_annotations(self, info: dict) -> list:
        index = self._2d_annotation_index
        scene_id = str(info.get(
            'scene_id', Path(info['img_paths'][0]).parent.name))
        annotations_by_view = []
        for view_index, image_path in enumerate(info['img_paths']):
            key = (scene_id, _normalise_image_key(image_path,
                                                   Path(self.data_root).resolve()))
            records = index.by_image_key.get(key)
            if records is None:
                image = index.images_by_view.get((scene_id, view_index))
                records = (index.annotations_by_image_id.get(
                    image['image_id'], []) if image is not None else [])
            parsed_records = []
            for record in records:
                category_id = int(record['category_id'])
                category_name = index.category_id_to_name.get(category_id)
                if category_name is None:
                    raise ValueError(
                        f'2D category_id {category_id} is missing from categories')
                classes = tuple(self.metainfo['classes'])
                if category_name not in classes:
                    raise ValueError(
                        f'2D category {category_name!r} is not in ScanNet classes')
                parsed = dict(record)
                parsed['bbox_label'] = classes.index(category_name)
                parsed_records.append(parsed)
            annotations_by_view.append(parsed_records)
        return annotations_by_view

    def parse_ann_info(self, info: dict) -> dict:

        ann_info = super().parse_ann_info(info)

        if self.remove_dontcare:
            ann_info = self._remove_dontcare(ann_info)

        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 6), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)

        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # count the numbers
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                # cat_name = self.metainfo['classes'][label]
                cat_name = label
                self.num_ins_per_cat[cat_name] += 1

        return ann_info


@DATASETS.register_module()
class MultiViewARKitDataset(Det3DDataset):
    METAINFO = {
        'classes':
        ("cabinet", "refrigerator", "shelf", "stove", "bed", # 0..5
            "sink", "washer", "toilet", "bathtub", "oven", # 5..10
            "dishwasher", "fireplace", "stool", "chair", "table", # 10..15
            "tv_monitor", "sofa")
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 box_type_3d: str = 'Depth',
                 filter_empty_gt: bool = True,
                 remove_dontcare: bool = False,
                 test_mode: bool = False,
                 **kwargs) -> None:

        self.remove_dontcare = remove_dontcare

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            pipeline=pipeline,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        assert 'use_camera' in self.modality and \
               'use_lidar' in self.modality
        assert self.modality['use_camera'] or self.modality['use_lidar']

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix']) # identiy
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_depth']:
            info['depth_info'] = []
        if self.modality['use_neuralrecon_depth']:
            info['depth_info'] = []

        if self.modality['use_lidar']:
            # implement lidar processing in the future
            raise NotImplementedError(
                'Please modified '
                '`MultiViewPipeline` to support lidar processing')

        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        info['img_info'] = []
        info['lidar2img'] = []
        intrinsics = []
        # load img and depth_img
        for i in range(len(info['img_paths'])):
            img_filename = osp.join(self.data_root, info['img_paths'][i])

            info['img_info'].append(dict(filename=img_filename))
            if 'depth_info' in info.keys():
                if self.modality['use_neuralrecon_depth']:
                    info['depth_info'].append(
                        dict(filename=img_filename[:-4] + '.npy'))
                else:
                    info['depth_info'].append(
                        dict(filename=osp.join(self.data_root, info['depth_paths'][i]))) # load depth from here
                    assert info['img_paths'][i].split('/')[-1] == info['depth_paths'][i].split('/')[-1] #
            # implement lidar_info in input.keys() in the future.
            extrinsic = np.linalg.inv(
                info['axis_align_matrix'] @ info['lidar2cam'][i])
            info['lidar2img'].append(extrinsic.astype(np.float32)) # w2c
            # intrisinc:
            intrinsic = info['cam2img'][i]
            new_intrinsic = np.eye(4)
            new_intrinsic[:3, :3] = intrinsic
            intrinsics.append(new_intrinsic.astype(np.float32))
            
            
        origin = np.array([.0, .0, .5])
        info['lidar2img'] = dict(
            extrinsic=info['lidar2img'],
            intrinsic=intrinsics,
            origin=origin.astype(np.float32))

        if not self.test_mode:
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['ann_info'] = self.parse_ann_info(info)
            info['eval_ann_info'] = self._remove_dontcare(info['ann_info'])

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Info dict.

        Returns:
            dict: Processed `ann_info`.
        """
        ann_info = super().parse_ann_info(info)

        if self.remove_dontcare:
            ann_info = self._remove_dontcare(ann_info)

        # empty gt
        if ann_info is None:
            ann_info = dict()
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros((0, ), dtype=np.int64)

        ann_info['gt_bboxes_3d'] = DepthInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            with_yaw=True,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d) # src mode is the same as tgt mode

        # count the numbers
        for label in ann_info['gt_labels_3d']:
            if label != -1:
                # cat_name = self.metainfo['classes'][label]
                cat_name = label
                self.num_ins_per_cat[cat_name] += 1

        return ann_info
