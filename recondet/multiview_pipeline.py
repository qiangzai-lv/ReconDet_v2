from pathlib import Path

import numpy as np
from mmcv.transforms import BaseTransform, Compose

from mmdet3d.registry import TRANSFORMS
from .scannet_multiview_dataset import build_view_2d_instances
from .vggt_ground_truth import load_and_resize_depth


def camera_intrinsic_for_view(value, view_index):
    matrices = np.asarray(value, dtype=np.float32)
    if matrices.ndim == 3:
        if view_index >= len(matrices):
            raise IndexError(f'view_index {view_index} is outside intrinsics')
        matrices = matrices[view_index]
    if matrices.shape not in ((3, 3), (4, 4)):
        raise ValueError(
            f'camera intrinsic must be 3x3, 4x4, or Vx..., got '
            f'{matrices.shape}')
    return matrices[:3, :3].copy()


def read_pose_matrix(file_path):

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        matrix = [list(map(float, line.strip().split())) for line in lines]

        pose_matrix = np.array(matrix)

        if pose_matrix.shape != (4, 4):
            raise ValueError("The input file does not contain a valid 4x4 pose matrix.")

        return pose_matrix

    except Exception as e:
        print(f"Error reading pose matrix: {e}")
        return None


@TRANSFORMS.register_module()
class LoadFirstFramePose(BaseTransform):
    def transform(self, results: dict) -> dict:
        first_img_path = results['img_path'][0]
        pose_matrix = read_pose_matrix(str(Path(first_img_path).with_suffix('.txt')))
        if pose_matrix is None:
            raise ValueError(f'Could not load first-frame pose for {first_img_path}')

        results['pose_matrix'] = pose_matrix.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):

    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 loading: str = 'random',
                 depth_scale: float = 1000.0):
        if n_images <= 0:
            raise ValueError('n_images must be positive')
        if loading not in ('random', 'uniform'):
            raise ValueError(
                f'Unsupported view loading strategy: {loading}')

        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.loading = loading
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError('depth_scale must be finite and positive')
        self.depth_scale = float(depth_scale)

    def _select_view_indices(self, num_views: int) -> np.ndarray:
        if num_views <= 0:
            raise ValueError('A scene must contain at least one image')

        if self.loading == 'random':
            return np.random.choice(
                num_views,
                self.n_images,
                replace=self.n_images > num_views)

        return np.rint(
            np.linspace(0, num_views - 1, self.n_images)).astype(np.int64)

    def transform(self, results: dict) -> dict:
        ids = self._select_view_indices(len(results['img_info']))
        imgs = []
        extrinsics = []
        src_img_paths = []
        view_2d_instances = []
        view_img_ids = []
        view_ori_shapes = []
        view_img_shapes = []
        view_scale_factors = []
        all_2d_annotations = results.get('ann_info_2d')
        has_depth = 'depth_info' in results
        depths_metric = []
        selected_c2w = []
        selected_intrinsics = []
        frame_metadata = {}
        for i in ids:
            view_index = int(i)
            img_path = results['img_info'][view_index]['filename']
            frame_results = self.transforms(dict(img_path=img_path))
            if frame_results is None:
                raise RuntimeError(f'Failed to load image: {img_path}')

            imgs.append(frame_results['img'])
            src_img_paths.append(img_path)
            view_ori_shapes.append(tuple(frame_results['ori_shape'][:2]))
            view_img_shapes.append(tuple(frame_results['img_shape'][:2]))
            scale_factor = frame_results.get('scale_factor')
            if scale_factor is None:
                ori_height, ori_width = frame_results['ori_shape'][:2]
                new_height, new_width = frame_results['img_shape'][:2]
                scale_factor = (
                    new_width / ori_width, new_height / ori_height)
            if all_2d_annotations is not None:
                view_2d_instances.append(build_view_2d_instances(
                    all_2d_annotations[view_index],
                    tuple(scale_factor[:2]), frame_results['img_shape']))
                view_img_ids.append(int(results['image_ids_2d'][view_index]))
            view_scale_factors.append(tuple(scale_factor[:4]))
            if has_depth:
                depth_path = results['depth_info'][view_index]['filename']
                depths_metric.append(load_and_resize_depth(
                    depth_path, frame_results['img_shape'][:2],
                    self.depth_scale))
                selected_c2w.append(np.asarray(
                    results['lidar2cam'][view_index], dtype=np.float32))
                intrinsic_source = results.get(
                    'cam2img', results['lidar2img']['intrinsic'])
                intrinsic = camera_intrinsic_for_view(
                    intrinsic_source, view_index)
                x_scale, y_scale = map(float, scale_factor[:2])
                intrinsic[0] *= x_scale
                intrinsic[1] *= y_scale
                selected_intrinsics.append(intrinsic)
            extrinsics.append(
                results['lidar2img']['extrinsic'][view_index])
            frame_metadata = {
                key: value for key, value in frame_results.items()
                if key not in ('img', 'img_info', 'img_path')
            }

        results.update(frame_metadata)
        results['img'] = imgs
        results['img_path'] = src_img_paths
        results['view_indices'] = np.asarray(ids, dtype=np.int64)
        results['view_ori_shapes'] = view_ori_shapes
        results['view_img_shapes'] = view_img_shapes
        results['view_scale_factors'] = view_scale_factors
        if all_2d_annotations is not None:
            results['gt_instances_2d'] = view_2d_instances
            results['view_img_ids'] = view_img_ids
        if has_depth:
            results['gt_depths_metric'] = np.stack(
                depths_metric).astype(np.float32)
            results['gt_c2w_metric'] = np.stack(
                selected_c2w).astype(np.float32)
            results['gt_intrinsics'] = np.stack(
                selected_intrinsics).astype(np.float32)
        results['lidar2img']['extrinsic'] = extrinsics
        return results
