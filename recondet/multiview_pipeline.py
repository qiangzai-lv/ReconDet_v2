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


def arkit_pose_path(image_path):
    image_path = Path(image_path)
    suffix = '_color.png'
    if not image_path.name.endswith(suffix):
        raise ValueError(f'Unexpected ARKit color image name: {image_path}')
    return image_path.with_name(image_path.name[:-len(suffix)] + '_pose.npy')


@TRANSFORMS.register_module()
class LoadFirstFramePose(BaseTransform):
    def transform(self, results: dict) -> dict:
        first_img_path = results['img_path'][0]
        pose_path = arkit_pose_path(first_img_path)
        if not pose_path.is_file():
            raise FileNotFoundError(f'ARKit pose does not exist: {pose_path}')
        pose_matrix = np.load(pose_path)
        if pose_matrix.shape != (4, 4) or not np.isfinite(pose_matrix).all():
            raise ValueError(f'Invalid ARKit pose matrix: {pose_path}')
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

    def _select_view_indices(self, num_views: int,
                             available_view_indices=None) -> np.ndarray:
        if num_views <= 0:
            raise ValueError('A scene must contain at least one image')

        candidates = np.arange(num_views, dtype=np.int64)
        if available_view_indices is not None:
            available = np.asarray(available_view_indices)
            if available.dtype == np.bool_:
                if available.shape != (num_views,):
                    raise ValueError('available-view mask must have shape [V]')
                candidates = np.flatnonzero(available)
            else:
                candidates = available.astype(np.int64).reshape(-1)
            if len(candidates) == 0:
                raise ValueError('No available views can be selected')
            if (candidates < 0).any() or (candidates >= num_views).any():
                raise ValueError('available view index is out of range')

        if self.loading == 'random':
            return np.random.choice(
                candidates,
                self.n_images,
                replace=self.n_images > len(candidates))

        positions = np.rint(np.linspace(
            0, len(candidates) - 1, self.n_images)).astype(np.int64)
        return candidates[positions]

    def transform(self, results: dict) -> dict:
        ids = self._select_view_indices(
            len(results['img_info']), results.get('available_view_indices'))
        imgs = []
        extrinsics = []
        src_img_paths = []
        view_2d_instances = []
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
        if all_2d_annotations is not None:
            results['gt_instances_2d'] = view_2d_instances
        if has_depth:
            results['gt_depths_metric'] = np.stack(
                depths_metric).astype(np.float32)
            results['gt_c2w_metric'] = np.stack(
                selected_c2w).astype(np.float32)
            results['gt_intrinsics'] = np.stack(
                selected_intrinsics).astype(np.float32)
        results['lidar2img']['extrinsic'] = extrinsics
        return results
