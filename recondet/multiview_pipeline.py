from pathlib import Path

import numpy as np
from mmcv.transforms import BaseTransform, Compose

from mmdet3d.registry import TRANSFORMS


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
                 loading: str = 'random'):
        if n_images <= 0:
            raise ValueError('n_images must be positive')
        if loading not in ('random', 'uniform'):
            raise ValueError(
                f'Unsupported view loading strategy: {loading}')

        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.loading = loading

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
        frame_metadata = {}
        for i in ids:
            view_index = int(i)
            img_path = results['img_info'][view_index]['filename']
            frame_results = self.transforms(dict(img_path=img_path))
            if frame_results is None:
                raise RuntimeError(f'Failed to load image: {img_path}')

            imgs.append(frame_results['img'])
            src_img_paths.append(img_path)
            extrinsics.append(
                results['lidar2img']['extrinsic'][view_index])
            frame_metadata = {
                key: value for key, value in frame_results.items()
                if key not in ('img', 'img_info', 'img_path')
            }

        results.update(frame_metadata)
        results['img'] = imgs
        results['img_path'] = src_img_paths
        results['lidar2img']['extrinsic'] = extrinsics
        return results
