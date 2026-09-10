from mmdet.evaluation.metrics import CocoMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SceneCocoMetric(CocoMetric):
    """Collect one record per scene before expanding views after DDP gather."""

    def process(self, data_batch, data_samples):
        for scene in data_samples:
            predictions = scene['pred_instances_2d']
            ids = scene['view_img_ids']
            shapes = scene['view_ori_shapes']
            if not len(predictions) == len(ids) == len(shapes):
                raise ValueError('Scene predictions and view metadata differ')
            records = []
            for image_id, shape, pred in zip(ids, shapes, predictions):
                image_id = int(image_id)
                # Skip placeholder views with missing 2D annotations
                if image_id == -1:
                    continue
                if image_id not in self._coco_api.imgs:
                    raise ValueError(f'Unknown COCO image id {image_id}')
                image = self._coco_api.imgs[image_id]
                if (image['height'], image['width']) != tuple(shape):
                    raise ValueError(f'COCO size mismatch for image {image_id}')
                gt = dict(img_id=image_id, height=shape[0], width=shape[1])
                result = dict(img_id=image_id, **{
                    key: pred[key].detach().cpu().numpy()
                    for key in ('bboxes', 'scores', 'labels')})
                records.append((gt, result))
            self.results.append(records)

    def compute_metrics(self, results):
        unique = {}
        for scene in results:
            for gt, pred in scene:
                unique.setdefault(gt['img_id'], (gt, pred))
        if not unique:
            raise ValueError('No sampled images to evaluate')
        # Reset every evaluation: a previous epoch must not retain its subset.
        self.img_ids = sorted(unique)
        return super().compute_metrics([unique[i] for i in self.img_ids])
