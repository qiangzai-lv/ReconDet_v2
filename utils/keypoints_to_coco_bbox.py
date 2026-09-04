"""Generate COCO 2D boxes from fixed-point keypoint annotations."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger('keypoints_to_coco_bbox')


def convert(args):
    source = Path(args.keypoint_json)
    with source.open() as handle:
        data = json.load(handle)
    visibility_by_ann = {}
    for annotation in data.get('annotations', []):
        points = annotation.get('keypoints_2d')
        if points is None:
            raw = annotation.get('keypoints', [])
            points = [coord for index in range(0, len(raw), 3)
                      for coord in raw[index:index + 2]]
        points = list(points)
        visibility = annotation.get('keypoints_visibility')
        if visibility is None:
            raw = annotation.get('keypoints', [])
            visibility = [raw[index] for index in range(2, len(raw), 3)]
        visible_points = []
        for index, value in enumerate(visibility):
            if index * 2 + 1 >= len(points) or not value:
                continue
            x, y = float(points[index * 2]), float(points[index * 2 + 1])
            if x == x and y == y:
                visible_points.append((x, y))
        visibility_by_ann[int(annotation['id'])] = visible_points

    output = {
        'info': dict(data.get('info', {}), description='COCO boxes from visible keypoints'),
        'licenses': data.get('licenses', []),
        'images': data.get('images', []),
        'annotations': [],
        'categories': data.get('categories', []),
    }
    images_by_id = {int(item['id']): item for item in output['images']}
    stats = Counter()
    for annotation in data.get('annotations', []):
        points = visibility_by_ann[int(annotation['id'])]
        if len(points) < args.min_visible_points:
            stats['skipped_few_points'] += 1
            continue
        xs, ys = zip(*points)
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        image = images_by_id.get(int(annotation['image_id']))
        if image is not None:
            x1 = max(0.0, min(x1, image['width']))
            y1 = max(0.0, min(y1, image['height']))
            x2 = max(0.0, min(x2, image['width']))
            y2 = max(0.0, min(y2, image['height']))
        width, height = x2 - x1, y2 - y1
        if width < args.min_size or height < args.min_size:
            stats['skipped_small'] += 1
            continue
        output['annotations'].append({
            'id': int(annotation['id']),
            'image_id': int(annotation['image_id']),
            'category_id': int(annotation['category_id']),
            'bbox': [x1, y1, width, height],
            'area': width * height,
            'iscrowd': 0,
            'source_keypoint_annotation_id': int(annotation['id']),
        })
        stats['written'] += 1

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w') as handle:
        json.dump(output, handle, indent=2)
    LOGGER.info('Loaded %d keypoint annotations from %s',
                len(data.get('annotations', [])), source)
    LOGGER.info('Wrote %d bbox annotations to %s; stats=%s',
                len(output['annotations']), target, dict(stats))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keypoint-json', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-visible-points', type=int, default=1)
    parser.add_argument('--min-size', type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')
    if args.min_visible_points <= 0 or args.min_visible_points > 4:
        raise ValueError('--min-visible-points must be in [1, 4]')
    if args.min_size < 0:
        raise ValueError('--min-size must be non-negative')
    convert(args)


if __name__ == '__main__':
    main()
