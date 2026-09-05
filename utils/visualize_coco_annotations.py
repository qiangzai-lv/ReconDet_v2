"""Visualize COCO boxes or fixed 4-point annotations without source edits."""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


LOGGER = logging.getLogger('visualize_coco_annotations')


def _font(size: int):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _color(category_id: int):
    # Deterministic, high-contrast colors without requiring matplotlib.
    return ((37 * category_id + 71) % 220 + 20,
            (97 * category_id + 43) % 220 + 20,
            (173 * category_id + 19) % 220 + 20)


def visualize(args):
    with Path(args.coco_json).open() as handle:
        coco = json.load(handle)

    categories = {
        int(category['id']): category['name']
        for category in coco.get('categories', [])
    }
    annotations = defaultdict(list)
    for annotation in coco.get('annotations', []):
        annotations[int(annotation['image_id'])].append(annotation)
    bbox_annotations = defaultdict(list)
    if args.bbox_json:
        with Path(args.bbox_json).open() as handle:
            bbox_data = json.load(handle)
        for annotation in bbox_data.get('annotations', []):
            bbox_annotations[int(annotation['image_id'])].append(annotation)
        LOGGER.info('Loaded %d bbox annotations from %s',
                    len(bbox_data.get('annotations', [])), args.bbox_json)
    images = list(coco.get('images', []))
    if args.shuffle:
        random.Random(args.seed).shuffle(images)
    if args.num_images > 0:
        images = images[:args.num_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.image_root).resolve() if args.image_root else None
    label_font = _font(args.font_size)
    LOGGER.info('Loaded %d images and %d annotations from %s',
                len(coco.get('images', [])), len(coco.get('annotations', [])),
                args.coco_json)
    LOGGER.info('Visualizing %d images to %s', len(images), output_dir)

    saved = 0
    skipped = 0
    for image_record in images:
        source = Path(image_record['file_name'])
        if not source.is_absolute() and root is not None:
            source = root / source
        if not source.exists():
            LOGGER.warning('Image does not exist, skipping: %s', source)
            skipped += 1
            continue
        try:
            image = Image.open(source).convert('RGB')
        except (OSError, ValueError) as error:
            LOGGER.warning('Could not read %s: %s', source, error)
            skipped += 1
            continue

        draw = ImageDraw.Draw(image)
        # Draw derived boxes first, so keypoints remain visible on top.
        for annotation in bbox_annotations.get(int(image_record['id']), []):
            if 'bbox' not in annotation:
                continue
            x, y, box_width, box_height = annotation['bbox']
            category_id = int(annotation['category_id'])
            draw.rectangle((x, y, x + box_width, y + box_height),
                           outline=_color(category_id),
                           width=max(1, args.line_width))
        for annotation in annotations.get(int(image_record['id']), []):
            category_id = int(annotation['category_id'])
            color = _color(category_id)
            if 'bbox' in annotation:
                x, y, box_width, box_height = annotation['bbox']
                draw.rectangle((x, y, x + box_width, y + box_height),
                               outline=color,
                               width=max(1, args.line_width))
            keypoints = annotation.get('keypoints_2d')
            if keypoints is not None and len(keypoints) >= 8:
                points = [(float(keypoints[i]), float(keypoints[i + 1]))
                          for i in range(0, 8, 2)]
                visibility = annotation.get('keypoints_visibility')
                if visibility is None and len(annotation.get('keypoints', [])) >= 12:
                    visibility = [int(annotation['keypoints'][i + 2])
                                  for i in range(0, 12, 3)]
                if visibility is None:
                    visibility = [2] * len(points)
                # New keypoint files use binary visibility (1/0); old COCO
                # files use 2/0. Both are treated as visible when non-zero.
                center = points[0]
                for point_id, point in enumerate(points[1:], start=1):
                    if not visibility[0] or not visibility[point_id]:
                        continue
                    draw.line((center[0], center[1], point[0], point[1]),
                              fill=color, width=max(1, args.line_width // 2))
                radius = max(2, args.point_radius)
                for point_id, (px, py) in enumerate(points):
                    point_color = ('red' if point_id == 0 else color) \
                        if visibility[point_id] else 'gray'
                    draw.ellipse((px - radius, py - radius, px + radius,
                                  py + radius), fill=point_color, outline='white')
                x, y = center
            elif 'bbox' in annotation:
                x, y, width, height = annotation['bbox']
                x2, y2 = x + width, y + height
                draw.rectangle((x, y, x2, y2), outline=color,
                               width=max(1, args.line_width))
            else:
                continue
            if args.no_labels:
                continue
            name = categories.get(category_id, str(category_id))
            text = f'{name} [{category_id}]'
            left, top, right, bottom = draw.textbbox((0, 0), text,
                                                      font=label_font)
            text_width, text_height = right - left, bottom - top
            text_y = max(0, y - text_height - 2)
            draw.rectangle((x, text_y, x + text_width + 4,
                            text_y + text_height + 4), fill=color)
            draw.text((x + 2, text_y + 2), text, fill='white',
                      font=label_font)

        stem = Path(source).stem
        scene = image_record.get('scene_id', 'scene')
        view = image_record.get('view_index', image_record['id'])
        output_path = output_dir / f'{scene}_view{int(view):05d}_{stem}.jpg'
        image.save(output_path, quality=95)
        saved += 1
        if saved % args.log_interval == 0 or saved == len(images) - skipped:
            LOGGER.info('Progress %d/%d images', saved, len(images))

    LOGGER.info('Finished: saved=%d, skipped=%d, output=%s',
                saved, skipped, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-json', required=True)
    parser.add_argument('--bbox-json', default=None,
                        help='Optional COCO bbox JSON to overlay on the same images')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-root', default=None,
                        help='Root for relative COCO file_name values')
    parser.add_argument('--num-images', type=int, default=20,
                        help='Number of images; <=0 means all images')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--line-width', type=int, default=3)
    parser.add_argument('--font-size', type=int, default=18)
    parser.add_argument('--point-radius', type=int, default=5)
    parser.add_argument('--no-labels', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    arguments = parse_args()
    if arguments.log_interval <= 0:
        raise ValueError('--log-interval must be positive')
    if arguments.line_width <= 0 or arguments.font_size <= 0:
        raise ValueError('--line-width and --font-size must be positive')
    visualize(arguments)
