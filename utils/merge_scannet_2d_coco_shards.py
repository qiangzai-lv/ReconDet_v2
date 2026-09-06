"""Merge per-scene ScanNet COCO annotations into one dataset JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


def _canonical_categories(categories):
    return tuple(
        (int(item['id']), str(item['name']), str(item.get('supercategory', '')))
        for item in categories)


def merge_scene_payloads(scene_payloads):
    scene_payloads = list(scene_payloads)
    if not scene_payloads:
        raise ValueError('No scene annotation JSON files were found')

    categories = scene_payloads[0].get('categories', [])
    category_signature = _canonical_categories(categories)
    merged = {
        'info': {'description':
                 'ScanNet boxes from visible 3D instance point projections'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': categories,
    }
    rejections = []
    seen_scenes = set()
    seen_file_names = set()
    next_image_id = 1
    next_annotation_id = 1

    for payload in scene_payloads:
        if _canonical_categories(payload.get('categories', [])) != category_signature:
            raise ValueError('Scene files contain inconsistent COCO categories')
        images = payload.get('images', [])
        annotations = payload.get('annotations', [])
        scene_ids = {str(image.get('scene_id', '')) for image in images}
        scene_ids.update(str(item.get('scene_id', ''))
                         for item in annotations)
        scene_ids.discard('')
        if len(scene_ids) != 1:
            raise ValueError(
                'Each scene JSON must contain exactly one non-empty scene_id')
        scene_id = next(iter(scene_ids))
        if scene_id in seen_scenes:
            raise ValueError(f'duplicate scene JSON: {scene_id}')
        seen_scenes.add(scene_id)

        image_id_map = {}
        for image in sorted(images, key=lambda item: int(item['id'])):
            old_id = int(image['id'])
            if old_id in image_id_map:
                raise ValueError(f'duplicate image id in scene {scene_id}: {old_id}')
            file_name = str(image['file_name'])
            if file_name in seen_file_names:
                raise ValueError(f'duplicate image path: {file_name}')
            seen_file_names.add(file_name)
            image_id_map[old_id] = next_image_id
            output_image = dict(image)
            output_image['id'] = next_image_id
            merged['images'].append(output_image)
            next_image_id += 1

        seen_annotation_ids = set()
        for annotation in sorted(
                annotations, key=lambda item: int(item['id'])):
            old_annotation_id = int(annotation['id'])
            if old_annotation_id in seen_annotation_ids:
                raise ValueError(
                    f'duplicate annotation id in scene {scene_id}: '
                    f'{old_annotation_id}')
            seen_annotation_ids.add(old_annotation_id)
            old_image_id = int(annotation['image_id'])
            if old_image_id not in image_id_map:
                raise ValueError(
                    f'annotation references unknown image {old_image_id} '
                    f'in scene {scene_id}')
            output_annotation = dict(annotation)
            output_annotation['id'] = next_annotation_id
            output_annotation['image_id'] = image_id_map[old_image_id]
            merged['annotations'].append(output_annotation)
            next_annotation_id += 1
        rejections.extend(payload.get('rejections', []))

    return merged, rejections


def _write_json_atomic(path: Path, payload, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload, separators=(',', ':') if compact else None,
        indent=None if compact else 2)
    with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def merge_scene_directory(input_dir: Path, output: Path,
                          rejections_output: Path) -> None:
    paths = sorted(input_dir.glob('*.json'))
    payloads = []
    for path in paths:
        with path.open('r', encoding='utf-8') as handle:
            payloads.append(json.load(handle))
    merged, rejections = merge_scene_payloads(payloads)
    _write_json_atomic(output, merged, compact=True)
    _write_json_atomic(rejections_output, rejections, compact=False)
    print(
        f'Merged {len(paths)} scenes, {len(merged["images"])} images and '
        f'{len(merged["annotations"])} annotations into {output}',
        flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rejections-output')
    return parser


def main():
    args = build_parser().parse_args()
    output = Path(args.output).resolve()
    rejections_output = (
        Path(args.rejections_output).resolve()
        if args.rejections_output else
        output.with_name(output.stem + '_rejections.json'))
    merge_scene_directory(
        Path(args.input_dir).resolve(), output, rejections_output)


if __name__ == '__main__':
    main()
