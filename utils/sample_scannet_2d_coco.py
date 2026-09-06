"""Sample a fixed number of images per scene from a merged COCO file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


def _uniform_indices(count: int, limit: int):
    if count <= limit:
        return list(range(count))
    if limit <= 0:
        raise ValueError('images_per_scene must be positive')
    if limit == 1:
        return [0]
    # Round linspace positions so both ends of the scene are represented.
    return [round(index * (count - 1) / (limit - 1))
            for index in range(limit)]


def sample_coco_payload(payload: dict, images_per_scene: int) -> dict:
    if images_per_scene <= 0:
        raise ValueError('images_per_scene must be positive')
    images = list(payload.get('images', []))
    annotations = list(payload.get('annotations', []))
    image_by_id = {int(image['id']): image for image in images}
    if len(image_by_id) != len(images):
        raise ValueError('input COCO contains duplicate image ids')

    grouped = {}
    for image in images:
        scene_id = image.get('scene_id')
        if scene_id is None:
            raise ValueError('every image must contain scene_id')
        grouped.setdefault(str(scene_id), []).append(image)

    selected_images = []
    for scene_id in sorted(grouped):
        scene_images = sorted(
            grouped[scene_id],
            key=lambda image: (int(image.get('view_index', image['id'])),
                               int(image['id'])))
        selected_images.extend(
            scene_images[index]
            for index in _uniform_indices(len(scene_images), images_per_scene))

    old_to_new = {}
    output_images = []
    for new_id, image in enumerate(selected_images, start=1):
        old_to_new[int(image['id'])] = new_id
        output_image = dict(image)
        output_image['id'] = new_id
        output_images.append(output_image)

    output_annotations = []
    next_annotation_id = 1
    for annotation in annotations:
        old_image_id = int(annotation['image_id'])
        if old_image_id not in old_to_new:
            continue
        output_annotation = dict(annotation)
        output_annotation['id'] = next_annotation_id
        output_annotation['image_id'] = old_to_new[old_image_id]
        output_annotations.append(output_annotation)
        next_annotation_id += 1

    output = {
        'info': dict(payload.get('info', {})),
        'licenses': list(payload.get('licenses', [])),
        'images': output_images,
        'annotations': output_annotations,
        'categories': list(payload.get('categories', [])),
    }
    return output


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, separators=(',', ':'))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--images-per-scene', type=int, required=True)
    args = parser.parse_args()
    with Path(args.input).open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    sampled = sample_coco_payload(payload, args.images_per_scene)
    _write_json_atomic(Path(args.output), sampled)
    scene_ids = {str(image['scene_id']) for image in sampled['images']}
    print(
        f'Sampled {len(sampled["images"])} images from {len(scene_ids)} scenes '
        f'and kept {len(sampled["annotations"])} annotations',
        flush=True)


if __name__ == '__main__':
    main()
