"""Add scene and stable per-scene instance ids to ScanNet 3D annotations."""

from __future__ import annotations

if __package__ in (None, ''):
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import pickle
import tempfile
from pathlib import Path


def scene_id_from_info(info: dict) -> str:
    """Infer the ScanNet scene id from the first image path."""
    image_paths = info.get('img_paths')
    if not image_paths:
        raise ValueError('annotation record has no img_paths')
    scene_id = Path(image_paths[0]).parent.name
    if not scene_id:
        raise ValueError(f'cannot infer scene id from {image_paths[0]!r}')
    return scene_id


def add_metadata(payload: dict) -> dict:
    """Return a metadata-enriched copy of a ScanNet info payload."""
    if not isinstance(payload, dict) or not isinstance(
            payload.get('data_list'), list):
        raise TypeError('expected a dict payload containing data_list')

    enriched = dict(payload)
    enriched_records = []
    for info in payload['data_list']:
        if not isinstance(info, dict):
            raise TypeError('every data_list item must be a dict')
        updated_info = dict(info)
        updated_info['scene_id'] = scene_id_from_info(info)
        instances = info.get('instances', [])
        if not isinstance(instances, list):
            raise TypeError('instances must be a list')
        updated_instances = []
        for instance_id, instance in enumerate(instances):
            if not isinstance(instance, dict):
                raise TypeError('every instance must be a dict')
            updated_instance = dict(instance)
            updated_instance['instance_id'] = instance_id
            updated_instances.append(updated_instance)
        updated_info['instances'] = updated_instances
        enriched_records.append(updated_info)
    enriched['data_list'] = enriched_records
    return enriched


def update_file(input_path: Path, output_path: Path) -> None:
    with input_path.open('rb') as handle:
        payload = pickle.load(handle)
    enriched = add_metadata(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode='wb', dir=output_path.parent,
            prefix=output_path.name + '.', suffix='.tmp', delete=False) as tmp:
        temporary_path = Path(tmp.name)
        pickle.dump(enriched, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.flush()
    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    update_file(args.input, args.output)
    print(f'Wrote metadata-enriched annotations to {args.output}')


if __name__ == '__main__':
    main()
