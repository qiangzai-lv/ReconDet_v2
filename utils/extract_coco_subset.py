#!/usr/bin/env python3
"""Extract a reproducible random subset from a COCO annotation file."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

PathLike = Union[str, Path]
DEFAULT_INPUT = Path(
    "/root/shared-nvme/data/scannet_coco/keypoints_bbox_val.json")


def _validate_coco(coco: Dict[str, Any]) -> None:
    images = coco.get("images") if isinstance(coco, dict) else None
    annotations = coco.get("annotations") if isinstance(coco, dict) else None
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(
            "Invalid COCO data: 'images' and 'annotations' must be lists")
    for image in images:
        if not isinstance(image, dict) or "id" not in image:
            raise ValueError(
                "Invalid COCO data: every image must contain an 'id'")
        try:
            hash(image["id"])
        except TypeError as error:
            raise ValueError(
                "Invalid COCO data: every image 'id' must be hashable") from error
    for annotation in annotations:
        if not isinstance(annotation, dict) or "image_id" not in annotation:
            raise ValueError(
                "Invalid COCO data: every annotation must contain an "
                "'image_id'")
        try:
            hash(annotation["image_id"])
        except TypeError as error:
            raise ValueError(
                "Invalid COCO data: every annotation 'image_id' must be "
                "hashable") from error


def extract_subset(coco: Dict[str, Any], count: int,
                   seed: int) -> Dict[str, Any]:
    """Return a COCO mapping containing a random subset of its images."""
    _validate_coco(coco)
    images = coco["images"]
    if not 1 <= count <= len(images):
        raise ValueError(
            f"count must be between 1 and {len(images)}, got {count}")

    selected_indices = set(random.Random(seed).sample(range(len(images)),
                                                       count))
    selected_images = [
        image for index, image in enumerate(images)
        if index in selected_indices
    ]
    selected_ids = {image["id"] for image in selected_images}

    subset = dict(coco)
    subset["images"] = selected_images
    subset["annotations"] = [
        annotation for annotation in coco["annotations"]
        if annotation["image_id"] in selected_ids
    ]
    return subset


def extract_file(input_path: PathLike, count: int, seed: int,
                 output_path: Optional[PathLike] = None
                 ) -> Tuple[Path, int, int]:
    """Read, subset, and write a COCO file; return path and result counts."""
    source = Path(input_path)
    destination = (Path(output_path) if output_path is not None else
                   source.with_name(f"{source.stem}_{count}{source.suffix}"))
    paths_match = source.resolve() == destination.resolve()
    links_match = destination.exists() and source.samefile(destination)
    if paths_match or links_match:
        raise ValueError("Output path must differ from input path")

    with source.open("r", encoding="utf-8") as file:
        coco = json.load(file)
    subset = extract_subset(coco, count=count, seed=seed)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(subset, file, ensure_ascii=False)

    return destination, len(subset["images"]), len(subset["annotations"])


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly extract N images and their annotations from COCO JSON.")
    parser.add_argument("count", type=int, help="number of images to extract")
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"source COCO JSON (default: {DEFAULT_INPUT})")
    parser.add_argument(
        "--output", type=Path,
        help="output JSON (default: <input_stem>_<count>.json)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="random seed for reproducible sampling (default: 42)")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    try:
        output_path, image_count, annotation_count = extract_file(
            args.input,
            count=args.count,
            seed=args.seed,
            output_path=args.output,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Images: {image_count}")
    print(f"Annotations: {annotation_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
