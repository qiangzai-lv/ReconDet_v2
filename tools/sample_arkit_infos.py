#!/usr/bin/env python3
"""Extract small ARKitScenes train/val info subsets."""
import argparse
import pickle
from pathlib import Path


def sample_file(source: Path, target: Path, count: int) -> None:
    with source.open('rb') as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or 'data_list' not in payload:
        raise ValueError(f'{source} must contain a dict with data_list')
    data_list = payload['data_list']
    if len(data_list) < count:
        raise ValueError(f'{source} has only {len(data_list)} entries, need {count}')
    output = dict(payload)
    output['data_list'] = list(data_list[:count])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{source} -> {target}: {count} entries')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path,
                        default=Path('/root/shared-nvme/data/ARKitScenes_processed'))
    parser.add_argument('--train-count', type=int, default=20)
    parser.add_argument('--val-count', type=int, default=10)
    parser.add_argument('--train-output', type=Path,
                        default=None)
    parser.add_argument('--val-output', type=Path, default=None)
    args = parser.parse_args()
    if args.train_count <= 0 or args.val_count <= 0:
        parser.error('counts must be positive')
    train_src = args.data_root / 'arkit_infos_train.pkl'
    val_src = args.data_root / 'arkit_infos_val.pkl'
    train_dst = args.train_output or args.data_root / 'arkit_infos_train_20.pkl'
    val_dst = args.val_output or args.data_root / 'arkit_infos_val_10.pkl'
    sample_file(train_src, train_dst, args.train_count)
    sample_file(val_src, val_dst, args.val_count)


if __name__ == '__main__':
    main()
