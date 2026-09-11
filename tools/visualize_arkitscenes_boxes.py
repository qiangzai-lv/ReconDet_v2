#!/usr/bin/env python3
"""Export one ARKitScenes point cloud and its 3D yaw boxes as a PLY.

The processed files store points as float32 [x,y,z,(r,g,b)] and boxes as
[cx,cy,cz,dx,dy,dz,yaw] in the scene (axis-aligned/world) coordinate frame.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np


def box_segments(box, transform=None):
    c = np.asarray(box[:3], dtype=np.float64)
    d = np.asarray(box[3:6], dtype=np.float64)
    yaw = float(box[6])
    x, y, z = d / 2
    corners = np.array([[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],
                        [-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]])
    co, si = np.cos(yaw), np.sin(yaw)
    R = np.array([[co,-si,0],[si,co,0],[0,0,1]])
    corners = corners @ R.T + c
    if transform is not None:
        corners = (corners @ transform[:3, :3].T + transform[:3, 3])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    return corners, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--ann', required=True, help='arkit_infos_*.pkl')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root, ann = Path(args.root), Path(args.ann)
    data = pickle.load(ann.open('rb'))['data_list'][args.index]
    axis_align = np.asarray(data.get('axis_align_matrix', np.eye(4)), dtype=np.float64)
    if axis_align.shape != (4, 4) or not np.isfinite(axis_align).all():
        raise ValueError('axis_align_matrix must be a finite 4x4 matrix')
    pf = root / data['lidar_points']['lidar_path']
    if not pf.exists():
        pf = root / ('train_points' if 'train' in ann.name else 'val_points') / pf.name
    raw = np.fromfile(pf, dtype=np.float32).reshape(-1, data['lidar_points'].get('num_pts_feats', 6))
    xyz = raw[:, :3]
    xyz = xyz @ axis_align[:3, :3].T + axis_align[:3, 3]
    colors = np.clip(raw[:, 3:6], 0, 255).astype(np.uint8) if raw.shape[1] >= 6 else np.full((len(xyz),3), 180, np.uint8)
    verts = [(*p, *rgb) for p, rgb in zip(xyz, colors)]
    lines = []
    for ins in data.get('instances', []):
        cs, es = box_segments(ins['bbox_3d'], axis_align)
        base = len(verts)
        verts += [(*p, 255, 40, 40) for p in cs]
        lines += [(base+a, base+b) for a,b in es]
    with open(args.output, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(f'element edge {len(lines)}\nproperty int vertex1\nproperty int vertex2\nend_header\n')
        for v in verts: f.write('%.6f %.6f %.6f %d %d %d\n' % v)
        for a,b in lines: f.write(f'{a} {b}\n')
    print(f'wrote {args.output}: {len(xyz)} points, {len(data.get("instances", []))} boxes')


if __name__ == '__main__': main()
