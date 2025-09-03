#!/usr/bin/env python3
"""
Visualize a mesh (.ply), a trajectory of poses (c2w 4x4 per line), and/or a point cloud (.ply)
in Rerun Viewer. Supports cloud downsampling, trajectory subsampling, and Y/Z axis flipping
to handle different conventions.

Examples:
  - Mesh only:
      python visualize.py --mesh data/room0_mesh.ply
  - Mesh + trajectory (subsample every 3 frames):
      python visualize.py --mesh data/room0_mesh.ply --traj out/trajectory.txt --every 3
  - Mesh + point cloud (with 1cm voxel):
      python visualize.py --mesh data/room0_mesh.ply --cloud out/fused_cloud.ply --voxel_size 0.01
  - Headless + save .rrd session:
      python visualize.py --mesh data/room0_mesh.ply --out_rrd viz.rrd --no_spawn

Trajectory file format:
  - Text, one line per frame, 16 floats = 4x4 row-major matrix (c2w).
  - Empty lines or lines starting with '#' are ignored.
"""

from __future__ import annotations
import argparse
import os
from typing import List, Optional

import numpy as np
import open3d as o3d
import rerun as rr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize mesh, trajectory, and/or point cloud in Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mesh", type=str, default=None,
                   help="Path to .ply mesh file")
    p.add_argument("--traj", type=str, default=None,
                   help="Trajectory file: each line with 16 floats (4x4 c2w, row-major)")
    p.add_argument("--every", type=int, default=1,
                   help="Subsample trajectory (1 = all frames)")
    p.add_argument("--flip_axes", action="store_true", default=True,
                   help="Flip Y and Z axes (TYPICAL Point-SLAM convention)")
    p.add_argument("--no_flip_axes", dest="flip_axes", action="store_false",
                   help="Disable axis flip (use trajectory as is)")
    p.add_argument("--cloud", type=str, default=None,
                   help="Final point cloud file (.ply)")
    p.add_argument("--voxel_size", type=float, default=0.0,
                   help="Point cloud voxel downsample (0 = no downsample)")
    p.add_argument("--spawn", dest="spawn", action="store_true", default=True,
                   help="Launch the Rerun Viewer (GUI)")
    p.add_argument("--no_spawn", dest="spawn", action="store_false",
                   help="Do not launch the GUI (useful on server/headless)")
    p.add_argument("--out_rrd", type=str, default=None,
                   help="If set, saves the Rerun session into this .rrd file")
    p.add_argument("--traj_color", type=str, choices=["rg", "rb", "gb"],
                   default="rb", help="Trajectory color gradient")
    p.add_argument("--normal_scale", type=float, default=0.03,
                   help="Scale of normals (arrows) for the point cloud")
    return p.parse_args()


def load_poses(path: str, flip_axes: bool = True) -> List[np.ndarray]:
    """Load c2w 4x4 poses (row-major) from a text file. Optionally flip Y/Z axes."""
    poses: List[np.ndarray] = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            vals = ln.split()
            if len(vals) < 16:
                continue
            c2w = np.array(list(map(float, vals[:16])), dtype=np.float32).reshape(4, 4)
            if flip_axes:
                c2w[:3, 1] *= -1.0
                c2w[:3, 2] *= -1.0
            poses.append(c2w)
    return poses


def _to_uint8_colors(x: np.ndarray) -> np.ndarray:
    """Convert Open3D colors [0..1] to uint8 [0..255] if needed."""
    if x.dtype != np.uint8:
        x = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
    return x


def _traj_gradient(n: int, mode: str = "rb") -> np.ndarray:
    """Create an n×3 uint8 gradient. mode: 'rb' (red→blue), 'rg', 'gb'."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    zeros = np.zeros_like(t)
    if mode == "rg":
        r, g, b = t, 1.0 - t, zeros
    elif mode == "gb":
        r, g, b = zeros, t, 1.0 - t
    else:  # "rb"
        r, g, b = t, zeros, 1.0 - t
    col = np.stack([r, g, b], axis=1)  # [n,3] in [0,1]
    return _to_uint8_colors(col)


def main() -> None:
    args = parse_args()

    if args.mesh is not None and not os.path.exists(args.mesh):
        raise FileNotFoundError(f"Mesh not found: {args.mesh}")
    if args.traj is not None and not os.path.exists(args.traj):
        print(f"[WARN] Trajectory file not found: {args.traj}")
    if args.cloud is not None and not os.path.exists(args.cloud):
        print(f"[WARN] Point cloud file not found: {args.cloud}")

    # --- init Rerun ---
    rr.init("ply_viewer", spawn=args.spawn)
    
    # --- mesh (optional) ---
    if args.mesh is not None:
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        if mesh.is_empty():
            raise RuntimeError(f"Mesh empty or unreadable: {args.mesh}")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        triangles = np.asarray(mesh.triangles, dtype=np.int32)
        colors: Optional[np.ndarray] = None
        if mesh.has_vertex_colors():
            colors = _to_uint8_colors(np.asarray(mesh.vertex_colors, dtype=np.float32))


        # --- log mesh ---
        rr.log(
            "scene/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=triangles,
                vertex_colors=colors,
            ),
        )

    # --- trajectory (optional) ---
    if args.traj and os.path.exists(args.traj):
        poses = load_poses(args.traj, flip_axes=args.flip_axes)
        print(f"[INFO] Loaded {len(poses)} poses")
        if len(poses) > 0:
            positions = np.stack([p[:3, 3] for p in poses], axis=0).astype(np.float32)
            positions = positions[::max(1, args.every)]
            n = positions.shape[0]
            traj_colors = _traj_gradient(n, mode=args.traj_color)

            rr.log("scene/trajectory", rr.LineStrips3D([positions], colors=[traj_colors]))

            # start / end markers
            rr.log("scene/trajectory/start",
                   rr.Points3D(positions[0:1], radii=0.03, colors=np.array([[0, 255, 0]], dtype=np.uint8)))
            rr.log("scene/trajectory/end",
                   rr.Points3D(positions[-1:], radii=0.03, colors=np.array([[255, 0, 0]], dtype=np.uint8)))

    # --- point cloud (optional) ---
    if args.cloud and os.path.exists(args.cloud):
        pcd = o3d.io.read_point_cloud(args.cloud)
        if pcd.is_empty():
            print(f"[WARN] Empty point cloud: {args.cloud}")
        else:
            if args.voxel_size and args.voxel_size > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

            pts = np.asarray(pcd.points, dtype=np.float32)

            cols: Optional[np.ndarray] = None
            if pcd.has_colors():
                cols = _to_uint8_colors(np.asarray(pcd.colors, dtype=np.float32))

            rr.log("scene/pointcloud", rr.Points3D(positions=pts, colors=cols))

            if pcd.has_normals():
                nrms = np.asarray(pcd.normals, dtype=np.float32)
                rr.log("scene/pointcloud/normals",
                       rr.Arrows3D(origins=pts, vectors=nrms * float(args.normal_scale)))

    # --- save .rrd session (optional) ---
    if args.out_rrd:
        try:
            rr.save(args.out_rrd)
            print(f"[INFO] Rerun session saved to: {args.out_rrd}")
        except Exception as e:
            print(f"[WARN] Could not save .rrd: {e}")


if __name__ == "__main__":
    main()
