#!/usr/bin/env python3

# ATTENZIONE: Questo script è molto pesante, la visualizzazione è possibile solo per un massimo di 100 frames


from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk
import rerun.blueprint as rrb
import trimesh

from pathlib import Path

import open3d as o3d

DESCRIPTION = """
# ARKitScenes
This example visualizes the [ARKitScenes dataset](https://github.com/apple/ARKitScenes/) using Rerun. The dataset
contains color images, depth images, the reconstructed mesh, and labeled bounding boxes around furniture.

The full source code for this example is available
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/arkit_scenes).
""".strip()


INTRINSIC = np.array([[600.0,   0.0, 599.5],
                      [  0.0, 600.0, 339.5],
                      [  0.0,   0.0,   1.0]])
RESOLUTION = [1200, 680]
DEPTH_SCALE = 6553.5


def load_poses(traj_file: Path) -> list[rr.Transform3D]:
    poses = []
    with open(traj_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 16:
                continue
            c2w = np.array(vals, dtype=np.float32).reshape(4, 4)

            # inverti per ottenere world->camera
            w2c = np.linalg.inv(c2w)

            t = w2c[:3, 3]

            q = trimesh.transformations.quaternion_from_matrix(w2c)  # [w, x, y, z]
            qx, qy, qz, qw = q[1], q[2], q[3], q[0]
            quat = rr.Quaternion(xyzw=[qx, qy, qz, qw])

            transform = rr.Transform3D(
                translation=t.tolist(),
                rotation=quat,
                relation=rr.TransformRelation.ChildFromParent,
            )
            poses.append(transform)
    return poses


def load_w2c_matrices(traj_file: Path) -> list[np.ndarray]:
    matrices = []
    with open(traj_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 16:
                continue
            c2w = np.array(vals, dtype=np.float64).reshape(4, 4)
            w2c = np.linalg.inv(c2w)
            matrices.append(w2c)
    return matrices

def log_office2(recording_dir: Path, frames: int) -> None:
    rr.log("description", rr.TextDocument("Replica office2 dataset", media_type=rr.MediaType.MARKDOWN), static=True)

    print(f"[DEBUG] Using dataset folder: {recording_dir}")
    print(f"[DEBUG] Looking for traj.txt in: {recording_dir / 'traj.txt'}")
    print(f"[DEBUG] Looking for mesh in: {recording_dir / 'office2_default.ply'}")
    print(f"[DEBUG] Looking for images in: {recording_dir / 'results'}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02,  
        sdf_trunc=0.15,      
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        RESOLUTION[0], RESOLUTION[1],
        INTRINSIC[0, 0], INTRINSIC[1, 1],
        INTRINSIC[0, 2], INTRINSIC[1, 2]
    )

    
    traj_file = recording_dir / "traj.txt"
    results_dir = recording_dir / "results"

    poses = load_poses(traj_file)
    w2c_matrices = load_w2c_matrices(traj_file=traj_file)

    depth_files = sorted(results_dir.glob("depth*.png"))
    rgb_files   = sorted(results_dir.glob("frame*.jpg")) + sorted(results_dir.glob("frame*.jpeg"))
    rgb_files   = sorted(rgb_files)

    print(f"[DEBUG] Found {len(rgb_files)} RGB images, {len(depth_files)} Depth images")


    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    
    for idx, (rgb_path, depth_path, pose, w2c) in enumerate(zip(rgb_files, depth_files, poses, w2c_matrices)):
        rr.set_time("frame", sequence=idx)

        # Log della camera in Rerun
        rr.log("world/camera", pose)
        rr.log("world/camera",
            rr.Pinhole(image_from_camera=INTRINSIC, resolution=RESOLUTION))

        img_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        img_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        rr.log("world/camera/rgb", rr.Image(img_bgr, color_model="BGR").compress(jpeg_quality=90))
        rr.log("world/camera/depth", rr.DepthImage(img_depth, meter=DEPTH_SCALE))

        color_o3d = o3d.io.read_image(str(rgb_path))
        depth_o3d = o3d.io.read_image(str(depth_path))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=DEPTH_SCALE,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, w2c)

        if idx % 50 == 0:
            mesh_temp = volume.extract_triangle_mesh()
            mesh_temp.compute_vertex_normals()
            rr.log(
                "world/mesh_progressive",
                rr.Mesh3D(
                    vertex_positions=np.asarray(mesh_temp.vertices),
                    triangle_indices=np.asarray(mesh_temp.triangles),
                    vertex_colors=np.asarray(mesh_temp.vertex_colors),
                )
            )






def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the Replica office2 dataset using Rerun.")
    parser.add_argument("--frames", type=int, default=99999, help="Max number of frames to visualize")
    rr.script_add_args(parser)
    args = parser.parse_args()

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D"),
        rrb.Vertical(
            rrb.Tabs(
                rrb.Spatial2DView(name="RGB", origin="world/camera", contents=["world/camera/rgb"]),
                rrb.Spatial2DView(name="Depth", origin="world/camera", contents=["world/camera/depth"]),
                name="2D",
            ),
            row_shares=[2, 1],
        ),
    )

    rr.script_setup(args, "rerun_example_office2", default_blueprint=blueprint)
    recording_dir = Path("datasets/office2")
    log_office2(recording_dir, frames=args.frames)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()