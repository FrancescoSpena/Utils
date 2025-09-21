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

            # # Replica fix
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1

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


def visible_points(points_world: np.ndarray, colors: np.ndarray,
                   c2w: np.ndarray, intrinsics: np.ndarray,
                   width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restituisce indici, punti e colori visibili dalla camera.
    """


    homog = np.hstack([points_world, np.ones((points_world.shape[0], 1))])
    points_cam = (c2w @ homog.T).T[:, :3]

    # davanti alla camera
    mask = points_cam[:, 2] > 0.1

    # proiezione
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    u = (fx * points_cam[:, 0] / points_cam[:, 2] + cx).astype(int)
    v = (fy * points_cam[:, 1] / points_cam[:, 2] + cy).astype(int)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.where(mask & inside)[0]

    return idx, points_world[idx], colors[idx]





def log_office2(recording_dir: Path, frames: int) -> None:
    rr.log("description", rr.TextDocument("Replica office2 dataset", media_type=rr.MediaType.MARKDOWN), static=True)

    print(f"[DEBUG] Using dataset folder: {recording_dir}")
    print(f"[DEBUG] Looking for traj.txt in: {recording_dir / 'traj.txt'}")
    print(f"[DEBUG] Looking for mesh in: {recording_dir / 'office2_default.ply'}")
    print(f"[DEBUG] Looking for images in: {recording_dir / 'results'}")

    
    # Paths
    traj_file = recording_dir / "traj.txt"
    mesh_file = recording_dir / "office2_default.ply"
    results_dir = recording_dir / "results"

    # Carica poses
    poses = load_poses(traj_file)

    # Carica mesh
    print(f"[DEBUG] Loaded {len(poses)} poses")
    if mesh_file.exists():
        mesh = trimesh.load(str(mesh_file))
        print(f"[DEBUG] Mesh loaded with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        rr.log(
            "world/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                vertex_colors=mesh.visual.vertex_colors,
                triangle_indices=mesh.faces,
            ),
            static=True,
        )

    # Carica immagini
    depth_files = sorted(results_dir.glob("depth*.png"))
    rgb_files   = sorted(results_dir.glob("frame*.jpg")) + sorted(results_dir.glob("frame*.jpeg"))
    rgb_files   = sorted(rgb_files)

    print(f"[DEBUG] Found {len(rgb_files)} RGB images, {len(depth_files)} Depth images")

    if frames < len(rgb_files):
        rgb_files = rgb_files[:frames]
        depth_files = depth_files[:frames]
        poses = poses[:frames]
    
    ply_file = recording_dir / "final_point_cloud.ply"
    points_world, colors = None, None

    if ply_file.exists():
        cloud = trimesh.load(str(ply_file))
        points_world = np.array(cloud.vertices)  # [N, 3] in world coordinates
        if hasattr(cloud.visual, "vertex_colors"):
            colors = np.array(cloud.visual.vertex_colors[:, :3]) / 255.0
        else:
            colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(points_world), 1))  # grigio se no colori
        print(f"[DEBUG] Loaded point cloud: {points_world.shape[0]} punti")
    else:
        print("[WARNING] final_point_cloud.ply non trovato!")
    


    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    seen_faces = set()

    
    for idx, (rgb_path, depth_path, pose) in enumerate(zip(rgb_files, depth_files, poses)):
        rr.set_time("frame", sequence=idx)

        rr.log("world/camera", pose)
        rr.log("world/camera",
            rr.Pinhole(image_from_camera=INTRINSIC, resolution=RESOLUTION))

        img_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        img_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        rr.log("world/camera/rgb", rr.Image(img_bgr, color_model="BGR").compress(jpeg_quality=90))
        rr.log("world/camera/depth", rr.DepthImage(img_depth, meter=DEPTH_SCALE))


        if points_world is not None:
            vals = [float(x) for x in open(traj_file).readlines()[idx].split()]
            c2w = np.array(vals, dtype=np.float32).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1

            visible_idx, _, _ = visible_points(mesh.vertices, mesh.vertices, c2w,
                                                        INTRINSIC, RESOLUTION[0], RESOLUTION[1])

            visible_face_mask = np.isin(mesh.faces, visible_idx).any(axis=1)
            new_faces = np.where(visible_face_mask)[0]
            seen_faces.update(new_faces)

            progressive_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[list(seen_faces)],
                process=False
            )

            rr.log(
                "world/progressive_mesh",
                rr.Mesh3D(
                    vertex_positions=progressive_mesh.vertices,
                    triangle_indices=progressive_mesh.faces,
                    vertex_colors=mesh.visual.vertex_colors if hasattr(mesh.visual, "vertex_colors") else None
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