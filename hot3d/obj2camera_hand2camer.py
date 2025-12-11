import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.rerun import Visualizer


def main(args):
    data_dir = Path(args.data_dir)
    obj_assets_dir = data_dir.parents[2] / "assets"
    image_paths = sorted(
        (path for path in data_dir.glob("214-1_*.png") if path.is_file()),
        key=lambda path: path.name,
    )
    cali_paths = [image_paths[0].parent / f"{image_path.stem}_cali.json" for image_path in image_paths]
    obj_pose_paths = [image_path.parent / f"object_poses/{image_path.stem.replace('214-1_', '')}.json" for image_path in image_paths]
    hand_pose_paths = [image_path.parent / f"hand_poses/{image_path.stem.replace('214-1_', '')}.json" for image_path in image_paths]

    vis = Visualizer(viewer_name="hot3d_in_camera", jpeg_quality=30)
    vis.log_axis("world", scale=0.1)
    logged_meshes = set()

    for idx, (image_path, cali_path, obj_pose_path) in enumerate(zip(image_paths, cali_paths, obj_pose_paths)):


        if idx % args.interval != 0:
            continue
        vis.set_time_sequence(idx)
        print(f"visualizing frame {idx}: {image_path.name}")
        with cali_path.open("r") as f:
            cali_data = json.load(f)

        c2w = np.asarray(cali_data["c2w"], dtype=float)
        intrinsics = cali_data["intrinsics"]
        fx, fy = intrinsics["focal_length"]
        cx, cy = intrinsics["principal_point"]
        resolution = intrinsics["resolution"]
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        w2c = np.linalg.inv(c2w)

        vis.log_image("world/image", str(image_path), static=False)
        vis.log_cam_pose("world/image", np.eye(4), static=False)
        vis.log_calibration("world/image", resolution, k_mat, static=False)
        if not obj_assets_dir.exists():
            print(f"[Warning] Object assets directory {obj_assets_dir} does not exist, skipping object visualization.")
            continue

        with obj_pose_path.open("r") as f:
            obj_data = json.load(f)

        for obj_id, obj_info in obj_data["objects"].items():
            obj_label = f"world/object/{obj_id}"
            mesh_path = obj_assets_dir / f"{obj_id}.glb"
            o2w = np.asarray(obj_info["T_world_object"]["matrix"], dtype=float)
            o2c = w2c @ o2w
            if mesh_path.exists():
                mesh = trimesh.load(mesh_path)
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = mesh.dump().sum()
                vertices = mesh.vertices
                # vertices = (o2w[:3, :3] @ vertices.T + o2w[:3, 3:4]).T
                vertices = (o2c[:3, :3] @ vertices.T + o2c[:3, 3:4]).T
                faces = mesh.faces
                if faces.shape[0] > 0:
                    keep = int(faces.shape[0] * 0.5)
                    if keep > 0:
                        indices = np.random.choice(faces.shape[0], keep, replace=False)
                        faces = faces[indices]
                # if obj_id not in logged_meshes:
                if 1:
                    rr.log(
                        f"{obj_label}/mesh",
                        rr.Mesh3D(vertex_positions=vertices, triangle_indices=faces),
                        static=False,
                    )
                logged_meshes.add(obj_id)
                # vis.log_cam_pose(obj_label, o2w, static=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP outputs with ReRun.")
    parser.add_argument("--data_dir", type=str, default="./dataset/P0012_ca1f6626/processed/undistorted", help="Path to the data directory.")
    parser.add_argument("--out_dir", type=str, default="./dataset/P0012_ca1f6626/processed/undistorted", help="Path to the output directory.")
    parser.add_argument("--interval", type=int, default=1, help="Interval between frames to process and save.")

    args = parser.parse_args()
    main(args)
