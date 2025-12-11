import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
import trimesh
import torch

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.rerun import Visualizer
from data_loaders.mano_layer import loadManoHandModel
from data_loaders.pytorch3d_rotation.rotation_conversions import matrix_to_axis_angle
from data_loaders.HandDataProviderBase import HandDataProviderBase


def main(args):
    data_dir = Path(args.data_dir)
    obj_assets_dir = data_dir.parents[2] / "assets"
    mano_model_dir = data_dir.parents[2] / "body_models"
    invalid_path = data_dir / "invalid_frames.txt"
    obj_pose_in_cam_path = data_dir / "object_poses_in_cam"
    hand_pose_in_cam_path = data_dir / "hand_poses_in_cam"
    obj_pose_in_cam_path.mkdir(parents=True, exist_ok=True)
    hand_pose_in_cam_path.mkdir(parents=True, exist_ok=True)

    mano_model = loadManoHandModel(str(mano_model_dir))
    invalid_frames = set()
    if invalid_path.exists():
        with invalid_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    invalid_frames.add(int(line))
                except ValueError:
                    continue
    image_paths = sorted(
        (path for path in data_dir.glob("214-1_*.png") if path.is_file()),
        key=lambda path: path.name,
    )
    cali_paths = [image_paths[0].parent / f"{image_path.stem}_cali.json" for image_path in image_paths]
    obj_pose_paths = [image_path.parent / f"object_poses/{image_path.stem.replace('214-1_', '')}.json" for image_path in image_paths]
    hand_pose_paths = [image_path.parent / f"hand_poses/{image_path.stem.replace('214-1_', '')}.json" for image_path in image_paths]

    vis = Visualizer(viewer_name="hot3d_in_camera", jpeg_quality=30)
    if not args.headless:
        vis.log_axis("world", scale=0.1)
    logged_meshes = set()

    for idx, (image_path, cali_path, obj_pose_path, hand_pose_path) in enumerate(
        zip(image_paths, cali_paths, obj_pose_paths, hand_pose_paths)
    ):
        if idx % args.interval != 0:
            continue        
        frame_num = None
        stem_parts = image_path.stem.split("_")
        if stem_parts and stem_parts[-1].isdigit():
            frame_num = int(stem_parts[-1])
        if frame_num is not None and frame_num in invalid_frames:
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

        if not args.headless:
            vis.log_image("world/image", str(image_path), static=False)
            # vis.log_cam_pose("world/image", c2w, static=False)
            vis.log_cam_pose("world/image", np.eye(4), static=False)
            vis.log_calibration("world/image", resolution, k_mat, static=False)
        if not obj_assets_dir.exists():
            print(f"[Warning] Object assets directory {obj_assets_dir} does not exist, skipping object visualization.")
            continue

        with obj_pose_path.open("r") as f:
            obj_data = json.load(f)

        obj_cam_payload = {"objects": {}, "k_mat": k_mat.tolist()}
        for obj_id, obj_info in obj_data["objects"].items():
            obj_label = f"world/object/{obj_id}"
            mesh_path = obj_assets_dir / f"{obj_id}.glb"
            o2w = np.asarray(obj_info["T_world_object"]["matrix"], dtype=float)
            o2c = w2c @ o2w
            obj_cam_payload["objects"][obj_id] = {
                "T_camera_object": {"matrix": o2c.tolist()}
            }
            if mesh_path.exists() and not args.headless:
                mesh = trimesh.load(mesh_path)
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = mesh.dump().sum()
                vertices = mesh.vertices
                obj_normals = mesh.vertex_normals
                obj_colors = None
                if (
                    hasattr(mesh, "visual")
                    and mesh.visual.kind == "vertex"
                    and mesh.visual.vertex_colors is not None
                ):
                    obj_colors = np.asarray(mesh.visual.vertex_colors)
                # vertices = (o2w[:3, :3] @ vertices.T + o2w[:3, 3:4]).T
                vertices = (o2c[:3, :3] @ vertices.T + o2c[:3, 3:4]).T
                obj_normals = (o2c[:3, :3] @ obj_normals.T).T
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
                        rr.Mesh3D(
                            vertex_positions=vertices,
                            triangle_indices=faces,
                            vertex_normals=obj_normals,
                            vertex_colors=obj_colors,
                        ),
                        static=False,
                    )
                logged_meshes.add(obj_id)
                # vis.log_cam_pose(obj_label, o2w, static=False)
        obj_pose_cam_path = obj_pose_in_cam_path / obj_pose_path.name
        with obj_pose_cam_path.open("w") as f:
            json.dump(obj_cam_payload, f, indent=2)

        with hand_pose_path.open("r") as f:
            hand_pose_data = json.load(f)
        hands_data = hand_pose_data["hand_poses"]

        hand_cam_payload = {"hand_poses": {}, "k_mat": k_mat.tolist()}
        for hand_id, hand_info in hands_data.items():
            wrist_world = np.array(hand_info["wrist_pose"]["matrix"], dtype=float)
            joint_angles = hand_info["joint_angles"]
            axis_angle = matrix_to_axis_angle(torch.from_numpy(wrist_world[:3, :3]).float())
            global_vec = torch.cat([axis_angle, torch.from_numpy(wrist_world[:3, 3]).float()])
            # betas = hand_info.get("betas")
            shape_params = torch.tensor(np.array(hand_info["betas"]), dtype=torch.float32)
            is_right = torch.tensor(
                [str(hand_info.get("handedness", hand_id)).lower() in ("1", "right")], dtype=torch.bool
            )
            wrist_cam = w2c @ wrist_world
            hand_cam_payload["hand_poses"][hand_id] = {
                "handedness": hand_info.get("handedness"),
                "wrist_pose": {"matrix": wrist_world.tolist()},
                "w2c": {"matrix": w2c.tolist()},
                "joint_angles": joint_angles,
                "betas": hand_info.get("betas"),
            }
            if args.headless:
                continue
            vertices, _ = mano_model.forward_kinematics(
                shape_params, torch.tensor(joint_angles, dtype=torch.float32), global_vec, is_right
            )
            vertices = (w2c[:3, :3] @ vertices.numpy().T + w2c[:3, 3:4]).T
            faces = mano_model.mano_layer_right.faces if is_right[0] else mano_model.mano_layer_left.faces
            hand_normals = HandDataProviderBase.get_triangular_mesh_normals(vertices, faces)
            color_rgba = np.array([255, 80, 80, 255] if is_right[0] else [80, 140, 255, 255], dtype=np.uint8)
            hand_colors = np.repeat(color_rgba[None, :], vertices.shape[0], axis=0)
            rr.log(
                f"world/hand/{'right' if is_right[0] else 'left'}/mesh",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_normals=hand_normals,
                    vertex_colors=hand_colors,
                ),
                static=False,
            )
        hand_pose_cam_path = hand_pose_in_cam_path / hand_pose_path.name
        with hand_pose_cam_path.open("w") as f:
            json.dump(hand_cam_payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP outputs with ReRun.")
    parser.add_argument("--data_dir", type=str, default="./dataset/P0012_ca1f6626/processed/undistorted", help="Path to the data directory.")
    parser.add_argument("--out_dir", type=str, default="./dataset/P0012_ca1f6626/processed/undistorted", help="Path to the output directory.")
    parser.add_argument("--interval", type=int, default=1, help="Interval between frames to process and save.")
    parser.add_argument("--headless", action="store_true", help="If set, skip logging camera/images/meshes.")

    args = parser.parse_args()
    main(args)
