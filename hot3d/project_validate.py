import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))
from data_loaders.mano_layer import loadManoHandModel
from data_loaders.pytorch3d_rotation.rotation_conversions import matrix_to_axis_angle


def load_invalid_frames(data_dir: Path) -> set[int]:
    invalid_path = data_dir / "invalid_frames.txt"
    invalid = set()
    if invalid_path.exists():
        for line in invalid_path.read_text().splitlines():
            line = line.strip()
            if line.isdigit():
                invalid.add(int(line))
    return invalid


def project_points(K: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    pixels_h = (K @ points_cam.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_pose_in_cam_path = data_dir / "object_poses_in_cam"
    hand_pose_in_cam_path = data_dir / "hand_poses_in_cam"
    project_dir = out_dir / "project_in_cam"
    project_dir.mkdir(parents=True, exist_ok=True)
    mano_model_dir = data_dir.parents[2] / "body_models"
    mano_model = loadManoHandModel(str(mano_model_dir))

    invalid_frames = load_invalid_frames(data_dir)
    images = sorted(data_dir.glob("214-1_*.png"))

    for image_path in tqdm(images):
        parts = image_path.stem.split("_")
        frame_idx = parts[-1] if parts else image_path.stem
        if frame_idx.isdigit() and int(frame_idx) in invalid_frames:
            continue

        obj_pose_file = obj_pose_in_cam_path / f"{frame_idx}.json"
        if not obj_pose_file.exists():
            continue

        with obj_pose_file.open("r") as f:
            pose_payload = json.load(f)

        k_mat = np.array(pose_payload.get("k_mat"), dtype=float)
        objects = pose_payload.get("objects", {})

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        overlay = image.copy()

        for obj_id, obj_entry in objects.items():
            mesh_path = data_dir.parents[2] / "assets" / f"{obj_id}.glb"
            if not mesh_path.exists():
                continue
            mesh = trimesh.load(mesh_path)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()

            vertices = mesh.vertices
            faces = mesh.faces
            if faces.shape[0] > 0:
                count = min(args.sample_vertices, vertices.shape[0])
                indices = np.random.choice(vertices.shape[0], count, replace=False)
                vertices = vertices[indices]

            o2c = np.array(obj_entry["T_camera_object"]["matrix"], dtype=float)
            verts_cam = (o2c[:3, :3] @ vertices.T + o2c[:3, 3:4]).T
            verts_2d = project_points(k_mat, verts_cam)

            for px, py in verts_2d.astype(int):
                cv2.circle(overlay, (px, py), 2, (0, 255, 0), -1)

        hand_pose_file = hand_pose_in_cam_path / f"{frame_idx}.json"
        if mano_model is not None and hand_pose_file.exists():
            with hand_pose_file.open("r") as f:
                hand_payload = json.load(f)
            hand_poses = hand_payload.get("hand_poses", {})
            k_hand = np.array(hand_payload.get("k_mat", k_mat.tolist()), dtype=float)
            for hand_id, hand_info in hand_poses.items():
                wrist_world = np.array(hand_info["wrist_pose"]["matrix"], dtype=float)
                w2c = np.array(hand_info["w2c"]["matrix"], dtype=float)
                joint_angles = hand_info.get("joint_angles", [])
                if len(joint_angles) < mano_model.num_pose_coeffs:
                    joint_angles = (joint_angles + [0.0] * mano_model.num_pose_coeffs)[
                        : mano_model.num_pose_coeffs
                    ]
                betas = hand_info.get("betas")
                if betas is None:
                    betas = [0.0] * mano_model.num_shape_params
                shape_params = torch.tensor(betas, dtype=torch.float32)
                axis_angle = matrix_to_axis_angle(
                    torch.from_numpy(wrist_world[:3, :3]).float()
                )
                global_vec = torch.cat(
                    [axis_angle, torch.from_numpy(wrist_world[:3, 3]).float()]
                )
                is_right = torch.tensor(
                    [
                        str(hand_info.get("handedness", hand_id)).lower()
                        in ("1", "right")
                    ],
                    dtype=torch.bool,
                )
                verts, _ = mano_model.forward_kinematics(
                    shape_params,
                    torch.tensor(joint_angles, dtype=torch.float32),
                    global_vec,
                    is_right,
                )
                w2c = torch.from_numpy(w2c).float()
                verts = (w2c[:3, :3] @ verts.T + w2c[:3, 3:4]).T
                verts_np = verts.numpy()
                if verts_np.shape[0] > args.sample_hand_vertices:
                    idxs = np.random.choice(
                        verts_np.shape[0], args.sample_hand_vertices, replace=False
                    )
                    verts_np = verts_np[idxs]
                verts_2d = project_points(k_hand, verts_np)
                color = (0, 0, 255) if is_right[0] else (255, 0, 0)
                for px, py in verts_2d.astype(int):
                    cv2.circle(overlay, (px, py), 2, color, -1)

        out_path = project_dir / f"{frame_idx}.png"
        cv2.imwrite(str(out_path), overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_vertices", type=int, default=500)
    parser.add_argument("--sample_hand_vertices", type=int, default=500)
    args = parser.parse_args()
    main(args)
