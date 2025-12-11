import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


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
    points_h = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
    pixels_h = (K @ points_h.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels


def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_pose_in_cam_path = data_dir / "object_poses_in_cam"
    project_dir = out_dir / "project_in_cam"
    project_dir.mkdir(parents=True, exist_ok=True)

    invalid_frames = load_invalid_frames(data_dir)
    images = sorted(data_dir.glob("214-1_*.png"))

    for image_path in images:
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

        out_path = project_dir / f"{frame_idx}.png"
        cv2.imwrite(str(out_path), overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sample_vertices", type=int, default=500)
    args = parser.parse_args()
    main(args)
