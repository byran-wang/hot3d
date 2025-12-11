import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.append("./third_party/utils_simba")
from utils_simba.rerun import Visualizer


def main(args):
    data_dir = Path(args.data_dir)
    obj_assets_dir = Path(args.data_dir) / "assets"
    image_paths = sorted(
        (path for path in data_dir.glob("214-1_*.png") if path.is_file()),
        key=lambda path: path.name,
    )
    cali_paths = [image_paths[0].parent / f"{image_path.stem}_cali.json" for image_path in image_paths]
    obj_pose_paths = [image_path.parent / f"object_poses/{image_path.stem}.json" for image_path in image_paths]

    vis = Visualizer(viewer_name="hot3d_in_camera", jpeg_quality=30)
    for idx, (image_path, cali_path) in enumerate(zip(image_paths, cali_paths)):
        vis.set_time_sequence(idx)

        with cali_path.open("r") as f:
            cali_data = json.load(f)

        c2w = np.asarray(cali_data["c2w"], dtype=float)
        intrinsics = cali_data["intrinsics"]
        fx, fy = intrinsics["focal_length"]
        cx, cy = intrinsics["principal_point"]
        resolution = intrinsics["resolution"]
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

        vis.log_image("world/image", str(image_path))
        vis.log_cam_pose("world/image", c2w)
        vis.log_calibration("world/image", resolution, k_mat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COLMAP outputs with ReRun.")
    parser.add_argument("--data_dir", type=str, default="./dataset/P0012_ca1f6626/processed/undistorted", help="Path to the data directory.")
    parser.add_argument("--out_dir", type=str, default="./dataset/P0012_ca1f6626/processed/undistorted", help="Path to the output directory.")

    args = parser.parse_args()
    main(args)
