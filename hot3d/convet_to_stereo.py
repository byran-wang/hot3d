from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class CameraData:
    stream_id: str
    image: np.ndarray
    image_size: Tuple[int, int]  # (width, height)
    intrinsic_matrix: np.ndarray  # 3x3
    extrinsics_world_from_cam: np.ndarray  # 4x4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hot3D triple cameras into rectified stereo pairs."
    )
    parser.add_argument(
        "--stereo_dir",
        type=Path,
        default=Path("./stereo"),
        help="Directory containing per-stream PNGs and *_calibration.json files.",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("./stereo_pairs"),
        help="Output directory for rectified stereo pairs.",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="1201-1:214-1,214-1:1201-2",
        help="Comma separated list of left:right stream id pairs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Stereo rectification zoom parameter passed to cv2.stereoRectify.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=640,
        help="Maximum size (width or height) for the output images.",
    )   
    return parser.parse_args()


def load_camera(stream_id: str, stereo_dir: Path) -> CameraData:
    image_path = stereo_dir / f"{stream_id}.png"
    calibration_path = stereo_dir / f"{stream_id}_calibration.json"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for {stream_id}: {image_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Missing calibration for {stream_id}: {calibration_path}"
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read {image_path}")

    with calibration_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    intrinsics = data["intrinsics"]
    fx, fy = intrinsics["focal_length"]
    cx, cy = intrinsics["principal_point"]
    width, height = intrinsics["resolution"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    extrinsics = np.array(data["extrinsics"], dtype=np.float64)

    return CameraData(
        stream_id=stream_id,
        image=image,
        image_size=(int(width), int(height)),
        intrinsic_matrix=K,
        extrinsics_world_from_cam=extrinsics,
    )


def invert_transform(world_from_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return rotation (world->cam) and translation for cv2 (camera frame)."""
    R_wc = world_from_cam[:3, :3]
    t_wc = world_from_cam[:3, 3:4]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def _resize_image_and_intrinsics(
    camera: CameraData, target_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if camera.image_size == target_size:
        return camera.image, camera.intrinsic_matrix

    target_w, target_h = target_size
    src_w, src_h = camera.image_size
    scale_x = target_w / src_w
    scale_y = target_h / src_h

    resized = cv2.resize(camera.image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scaled_K = camera.intrinsic_matrix.copy()
    scaled_K[0, 0] *= scale_x
    scaled_K[0, 2] *= scale_x
    scaled_K[1, 1] *= scale_y
    scaled_K[1, 2] *= scale_y
    return resized, scaled_K


def stereo_rectify_pair(
    left: CameraData,
    right: CameraData,
    save_dir: Path,
    alpha: float,
    max_size: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    target_size = (
        min(max(left.image_size[0], right.image_size[0]), max_size),
        min(max(left.image_size[1], right.image_size[1]), max_size),
    )
    left_image, left_K = _resize_image_and_intrinsics(left, target_size)
    right_image, right_K = _resize_image_and_intrinsics(right, target_size)
    # plot the left and right image with plt
    if 0:
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Left: {left.stream_id}")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Right: {right.stream_id}")
        plt.show()
    if 0:
        R_cw_l, t_cw_l = invert_transform(left.extrinsics_world_from_cam)
        R_cw_r, t_cw_r = invert_transform(right.extrinsics_world_from_cam)

        R = R_cw_r @ np.linalg.inv(R_cw_l)
        T = (
            R_cw_r
            @ (
                left.extrinsics_world_from_cam[:3, 3]
                - right.extrinsics_world_from_cam[:3, 3]
            )
        ).reshape(3, 1)
    else:
        R_T = np.linalg.inv(right.extrinsics_world_from_cam) @ left.extrinsics_world_from_cam
        R = R_T[:3, :3]
        T = R_T[:3, 3:].reshape(3, 1)
        
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=left_K,
        distCoeffs1=None,
        cameraMatrix2=right_K,
        distCoeffs2=None,
        imageSize=target_size,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
    )

    map1_x, map1_y = cv2.initUndistortRectifyMap(
        cameraMatrix=left_K,
        distCoeffs=None,
        R=R1,
        newCameraMatrix=P1[:, :3],
        size=target_size,
        m1type=cv2.CV_32FC1,
    )
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        cameraMatrix=right_K,
        distCoeffs=None,
        R=R2,
        newCameraMatrix=P2[:, :3],
        size=target_size,
        m1type=cv2.CV_32FC1,
    )

    left_rect = cv2.remap(left_image, map1_x, map1_y, interpolation=cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, map2_x, map2_y, interpolation=cv2.INTER_LINEAR)

    left_output_path = save_dir / f"{left.stream_id}_rectified.png"
    right_output_path = save_dir / f"{right.stream_id}_rectified.png"
    cv2.imwrite(str(left_output_path), left_rect)
    cv2.imwrite(str(right_output_path), right_rect)

    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    baseline = float(-P2[0, 3] / P2[0, 0])

    rectified_intrinsics = {
        "resolution": [int(target_size[0]), int(target_size[1])],
        "focal_length": [fx, fy],
        "principal_point": [cx, cy],
    }
    rectified_extrinsics = {
        "left": np.eye(4).tolist(),
        "right": np.array(
            [[1.0, 0.0, 0.0, baseline], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        ).tolist(),
    }

    metadata = {
        "pair": [left.stream_id, right.stream_id],
        "intrinsics": rectified_intrinsics,
        "extrinsics": rectified_extrinsics,
        "projection_matrices": {
            "left": P1.tolist(),
            "right": P2.tolist(),
        },
        "rectification_rotation": {
            "left": R1.tolist(),
            "right": R2.tolist(),
        },
        "Q": Q.tolist(),
        "baseline_m": baseline,
    }

    with (save_dir / "rectified_calibration.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    intrinsic_pickle_path = save_dir / "0000.pkl"
    K = np.array([[fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
        ],dtype=np.float32)
    with intrinsic_pickle_path.open("wb") as file:
        pickle.dump(
            {
                "stereo_camMat": K.tolist(),
                "stereo_baseline": baseline,
            },
            file,
        )


def main(args: argparse.Namespace) -> None:
    stereo_dir: Path = args.stereo_dir
    save_root: Path = args.save_dir
    pairs = [tuple(pair.split(":")) for pair in args.pairs.split(",") if pair]
    if not pairs:
        raise ValueError("No stereo pairs were provided.")

    save_root.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, CameraData] = {}
    for left_id, right_id in pairs:
        if left_id not in cache:
            cache[left_id] = load_camera(left_id, stereo_dir)
        if right_id not in cache:
            cache[right_id] = load_camera(right_id, stereo_dir)

        pair_dir = save_root / f"{left_id}_{right_id}"
        stereo_rectify_pair(
            left=cache[left_id],
            right=cache[right_id],
            save_dir=pair_dir,
            alpha=args.alpha,
            max_size=args.max_size,
        )


if __name__ == "__main__":
    main(parse_args())
