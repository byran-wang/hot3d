# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from typing import Optional, Type

import rerun as rr  # @manual
from data_loaders.loader_hand_poses import HandType
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import loadManoHandModel
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
try:
    from dataset_api import Hot3dDataProvider  # @manual
except ImportError:
    from hot3d.dataset_api import Hot3dDataProvider

try:
    from Hot3DVisualizer_stereo import Hot3DVisualizer
except ImportError:
    from hot3d.Hot3DVisualizer_stereo import Hot3DVisualizer

from tqdm import tqdm
import cv2
import numpy as np
import json

import sys
from pathlib import Path
sys.path.append('third_party/utils_simba')
from utils_simba.depth import get_depth
from utils_simba.vis import rotation_matrix_to_quaternion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="path to hot3d data sequence",
        required=True,
    )
    parser.add_argument(
        "--object_library_folder",
        type=str,
        help="path to object library folder containing instance.json and *.glb cad files",
        required=True,
    )
    parser.add_argument(
        "--mano_model_folder",
        type=str,
        default=None,
        help="path to MANO models containing the MANO_RIGHT/LEFT.pkl files",
        required=False,
    )
    parser.add_argument(
        "--hand_type",
        type=str,
        default="UMETRACK",
        help="type of HAND (MANO or UMETRACK)",
        required=False,
    )

    parser.add_argument("--jpeg_quality", type=int, default=75, help=argparse.SUPPRESS)

    # If this path is set, we will save the rerun (.rrd) file to the given path
    parser.add_argument(
        "--rrd_output_path", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./stereo",
        help="path to output directory to save stereo images",
        required=False,
    )
    parser.add_argument(
        "--headless",
        action='store_true',
        help="run in headless mode without visualizing in the rerun app",
    )

    return parser.parse_args()


def log_points(label: str, 
                points: np.ndarray, 
                colors: np.ndarray = None, 
                sizes: np.ndarray = None,
                radii: float = 0.001,
                static=False,
                ) -> None:
    if colors is None:
        rr.log(label, rr.Points3D(positions=points, radii=radii), static=static)
    else:
        if sizes is None:
            rr.log(label, rr.Points3D(positions=points, colors=colors, radii=radii), static=static)
        else:
            rr.log(label, rr.Points3D(positions=points, colors=colors, radii=sizes), static=static)

def log_image(self,label: str, 
                image_file: str, 
                jpeg_quality: int = 75, 
                static=False,
                ) -> None:
    assert os.path.exists(image_file), f"Image file {image_file} does not exist"
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rr.log(label, rr.Image(image).compress(jpeg_quality=jpeg_quality), static=static)

def unproject_to_cam(xy_depth, K):
    """Unproject 2D points with depth to 3D points in camera coordinates."""
    points_3d = np.linalg.inv(K) @ xy_depth
    points_3d = np.vstack((points_3d, np.ones((1, points_3d.shape[1]))))
    return points_3d.T

def unproject_to_world(xy_depth, K, H):
    """Unproject 2D points with depth to 3D points in world coordinates."""
    points_3d = unproject_to_cam(xy_depth, K)
    points_3d = (H @ points_3d.T).T[:, :3]
    return points_3d

def unproject_depth_map_to_world(depth, K, H, mask=None):
    """Unproject depth map to 3D points in world coordinates."""
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()
    if mask is not None:
        mask = mask.flatten()
        x = x[mask]
        y = y[mask]
        depth = depth[mask]

    xy_depth = np.vstack((x * depth, y * depth, depth))
    points_3d = unproject_to_world(xy_depth, K, H)
    return points_3d            

def execute_rerun(
    sequence_folder: str,
    object_library_folder: str,
    mano_model_folder: Optional[str],
    rrd_output_path: Optional[str],
    jpeg_quality: int,
    timestamps_slice: Type[slice],
    fail_on_missing_data: bool,
    hand_type: str,
    out_dir: str,
    headless: bool,
):
    if not os.path.exists(sequence_folder):
        raise RuntimeError(f"Sequence folder {sequence_folder} does not exist")
    if not os.path.exists(object_library_folder):
        raise RuntimeError(
            f"Object Library folder {object_library_folder} does not exist"
        )

    hand_enum_type = HandType.Umetrack if hand_type == "UMETRACK" else HandType.Mano
    if hand_type not in ["UMETRACK", "MANO"]:
        raise RuntimeError(
            f"Invalid hand type: {hand_type}. hand_type must be either UMETRACK or MANO"
        )

    object_library = load_object_library(
        object_library_folderpath=object_library_folder
    )

    mano_hand_model = loadManoHandModel(mano_model_folder)

    #
    # Initialize hot3d data provider
    #
    data_provider = Hot3dDataProvider(
        sequence_folder=sequence_folder,
        object_library=object_library,
        mano_hand_model=mano_hand_model,
        fail_on_missing_data=fail_on_missing_data,
    )
    print(f"data_provider statistics: {data_provider.get_data_statistics()}")
    #
    # Prepare the rerun rerun log configuration
    #

    rr.init("hot3d Data Viewer", spawn=(rrd_output_path is None))
    
    if rrd_output_path is not None:
        print(f"Saving .rrd file to {rrd_output_path}")
        rr.save(rrd_output_path)

    #
    # Initialize the rerun hot3d visualizer interface
    #
    os.makedirs(out_dir, exist_ok=True)
    rr_visualizer = Hot3DVisualizer(data_provider, hand_enum_type, out_dir=out_dir)


    # Define which image stream will be shown
    image_stream_ids = data_provider.device_data_provider.get_image_stream_ids()

    #
    # Log static assets (aka Timeless assets)

    rr_visualizer.log_static_assets(image_stream_ids)

    timestamps = data_provider.device_data_provider.get_sequence_timestamps()
    #
    # Visualize the dataset sequence
    #
    # Loop over the timestamps of the sequence and visualize corresponding data
    idx_range = list(range(len(timestamps[timestamps_slice])))[200::10]
    for idx, timestamp in enumerate(tqdm(timestamps[timestamps_slice])):
        # if idx <= 308:  # for testing purposes only
        #     continue
        # rr.set_time_nanos("synchronization_time", int(timestamp))
        # rr.set_time_sequence("timestamp", timestamp)
        if idx == 0:
            object_poses_with_dt = (
                    rr_visualizer._object_pose_data_provider.get_pose_at_timestamp(
                    timestamp_ns=timestamp,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE,
                    acceptable_time_delta=0,
                )
            )

            Hot3DVisualizer.log_object_poses(
                "world/objects",
                object_poses_with_dt,
                rr_visualizer._object_pose_data_provider,
                rr_visualizer._object_library,
                rr_visualizer._object_cache_status,
            )

        if idx not in idx_range:
            continue
        print(f"Processing frame idx: {idx}, timestamp: {timestamp}")

        base_dir = "/home/simba/Documents/dataset/HOT3D/P0003_c701bd11/processed"

        dmap_rescale = 1.0


        image_path = f"{base_dir}/images_raw/{idx:04d}.png"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            None,
            fx=dmap_rescale,
            fy=dmap_rescale,
            interpolation=cv2.INTER_NEAREST,
        )

        depth_path = f"{base_dir}/depth_fs/{idx:04d}.png"
        depth = get_depth(depth_path)
        depth = cv2.resize(
            depth,
            None,
            fx=dmap_rescale,
            fy=dmap_rescale,
            interpolation=cv2.INTER_NEAREST,
        )

        cali_path = f"{base_dir}/stereo/214-1_1201-2/{idx:04d}_cali.json"
        with open(cali_path, "r") as f:
            cali = json.load(f)
            left_K = cali.get("left_intrinsic")
            K = np.eye(3)
            K[0, 0] = left_K["focal_length"][0]
            K[1, 1] = left_K["focal_length"][1]
            K[0, 2] = left_K["principal_point"][0]
            K[1, 2] = left_K["principal_point"][1]
            K[:2, :] *= dmap_rescale

            c2w = np.asarray(cali.get("left_c2w"), dtype=float)

        mask_path = f"{base_dir}/images/{idx:04d}.png"
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)[...,3]  # alpha channel   
        mask = cv2.resize(
            mask,
            None,
            fx=dmap_rescale,
            fy=dmap_rescale,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        points_3d = unproject_depth_map_to_world(depth, K, c2w, mask=mask)

        log_points(f"world/cube/{idx:04d}", points_3d, colors=image[mask], radii=0.0003, static=True)
        # rr.log(f"world/cameras/{idx:04d}", rr.Image(image).compress(jpeg_quality=jpeg_quality), static=True)
        rr.log(
            f"world/cameras/{idx:04d}",
            rr.Pinhole(
                resolution=image.shape[:2],
                focal_length=[K[0,0], K[1,1]],
                principal_point=[K[0,2], K[1,2]],
                image_plane_distance=0.02,
            ),
            static=True,
        )
        tvec = c2w[:3, 3]
        quat_xyzw = rotation_matrix_to_quaternion(c2w[:3, :3])
        rr.log(f"world/cameras/{idx:04d}", rr.Transform3D(translation=tvec, rotation=rr.Quaternion(xyzw=quat_xyzw)), static=True)


        




        # rr_visualizer.log_dynamic_assets(image_stream_ids, timestamp, frame_idx=idx, headless=headless)



def main():
    args = parse_args()
    print(f"args provided: {args}")
    try:
        execute_rerun(
            sequence_folder=args.sequence_folder,
            object_library_folder=args.object_library_folder,
            mano_model_folder=args.mano_model_folder,
            rrd_output_path=args.rrd_output_path,
            jpeg_quality=args.jpeg_quality,
            timestamps_slice=slice(None, None, None),
            fail_on_missing_data=False,
            hand_type=args.hand_type,
            out_dir=args.out_dir,
            headless=args.headless
        )
    except Exception as error:
        print(f"An exception occurred: {error}")


if __name__ == "__main__":
    main()
