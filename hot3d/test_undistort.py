from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from projectaria_tools.core.image import InterpolationMethod
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

vrsfile = "./dataset/P0003_c701bd11/recording.vrs"  # replace with your VRS file path
print(f"Creating data provider from {vrsfile}")
provider = data_provider.create_vrs_data_provider(vrsfile)
if not provider:
    print("Invalid vrs data provider")

# input: retrieve image as a numpy array
sensor_name = "camera-rgb"
sensor_stream_id = provider.get_stream_id_from_label(sensor_name)
image_data = provider.get_image_data_by_index(sensor_stream_id, 0)
image_array = image_data[0].to_numpy_array()
# input: retrieve image distortion
device_calib = provider.get_device_calibration()
src_calib = device_calib.get_camera_calib(sensor_name)




## Check device version
# Example variables used in this notebook
rgb_stream_id = StreamId('214-1')

# Some example variables are different for Gen1 and Gen2,
# because they have different HW configs, sensor label names, etc.

# Gen1 images are rotated 90 degrees for better visualization
ROTATE_90_FLAG = True

# A linear camera model used in undistortion example: [width, height, focal]
example_linear_rgb_camera_model_params = [800, 800, 200]


# create output calibration: a linear model of image example_linear_rgb_camera_model_params.
# Invisible pixels are shown as black.
camera_name = "camera-rgb"
dst_calib = calibration.get_linear_camera_calibration(example_linear_rgb_camera_model_params[0], example_linear_rgb_camera_model_params[1], example_linear_rgb_camera_model_params[2], camera_name)

# distort image
rectified_array = calibration.distort_by_calibration(image_array, dst_calib, src_calib, InterpolationMethod.BILINEAR)

# visualize input and results
plt.figure()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Image undistortion (focal length = {dst_calib.get_focal_lengths()})")

axes[0].imshow(image_array, cmap="gray", vmin=0, vmax=255)
axes[0].title.set_text(f"sensor image ({sensor_name})")
axes[0].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
axes[1].imshow(rectified_array, cmap="gray", vmin=0, vmax=255)
axes[1].title.set_text(f"undistorted image ({sensor_name})")
axes[1].tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
plt.show()