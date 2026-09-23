#!/usr/bin/env python3
"""Convert a ROS1 rosbag into a standard NuScenes v1.0-trainval dataset.

    python bag2nuscenes.py /path/to.bag --out /data/tcar_nuscenes --calib /path/to/calib

Shaped like the real nuScenes so the devkit and nuScenes-based training code
work unchanged:
  - LIDAR_TOP (/middle/rslidar_points) and the six standard cameras, with
    images rectified to pinhole so `camera_intrinsic` is exact;
  - 2 Hz samples, every LiDAR frame and every 30 fps camera frame as sweeps;
  - scenes of exactly 20 s, named from the official nuScenes train/val lists;
  - GNSS/INS as CAN bus expansion files (can_bus/<scene>_pose.json, _ms_imu.json).

The traffic-light camera (camera_4), the bottom LiDARs, the radar and all other
topics are left out; bag2nuscenes_full.py carries them.

Running it again on another bag appends: scene names continue down the
official list, sensor and category tokens are reused, and re-importing the
same bag is refused.
"""
from converter import STANDARD, run

if __name__ == "__main__":
    run(STANDARD, __doc__)
