#!/usr/bin/env python3
"""Convert a ROS1 rosbag into a NuScenes dataset carrying every sensor and topic.

    python bag2nuscenes_full.py /path/to.bag --out /data/tcar_nuscenes_full --calib /path/to/calib

The standard converter (bag2nuscenes.py) with more kinds of data: the same
frame selection, the same 20 s scenes and scene names, the same rectified six
cameras, LIDAR_TOP and CAN bus files, plus

  - CAM_TRAFFIC (camera_4, rectified), best-effort: in a sample when its frame
    is within --sync-ms, never a reason to drop one;
  - LIDAR_BOTTOM_FRONT/REAR (M1) and LIDAR_BOTTOM_LEFT/RIGHT (Bpearl) as .pcd.bin;
  - RADAR_FRONT (ARS548) as NuScenes radar .pcd, ego-motion compensated;
  - next to every LiDAR file, <token>.time.bin: float32 seconds of each point
    relative to the frame timestamp, same order as the .pcd.bin;
  - every other topic in the bag (vehicle CAN, commands, AD state, corner and
    vehicle radar objects, raw CAN frames, INS logs, diagnostics) as
    ext/<scene>/<topic>.json.

Every included sensor needs calibration; --skip-uncalibrated leaves out the
optional ones that have none.
"""
from converter import FULL, run

if __name__ == "__main__":
    run(FULL, __doc__)
