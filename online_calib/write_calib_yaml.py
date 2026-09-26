#!/usr/bin/env python3
"""Write the online calibration as per-camera YAML files in the T-Car team format.

    python online_calib/write_calib_yaml.py --results online_calib/results_20260923 --out /hdd/T-Car/online_calib_20260923

Follows the layout of /hdd/T-Car/intrinsic and /hdd/T-Car/extrinsic:
  intrinsic/<pos>_camera_<n>_intrinsic.yaml     ROS camera_info (distortion_model fisheye = KB,
                                                  D = k1..k4), plus an `online_calib` block with the
                                                  full model (KB + windshield field), time offset and
                                                  rolling-shutter readout
  extrinsic/<pos>_lidar_to_camera_<n>.yaml       `transform`: x_link = R x_lidar + t, camera link frame
                                                  (REP-103: x forward, y left, z up), as the team files;
                                                  `opencv`: x_cam = R x_lidar + t, OpenCV optical frame;
                                                  `camera_in_ego`: the camera's pose on the car
  rig.yaml                                       LiDAR -> ego (fused over the seven cameras)
  full_model/final_<CHANNEL>.json                the solver's complete output
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMS = {  # channel -> (team position name, camera number)
    "CAM_FRONT": ("front", 3), "CAM_FRONT_RIGHT": ("frontright", 1), "CAM_FRONT_LEFT": ("frontleft", 6),
    "CAM_BACK": ("rear", 2), "CAM_BACK_LEFT": ("rearleft", 5), "CAM_BACK_RIGHT": ("rearright", 0),
    "CAM_TRAFFIC": ("traffic", 4),
}
# camera link (x forward, y left, z up) <- OpenCV optical (x right, y down, z forward)
M_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def mat(rows: int, cols: int, a) -> dict:
    return {"rows": rows, "cols": cols, "data": [float(x) for x in np.asarray(a).ravel()]}


def rot_block(R: np.ndarray) -> dict:
    return {"quaternion_xyzw": [float(x) for x in Rotation.from_matrix(R).as_quat()],
            "matrix": [[float(x) for x in r] for r in R]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    fused = json.loads((a.results / "final/lidar_ego_fused.json").read_text())
    T_el = np.array(fused["T_ego_lidar"])
    export = json.loads((a.results / "calib_tcar_20260923/export_report.json").read_text())
    for d in ("intrinsic", "extrinsic", "full_model"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    source = "online targetless calibration, calibrated on bag A-9 (2026-09-23), checked on A-10 and A-8"

    for ch, (pos, n) in CAMS.items():
        c = json.loads((a.results / f"final/final_{ch}.json").read_text())
        ex = a.results / "calib_tcar_20260923" / ch
        K = np.loadtxt(ex / "intrinsic.txt")
        D = np.loadtxt(ex / "distortion.txt")
        fx, fy, cx, cy, *k = c["intr"]
        intr = {
            "image_width": c["width"], "image_height": c["height"], "camera_name": f"camera_{n}",
            "camera_matrix": mat(3, 3, K),
            "distortion_model": "fisheye",
            "distortion_coefficients": mat(1, 4, D),
            "rectification_matrix": mat(3, 3, np.eye(3)),
            "projection_matrix": mat(3, 4, np.c_[K, np.zeros(3)]),
            "online_calib": {
                "source": source,
                "note": ("camera_matrix / distortion_coefficients are a Kannala-Brandt (OpenCV fisheye) lens "
                         "refitted to reproduce the full model below without the windshield field: it differs "
                         f"from the full model by {export[ch]['kb_fit_median_px']:.2f} px on median, "
                         f"{export[ch]['kb_fit_max_px']:.2f} px at most"),
                "full_model": {
                    "lens": {"model": "kannala_brandt", "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                             "k1": k[0], "k2": k[1], "k3": k[2], "k4": k[3]},
                    "windshield_field": ("pixel offset S_inf(u) + S_near(u) / range added after the lens, "
                                         f"{c['grid']['gx']}x{c['grid']['gy']} cubic B-spline grid; see "
                                         f"full_model/final_{ch}.json (grid.G)"),
                },
                "time_offset_s": c["dt_s"],
                "rolling_shutter_readout_s": c["rs_s"],
                "timing_note": ("row r (0 at the top, H rows) is exposed at header stamp + time_offset_s "
                                "+ rolling_shutter_readout_s * (r / H - 0.5)"),
            },
        }
        T_cl = np.array(c["T_cam_lidar"])                      # x_opt = R x_lidar + t
        R_link, t_link = M_LINK_OPT @ T_cl[:3, :3], M_LINK_OPT @ T_cl[:3, 3]
        T_ec = T_el @ np.linalg.inv(T_cl)                     # camera (optical) pose in ego
        fin = c.get("final_stages", [{}])[-1]
        extr = {
            "parent_frame": "rslidar",
            "child_frame": f"camera_{n}",
            "method": "online_targetless (KLT bundle adjustment + LiDAR reflectivity matching, joint)",
            "n_frames_used": int(c.get("pnp_refinement", [{}])[-1].get("n_frames", 0)) or None,
            "transform": {"translation": [float(x) for x in t_link], "rotation": rot_block(R_link)},
            "residuals": {"feature_track_median_px": fin.get("reproj_median_px"),
                          "lidar_match_median_px": fin.get("match_median_px")},
            "notes": ("REP-103 link frame (X fwd, Y left, Z up), x_link = R x_lidar + t. For OpenCV-frame "
                      "projection use `opencv` below (x_cam = R x_lidar + t, x right, y down, z forward)."),
            "opencv": {"translation": [float(x) for x in T_cl[:3, 3]], "rotation": rot_block(T_cl[:3, :3])},
            "camera_in_ego": {
                "frame": "ego = base_link at the INS output point (INSPVA), x forward, y left, z up",
                "translation": [float(x) for x in T_ec[:3, 3]],
                "rotation_optical_to_ego": rot_block(T_ec[:3, :3]),
            },
            "source": source,
        }
        (a.out / "intrinsic" / f"{pos}_camera_{n}_intrinsic.yaml").write_text(yaml.safe_dump(intr, sort_keys=False))
        (a.out / "extrinsic" / f"{pos}_lidar_to_camera_{n}.yaml").write_text(yaml.safe_dump(extr, sort_keys=False))
        shutil.copy(a.results / f"final/final_{ch}.json", a.out / "full_model" / f"final_{ch}.json")

    rpy = Rotation.from_matrix(T_el[:3, :3]).as_euler("xyz", degrees=True)
    rig = {
        "parent_frame": "ego (base_link at the INS output point, x forward, y left, z up)",
        "child_frame": "rslidar",
        "transform": {"translation": [float(x) for x in T_el[:3, 3]], "rotation": rot_block(T_el[:3, :3]),
                      "rpy_deg": [float(x) for x in rpy]},
        "notes": "x_ego = R x_lidar + t. Median of the seven cameras' estimates.",
        "spread_across_cameras": {"translation_std_m": fused["spread_t_m_std"],
                                  "rotation_std_deg": fused["spread_rot_deg_std"]},
        "source": source,
    }
    (a.out / "rig.yaml").write_text(yaml.safe_dump(rig, sort_keys=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
