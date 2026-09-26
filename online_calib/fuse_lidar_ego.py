#!/usr/bin/env python3
"""Fuse the LiDAR -> ego transforms that each camera's final stage estimated.

    python online_calib/fuse_lidar_ego.py --results DIR --initial LIDAR_EGO.json --out FUSED.json \
        [--channels CAM_FRONT CAM_BACK ...]

Every camera sees the LiDAR's place on the car through its own tracks (camera on
the car) and matches (camera relative to the LiDAR), so each final_<CHANNEL>.json
holds an independent estimate. The fused transform is the per-axis median of
their small corrections to the initial one (rotation vector and translation);
the spread across cameras is the honest uncertainty. Writes FUSED.json with
T_ego_lidar and the per-camera deviations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_TRAFFIC"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--channels", nargs="*", default=CAMS)
    a = p.parse_args()
    T0 = np.array(json.loads(a.initial.read_text())["T_ego_lidar"])
    R0 = Rotation.from_matrix(T0[:3, :3])
    per = {}
    for ch in a.channels:
        f = a.results / f"final_{ch}.json"
        if not f.exists():
            continue
        T = np.array(json.loads(f.read_text())["T_ego_lidar"])
        per[ch] = {"rotvec_deg": np.degrees((R0.inv() * Rotation.from_matrix(T[:3, :3])).as_rotvec()),
                   "t_m": T[:3, 3] - T0[:3, 3]}
    rv = np.array([v["rotvec_deg"] for v in per.values()])
    tt = np.array([v["t_m"] for v in per.values()])
    rv_f, t_f = np.median(rv, 0), np.median(tt, 0)
    T = np.eye(4)
    T[:3, :3] = (R0 * Rotation.from_rotvec(np.radians(rv_f))).as_matrix()
    T[:3, 3] = T0[:3, 3] + t_f
    dev = {ch: {"rot_deg": float(np.linalg.norm(v["rotvec_deg"] - rv_f)), "trans_cm": float(100 * np.linalg.norm(v["t_m"] - t_f)),
                "correction_rotvec_deg": v["rotvec_deg"].round(4).tolist(), "correction_t_m": v["t_m"].round(4).tolist()}
           for ch, v in per.items()}
    out = {"T_ego_lidar": T.tolist(), "from_cameras": list(per), "correction_rotvec_deg": rv_f.tolist(),
           "correction_t_m": t_f.tolist(), "spread_rot_deg_std": rv.std(0).tolist(), "spread_t_m_std": tt.std(0).tolist(),
           "per_camera": dev, "initial": str(a.initial)}
    a.out.write_text(json.dumps(out, indent=1))
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    print(f"fused from {len(per)} cameras: t {T[:3, 3].round(3)} m, rpy {rpy.round(3)} deg")
    print(f"  correction vs initial: t {t_f.round(3)} m, rot {rv_f.round(3)} deg")
    print(f"  spread (std across cameras): t {tt.std(0).round(3)} m, rot {rv.std(0).round(3)} deg")
    for ch, d in dev.items():
        print(f"  {ch:16s} off fused by {d['rot_deg']:.3f} deg, {d['trans_cm']:.1f} cm")


if __name__ == "__main__":
    main()
