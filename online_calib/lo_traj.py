#!/usr/bin/env python3
"""LiDAR odometry -> a trajectory file the calibration stages read in place of the INS.

    python online_calib/lo_traj.py REFINED.npz [REFINED2.npz ...] --out LO.npz

REFINED.npz is a continuous-time LiDAR odometry of one clip (/hdd/DM_calib/lidar_odo:
lo_kiss.py, then lo_refine.py): knots T_w_L at tau_k = sweep header + 0.1 s, SLERP /
linear in between — which is how traj.Trajectory interpolates too, so the knots are
written as they are. The frame the stages call "ego" is then the LiDAR itself:
calibrate with an identity --lidar-ego; the LiDAR -> ego transform is not estimated
(Trajectory.lidar_frame). Velocity and body rate come from the knots' differences.

Several clips go into one file (disjoint in time). Each odometry has its own world
(the LiDAR at its first knot); clip k's is shifted by k km so that nothing can
associate across clips by accident. Only clip-local geometry is used downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

CLIP_SHIFT_M = 1000.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("lo", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    segs = []
    for k, f in enumerate(a.lo):
        z = np.load(f)
        t = z["tau"].astype(np.int64)
        T = z["T_w_L"].astype(np.float64)
        pos = T[:, :3, 3] + [CLIP_SHIFT_M * k, 0.0, 0.0]
        rot = Rotation.from_matrix(T[:, :3, :3])
        ts = t * 1e-9
        vel_ego = rot.inv().apply(np.gradient(pos, ts, axis=0))
        rate = (rot[:-1].inv() * rot[1:]).as_rotvec() / np.diff(ts)[:, None]
        segs.append((t, pos, rot.as_quat()[:, [3, 0, 1, 2]], vel_ego, (t[:-1] + t[1:]) // 2, rate))
    segs.sort(key=lambda s: s[0][0])
    for s0, s1 in zip(segs, segs[1:]):
        if s1[0][0] <= s0[0][-1]:
            raise SystemExit("the odometries overlap in time")
    cat = [np.concatenate([s[i] for s in segs]) for i in range(6)]
    np.savez(a.out, lo=True, t=cat[0], origin=np.zeros(3), pos=cat[1], quat=cat[2], vel_ego=cat[3],
             imu_t=cat[4], imu_rate=cat[5], imu_hz=10.0, sources=[str(f) for f in a.lo])
    sp = np.hypot(cat[3][:, 0], cat[3][:, 1])
    print(f"-> {a.out}: {len(cat[0])} knots from {len(segs)} clips, speed median {np.median(sp):.1f} m/s, "
          f"max yaw rate {np.degrees(np.abs(cat[5][:, 2]).max()):.1f} deg/s")


if __name__ == "__main__":
    main()
