#!/usr/bin/env python3
"""Held-out check of LiDAR -> image projection.

    python online_calib/eval_lidar.py CALIB.json --clip CLIP --ins INS.npz --mask MASK.png

On a drive the calibration never saw: LiDAR renderings are matched against the
images and the matches are verified geometrically by a RANSAC-PnP of their own
(the held-out extrinsic). Reported:
  cross-validated error   pixel error of those verified matches under the
                          calibration being evaluated — it was not fitted to them
  self-fit error          the same under the held-out extrinsic (the floor set by
                          matching noise)
  repeatability           rotation and translation between the calibration and
                          the held-out extrinsic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ba import DEV, LidarMatches, State, match_residuals
from camera_model import Grid
from evaluate import Calib
from refine_pnp import gather, solve
from traj import Trajectory


def match_errors(calib_json: dict, T_cl: np.ndarray, X0, DX, U) -> np.ndarray:
    c = calib_json
    g = c.get("grid")
    grid = Grid(c["width"], c["height"], g["gx"], g["gy"]) if g else None
    st = State(intr=np.array(c["intr"]), T_cl=T_cl, dt=c.get("dt_s", 0.0), rs=c.get("rs_s", 0.0),
               G=np.array(g["G"]) if g else None)
    t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=DEV)
    m = LidarMatches(t(X0), t(DX), t(U))
    r, _ = match_residuals(m, st, grid, {"P": 0}, c["model"], c["height"], 1.0)
    return np.hypot(r[0::2], r[1::2])


def evaluate_lidar(calib_path: Path, clip: Path, traj: Trajectory, mask: np.ndarray, n_frames: int = 20,
                   min_speed: float = 1.5, save_matches: Path | None = None, window_s: float = 0.5) -> dict:
    cal = Calib(calib_path)
    X, U, stats, X0, DX = gather(cal, [clip], {clip: traj}, mask, n_frames, min_speed, window_s)
    T_ho, inl = solve(cal, X, U)
    if save_matches is not None:
        # the held-out-verified 2D-3D matches, in the same form refine_pnp saves
        # (LiDAR frame at the image stamp + its motion per second of delay), plus the
        # held-out extrinsic they came with
        np.savez(save_matches, X_l=X0[inl], dX_l=DX[inl], uv=U[inl], T_heldout=T_ho,
                 calib=str(calib_path), clip=str(clip))
    c = cal.c
    e_cv = match_errors(c, cal.T_cl, X0[inl], DX[inl], U[inl])
    e_self = match_errors(c, T_ho, X0[inl], DX[inl], U[inl])
    dT = T_ho @ np.linalg.inv(cal.T_cl)
    rot = float(np.degrees(np.linalg.norm(cv2.Rodrigues(dT[:3, :3])[0])))
    cam_a = -cal.T_cl[:3, :3].T @ cal.T_cl[:3, 3]
    cam_b = -T_ho[:3, :3].T @ T_ho[:3, 3]
    rng = np.linalg.norm(X0[inl], axis=1)
    by_range = []
    for lo, hi in ((0, 8), (8, 15), (15, 30), (30, 200)):
        m = (rng >= lo) & (rng < hi)
        by_range.append({"range_m": [lo, hi], "n": int(m.sum()),
                         "cv_median_px": float(np.median(e_cv[m])) if m.sum() >= 10 else None,
                         "self_median_px": float(np.median(e_self[m])) if m.sum() >= 10 else None})
    return {"n_frames": len(stats), "n_matches": int(len(U)), "n_verified": int(len(inl)), "by_range": by_range,
            "cv_median_px": float(np.median(e_cv)), "cv_p90_px": float(np.percentile(e_cv, 90)),
            "cv_within_2px": float(np.mean(e_cv < 2)), "self_median_px": float(np.median(e_self)),
            "repeat_rot_deg": rot, "repeat_trans_cm": float(100 * np.linalg.norm(cam_a - cam_b)),
            "heldout_cam_in_lidar": cam_b.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("calib", type=Path)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--min-speed", type=float, default=1.5,
                   help="only frames with the car faster than this (m/s); -1 keeps standing frames")
    p.add_argument("--save-matches", type=Path, default=None,
                   help="also write the verified matches (npz) for reuse, e.g. by a joint rig solve")
    p.add_argument("--window-s", type=float, default=0.5, help="render window, see refine_pnp.py --window-s")
    a = p.parse_args()
    r = evaluate_lidar(a.calib, a.clip, Trajectory(a.ins), cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE), a.frames,
                       a.min_speed, a.save_matches, a.window_s)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
