#!/usr/bin/env python3
"""Held-out check of a camera's internal geometry: track reprojection error.

    python online_calib/eval_tracks.py CALIB.json TRACKS.npz --ins INS.npz

The calibration is frozen (lens, windshield field, extrinsics, timing); the
landmarks of tracks from a drive the calibration never saw are re-triangulated
and refined. Two numbers, each overall and by distance from the image centre
(where lens and windshield models differ most):
  ins_poses    camera poses from the INS and the calibrated extrinsic: the whole
               chain, including INS attitude noise and mount vibration
  free_poses   every frame's pose also refined (weak prior to the INS): what is
               left is the camera model's own error plus tracking noise
Tracking noise dominates those medians, so each also carries the systematic
residual: the mean residual vector in each cell of a 16x9 grid over the image.
Noise averages out of it (~0.02 px with a thousand observations per cell); a
lens or windshield model that is wrong for this camera leaves a spatial pattern.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ba import State, build_problem, lm_solve, reprojection_stats, residuals_and_jacobians, triangulate
from calibrate import compact, prune
from camera_model import Grid
from traj import Trajectory

BINS = (0.0, 0.3, 0.5, 0.7, 0.85, 1.01)
CELLS = (16, 9)
MIN_PER_CELL = 50


def error_stats(pb, st, grid, W: int, H: int) -> dict:
    e = reprojection_stats(pb, st, grid)["err"]
    uv = pb.obs_uv.cpu().numpy()
    rad = np.hypot(uv[:, 0] - W / 2, uv[:, 1] - H / 2) / (np.hypot(W, H) / 2)
    by = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (rad >= lo) & (rad < hi)
        by.append({"radius": [lo, hi], "n": int(m.sum()),
                   "median_px": float(np.median(e[m])) if m.any() else None,
                   "rms_px": float(np.sqrt(np.mean(np.minimum(e[m], 5) ** 2))) if m.any() else None})
    # systematic residual per cell, from inlier observations
    r = residuals_and_jacobians(pb, st, grid, {"P": 0})[0].reshape(-1, 2)
    inl = e < 3
    cx = np.clip((uv[:, 0] / W * CELLS[0]).astype(int), 0, CELLS[0] - 1)
    cy = np.clip((uv[:, 1] / H * CELLS[1]).astype(int), 0, CELLS[1] - 1)
    key = (cy * CELLS[0] + cx)[inl]
    n = np.bincount(key, minlength=CELLS[0] * CELLS[1])
    mean = np.stack([np.bincount(key, r[inl, k], minlength=n.size) for k in (0, 1)], 1) / np.maximum(n, 1)[:, None]
    ok = n >= MIN_PER_CELL
    mag = np.hypot(mean[ok, 0], mean[ok, 1])
    return {"n_obs": int(len(e)), "n_landmarks": pb.n_lm, "median_px": float(np.median(e)),
            "rms_clipped5_px": float(np.sqrt(np.mean(np.minimum(e, 5) ** 2))),
            "within_1px": float(np.mean(e < 1)), "by_radius": by,
            "systematic_rms_px": float(np.sqrt(np.mean(mag ** 2))) if ok.any() else None,
            "systematic_p90_px": float(np.percentile(mag, 90)) if ok.any() else None,
            "systematic_cells": {"shape": list(CELLS[::-1]), "n": n.tolist(), "mean": mean.round(4).tolist()}}


def evaluate_tracks(calib_path: Path, tracks_path: Path, traj: Trajectory) -> dict:
    c = json.loads(Path(calib_path).read_text())
    tr = dict(np.load(tracks_path))
    g = c.get("grid")
    grid = Grid(c["width"], c["height"], g["gx"], g["gy"]) if g else None
    G = np.array(g["G"]) if g else None
    if g and not g.get("near", True):
        G[2:] = 0
    st = State(intr=np.array(c["intr"]), T_cl=np.array(c["T_cam_lidar"]),
               dt=c.get("dt_s", 0.0), rs=c.get("rs_s", 0.0), G=G)
    pb = build_problem([tr], traj, np.array(c["T_ego_lidar"]), c["model"], frame_step=1)
    X, ok = triangulate(pb, st, grid)
    st.X = X
    pb = prune(pb, ok[pb.obs_j.cpu().numpy()])
    pb, st = compact(pb, st)
    st, _ = lm_solve(pb, st, grid, {}, iters=8, verbose=False)     # landmarks only
    pb = prune(pb, reprojection_stats(pb, st, grid)["err"] < 20.0)  # gross track failures
    pb, st = compact(pb, st)
    st, _ = lm_solve(pb, st, grid, {}, iters=8, verbose=False)
    out = {"ins_poses": error_stats(pb, st, grid, c["width"], c["height"])}
    # free the poses (weak prior to the INS), then drop what even that cannot explain
    F = len(pb.fr_R)
    st.pf = np.zeros((F, 6))
    sig = np.tile([np.radians(2.0)] * 3 + [0.5] * 3, F)
    for thresh in (8.0, None):
        st, _ = lm_solve(pb, st, grid, {"pf": True}, iters=10, priors={"pf": sig}, verbose=False)
        if thresh:
            pb = prune(pb, reprojection_stats(pb, st, grid)["err"] < thresh)
            pb, st = compact(pb, st)
    out["free_poses"] = error_stats(pb, st, grid, c["width"], c["height"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("calib", type=Path)
    p.add_argument("tracks", type=Path)
    p.add_argument("--ins", type=Path, required=True)
    a = p.parse_args()
    r = evaluate_tracks(a.calib, a.tracks, Trajectory(a.ins))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
