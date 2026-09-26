#!/usr/bin/env python3
"""Camera <- LiDAR extrinsic from pooled LiDAR-image matches (multi-frame RANSAC-PnP).

    python online_calib/refine_pnp.py CALIB.json --clips CLIP [...] --ins INS.npz [...] \
        --mask MASK.png --out OUT.json [--rounds 2]

Per frame, the LiDAR around the image is rendered into the camera with the
current calibration and matched against the image with LoFTR (lidar_match.py).
Single-frame cross-modal matches are mostly wrong, but the right ones all agree
on one extrinsic: the matches of many frames are pooled, every LiDAR point is
expressed in the LiDAR frame at its image's time (so the LiDAR's position on
the car no longer matters), and RANSAC-PnP on the camera's own rays (the lens
and windshield model from the bundle adjustment, held fixed) keeps the
consistent set; Levenberg-Marquardt polishes it. Re-rendering with the new
extrinsic and matching again (--rounds) tightens it further.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from camera_model import unproject
from evaluate import DEV, Calib
from lidar_match import correspondences
from traj import Trajectory


def lidar_frame_at_image(calib: Calib, traj: Trajectory, Pw: np.ndarray, t_img: int):
    """World points -> (X_l, dX_l): the LiDAR frame at the image stamp, and how a
    point moves in it per second of delay (ego motion, constant velocity), so
    X_l(delta) = X_l - delta * dX_l."""
    R, p = traj.R(t_img)[0], traj.p(t_img)[0]
    w, v = traj.omega_ego(t_img)[0], traj.vel_world(t_img)[0]
    y = (Pw - p) @ R
    dy = np.cross(w, y) + R.T @ v
    T_le = np.linalg.inv(calib.T_el)
    return y @ T_le[:3, :3].T + T_le[:3, 3], dy @ T_le[:3, :3].T


def lidar_at_image_time(calib: Calib, traj: Trajectory, Pw: np.ndarray, uv: np.ndarray, t_img: int):
    """World points -> LiDAR frame at the exposure time of their image row."""
    row = uv[:, 1].astype(np.float64) / calib.H - 0.5      # float64: ns timestamps need it
    t = t_img + (calib.dt + calib.rs * row) * 1e9
    lo, hi = float(traj.t[0]), float(traj.t[-1])
    if (t < lo).any() or (t > hi).any():
        print(f"    [!] {int(((t < lo) | (t > hi)).sum())} times outside the trajectory at {t_img}, "
              f"row range {row.min():.2f}..{row.max():.2f}")
        t = np.clip(t, lo, hi)
    R, p = traj.R(t), traj.p(t)
    Xe = np.einsum("nji,nj->ni", R, Pw - p)
    T_le = np.linalg.inv(calib.T_el)
    return Xe @ T_le[:3, :3].T + T_le[:3, 3]


def gather(calib: Calib, clips, trajs, mask, n_per_clip: int, min_speed: float = 1.5,
           window_s: float = 0.5):
    X, U, X0, DX = [], [], [], []
    stats = []
    for clip in clips:
        traj = trajs[clip]
        files = sorted((clip / "cam" / calib.c["channel"]).glob("*.jpg"))
        files = [f for f in files if traj.covers(int(f.stem))[0] and traj.speed(int(f.stem))[0] > min_speed]
        for f in files[:: max(1, len(files) // n_per_clip)][:n_per_clip]:
            t = int(f.stem)
            uv, Pw, info = correspondences(calib, clip, t, traj, mask, window_s=window_s)
            torch.cuda.empty_cache()
            fin = np.isfinite(uv).all(1) & np.isfinite(Pw).all(1)
            uv, Pw = uv[fin].astype(np.float64), Pw[fin]
            if len(uv) < 6:
                continue
            X.append(lidar_at_image_time(calib, traj, Pw, uv, t))
            x0, dx = lidar_frame_at_image(calib, traj, Pw, t)
            X0.append(x0)
            DX.append(dx)
            U.append(uv)
            stats.append(info["n_kept"])
    return np.concatenate(X), np.concatenate(U), stats, np.concatenate(X0), np.concatenate(DX)


def solve(calib: Calib, X_l: np.ndarray, uv: np.ndarray, thresh_px: float = 3.0):
    """RANSAC-PnP on normalized rays, then LM on the inliers. Returns (T_cam_lidar, inliers)."""
    rays = unproject(torch.tensor(uv, dtype=torch.float64, device=DEV), calib.intr, calib.model,
                     calib.grid, calib.G).cpu().numpy()
    fwd = rays[:, 2] > 0.25                           # pinhole-normalizable rays
    n = rays[fwd, :2] / rays[fwd, 2:]
    obj = X_l[fwd].astype(np.float64)
    f = float(calib.intr[0])
    R0 = calib.T_cl[:3, :3]
    rvec0, _ = cv2.Rodrigues(R0)
    tvec0 = calib.T_cl[:3, 3].reshape(3, 1)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, n, np.eye(3), None, rvec0.copy(), tvec0.copy(), useExtrinsicGuess=True,
        iterationsCount=20000, reprojectionError=thresh_px / f, confidence=0.9999,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None:
        return calib.T_cl, np.zeros(0, int)
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], n[inl], np.eye(3), None, rvec, tvec)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, np.flatnonzero(fwd)[inl]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("calib", type=Path)
    p.add_argument("--clips", type=Path, nargs="+", required=True)
    p.add_argument("--ins", type=Path, nargs="+", required=True)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames-per-clip", type=int, default=15)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--min-speed", type=float, default=1.5,
                   help="only frames with the car faster than this (m/s); -1 keeps standing frames")
    p.add_argument("--window-s", type=float, default=0.5,
                   help="LiDAR sweeps within +-this of the image are rendered (default 0.5 s). Every sweep "
                        "is placed with the INS, so INS error over the window lands in the extrinsic: on "
                        "the DM rig +-0.5 s moves the camera 10-20 cm along the direction of travel, "
                        "+-0.11 s (the nearest one or two sweeps) does not")
    a = p.parse_args()
    cal = Calib(a.calib)
    ins = [a.ins[i] if len(a.ins) > 1 else a.ins[0] for i in range(len(a.clips))]
    cache = {}
    trajs = {c: cache.setdefault(i, Trajectory(i)) for c, i in zip(a.clips, ins)}
    mask = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE)
    hist = []
    for r in range(a.rounds):
        X, U, stats, X0, DX = gather(cal, a.clips, trajs, mask, a.frames_per_clip, a.min_speed, a.window_s)
        T_before = cal.T_cl.copy()
        T, inl = solve(cal, X, U)
        dT = T @ np.linalg.inv(T_before)
        ang = np.degrees(np.linalg.norm(cv2.Rodrigues(dT[:3, :3])[0]))
        cam_before = -T_before[:3, :3].T @ T_before[:3, 3]
        cam_after = -T[:3, :3].T @ T[:3, 3]
        print(f"round {r}: {len(stats)} frames, {len(U)} matches, {len(inl)} inliers "
              f"({100 * len(inl) / max(len(U), 1):.0f}%); rotation change {ang:.3f} deg, "
              f"camera position in LiDAR {np.round(cam_before, 3)} -> {np.round(cam_after, 3)}")
        hist.append({"n_frames": len(stats), "n_matches": int(len(U)), "n_inliers": int(len(inl)),
                     "rot_change_deg": float(ang), "cam_in_lidar": cam_after.tolist()})
        cal.T_cl = T
    np.savez(a.out.with_suffix(".matches.npz"), X_l=X0[inl], dX_l=DX[inl], uv=U[inl])
    out = dict(cal.c)
    out["T_cam_lidar"] = cal.T_cl.tolist()
    out["T_lidar_cam"] = np.linalg.inv(cal.T_cl).tolist()
    out["pnp_refinement"] = hist
    out["render_window_s"] = a.window_s
    a.out.write_text(json.dumps(out))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
