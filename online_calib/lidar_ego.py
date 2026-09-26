#!/usr/bin/env python3
"""LiDAR -> ego (base_link) extrinsic from driving data, no target.

    python online_calib/lidar_ego.py WORK/<clip> [<clip> ...] --ins WORK/<bag>_ins.npz --out lidar_ego.json

1. Roll/pitch: the ground plane seen by the LiDAR is the ego x-y plane.
2. Yaw: frame-to-frame ICP gives the LiDAR's own motion direction; the ego
   moves along +x (odom twist), so the two directions fix the heading.
3. Refinement: every sweep is deskewed into the world with the INS trajectory
   and the candidate extrinsic; the extrinsic that makes the accumulated map
   most self-consistent (point-to-plane distance between sweeps taken from
   different poses, through turns) wins. z is barely observable from planar
   driving and is kept at its prior.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj import (  # noqa: E402
    Trajectory, deskew_to_world, load_sweep, normals_pca, se3, se3_exp,
    sweep_xyz, transform, voxel_down,
)


def ground_rotation(sweeps: list[np.ndarray]) -> np.ndarray:
    """Rotation that levels the LiDAR (its ground normal -> +z), yaw left at 0."""
    normals = []
    rng = np.random.default_rng(0)
    for s in sweeps:
        p = sweep_xyz(s)
        p = p[np.hypot(p[:, 0], p[:, 1]) < 25]
        best, best_n = 0, None
        low = p[p[:, 2] < np.percentile(p[:, 2], 30)]
        for _ in range(200):
            a, b, c = low[rng.choice(len(low), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-6:
                continue
            n /= np.linalg.norm(n)
            if n[2] < 0:
                n = -n
            inl = np.abs((p - a) @ n) < 0.05
            if inl.sum() > best:
                best, best_n = inl.sum(), (n, inl)
        n, inl = best_n
        q = p[inl]
        w, v = np.linalg.eigh(np.cov((q - q.mean(0)).T))
        n = v[:, 0] * np.sign(v[2, 0])
        normals.append(n)
    n = np.mean(normals, axis=0)
    n /= np.linalg.norm(n)
    # smallest rotation taking n to z
    axis = np.cross(n, [0, 0, 1.0])
    ang = np.arccos(np.clip(n[2], -1, 1))
    return Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-12) * ang).as_matrix()


def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, T0: np.ndarray,
                       iters: int = 30, max_dist: float = 1.0) -> np.ndarray:
    tree = cKDTree(dst)
    nrm, planar = normals_pca(dst, k=10, tree=tree)
    T = T0.copy()
    for _ in range(iters):
        s = transform(T, src)
        d, idx = tree.query(s, distance_upper_bound=max_dist)
        ok = np.isfinite(d) & (planar[np.minimum(idx, len(dst) - 1)] > 0.3)
        s, q, n = s[ok], dst[idx[ok]], nrm[idx[ok]]
        r = np.einsum("ij,ij->i", s - q, n)
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.1)          # Huber
        A = np.hstack([np.cross(s, n), n])
        dx = np.linalg.solve((A * w[:, None]).T @ A + 1e-6 * np.eye(6), -(A * w[:, None]).T @ r)
        T = se3_exp(dx) @ T
        if np.linalg.norm(dx) < 1e-6:
            break
    return T


def heading_yaw(sweeps, stamps, traj: Trajectory, R_level: np.ndarray) -> float:
    """Yaw of the levelled LiDAR relative to ego x, from ICP motion directions."""
    angles, weights = [], []
    for i in range(0, len(sweeps) - 2, 3):
        a = voxel_down(sweep_xyz(sweeps[i]) @ R_level.T, 0.2)
        b = voxel_down(sweep_xyz(sweeps[i + 2]) @ R_level.T, 0.2)
        if traj.speed(stamps[i])[0] < 3.0:
            continue
        T = icp_point_to_plane(b, a, np.eye(4))       # pose of b in a's frame
        t = T[:3, 3]
        if np.linalg.norm(t[:2]) < 0.3:
            continue
        # ego moves along its own +x (lateral slip is ~0); the lidar sees t
        angles.append(np.arctan2(t[1], t[0]))
        weights.append(np.linalg.norm(t[:2]))
    ang = np.angle(np.average(np.exp(1j * np.array(angles)), weights=weights))
    return float(ang)      # LiDAR x points this far *from* the motion direction, negated below


def refine(sweeps, stamps, traj: Trajectory, T0: np.ndarray, voxel: float = 0.3,
           max_dist: float = 0.5) -> tuple[np.ndarray, dict]:
    """Map self-consistency over the extrinsic (x, y, roll, pitch, yaw; z fixed)."""
    # keep per-point times for deskewing: recompute on the voxel subset
    subs = []
    for s in sweeps:
        p = sweep_xyz(s)
        _, idx = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
        subs.append(s[np.sort(idx)])
    n = len(subs)
    pairs = [(i, j) for i in range(n) for j in (i - 4, i - 2, i + 2, i + 4) if 0 <= j < n]

    def world(T_el):
        return [deskew_to_world(s, t, traj, T_el) for s, t in zip(subs, stamps)]

    def residuals(x, assoc=None):
        T_el = T0 @ se3_exp(np.array([x[2], x[3], x[4], x[0], x[1], 0.0]))
        W = world(T_el)
        res = []
        for (i, j), (idx_i, q_idx, nrm) in zip(pairs, assoc):
            r = np.einsum("ij,ij->i", W[j][idx_i] - W[i][q_idx], nrm)
            res.append(r / (1.0 + np.abs(r) / 0.05) ** 0.5)
        return np.concatenate(res)

    x = np.zeros(5)
    info = {}
    for outer in range(4):
        T_el = T0 @ se3_exp(np.array([x[2], x[3], x[4], x[0], x[1], 0.0]))
        W = world(T_el)
        trees = {i: cKDTree(W[i]) for i in {p[0] for p in pairs}}
        nrms = {i: normals_pca(W[i], k=10, tree=trees[i]) for i in trees}
        assoc = []
        for i, j in pairs:
            d, idx = trees[i].query(W[j], distance_upper_bound=max_dist)
            ok = np.flatnonzero(np.isfinite(d) & (nrms[i][1][np.minimum(idx, len(W[i]) - 1)] > 0.5))
            assoc.append((ok, idx[ok], nrms[i][0][idx[ok]]))
        sol = least_squares(residuals, x, args=(assoc,), x_scale=[0.05, 0.05, 0.002, 0.002, 0.002],
                            diff_step=1e-4, max_nfev=40)
        x = sol.x
        info[f"iter{outer}"] = {"rms_m": float(np.sqrt(np.mean(sol.fun ** 2))), "x": x.tolist(),
                                "n": int(len(sol.fun))}
        max_dist = max(0.2, max_dist * 0.6)
    T_el = T0 @ se3_exp(np.array([x[2], x[3], x[4], x[0], x[1], 0.0]))
    return T_el, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("clips", type=Path, nargs="+")
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--z", type=float, default=1.9, help="prior LiDAR height above base_link (m)")
    p.add_argument("--stride", type=int, default=10, help="use every Nth sweep for refinement")
    args = p.parse_args()

    traj = Trajectory(args.ins)
    files = sorted(f for c in args.clips for f in (c / "lidar").glob("*.npy"))
    stamps_all = np.array([int(f.stem) for f in files])
    sweeps_all = [load_sweep(f) for f in files]

    R_level = ground_rotation(sweeps_all[::40])
    yaw_motion = heading_yaw(sweeps_all[:120], stamps_all[:120], traj, R_level)
    # A LiDAR moving along +x_ego sees translation at angle yaw_motion in its own
    # frame, so its x axis is rotated by -yaw_motion from ego x.
    R_el = Rotation.from_euler("z", -yaw_motion).as_matrix() @ R_level
    # this maps lidar coordinates to a frame whose x is the motion direction:
    T0 = se3(R_el, np.array([0.0, 0.0, args.z]))
    print(f"init: level {np.degrees(Rotation.from_matrix(R_level).as_rotvec())} deg, "
          f"yaw {np.degrees(-yaw_motion):.2f} deg")

    moving = np.flatnonzero(traj.speed(stamps_all) > 2.0)[::args.stride]
    T_el, info = refine([sweeps_all[i] for i in moving], stamps_all[moving], traj, T0)
    rpy = Rotation.from_matrix(T_el[:3, :3]).as_euler("xyz", degrees=True)
    out = {"T_ego_lidar": T_el.tolist(), "rpy_deg": rpy.tolist(), "t": T_el[:3, 3].tolist(),
           "z_fixed_prior_m": args.z, "refine": info}
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "refine"}, indent=1))
    for k, v in info.items():
        print(k, v)


if __name__ == "__main__":
    main()
