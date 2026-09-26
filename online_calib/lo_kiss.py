#!/usr/bin/env python3
"""Stage 1 of the LiDAR odometry: KISS-ICP on one extracted clip, per-point deskew.

    PY online_calib/lo_kiss.py CLIP --out KISS.npz [--voxel 0.5] [--threads N]

PY is a Python with kiss-icp 1.3 (pip install kiss-icp==1.3.0; the DM rig's
/hdd/DM_calib/lidar_odo/venv has it). Taken from /hdd/DM_calib/lidar_odo/lo_kiss.py
unchanged except that it uses every core by default; stage 2 is lo_refine.py.

Input: CLIP/lidar/<header ns>.npy (extract.py layout: x y z intensity ring t,
t = seconds since the sweep's first column, 0 .. 0.0999 s; on the Ruby128 the
header stamp is the sweep's start).

KISS-ICP (v1.3) deskews a frame to the time of normalised stamp 1.0 (verified on a
synthetic frame: the point at stamp 1.0 is left unchanged, the one at 0 moves by the
full relative motion). We pass stamp = t / 0.1 s, so the pose returned for sweep k is
the LiDAR pose at tau_k = header_k + 0.1 s (= the end of that sweep, ~ the next
header). Written: knot times tau (ns, int64) and T_w_L (N,4,4), world = first pose.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

SWEEP_S = 0.1


def load(path: Path, rmin: float, rmax: float):
    s = np.load(path)
    p = np.stack([s["x"], s["y"], s["z"]], -1).astype(np.float64)
    r = np.linalg.norm(p, axis=1)
    k = (r > rmin) & (r < rmax)
    return p[k], s["t"][k].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seg", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voxel", type=float, default=0.5)
    ap.add_argument("--rmin", type=float, default=2.5)
    ap.add_argument("--rmax", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = a.rmax, a.rmin, True
    cfg.mapping.voxel_size = a.voxel
    cfg.registration.max_num_threads = a.threads
    odo = KissICP(cfg)
    files = sorted((a.seg / "lidar").glob("*.npy"))
    hdr = np.array([int(f.stem) for f in files], np.int64)
    poses = []
    t0 = time.time()
    for f in files:
        p, t = load(f, a.rmin, a.rmax)
        odo.register_frame(p, t / SWEEP_S)
        poses.append(odo.last_pose.copy())
    wall = time.time() - t0
    tau = hdr + int(SWEEP_S * 1e9)
    np.savez(a.out, tau=tau, header=hdr, T_w_L=np.array(poses), wall_s=wall, voxel=a.voxel)
    P = np.array(poses)
    d = np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)
    print(f"{a.seg.name}: {len(files)} sweeps in {wall:.1f} s ({1e3 * wall / len(files):.0f} ms/sweep); "
          f"path {d.sum():.1f} m, mean speed {d.mean() / SWEEP_S:.2f} m/s", flush=True)


if __name__ == "__main__":
    main()
