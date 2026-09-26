#!/usr/bin/env python3
"""Write online calibration results as a converter calibration directory.

    python online_calib/export_calib.py --results DIR --lidar-ego LIDAR_EGO.json --out CALIB_DIR

For every final_<CHANNEL>.json in DIR, writes CALIB_DIR/<CHANNEL>/ with
intrinsic.txt, distortion.txt (4 coefficients: OpenCV fisheye = Kannala-Brandt),
quat_r.txt and t.txt (camera <- ego, OpenCV convention), which is the layout
`bag2nuscenes.py --calib` reads, plus full_model.json (the lens, the windshield
field, timing) for tools that can use it. LIDAR_TOP/ gets r.txt and t.txt from
the LiDAR -> ego transform in LIDAR_EGO.json.

The converter's rectification knows only the lens, so the KB parameters written
are refitted to reproduce the full model (lens + far windshield field) as closely
as a KB lens can; what is left over is reported per camera.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from camera_model import Grid, project, unproject


def kb_approximation(c: dict, step: int = 24) -> tuple[np.ndarray, dict]:
    """KB intrinsics that best reproduce the full model for distant points."""
    W, H = c["width"], c["height"]
    intr = torch.tensor(c["intr"], dtype=torch.float64)
    g = c.get("grid")
    grid = Grid(W, H, g["gx"], g["gy"]) if g else None
    G = torch.tensor(g["G"], dtype=torch.float64) if g else None
    if G is not None:
        G = G.clone()
        G[2:] = 0                                    # far field: the near term needs a range
    u, v = np.meshgrid(np.arange(step / 2, W, step), np.arange(step / 2, H, step))
    uv = torch.tensor(np.c_[u.ravel(), v.ravel()], dtype=torch.float64)
    rays = unproject(uv, intr, c["model"], grid, G)
    ok = torch.isfinite(rays).all(1) & (rays[:, 2] > 0.05)
    rays, uv = rays[ok], uv[ok].numpy()
    if c["model"] != "kb":
        raise SystemExit(f"{c['channel']}: only KB lenses can be exported as OpenCV fisheye")

    def res(p):
        return (project(rays, torch.tensor(p, dtype=torch.float64), "kb").numpy() - uv).ravel()

    p = least_squares(res, np.array(c["intr"]), method="lm").x
    e = np.hypot(*res(p).reshape(-1, 2).T)
    e0 = np.hypot(*res(np.array(c["intr"])).reshape(-1, 2).T)
    return p, {"kb_fit_median_px": float(np.median(e)), "kb_fit_max_px": float(e.max()),
               "lens_only_median_px": float(np.median(e0)), "lens_only_max_px": float(e0.max())}


def write_vec(path: Path, v) -> None:
    np.savetxt(path, np.asarray(v, dtype=np.float64).reshape(-1), fmt="%.10e")


def quat_wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--lidar-ego", type=Path, required=True, help="json with T_ego_lidar (the fused one)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    T_el = np.array(json.loads(a.lidar_ego.read_text())["T_ego_lidar"])
    T_le = np.linalg.inv(T_el)
    report = {}
    for f in sorted(a.results.glob("final_CAM_*.json")):
        ch = f.stem.replace("final_", "")
        c = json.loads(f.read_text())
        if c.get("channel") != ch:                  # model variants: final_<CHANNEL>_<variant>.json
            continue
        kb, fit = kb_approximation(c)
        T_ce = np.array(c["T_cam_lidar"]) @ T_le
        d = a.out / ch
        d.mkdir(parents=True, exist_ok=True)
        K = np.array([[kb[0], 0, kb[2]], [0, kb[1], kb[3]], [0, 0, 1]])
        np.savetxt(d / "intrinsic.txt", K, fmt="%.10e")
        write_vec(d / "distortion.txt", kb[4:8])
        write_vec(d / "quat_r.txt", quat_wxyz(T_ce[:3, :3]))
        write_vec(d / "t.txt", T_ce[:3, 3])
        (d / "full_model.json").write_text(json.dumps({**c, "T_ego_lidar": T_el.tolist(),
                                                       "T_cam_ego": T_ce.tolist()}, indent=1))
        report[ch] = fit
        print(f"{ch}: KB refit reproduces the full far-field model to median {fit['kb_fit_median_px']:.2f} px, "
              f"max {fit['kb_fit_max_px']:.2f} px (lens alone: {fit['lens_only_max_px']:.2f} px)")
    d = a.out / "LIDAR_TOP"
    d.mkdir(parents=True, exist_ok=True)
    write_vec(d / "r.txt", quat_wxyz(T_le[:3, :3]))
    write_vec(d / "t.txt", T_le[:3, 3])
    (a.out / "export_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
