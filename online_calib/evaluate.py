#!/usr/bin/env python3
"""Independent check of a camera calibration: LiDAR edges vs image edges.

    python online_calib/evaluate.py CALIB.json --clip CLIP --ins INS.npz [--overlays DIR]

Nothing here is used by the optimization. For each sampled frame, the LiDAR
sweeps around it are deskewed into the world and brought into the camera at
the exposure time of each point's image row (time offset and rolling shutter
from the calibration), then projected with the full camera model.
  edge error   LiDAR depth discontinuities (the near side of a range jump along
               a ring) should fall on image edges: the distance transform of
               the Canny edge map at those points, in pixels
  overlays     the points coloured by range on the image
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from camera_model import Grid, project
from traj import Trajectory, load_sweep, sweep_xyz

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Calib:
    def __init__(self, path: Path):
        c = json.loads(Path(path).read_text())
        self.c = c
        self.model = c["model"]
        self.W, self.H = c["width"], c["height"]
        self.intr = torch.tensor(c["intr"], dtype=torch.float64, device=DEV)
        self.T_cl = np.array(c["T_cam_lidar"])
        self.T_el = np.array(c["T_ego_lidar"])
        self.dt, self.rs = c.get("dt_s", 0.0), c.get("rs_s", 0.0)
        g = c.get("grid")
        self.grid = Grid(self.W, self.H, g["gx"], g["gy"]) if g else None
        self.G = torch.tensor(g["G"], dtype=torch.float64, device=DEV) if g else None
        if g and not g.get("near", True):
            self.G[2:] = 0

    def project_world(self, Pw: np.ndarray, t_img: int, traj: Trajectory, iters: int = 2):
        """World points -> pixels at image time t_img, honouring dt and rolling shutter.

        Exposure times are taken to the microsecond (0.02 mm at 20 m/s): the poses
        are evaluated once per distinct microsecond, and the points moved on the device."""
        T_ce = torch.as_tensor(self.T_cl @ np.linalg.inv(self.T_el), dtype=torch.float64, device=DEV)
        P = torch.as_tensor(Pw, dtype=torch.float64, device=DEV)
        row = np.zeros(len(Pw))
        for _ in range(iters):
            q = np.round((self.dt + self.rs * row) * 1e6).astype(np.int64)
            q0 = int(q.min()) if len(q) else 0
            ts = t_img + (q0 + np.arange(int(q.max()) - q0 + 1 if len(q) else 1)) * 1000
            k = torch.as_tensor(q - q0, device=DEV)
            R = torch.as_tensor(traj.R(ts), dtype=torch.float64, device=DEV)[k]
            p = torch.as_tensor(traj.p(ts), dtype=torch.float64, device=DEV)[k]
            Xe = torch.einsum("nji,nj->ni", R, P - p)
            del R, p
            Xc = Xe @ T_ce[:3, :3].T + T_ce[:3, 3]
            front = Xc[:, 2] > 0.5
            uv = torch.full((len(Pw), 2), float("nan"), dtype=torch.float64, device=DEV)
            if bool(front.any()):
                uv[front] = project(Xc[front], self.intr, self.model, self.grid, self.G)
            # rows off the sensor do not exist; a lens polynomial far outside its
            # valid range (radtan at wide angles) can put them millions of pixels away
            row = np.clip(np.nan_to_num(uv[:, 1].cpu().numpy() / self.H - 0.5), -0.5, 0.5)
        return uv.cpu().numpy(), Xc.cpu().numpy()


def sweep_edges(s: np.ndarray, jump: float = 0.3, rel: float = 0.05) -> np.ndarray:
    """Mask of points on the near side of a range discontinuity along their ring."""
    r = np.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2)
    edge = np.zeros(len(s), bool)
    order = np.lexsort((s["t"], s["ring"]))
    rr, ring = r[order], s["ring"][order]
    same = ring[1:] == ring[:-1]
    d = rr[1:] - rr[:-1]
    thr = np.maximum(jump, rel * np.minimum(rr[1:], rr[:-1]))
    big = same & (np.abs(d) > thr)
    near_prev = big & (d > 0)          # next point is farther: prev is the near side
    near_next = big & (d < 0)
    e = np.zeros(len(s), bool)
    e[:-1] |= near_prev
    e[1:] |= near_next
    edge[order] = e
    return edge


def frame_points(clip: Path, t_img: int, traj: Trajectory, T_el: np.ndarray, n_sweeps: int = 3):
    files = sorted((clip / "lidar").glob("*.npy"))
    stamps = np.array([int(f.stem) for f in files])
    i = int(np.clip(np.searchsorted(stamps, t_img) - 1, 0, len(files) - 1))
    sel = [k for k in range(max(0, i - n_sweeps // 2), min(len(files), i + n_sweeps // 2 + 1))
           if traj.covers(stamps[k], stamps[k] + int(1e8))[0]]
    P, E, I, D = [], [], [], []
    for k in sel:
        s = load_sweep(files[k], rmin=2.0, rmax=80.0)
        t = stamps[k] + s["t"].astype(np.float64) * 1e9
        pe = sweep_xyz(s) @ T_el[:3, :3].T + T_el[:3, 3]
        P.append(np.einsum("nij,nj->ni", traj.R(t), pe) + traj.p(t))
        E.append(sweep_edges(s))
        I.append(s["intensity"])
        D.append(np.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2))
    return np.concatenate(P), np.concatenate(E), np.concatenate(I), np.concatenate(D)


def zbuffer_visible(uv: np.ndarray, depth: np.ndarray, W: int, H: int, cell: int = 6, tol: float = 0.08):
    """Drop points hidden behind nearer ones (the LiDAR sees around corners the camera cannot)."""
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    idx = np.flatnonzero(ok)
    cx = (uv[idx, 0] // cell).astype(int)
    cy = (uv[idx, 1] // cell).astype(int)
    key = cy * (W // cell + 1) + cx
    zmin = np.full(key.max() + 1, np.inf)
    np.minimum.at(zmin, key, depth[idx])
    vis = np.zeros(len(uv), bool)
    vis[idx] = depth[idx] <= zmin[key] * (1 + tol) + 0.2
    return vis


def edge_error(img_gray: np.ndarray, uv: np.ndarray, mask_edge: np.ndarray, valid_mask: np.ndarray):
    g = cv2.GaussianBlur(img_gray, (0, 0), 1.2)
    edges = cv2.Canny(g, 40, 110)
    dtf = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    sel = mask_edge
    u = uv[sel]
    ui, vi = u[:, 0].astype(int), u[:, 1].astype(int)
    inside = valid_mask[vi, ui] > 0
    return dtf[vi[inside], ui[inside]]


def overlay(img: np.ndarray, uv: np.ndarray, rng: np.ndarray, vis: np.ndarray, radius: int = 2,
            max_range: float = 40.0) -> np.ndarray:
    out = img.copy()
    sel = np.flatnonzero(vis)
    sel = sel[np.argsort(-rng[sel])]
    col = cv2.applyColorMap(np.clip(255 * (1 - rng[sel] / max_range), 0, 255).astype(np.uint8)[:, None],
                            cv2.COLORMAP_TURBO)[:, 0]
    for (u, v), c in zip(uv[sel].astype(int), col):
        cv2.circle(out, (int(u), int(v)), radius, tuple(int(x) for x in c), -1, lineType=cv2.LINE_AA)
    return out


def evaluate(calib: Calib, clip: Path, traj: Trajectory, channel: str, mask: np.ndarray | None,
             n_frames: int = 20, overlays: Path | None = None, tag: str = "") -> dict:
    files = sorted((clip / "cam" / channel).glob("*.jpg"))
    moving = [f for f in files if traj.speed(int(f.stem))[0] > 1.0] or files
    pick = moving[:: max(1, len(moving) // n_frames)][:n_frames]
    errs, per_frame = [], []
    for k, f in enumerate(pick):
        t_img = int(f.stem)
        img = cv2.imread(str(f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        P, E, I, D = frame_points(clip, t_img, traj, calib.T_el)
        uv, Xc = calib.project_world(P, t_img, traj)
        dist = np.linalg.norm(Xc, axis=1)
        vis = zbuffer_visible(uv, dist, calib.W, calib.H)
        vm = mask if mask is not None else np.full((calib.H, calib.W), 255, np.uint8)
        e = edge_error(gray, uv, vis & E & (dist < 40), vm)
        errs.append(e)
        per_frame.append(float(np.median(e)) if len(e) else np.nan)
        if overlays is not None and k % max(1, n_frames // 4) == 0:
            overlays.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlays / f"{tag}{channel}_{clip.name}_{t_img}.jpg"),
                        overlay(img, uv, dist, vis & (dist < 60)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    e = np.concatenate(errs)
    return {"n_edge_points": int(len(e)), "median_px": float(np.median(e)), "mean_px": float(np.mean(np.minimum(e, 20))),
            "within_2px": float(np.mean(e <= 2)), "within_1px": float(np.mean(e <= 1)),
            "per_frame_median": per_frame}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("calib", type=Path)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--overlays", type=Path, default=None)
    a = p.parse_args()
    cal = Calib(a.calib)
    mask = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE) if a.mask else None
    r = evaluate(cal, a.clip, Trajectory(a.ins), cal.c["channel"], mask, a.frames, a.overlays,
                 tag=a.calib.stem.replace("calib_", "") + "__")
    print(json.dumps({k: v for k, v in r.items() if k != "per_frame_median"}))


if __name__ == "__main__":
    main()
