#!/usr/bin/env python3
"""Link KLT tracks of different cameras that follow the same scene point.

    python online_calib/cross_tracks.py --clips CLIP [...] --calib DIR --ins TRAJ \
        --tracks DIR --masks DIR --out LINKS.npz [--stride 10]

For each pair of cameras whose views overlap (from the calibrations in DIR/final_<CH>.json),
every --stride-th frame of the first camera and the second camera's frame nearest in
time are both warped into one virtual pinhole camera looking at the middle of their
overlap -- so lens, focal length and windshield no longer differ between the two -- and
matched there with LoFTR; matches inconsistent with the calibrated rig and the
trajectory (epipolar error >= --epi-px pixels) are dropped. The two cameras' own KLT
features almost never sit on the same corner (different resolution, independent
detection), so instead each KLT point of the first camera is carried into the second:
through the affine fitted to the LoFTR matches around it, refined by normalized
cross-correlation in the virtual views, and kept if the pair passes the epipolar check.
That point then starts a new track in the second camera (pyramidal Lucas-Kanade,
forward-backward checked, up to --lk-frames frames each way), linked to the first
camera's track.

Writes LINKS.npz for rig_final.py --links:
  clip, cam_a, track_a, cam_b, track_b      one row per link (track_b is a new track)
  x_clip, x_cam, x_track, x_frame, x_xy     the new tracks' observations (frame index
                                            into that clip's tracks file frame_ns)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from camera_model import Grid, project, unproject
from lidar_match import loftr
from traj import Trajectory

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VW, VH = 1024, 768                                   # virtual view


class Cam:
    def __init__(self, path: Path):
        c = json.loads(path.read_text())
        self.c = c
        self.W, self.H = c["width"], c["height"]
        g = c.get("grid")
        self.grid = Grid(self.W, self.H, g["gx"], g["gy"]) if g else None
        t = lambda a: torch.tensor(a, dtype=torch.float64, device=DEV)
        self.intr, self.G = t(c["intr"]), (t(g["G"]) if g else None)
        self.T_lc = np.linalg.inv(np.array(c["T_cam_lidar"]))       # lidar <- camera
        self.T_el = np.array(c["T_ego_lidar"])
        self.dt, self.rs = c.get("dt_s", 0.0), c.get("rs_s", 0.0)
        self.f = float(c["intr"][0])

    def rays_lidar(self, uv: np.ndarray) -> np.ndarray:
        d = unproject(torch.tensor(uv, dtype=torch.float64, device=DEV), self.intr, self.c["model"],
                      self.grid, self.G).cpu().numpy()
        return d @ self.T_lc[:3, :3].T

    def project_lidar_dirs(self, d_l: np.ndarray, rng: float = 30.0) -> np.ndarray:
        """Pixels of directions given in the LiDAR frame (points rng metres out)."""
        dc = d_l @ self.T_lc[:3, :3]
        uv = np.full((len(dc), 2), np.nan)
        ok = dc[:, 2] > 0.1
        uv[ok] = project(torch.tensor(dc[ok] * rng, device=DEV), self.intr, self.c["model"],
                         self.grid, self.G).cpu().numpy()
        return uv


def virtual_view(a: Cam, b: Cam, mask_a, mask_b):
    """Virtual camera (R_lidar_virtual, f) over the overlap of a and b, and both remap tables."""
    uu, vv = np.meshgrid(np.linspace(0, a.W - 1, 64), np.linspace(0, a.H - 1, 40))
    uv = np.stack([uu.ravel(), vv.ravel()], 1)
    ra = a.rays_lidar(uv)
    ra /= np.linalg.norm(ra, axis=1, keepdims=True)
    ub = b.project_lidar_dirs(ra)
    inb = np.isfinite(ub).all(1) & (ub[:, 0] >= 0) & (ub[:, 0] < b.W) & (ub[:, 1] >= 0) & (ub[:, 1] < b.H)
    ia = uv.astype(int)
    inb &= mask_a[ia[:, 1], ia[:, 0]] > 0
    ib = np.clip(np.nan_to_num(ub).astype(int), 0, [b.W - 1, b.H - 1])
    inb &= mask_b[ib[:, 1], ib[:, 0]] > 0
    if inb.sum() < 50:
        return None
    ov = ra[inb]
    z = ov.mean(0)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)                  # horizontal, LiDAR z is up
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R_lv = np.stack([-x, -y, z], 1)                     # virtual: x right, y down, z forward
    dv = ov @ R_lv
    ang = np.abs(dv[:, :2] / dv[:, 2:])
    fx = (VW / 2) / max(np.percentile(ang[:, 0], 98), 0.1)
    fy = (VH / 2) / max(np.percentile(ang[:, 1], 98), 0.1)
    f = float(np.clip(min(fx, fy), 250.0, 1500.0))
    xs, ys = np.meshgrid(np.arange(VW), np.arange(VH))
    dv = np.stack([(xs - VW / 2) / f, (ys - VH / 2) / f, np.ones_like(xs, float)], -1).reshape(-1, 3)
    d_l = dv @ R_lv.T
    maps = []
    for cam, m in ((a, mask_a), (b, mask_b)):
        u = cam.project_lidar_dirs(d_l)
        ok = np.isfinite(u).all(1) & (u[:, 0] >= 0) & (u[:, 0] < cam.W - 1) & (u[:, 1] >= 0) & (u[:, 1] < cam.H - 1)
        iu = np.clip(np.nan_to_num(u).astype(int), 0, [cam.W - 1, cam.H - 1])
        ok &= m[iu[:, 1], iu[:, 0]] > 0
        u[~ok] = -1
        maps.append(u.reshape(VH, VW, 2).astype(np.float32))
    return R_lv, f, maps


def warp(img: np.ndarray, mp: np.ndarray) -> np.ndarray:
    return cv2.remap(img, mp[..., 0], mp[..., 1], cv2.INTER_LINEAR, borderValue=0)


def sample_map(mp: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Original-image pixel of virtual pixels (bilinear in the remap table)."""
    out = cv2.remap(mp, xy[:, 0].astype(np.float32)[None], xy[:, 1].astype(np.float32)[None],
                    cv2.INTER_LINEAR)[0]
    return out


def to_virtual(cam: Cam, uv: np.ndarray, R_lv: np.ndarray, f: float) -> np.ndarray:
    """Original pixels of a camera -> virtual view pixels (NaN behind it)."""
    d = cam.rays_lidar(uv) @ R_lv
    out = np.full((len(uv), 2), np.nan)
    ok = d[:, 2] > 0.05
    out[ok] = d[ok, :2] / d[ok, 2:] * f + [VW / 2, VH / 2]
    return out


def local_affine(src: np.ndarray, dst: np.ndarray, q: np.ndarray, radius: float, min_n: int = 5,
                 max_rms: float = 1.5):
    """Map query points q through the affine fitted to the (src -> dst) matches within
    `radius` of each; (predictions, usable mask)."""
    pred = np.full((len(q), 2), np.nan)
    good = np.zeros(len(q), bool)
    tree = cKDTree(src)
    qq = np.nan_to_num(q, nan=-1e6)
    for k, nb in enumerate(tree.query_ball_point(qq, radius)):
        if len(nb) < min_n:
            continue
        A_ = np.c_[src[nb], np.ones(len(nb))]
        M, *_ = np.linalg.lstsq(A_, dst[nb], rcond=None)
        rms = np.sqrt(np.mean(np.sum((A_ @ M - dst[nb]) ** 2, 1)))
        if rms > max_rms:
            continue
        pred[k] = np.r_[qq[k], 1.0] @ M
        good[k] = True
    return pred, good


def match_pair(img_a, img_b, valid_a, valid_b):
    m = loftr()
    prep = lambda g: torch.from_numpy(cv2.createCLAHE(3.0, (8, 8)).apply(g)).float()[None, None].to(DEV) / 255.0
    with torch.no_grad():
        out = m({"image0": prep(img_a), "image1": prep(img_b)})
    ka, kb = out["keypoints0"].cpu().numpy(), out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    ia, ib = ka.astype(int), kb.astype(int)
    ok = (conf > 0.3) & valid_a[ia[:, 1], ia[:, 0]] & valid_b[ib[:, 1], ib[:, 0]]
    return ka[ok], kb[ok]


def epipolar_px(a: Cam, b: Cam, ua, ub, ta: int, tb: int, traj: Trajectory):
    """Angular distance of b's ray from the epipolar plane of a's ray, in b's pixels."""
    ra, rb = a.rays_lidar(ua), b.rays_lidar(ub)
    T_wa = traj.T(np.array([ta]))[0] @ a.T_el
    T_wb = traj.T(np.array([tb]))[0] @ b.T_el
    ca = T_wa[:3, :3] @ a.T_lc[:3, 3] + T_wa[:3, 3]
    cb = T_wb[:3, :3] @ b.T_lc[:3, 3] + T_wb[:3, 3]
    da, db = ra @ T_wa[:3, :3].T, rb @ T_wb[:3, :3].T
    da /= np.linalg.norm(da, axis=1, keepdims=True)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    base = cb - ca
    n = np.cross(base, da)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    small = nn[:, 0] < 1e-9
    n = n / np.maximum(nn, 1e-12)
    ang = np.abs(np.arcsin(np.clip(np.einsum("ni,ni->n", n, db), -1, 1)))
    ang[small] = np.arccos(np.clip(np.einsum("ni,ni->n", da, db), -1, 1))[small]
    # the point must be in front of both cameras: b's ray on a's side of the baseline
    front = np.einsum("ni,ni->n", da, db) > 0.5
    return ang * b.f, front


NEW_TRACK0 = 10_000_000                              # ids of the new tracks, clear of KLT's


def ncc_refine(va, vb, xa, pred, half=7, search=4, min_ncc=0.8):
    """Refine predicted positions in vb of points xa in va by normalized cross-correlation."""
    out = np.full_like(pred, np.nan)
    H, W = va.shape
    for k, ((x0, y0), (x1, y1)) in enumerate(zip(xa, pred)):
        xi, yi, xj, yj = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        if not (half <= xi < W - half and half <= yi < H - half and
                half + search <= xj < W - half - search and half + search <= yj < H - half - search):
            continue
        tpl = va[yi - half:yi + half + 1, xi - half:xi + half + 1]
        win = vb[yj - half - search:yj + half + search + 1, xj - half - search:xj + half + search + 1]
        if tpl.std() < 4 or win.std() < 4:
            continue
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (px, py) = cv2.minMaxLoc(r)
        if mx < min_ncc or not (0 < px < r.shape[1] - 1 and 0 < py < r.shape[0] - 1):
            continue
        dx = 0.5 * (r[py, px - 1] - r[py, px + 1]) / (r[py, px - 1] - 2 * mx + r[py, px + 1] + 1e-12)
        dy = 0.5 * (r[py - 1, px] - r[py + 1, px]) / (r[py - 1, px] - 2 * mx + r[py + 1, px] + 1e-12)
        # sub-pixel offset of the template centre relative to the rounded query position
        out[k] = (xj - search + px + dx + (x0 - xi), yj - search + py + dy + (y0 - yi))
    return out


class Frames:
    """Grayscale frames of one camera in one clip, decoded once."""

    def __init__(self, clip: Path, ch: str, frame_ns: np.ndarray):
        self.clip, self.ch, self.ns, self.cache = clip, ch, frame_ns, {}

    def __getitem__(self, k: int):
        if k not in self.cache:
            if len(self.cache) > 80:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = cv2.imread(str(self.clip / "cam" / self.ch / f"{self.ns[k]}.jpg"), cv2.IMREAD_GRAYSCALE)
        return self.cache[k]


LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def lk_track(frames: Frames, j: int, pts: np.ndarray, mask: np.ndarray, n: int, fb_max: float = 0.5):
    """Track points seeded in frame j both ways; per point a list of (frame, xy)."""
    obs = [[(j, p.copy())] for p in pts]
    for step in (1, -1):
        cur = pts.astype(np.float32).copy()
        alive = np.arange(len(pts))
        k = j
        for _ in range(n):
            k2 = k + step
            if k2 < 0 or k2 >= len(frames.ns) or not len(alive):
                break
            g0, g1 = frames[k], frames[k2]
            if g0 is None or g1 is None:
                break
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, cur, None, **LK)
            p0, st0, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
            ok = (st1[:, 0] == 1) & (st0[:, 0] == 1) & (np.linalg.norm(p0 - cur, axis=1) < fb_max)
            h, w = mask.shape
            inside = (p1[:, 0] >= 2) & (p1[:, 0] < w - 2) & (p1[:, 1] >= 2) & (p1[:, 1] < h - 2)
            ok &= inside
            ok[inside] &= mask[p1[inside, 1].astype(int), p1[inside, 0].astype(int)] > 0
            for q, xy in zip(alive[ok], p1[ok]):
                obs[q].append((k2, xy.astype(np.float64)))
            cur, alive, k = p1[ok], alive[ok], k2
    return obs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clips", type=Path, nargs="+", required=True)
    p.add_argument("--calib", type=Path, required=True, help="dir with final_<CHANNEL>.json")
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", nargs="+", default=["CAM_FRONT:CAM_FRONT_LEFT", "CAM_FRONT:CAM_FRONT_RIGHT",
                                                  "CAM_FRONT:CAM_TRAFFIC", "CAM_FRONT_LEFT:CAM_BACK_LEFT",
                                                  "CAM_FRONT_RIGHT:CAM_BACK_RIGHT", "CAM_BACK:CAM_BACK_LEFT",
                                                  "CAM_BACK:CAM_BACK_RIGHT"])
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--epi-px", type=float, default=3.0)
    p.add_argument("--affine-radius", type=float, default=24.0, help="virtual-view pixels")
    p.add_argument("--lk-frames", type=int, default=10)
    p.add_argument("--min-len", type=int, default=5, help="shortest new track kept (observations)")
    a = p.parse_args()
    t0 = time.time()
    traj = Trajectory(a.ins)
    links, xobs = [], []
    next_id = NEW_TRACK0
    for pair in a.pairs:
        ca, cb = pair.split(":")
        A, B = Cam(a.calib / f"final_{ca}.json"), Cam(a.calib / f"final_{cb}.json")
        ma = cv2.imread(str(a.masks / f"{ca}.png"), cv2.IMREAD_GRAYSCALE)
        mb = cv2.imread(str(a.masks / f"{cb}.png"), cv2.IMREAD_GRAYSCALE)
        vv = virtual_view(A, B, ma, mb)
        if vv is None:
            print(f"{pair}: no overlap"); continue
        R_lv, fv, (map_a, map_b) = vv
        valid_a, valid_b = map_a[..., 0] >= 0, map_b[..., 0] >= 0
        for clip in a.clips:
            ta_ = dict(np.load(a.tracks / f"tracks_{clip.name}_{ca}.npz"))
            tb_ = dict(np.load(a.tracks / f"tracks_{clip.name}_{cb}.npz"))
            fa, fb = ta_["frame_ns"], tb_["frame_ns"]
            by_a = np.split(np.argsort(ta_["obs_frame"], kind="stable"),
                            np.cumsum(np.bincount(ta_["obs_frame"], minlength=len(fa)))[:-1])
            frames_b = Frames(clip, cb, fb)
            seen: dict = {}                      # track of a -> new tracks in b (frame spans)
            n_m = n_epi = n_seed = n_new = 0
            for i in range(0, len(fa), a.stride):
                j = int(np.argmin(np.abs(fb - fa[i])))
                if abs(int(fb[j]) - int(fa[i])) > 20_000_000:
                    continue
                ta_t, tb_t = int(fa[i]) + int(A.dt * 1e9), int(fb[j]) + int(B.dt * 1e9)
                if not traj.covers([ta_t, tb_t]).all():
                    continue
                ga = cv2.imread(str(clip / "cam" / ca / f"{fa[i]}.jpg"), cv2.IMREAD_GRAYSCALE)
                gb = frames_b[j]
                if ga is None or gb is None:
                    continue
                va, vb = warp(ga, map_a), warp(gb, map_b)
                ka, kb = match_pair(va, vb, valid_a, valid_b)
                if len(ka) < 8:
                    continue
                ua, ub = sample_map(map_a, ka), sample_map(map_b, kb)
                ok = (ua >= 0).all(1) & (ub >= 0).all(1)
                ka, kb, ua, ub = ka[ok], kb[ok], ua[ok].astype(np.float64), ub[ok].astype(np.float64)
                n_m += len(ua)
                e, front = epipolar_px(A, B, ua, ub, ta_t, tb_t, traj)
                ok = (e < a.epi_px) & front
                ka, kb = ka[ok], kb[ok]
                n_epi += len(ka)
                oa = by_a[i]
                # only a-tracks not yet carried over near this time
                oa = np.array([o for o in oa if not any(lo <= j <= hi for lo, hi in seen.get(int(ta_["obs_track"][o]), []))],
                              dtype=np.int64)
                if len(ka) < 8 or not len(oa):
                    continue
                xa_v = to_virtual(A, ta_["obs_xy"][oa].astype(np.float64), R_lv, fv)
                pred, good = local_affine(ka, kb, xa_v, radius=a.affine_radius)
                ref = np.full_like(pred, np.nan)
                ref[good] = ncc_refine(va, vb, xa_v[good], pred[good])
                ok = np.isfinite(ref).all(1)
                if not ok.any():
                    continue
                ubk = sample_map(map_b, ref[ok])
                okb = (ubk >= 0).all(1)
                sel = np.flatnonzero(ok)[okb]
                ubk = ubk[okb].astype(np.float64)
                e2, fr2 = epipolar_px(A, B, ta_["obs_xy"][oa[sel]].astype(np.float64), ubk, ta_t, tb_t, traj)
                keep = (e2 < a.epi_px) & fr2
                sel, ubk = sel[keep], ubk[keep]
                n_seed += len(sel)
                if not len(sel):
                    continue
                for tid, ob in zip(ta_["obs_track"][oa[sel]], lk_track(frames_b, j, ubk, mb, a.lk_frames)):
                    if len(ob) < a.min_len:
                        continue
                    fr = [f for f, _ in ob]
                    seen.setdefault(int(tid), []).append((min(fr), max(fr)))
                    links.append((clip.name, ca, int(tid), cb, next_id))
                    xobs += [(clip.name, cb, next_id, f, xy) for f, xy in ob]
                    next_id += 1
                    n_new += 1
            print(f"{pair} {clip.name}: {n_m} matches, {n_epi} epipolar-consistent, {n_seed} carried over, "
                  f"{n_new} new tracks ({time.time() - t0:.0f}s)", flush=True)
    L = list(zip(*links)) if links else [[]] * 5
    Xo = list(zip(*xobs)) if xobs else [[]] * 5
    np.savez(a.out, clip=np.array(L[0]), cam_a=np.array(L[1]), track_a=np.array(L[2], np.int64),
             cam_b=np.array(L[3]), track_b=np.array(L[4], np.int64),
             x_clip=np.array(Xo[0]), x_cam=np.array(Xo[1]), x_track=np.array(Xo[2], np.int64),
             x_frame=np.array(Xo[3], np.int64), x_xy=np.array(Xo[4], np.float64).reshape(-1, 2))
    print(f"-> {a.out}: {len(links)} links, {len(xobs)} new observations ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
