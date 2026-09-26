"""2D-3D correspondences between camera images and LiDAR, by learned matching.

The LiDAR sweeps around an image are deskewed into the world, projected with
the current camera calibration and splatted into a reflectivity image of the
camera's view. LoFTR (Sun et al., CVPR 2021; the cross-modal use follows Koide
et al., ICRA 2023) matches that rendering against the real image; since every
rendered pixel remembers the LiDAR point behind it, each match is a 2D-3D
correspondence: an image pixel and a world point.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from evaluate import Calib
from traj import Trajectory, load_sweep, sweep_xyz

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_LOFTR = None


def loftr():
    global _LOFTR
    if _LOFTR is None:
        import kornia.feature as KF
        _LOFTR = KF.LoFTR(pretrained="outdoor").to(DEV).eval()
    return _LOFTR


def window_points(clip, t_img: int, traj: Trajectory, T_el: np.ndarray, window_s: float = 0.3):
    """World points (and intensity) of the sweeps within +-window_s of t_img."""
    files = sorted((clip / "lidar").glob("*.npy"))
    stamps = np.array([int(f.stem) for f in files])
    sel = np.flatnonzero((np.abs(stamps + 5e7 - t_img) <= window_s * 1e9) & traj.covers(stamps, stamps + int(1e8)))
    P, I = [], []
    for k in sel:
        s = load_sweep(files[k], rmin=2.5, rmax=60.0)
        ut, inv = np.unique(s["t"], return_inverse=True)      # ~1000 firing times per sweep
        t = stamps[k] + ut.astype(np.float64) * 1e9
        pe = sweep_xyz(s) @ T_el[:3, :3].T + T_el[:3, 3]
        P.append(np.einsum("nij,nj->ni", traj.R(t)[inv], pe) + traj.p(t)[inv])
        I.append(s["intensity"])
    return np.concatenate(P), np.concatenate(I)


def render(calib: Calib, P: np.ndarray, I: np.ndarray, t_img: int, traj: Trajectory, radius: int = 2):
    """Reflectivity image of the camera's view, and the index of the point behind each pixel."""
    uv, Xc = calib.project_world(P, t_img, traj)
    d = np.linalg.norm(Xc, axis=1)
    W, H = calib.W, calib.H
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(-d[idx])]                    # far first, near overwrite
    img = np.zeros((H, W), np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), -1, np.int64)
    u = uv[idx, 0].astype(int)
    v = uv[idx, 1].astype(int)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            uu, vv = np.clip(u + dx, 0, W - 1), np.clip(v + dy, 0, H - 1)
            # later (nearer) writes win because of the far-to-near order
            img[vv, uu] = I[idx]
            zbuf[vv, uu] = d[idx]
            owner[vv, uu] = idx
    # fill the gaps between rings from the nearest rendered pixel (<= fill px away)
    empty = owner < 0
    if empty.any():
        dist, lab = cv2.distanceTransformWithLabels(empty.astype(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(~empty)
        src = np.zeros(lab.max() + 1, np.int64)
        src[lab[~empty]] = ys * W + xs
        fill = empty & (dist <= 3.0)
        flat = src[lab[fill]]
        img[fill] = img.reshape(-1)[flat]
        owner[fill] = owner.reshape(-1)[flat]
        zbuf[fill] = zbuf.reshape(-1)[flat]
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return img, owner, zbuf


def match(gray: np.ndarray, rend: np.ndarray, valid: np.ndarray, scale: float = 0.5,
          min_conf: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LoFTR between the camera image and the LiDAR rendering; returns (uv_img, uv_rend, conf)."""
    def prep(a):
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = a.shape
        a = a[: h - h % 8, : w - w % 8]
        return torch.from_numpy(a).float()[None, None].to(DEV) / 255.0
    r = rend.copy()
    lo, hi = np.percentile(r[valid], [2, 98]) if valid.any() else (0, 1)
    r = np.clip((r - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    r[~valid] = 0
    clahe = cv2.createCLAHE(3.0, (8, 8))           # equalize both: brightness vs reflectivity
    g = clahe.apply(gray.astype(np.uint8)).astype(np.float32)
    r = clahe.apply(r.astype(np.uint8)).astype(np.float32)
    with torch.no_grad():
        out = loftr()({"image0": prep(g), "image1": prep(r)})
    k0 = out["keypoints0"].cpu().numpy() / scale
    k1 = out["keypoints1"].cpu().numpy() / scale
    c = out["confidence"].cpu().numpy()
    keep = c >= min_conf
    return k0[keep], k1[keep], c[keep]


def correspondences(calib: Calib, clip, t_img: int, traj: Trajectory, mask: np.ndarray,
                    window_s: float = 0.5, max_shift: float = 120.0):
    """(uv_image (K,2), world points (K,3)) for one frame, filtered by a
    consistency check: the match must move the LiDAR pixel by a similar amount
    as its neighbours (median-shift gating) and land in the usable mask."""
    gray = cv2.cvtColor(cv2.imread(str(clip / "cam" / calib.c["channel"] / f"{t_img}.jpg")), cv2.COLOR_BGR2GRAY)
    P, I = window_points(clip, t_img, traj, calib.T_el, window_s)
    rend, owner, _ = render(calib, P, I, t_img, traj)
    valid = owner >= 0
    k_img, k_rend, conf = match(gray, rend, valid)
    ri = np.clip(k_rend.round().astype(int), [0, 0], [calib.W - 1, calib.H - 1])
    own = owner[ri[:, 1], ri[:, 0]]
    ii = np.clip(k_img.round().astype(int), [0, 0], [calib.W - 1, calib.H - 1])
    ok = (own >= 0) & (mask[ii[:, 1], ii[:, 0]] > 0)
    shift = k_img - k_rend
    ok &= np.linalg.norm(shift, axis=1) < max_shift
    if ok.sum() > 10:
        med = np.median(shift[ok], axis=0)
        ok &= np.linalg.norm(shift - med, axis=1) < 25.0
    return k_img[ok], P[own[ok]], {"n_raw": int(len(conf)), "n_kept": int(ok.sum()),
                                     "median_shift": np.median(shift[ok], axis=0).tolist() if ok.any() else None,
                                     "rend": rend, "gray": gray, "k_img": k_img[ok], "k_rend": k_rend[ok]}
