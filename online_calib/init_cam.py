"""Initial focal length and camera orientation on the car, from tracks + INS.

No SfM: the INS already knows how the car moved. For frame pairs from turns,
the camera's rotation angle (from the essential matrix of its tracks) must equal
the car's — whatever the mounting — and that angle scales with the assumed
focal length, so scanning the focal length finds it. The mounting rotation then
follows from hand-eye alignment: the rotation axes (turns) and the translation
directions (straights) seen by the camera are the car's, rotated by R_ce.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from traj import Trajectory


def _clipped(frame_ns, traj: Trajectory):
    """Frame times clamped into the trajectory, and which were inside it."""
    return np.clip(frame_ns, traj.t[0], traj.t[-1]), traj.covers(frame_ns)


def _pairs(frame_ns, traj: Trajectory, min_rot_deg=4.0, min_trans=2.0, max_gap=30):
    t, cov = _clipped(frame_ns, traj)
    R = traj.R(t)
    p = traj.p(t)
    out = []
    for i in range(0, len(frame_ns) - 1, 2):
        if not cov[i]:
            continue
        for j in range(i + 1, min(len(frame_ns), i + max_gap)):
            if not cov[j]:
                break
            ang = np.degrees(Rotation.from_matrix(R[i].T @ R[j]).magnitude())
            if ang >= min_rot_deg or np.linalg.norm(p[j] - p[i]) >= min_trans:
                out.append((i, j))
                break
    return out


def _common(obs_frame, obs_track, obs_xy, i, j):
    a = obs_frame == i
    b = obs_frame == j
    ta, tb = obs_track[a], obs_track[b]
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    return obs_xy[a][ia], obs_xy[b][ib]


def _rays_equidistant(xy, f, W, H):
    m = (xy - [W / 2, H / 2]) / f
    th = np.linalg.norm(m, axis=1)
    s = np.sin(th) / np.maximum(th, 1e-12)
    return np.stack([s * m[:, 0], s * m[:, 1], np.cos(th)], axis=1), th


def _relative_camera_motion(x1, x2, f, W, H):
    r1, th1 = _rays_equidistant(x1, f, W, H)
    r2, th2 = _rays_equidistant(x2, f, W, H)
    ok = (th1 < 1.1) & (th2 < 1.1)
    if ok.sum() < 30:
        return None
    p1 = r1[ok, :2] / r1[ok, 2:]
    p2 = r2[ok, :2] / r2[ok, 2:]
    cv2.setRNGSeed(0)            # RANSAC's RNG is per thread: seed it so calls are reproducible in any thread
    E, inl = cv2.findEssentialMat(p1, p2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / f)
    if E is None or E.shape != (3, 3):
        return None
    n, R, t, _ = cv2.recoverPose(E, p1, p2, np.eye(3), mask=inl)
    if n < 20:
        return None
    # recoverPose: x2 = R x1 + t  -> pose of camera j in camera i
    return R.T, (-R.T @ t).ravel()


def init_focal_and_rotation(tr: dict, traj: Trajectory, f_grid=None):
    frame_ns, of, ot, oxy = tr["frame_ns"], tr["obs_frame"], tr["obs_track"], tr["obs_xy"]
    W, H = (int(v) for v in tr["image_wh"])
    pairs = _pairs(frame_ns, traj)
    Re = traj.R(_clipped(frame_ns, traj)[0])
    pe = traj.p(_clipped(frame_ns, traj)[0])
    corr = [(i, j, *_common(of, ot, oxy, i, j)) for i, j in pairs]
    corr = [c for c in corr if len(c[2]) >= 60]
    turning = [c for c in corr
               if np.degrees(Rotation.from_matrix(Re[c[0]].T @ Re[c[1]]).magnitude()) >= 4.0]
    rng = np.random.default_rng(0)
    sample = [turning[k] for k in rng.choice(len(turning), min(40, len(turning)), replace=False)]
    f_grid = f_grid if f_grid is not None else np.arange(400, 2800, 50)

    def score(f):
        errs = []
        for i, j, x1, x2 in sample:
            m = _relative_camera_motion(x1, x2, f, W, H)
            if m is None:
                continue
            ang_c = Rotation.from_matrix(m[0]).magnitude()
            ang_e = Rotation.from_matrix(Re[i].T @ Re[j]).magnitude()
            errs.append(abs(ang_c - ang_e) / ang_e)
        return np.median(errs) if errs else np.inf

    # OpenCV releases the GIL: score the focal lengths on all cores
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        scores = np.array(list(ex.map(score, f_grid)))
        f0 = f_grid[np.argmin(scores)]
        fine = np.arange(f0 - 60, f0 + 61, 10)
        fs = np.array(list(ex.map(score, fine)))
        f = float(fine[np.argmin(fs)])
        motions = list(ex.map(lambda c: _relative_camera_motion(c[2], c[3], f, W, H), corr[::2]))

    # hand-eye rotation from axes (turns) and translation directions (all pairs)
    A, B, w = [], [], []
    for (i, j, x1, x2), m in zip(corr[::2], motions):
        if m is None:
            continue
        R_c, t_c = m
        R_e = Re[i].T @ Re[j]
        t_e = Re[i].T @ (pe[j] - pe[i])
        ang = Rotation.from_matrix(R_e).magnitude()
        if ang > np.radians(3):
            A.append(Rotation.from_matrix(R_c).as_rotvec() / max(np.linalg.norm(Rotation.from_matrix(R_c).as_rotvec()), 1e-9))
            B.append(Rotation.from_matrix(R_e).as_rotvec() / ang)
            w.append(1.0)
        if np.linalg.norm(t_e) > 1.0 and ang < np.radians(6):
            A.append(t_c / np.linalg.norm(t_c))
            B.append(t_e / np.linalg.norm(t_e))
            w.append(1.0)
    A, B, w = np.array(A), np.array(B), np.array(w)
    # R_ce with A ~ R_ce B, robustly (reweight twice)
    for _ in range(3):
        Hm = (B * w[:, None]).T @ A
        U, _, Vt = np.linalg.svd(Hm)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R_ce = Vt.T @ D @ U.T
        res = np.linalg.norm(A - B @ R_ce.T, axis=1)
        w = 1.0 / (1.0 + (res / 0.05) ** 2)
    return {"f": f, "f_scores": dict(zip(map(float, f_grid), map(float, scores))),
            "R_ec": R_ce.T, "n_pairs": len(corr), "n_vectors": len(A),
            "handeye_residual_median": float(np.median(res))}
