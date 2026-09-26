"""Joint bundle adjustment of one camera against the INS trajectory and the LiDAR map.

Unknowns: lens intrinsics, the windshield field (optional), the camera <- LiDAR
extrinsic, the camera time offset and rolling-shutter readout, and every
landmark. Camera poses are not unknowns: they are the INS trajectory composed
with the (fixed, separately calibrated) LiDAR -> ego extrinsic and the camera
extrinsic being estimated, evaluated at each observation's own time.

Residuals
  reprojection   tracked pixel vs projected landmark (Cauchy, 1 px scale)
  LiDAR          landmark to the local plane of the accumulated LiDAR map
                 (Cauchy, 5 cm scale) — this is what ties the camera to the
                 LiDAR, and makes scale and translation observable
  regularizer    windshield grids: smooth, and with no constant/linear part
                 (those belong to the lens model and the extrinsic)

Solved with Levenberg-Marquardt; landmarks are eliminated with the Schur
complement, so each step solves only the camera's few hundred parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from camera_model import N_INTR, Grid, project, unproject
from traj import Trajectory, deskew_to_world, load_sweep, voxel_down

SWEEP_NS = 100_000_000                 # a sweep's points span [header, header + 0.1 s]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64


def rodrigues(w: torch.Tensor) -> torch.Tensor:
    """(N,3) rotation vectors -> (N,3,3)."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + s * K + (1 - c) * (K @ K)


def skew(v: torch.Tensor) -> torch.Tensor:
    K = torch.zeros(v.shape[0], 3, 3, dtype=v.dtype, device=v.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -v[:, 2], v[:, 1], v[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 0], -v[:, 1], v[:, 0]
    return K


# --------------------------------------------------------------------- data
@dataclass
class Problem:
    model: str
    W: int
    H: int
    T_el0: np.ndarray                 # ego <- lidar, as calibrated beforehand (State.tel corrects it)
    fr_R: torch.Tensor                # (F,3,3) world <- ego at frame time
    fr_p: torch.Tensor                # (F,3)
    fr_w: torch.Tensor                # (F,3) ego angular rate (ego frame)
    fr_v: torch.Tensor                # (F,3) world velocity
    fr_clip: np.ndarray               # (F,) clip index
    obs_f: torch.Tensor               # (N,) frame index
    obs_j: torch.Tensor               # (N,) landmark index
    obs_uv: torch.Tensor              # (N,2)
    n_lm: int
    maps: list = field(default_factory=list)       # per clip (points, tree)
    lm_clip: np.ndarray | None = None
    lm_track: np.ndarray | None = None             # the landmark's track id in its clip's tracks file


@dataclass
class State:
    intr: np.ndarray
    T_cl: np.ndarray                  # camera <- lidar (4x4)
    dt: float = 0.0
    rs: float = 0.0
    G: np.ndarray | None = None       # (4, n) windshield grid, or None
    X: np.ndarray | None = None       # (M,3) landmarks (world)
    # Correction to the LiDAR -> ego translation. Tracks fix the camera on the
    # car, the LiDAR planes fix it relative to the LiDAR; the LiDAR's own
    # position on the car is what reconciles the two, so it is estimated here.
    tel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # ... and its rotation (right perturbation, LiDAR frame). Tracks fix the
    # camera's rotation on the car, LiDAR matches its rotation relative to the
    # LiDAR; together they fix the LiDAR's rotation on the car.
    rel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Per-frame camera pose corrections (F,6: rotation vector, translation; a left
    # perturbation in the camera frame). Only for evaluating a frozen calibration
    # with poses freed from the INS; never estimated together with the other blocks.
    pf: np.ndarray | None = None


def build_problem(tracks: list[dict], traj: Trajectory, T_el: np.ndarray, model: str,
                  frame_step: int = 2, min_obs: int = 3) -> Problem:
    frs, obs = [], []
    f_off = j_off = 0
    fr_clip = []
    lm_clip = []
    lm_track = []
    for ci, tr in enumerate(tracks):
        fns = tr["frame_ns"]
        of, ot, oxy = tr["obs_frame"], tr["obs_track"], tr["obs_xy"]
        keep = (of % frame_step == 0) & traj.covers(fns)[of]
        of, ot, oxy = of[keep], ot[keep], oxy[keep]
        cnt = np.bincount(ot)
        keep = cnt[ot] >= min_obs
        of, ot, oxy = of[keep], ot[keep], oxy[keep]
        uniq, oj = np.unique(ot, return_inverse=True)
        frs.append(fns)
        fr_clip.append(np.full(len(fns), ci))
        obs.append((of + f_off, oj + j_off, oxy))
        lm_clip.append(np.full(len(uniq), ci))
        lm_track.append(uniq)
        f_off += len(fns)
        j_off += len(uniq)
    fns = np.concatenate(frs)
    fq = np.clip(fns, traj.t[0], traj.t[-1])          # frames outside the trajectory have no observations
    W, H = (int(v) for v in tracks[0]["image_wh"])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=DEV)
    return Problem(
        model=model, W=W, H=H, T_el0=T_el.copy(),
        fr_R=t(traj.R(fq)), fr_p=t(traj.p(fq)), fr_w=t(traj.omega_ego(fq)), fr_v=t(traj.vel_world(fq)),
        fr_clip=np.concatenate(fr_clip),
        obs_f=torch.as_tensor(np.concatenate([o[0] for o in obs]), device=DEV),
        obs_j=torch.as_tensor(np.concatenate([o[1] for o in obs]), device=DEV),
        obs_uv=t(np.concatenate([o[2] for o in obs])),
        n_lm=j_off, lm_clip=np.concatenate(lm_clip), lm_track=np.concatenate(lm_track))


def build_map(clip_dir, traj: Trajectory, T_el: np.ndarray, stride: int = 2, voxel: float = 0.08):
    """World map of a clip: (points, kd-tree, sweep index per point, R_world_ego per sweep).

    The sweep rotations let the plane residuals follow a change of the LiDAR's
    position on the car: a map point moves by R_we(t_sweep) * delta_t_el."""
    pts, sid = [], []
    files = sorted((clip_dir / "lidar").glob("*.npy"))[::stride]
    files = [f for f in files if traj.covers(int(f.stem), int(f.stem) + SWEEP_NS)[0]]
    for k, f in enumerate(files):
        s = load_sweep(f, rmin=3.0, rmax=60.0)
        pts.append(deskew_to_world(s, int(f.stem), traj, T_el))
        sid.append(np.full(len(s), k, np.int32))
        if len(pts) % 20 == 0:                 # keep memory bounded
            P, S = voxel_down(np.concatenate(pts), voxel, np.concatenate(sid))
            pts, sid = [P], [S]
    P, S = voxel_down(np.concatenate(pts), voxel, np.concatenate(sid))
    R = traj.R(np.array([int(f.stem) for f in files]))
    return P, cKDTree(P), S, R


# ---------------------------------------------------------------- geometry
def T_el_of(T_el0: np.ndarray, tel, rel) -> np.ndarray:
    T = T_el0.copy()
    T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(np.asarray(rel)).as_matrix()
    T[:3, 3] += tel
    return T


def T_le_of(pb: Problem, st: State) -> np.ndarray:
    return np.linalg.inv(T_el_of(pb.T_el0, st.tel, st.rel))


def camera_points(pb: Problem, st: State, X: torch.Tensor, idx=None):
    """Landmarks in the camera frame for every observation, plus what the
    Jacobians need: dXc/dX (N,3,3), dXc/dDelta (N,3), frac row (N,)."""
    of = pb.obs_f if idx is None else pb.obs_f[idx]
    oj = pb.obs_j if idx is None else pb.obs_j[idx]
    uv = pb.obs_uv if idx is None else pb.obs_uv[idx]
    row = uv[:, 1] / pb.H - 0.5
    delta = st.dt + st.rs * row
    R, p, w, v = pb.fr_R[of], pb.fr_p[of], pb.fr_w[of], pb.fr_v[of]
    Xw = X[oj]
    y = torch.einsum("nji,nj->ni", R, Xw - p - v * delta[:, None])      # R^T (...)
    Rw = rodrigues(-w * delta[:, None])
    Xe = torch.einsum("nij,nj->ni", Rw, y)
    T_ce = torch.as_tensor(st.T_cl @ T_le_of(pb, st), dtype=DT, device=DEV)
    Rce, tce = T_ce[:3, :3], T_ce[:3, 3]
    Xc = Xe @ Rce.T + tce
    M = Rce @ Rw @ R.transpose(1, 2)                                      # dXc/dXw
    dXe = -torch.cross(w, Xe, dim=-1) - torch.einsum("nij,njk,nk->ni", Rw, R.transpose(1, 2), v)
    dXc_dDelta = dXe @ Rce.T
    if st.pf is not None:
        pf = torch.as_tensor(st.pf, dtype=DT, device=DEV)[of]
        Rf = rodrigues(pf[:, :3])
        Xc = torch.einsum("nij,nj->ni", Rf, Xc) + pf[:, 3:]
        M = Rf @ M
        dXc_dDelta = torch.einsum("nij,nj->ni", Rf, dXc_dDelta)
    return Xc, M, dXc_dDelta, row


def param_layout(pb: Problem, st: State, active: dict) -> dict:
    """Column offsets of the active parameter blocks."""
    lay, off = {}, 0
    for name, size in [("intr", N_INTR[pb.model]), ("xi", 6), ("dt", 1), ("rs", 1), ("tel", 3), ("rel", 3),
                       ("Ginf", 2 * (st.G.shape[1] if st.G is not None else 0)),
                       ("Gnear", 2 * (st.G.shape[1] if st.G is not None else 0)),
                       ("pf", st.pf.size if st.pf is not None else 0)]:
        if active.get(name) and size:
            lay[name] = (off, size)
            off += size
    lay["P"] = off
    return lay


@dataclass
class RowJacobian:
    """d residual / d camera parameters, on the device, split by structure:
    Jg (2N, Kg) in columns cg (the same for every row), and Vs (2N, Ks) in the
    per-row columns Cs (2N, Ks)."""
    P: int
    cg: torch.Tensor
    Jg: torch.Tensor
    Cs: torch.Tensor
    Vs: torch.Tensor


def residuals_and_jacobians(pb: Problem, st: State, grid: Grid | None, lay: dict, sigma_px=1.0,
                            dense: bool = False, idx=None, X=None):
    """Reprojection residuals (2N) with J_theta (scipy csr) and J_X (N,2,3).
    dense=True: all three stay on the device and J_theta is a RowJacobian.
    idx: only these observations (a slice or index tensor)."""
    if X is None:
        X = torch.as_tensor(st.X, dtype=DT, device=DEV)
    Xc, M, dXc_dD, row = camera_points(pb, st, X, idx)
    R_ce = torch.as_tensor((st.T_cl @ T_le_of(pb, st))[:3, :3], dtype=DT, device=DEV)
    T_cl = torch.as_tensor(st.T_cl, dtype=DT, device=DEV)
    Xl = (Xc - T_cl[:3, 3]) @ T_cl[:3, :3]                      # the point in the LiDAR frame
    uv = pb.obs_uv if idx is None else pb.obs_uv[idx]
    of = pb.obs_f if idx is None else pb.obs_f[idx]
    r, Jt, JXc = theta_jacobians(pb.model, st, grid, lay, Xc, dXc_dD, row, uv, sigma_px, R_ce,
                                 Xl_for_rel=Xl, R_cl=T_cl[:3, :3], obs_f=of, dense=dense)
    JX = JXc @ M
    if dense:
        return r, Jt, JX, Xc
    return r, Jt, JX.cpu().numpy(), Xc.cpu().numpy()


def theta_jacobians(model: str, st: State, grid, lay: dict, Xc, dXc_dD, row, uv_obs, sigma_px,
                    R_ce_for_tel=None, Xl_for_rel=None, R_cl=None, obs_f=None, dense: bool = False):
    """Residuals (2N) and their Jacobian w.r.t. the camera parameters (csr), plus
    d pixel / d Xc (N,2,3), for points already in the camera frame.

    dense=True keeps everything on the device: residuals (2N,) and the Jacobian as
    RowJacobian — the blocks every row has in the same columns (lens, extrinsic,
    time, LiDAR-ego) as one dense matrix, the rest (windshield grid, per-frame
    poses) as per-row column indices and values."""
    intr = torch.as_tensor(st.intr, dtype=DT, device=DEV)
    G = torch.as_tensor(st.G, dtype=DT, device=DEV) if st.G is not None else None
    f = lambda xc, it: project(xc, it, model, grid, G)
    pix = f(Xc, intr)
    r = (pix - uv_obs) / sigma_px
    N = r.shape[0]
    # d pix / d Xc: three forward-mode passes
    JXc = torch.stack([torch.func.jvp(f, (Xc, intr), (e.expand_as(Xc), torch.zeros_like(intr)))[1]
                       for e in torch.eye(3, dtype=DT, device=DEV)], dim=-1) / sigma_px   # (N,2,3)
    cols, vals, same = [], [], []           # same: the block has the same columns in every row
    if "intr" in lay:
        Ji = torch.stack([torch.func.jvp(f, (Xc, intr), (torch.zeros_like(Xc), e))[1]
                          for e in torch.eye(len(st.intr), dtype=DT, device=DEV)], dim=-1) / sigma_px
        cols.append(lay["intr"][0] + torch.arange(Ji.shape[-1], device=DEV).expand(N, 2, -1))
        same.append(True)
        vals.append(Ji)
    if "xi" in lay:
        dXc = torch.cat([-skew(Xc), torch.eye(3, dtype=DT, device=DEV).expand(N, 3, 3)], dim=-1)
        Jx = JXc @ dXc
        cols.append(lay["xi"][0] + torch.arange(6, device=DEV).expand(N, 2, -1))
        same.append(True)
        vals.append(Jx)
    if "tel" in lay and R_ce_for_tel is not None:
        cols.append(lay["tel"][0] + torch.arange(3, device=DEV).expand(N, 2, -1))
        same.append(True)
        vals.append(JXc @ (-R_ce_for_tel))
    if "rel" in lay and Xl_for_rel is not None:
        # X_l = Exp(-rel) R_el0^T (X_e - t_el): d X_l / d rel = [X_l]x
        cols.append(lay["rel"][0] + torch.arange(3, device=DEV).expand(N, 2, -1))
        same.append(True)
        vals.append(JXc @ (R_cl @ skew(Xl_for_rel)))
    if "pf" in lay and obs_f is not None:
        dXc = torch.cat([-skew(Xc), torch.eye(3, dtype=DT, device=DEV).expand(N, 3, 3)], dim=-1)
        cols.append(lay["pf"][0] + 6 * obs_f[:, None, None] + torch.arange(6, device=DEV).expand(N, 2, -1))
        same.append(False)
        vals.append(JXc @ dXc)
    if "dt" in lay or "rs" in lay:
        Jd = torch.einsum("nij,nj->ni", JXc, dXc_dD)
        if "dt" in lay:
            cols.append(torch.full((N, 2, 1), lay["dt"][0], device=DEV))
            same.append(True)
            vals.append(Jd[..., None])
        if "rs" in lay:
            cols.append(torch.full((N, 2, 1), lay["rs"][0], device=DEV))
            same.append(True)
            vals.append((Jd * row[:, None])[..., None])
    if ("Ginf" in lay or "Gnear" in lay) and grid is not None:
        uv = project(Xc, intr, model)                    # lens-only pixel: where the grid is sampled
        gidx, gw = grid.basis(uv[:, 0], uv[:, 1])
        rho = 1.0 / Xc.norm(dim=-1)
        n = grid.n
        for name, scale in (("Ginf", None), ("Gnear", rho)):
            if name not in lay:
                continue
            w = gw / sigma_px if scale is None else gw * scale[:, None] / sigma_px
            o = lay[name][0]
            # row 0 (u) uses component 0, row 1 (v) uses component 1
            c = torch.stack([o + gidx, o + n + gidx], dim=1)          # (N,2,16)
            v_ = torch.stack([w, w], dim=1)
            cols.append(c)
            same.append(False)
            vals.append(v_)
    if dense:
        u = [k for k, sm in enumerate(same) if sm]
        v = [k for k, sm in enumerate(same) if not sm]
        e_i = torch.zeros(2 * N, 0, dtype=torch.long, device=DEV)
        e_v = torch.zeros(2 * N, 0, dtype=DT, device=DEV)
        jac = RowJacobian(
            P=lay["P"],
            cg=torch.cat([cols[k][0, 0] for k in u]) if u else e_i[0],
            Jg=torch.cat([vals[k] for k in u], dim=-1).reshape(2 * N, -1) if u else e_v,
            Cs=torch.cat([cols[k] for k in v], dim=-1).reshape(2 * N, -1) if v else e_i,
            Vs=torch.cat([vals[k] for k in v], dim=-1).reshape(2 * N, -1) if v else e_v)
        return r.reshape(-1), jac, JXc
    if cols:
        C = torch.cat(cols, dim=-1).reshape(2 * N, -1).cpu().numpy()
        V = torch.cat(vals, dim=-1).reshape(2 * N, -1).cpu().numpy()
        nnz = C.shape[1]
        Jt = sp.csr_matrix((V.ravel(), C.ravel(), np.arange(0, 2 * N * nnz + 1, nnz)),
                           shape=(2 * N, lay["P"]))
    else:
        Jt = sp.csr_matrix((2 * N, lay["P"]))
    return r.reshape(-1).cpu().numpy(), Jt, JXc


# ------------------------------------------------------ LiDAR correspondences
@dataclass
class LidarMatches:
    """2D-3D matches from refine_pnp.py: LiDAR points in the LiDAR frame at their
    image's time, the rate at which that point moves in the LiDAR frame (ego
    motion), and the image pixel. They tie the camera to the LiDAR directly."""
    X_l: torch.Tensor          # (K,3)
    dX_l: torch.Tensor         # (K,3) d X_l / d delta
    uv: torch.Tensor           # (K,2)

    @staticmethod
    def load(path) -> "LidarMatches":
        z = np.load(path)
        t = lambda a: torch.as_tensor(a, dtype=DT, device=DEV)
        return LidarMatches(t(z["X_l"]), t(z["dX_l"]), t(z["uv"]))


def match_residuals(lm: LidarMatches, st: State, grid, lay: dict, model: str, H: int, sigma_px: float):
    row = lm.uv[:, 1] / H - 0.5
    delta = st.dt + st.rs * row
    T_cl = torch.as_tensor(st.T_cl, dtype=DT, device=DEV)
    Xl = lm.X_l - delta[:, None] * lm.dX_l
    Xc = Xl @ T_cl[:3, :3].T + T_cl[:3, 3]
    dXc_dD = -(lm.dX_l @ T_cl[:3, :3].T)
    r, Jt, _ = theta_jacobians(model, st, grid, lay, Xc, dXc_dD, row, lm.uv, sigma_px, None)
    return r, Jt


def cauchy_w(r2: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + r2 / (c * c))


def cauchy_cost(r2: np.ndarray, c: float) -> float:
    return float(np.sum(c * c * np.log1p(r2 / (c * c))))


# --------------------------------------------------------------- landmarks
def camera_centers_and_rays(pb: Problem, st: State, grid, idx=None):
    of = pb.obs_f if idx is None else pb.obs_f[idx]
    uv = pb.obs_uv if idx is None else pb.obs_uv[idx]
    G = torch.as_tensor(st.G, dtype=DT, device=DEV) if st.G is not None else None
    ray = unproject(uv, torch.as_tensor(st.intr, dtype=DT, device=DEV), pb.model, grid, G)
    T_ec = np.linalg.inv(st.T_cl @ T_le_of(pb, st))
    Rec = torch.as_tensor(T_ec[:3, :3], dtype=DT, device=DEV)
    tec = torch.as_tensor(T_ec[:3, 3], dtype=DT, device=DEV)
    R, p = pb.fr_R[of], pb.fr_p[of]
    delta = st.dt
    p = p + pb.fr_v[of] * delta
    c = torch.einsum("nij,j->ni", R, tec) + p
    d = torch.einsum("nij,jk,nk->ni", R, Rec, ray)
    return c, d


def triangulate(pb: Problem, st: State, grid, min_parallax_deg=1.0):
    """Least-squares ray intersection per landmark, in chunks of observations (two
    passes: the rays are recomputed for the parallax and depth checks)."""
    M = pb.n_lm
    I3 = torch.eye(3, dtype=DT, device=DEV)
    A = torch.zeros(M, 3, 3, dtype=DT, device=DEV)
    b = torch.zeros(M, 3, dtype=DT, device=DEV)
    dm = torch.zeros(M, 3, dtype=DT, device=DEV)
    chunks = _obs_chunks(len(pb.obs_f))
    for sl in chunks:
        c, d = camera_centers_and_rays(pb, st, grid, sl)
        oj = pb.obs_j[sl]
        Pm = I3[None] - d[:, :, None] * d[:, None, :]
        A.index_add_(0, oj, Pm)
        b.index_add_(0, oj, torch.einsum("nij,nj->ni", Pm, c))
        dm.index_add_(0, oj, d)
    X = torch.linalg.solve(A + 1e-9 * I3, b)
    # parallax: angle spread of each track's rays around their mean direction
    dm = dm / dm.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    spread = torch.zeros(M, dtype=DT, device=DEV)
    min_depth = torch.full((M,), 1e9, dtype=DT, device=DEV)
    for sl in chunks:
        c, d = camera_centers_and_rays(pb, st, grid, sl)
        oj = pb.obs_j[sl]
        cosang = (d * dm[oj]).sum(-1).clamp(-1, 1)
        spread.scatter_reduce_(0, oj, torch.arccos(cosang), "amax")
        min_depth.scatter_reduce_(0, oj, ((X[oj] - c) * d).sum(-1), "amin")
    ok = (spread * 2 > np.radians(min_parallax_deg)) & (min_depth > 1.0) & X.isfinite().all(-1)
    return X.cpu().numpy(), ok.cpu().numpy()


def associate_planes(pb: Problem, X: np.ndarray, valid: np.ndarray, gate: float,
                     k: int = 16, radius: float = 0.4, min_planarity: float = 0.6):
    """(landmark ids, normals, centroids) of landmarks lying on a LiDAR plane."""
    ids, nrm, cen, rot = [], [], [], []
    for ci, (P, tree, sidx, sR) in enumerate(pb.maps):
        sel = np.flatnonzero(valid & (pb.lm_clip == ci))
        if not len(sel):
            continue
        d, nn = tree.query(X[sel], k=k, distance_upper_bound=radius)
        full = np.isfinite(d).all(axis=1)
        sel, nn = sel[full], nn[full]
        Q = P[nn]
        mu = Q.mean(axis=1)
        C = np.einsum("nki,nkj->nij", Q - mu[:, None], Q - mu[:, None]) / k
        w, v = np.linalg.eigh(C)
        n = v[:, :, 0]
        planar = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
        thick = np.sqrt(np.maximum(w[:, 0], 0))
        dist = np.abs(np.einsum("ni,ni->n", X[sel] - mu, n))
        ok = (planar > min_planarity) & (thick < 0.03) & (dist < gate)
        med = np.median(sidx[nn], axis=1).astype(int)
        ids.append(sel[ok]); nrm.append(n[ok]); cen.append(mu[ok]); rot.append(sR[med[ok]])
    if not ids:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3, 3))
    return np.concatenate(ids), np.concatenate(nrm), np.concatenate(cen), np.concatenate(rot)


# ----------------------------------------------------------------------- LM
def regularizer_rows(lay: dict, grid: Grid, lam_smooth: float, lam_affine: float, lam_ridge: float,
                     lam_near_scale: float = 1.0):
    """Linear residual rows R theta (for the grid blocks) as a csr matrix."""
    if grid is None or ("Ginf" not in lay and "Gnear" not in lay):
        return None
    D, A = grid.regularizer()
    n = grid.n
    blocks = []
    for name in ("Ginf", "Gnear"):
        if name not in lay:
            continue
        s = 1.0 if name == "Ginf" else lam_near_scale
        for comp in range(2):
            off = lay[name][0] + comp * n
            for mat, lam in ((D, lam_smooth), (A, lam_affine), (np.eye(n), lam_ridge)):
                m = sp.lil_matrix((mat.shape[0], lay["P"]))
                m[:, off:off + n] = lam * s * mat
                blocks.append(m.tocsr())
    return sp.vstack(blocks).tocsr()


class CamLM:
    """One camera's problem for Levenberg-Marquardt: its cost, its Gauss-Newton system
    and its update. lm_solve drives one of them; joint_lm_solve drives several that
    share unknowns (the LiDAR -> ego / motion-frame offset, landmarks seen by more
    than one camera).

    pp_prior: if given, a Gaussian prior of that width (pixels) pulling the principal
    point towards `pp_centre` (default: the image centre). The same forward-driving
    geometry that leaves fy free also makes cy trade almost exactly against the
    camera's pitch -- d pixels of cy is d/f radians of pitch for anything far away --
    so a camera that only yaws cannot separate them, and the fit wanders off with the
    extrinsic following it. A lens sitting in a machined mount is centred to a few tens
    of pixels, which is the information the fit is missing.

    tie_aspect: if given, a Gaussian prior of that width (pixels) on fy - fx, i.e.
    square pixels. A forward-looking camera on a car that mostly yaws and drives on
    the level leaves fy nearly free: stretching the world vertically by a and
    dividing fy by a leaves every projection unchanged, because yaw is a rotation
    about the very axis being stretched. Yaw pins fx, nothing pins fy. The sensor's
    pixels are square, so tying the two is the honest way to remove that gauge.
    """

    def __init__(self, pb: Problem, st: State, grid, active: dict, planes=None,
                 c_px: float = 2.0, sigma_plane: float = 0.05, c_plane: float = 2.0,
                 lam_smooth=30.0, lam_affine=300.0, lam_ridge=0.05, priors=None,
                 matches: LidarMatches | None = None, sigma_match: float = 2.0, c_match: float = 2.0,
                 tie_aspect: float | None = None, pp_prior: float | None = None, pp_centre=None):
        self.pb, self.grid, self.planes = pb, grid, planes
        self.c_px, self.sigma_plane, self.c_plane = c_px, sigma_plane, c_plane
        self.matches, self.sigma_match, self.c_match = matches, sigma_match, c_match
        self.tie_aspect, self.pp_prior = tie_aspect, pp_prior
        self.lay = param_layout(pb, st, active)
        self.Rreg = regularizer_rows(self.lay, grid, lam_smooth, lam_affine, lam_ridge)
        self.priors = priors or {}
        self.pp0 = (np.array([pb.W / 2.0, pb.H / 2.0]) if pp_centre is None else np.asarray(pp_centre, float))
        P, M = self.lay["P"], pb.n_lm
        self.on_dev = 8 * P * 3 * M <= DEVICE_F_BYTES

    # ------------------------------------------------------------ parameters
    def pack(self, s: State) -> np.ndarray:
        lay = self.lay
        v = np.zeros(lay["P"])
        if "intr" in lay: v[slice(lay["intr"][0], lay["intr"][0] + lay["intr"][1])] = s.intr
        if "dt" in lay: v[lay["dt"][0]] = s.dt
        if "rs" in lay: v[lay["rs"][0]] = s.rs
        if "tel" in lay: v[lay["tel"][0]:lay["tel"][0] + 3] = s.tel
        if "rel" in lay: v[lay["rel"][0]:lay["rel"][0] + 3] = s.rel
        if "Ginf" in lay: v[lay["Ginf"][0]:lay["Ginf"][0] + lay["Ginf"][1]] = s.G[:2].ravel()
        if "Gnear" in lay: v[lay["Gnear"][0]:lay["Gnear"][0] + lay["Gnear"][1]] = s.G[2:].ravel()
        return v

    def apply(self, s: State, dth: np.ndarray, dX: np.ndarray) -> State:
        lay = self.lay
        n = State(intr=s.intr.copy(), T_cl=s.T_cl.copy(), dt=s.dt, rs=s.rs,
                  G=None if s.G is None else s.G.copy(), X=s.X + dX, tel=s.tel.copy(), rel=s.rel.copy(),
                  pf=None if s.pf is None else s.pf.copy())
        if "pf" in lay:
            d = dth[lay["pf"][0]:lay["pf"][0] + lay["pf"][1]].reshape(-1, 6)
            dR = Rotation.from_rotvec(d[:, :3])
            n.pf = np.hstack([(dR * Rotation.from_rotvec(s.pf[:, :3])).as_rotvec(), dR.apply(s.pf[:, 3:]) + d[:, 3:]])
        if "tel" in lay: n.tel += dth[lay["tel"][0]:lay["tel"][0] + 3]
        if "rel" in lay: n.rel += dth[lay["rel"][0]:lay["rel"][0] + 3]
        if "intr" in lay: n.intr += dth[lay["intr"][0]:lay["intr"][0] + lay["intr"][1]]
        if "xi" in lay:
            xi = dth[lay["xi"][0]:lay["xi"][0] + 6]
            T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix(); T[:3, 3] = xi[3:]
            n.T_cl = T @ n.T_cl
        if "dt" in lay: n.dt += dth[lay["dt"][0]]
        if "rs" in lay: n.rs += dth[lay["rs"][0]]
        if "Ginf" in lay:
            o, k = lay["Ginf"]; n.G[:2] += dth[o:o + k].reshape(2, -1)
        if "Gnear" in lay:
            o, k = lay["Gnear"]; n.G[2:] += dth[o:o + k].reshape(2, -1)
        return n

    # ------------------------------------------------------------ camera-only rows
    def pp_rows(self, s: State):
        """(weight, residuals) of the principal-point prior, or None."""
        if self.pp_prior is None or "intr" not in self.lay:
            return None
        return 1.0 / self.pp_prior, (s.intr[2:4] - self.pp0) / self.pp_prior

    def aspect_row(self, s: State):
        """(weight, residual) of the fy == fx prior, or None."""
        if self.tie_aspect is None or "intr" not in self.lay:
            return None
        w_ = 1.0 / self.tie_aspect
        return w_, (s.intr[0] - s.intr[1]) * w_

    def prior_rows(self, s: State):
        """Zero-mean Gaussian priors, one row per parameter of the named blocks."""
        rows, vals, rhs = [], [], []
        cur = {"dt": [s.dt], "rs": [s.rs], "tel": list(s.tel), "rel": list(s.rel)}
        if s.pf is not None:
            cur["pf"] = list(s.pf.ravel())
        for name, sigma in self.priors.items():
            if name in self.lay:
                sg = np.broadcast_to(np.ravel(sigma), (len(cur[name]),)) if np.ndim(sigma) else [sigma] * len(cur[name])
                for k, value in enumerate(cur[name]):
                    rows.append(self.lay[name][0] + k); vals.append(1.0 / sg[k]); rhs.append(value / sg[k])
        return rows, vals, rhs

    def plane_res(self, s: State):
        ids, n, q, Rk = self.planes
        return np.einsum("ni,ni->n", s.X[ids] - q - np.einsum("nij,j->ni", Rk, s.tel), n) / self.sigma_plane

    def has_planes(self) -> bool:
        return self.planes is not None and len(self.planes[0]) > 0

    # ------------------------------------------------------------ cost
    def cost(self, s: State) -> float:
        pb, grid = self.pb, self.grid
        X = torch.as_tensor(s.X, dtype=DT, device=DEV)
        G = torch.as_tensor(s.G, dtype=DT, device=DEV) if s.G is not None else None
        r2 = []
        for sl in _obs_chunks(len(pb.obs_f)):
            Xc, *_ = camera_points(pb, s, X, sl)
            pix = project(Xc, torch.as_tensor(s.intr, dtype=DT, device=DEV), pb.model, grid, G)
            r2.append(((pix - pb.obs_uv[sl]) ** 2).sum(-1).cpu().numpy())
        cost = cauchy_cost(np.concatenate(r2) if r2 else np.zeros(0), self.c_px)
        if self.has_planes():
            cost += cauchy_cost(self.plane_res(s) ** 2, self.c_plane)
        th = self.pack(s)
        if self.matches is not None:
            rm, _ = match_residuals(self.matches, s, grid, {"P": 0}, pb.model, pb.H, self.sigma_match)
            cost += cauchy_cost(rm[0::2] ** 2 + rm[1::2] ** 2, self.c_match)
        if self.Rreg is not None:
            cost += float(np.sum((self.Rreg @ th) ** 2))
        pr, pv, prh = self.prior_rows(s)
        cost += float(np.sum(np.square(prh)))
        ar = self.aspect_row(s)
        if ar is not None:
            cost += float(ar[1] ** 2)
        pp = self.pp_rows(s)
        if pp is not None:
            cost += float(np.sum(pp[1] ** 2))
        return cost

    # ------------------------------------------------------------ Gauss-Newton system
    def linearize(self, st: State) -> "Linearization":
        """The full system [[H_theta, F], [F^T, H_XX]] [dtheta; dX] = -[g_theta; g_X]."""
        pb, lay = self.pb, self.lay
        M = pb.n_lm
        if self.on_dev:
            Hth, gth, HXX, gX, F = normal_equations_device(pb, st, self.grid, lay, M, self.c_px)
            if self.has_planes():
                ids, n, q, Rk = self.planes
                rp = self.plane_res(st)
                wp = cauchy_w(rp ** 2, self.c_plane)
                Jp = n / self.sigma_plane
                tj = lambda a: torch.as_tensor(a, dtype=DT, device=DEV)
                ids_t, Jp_t, wp_t = torch.as_tensor(ids, device=DEV), tj(Jp), tj(wp)
                HXX.index_add_(0, ids_t, Jp_t[:, :, None] * Jp_t[:, None, :] * wp_t[:, None, None])
                gX.index_add_(0, ids_t, Jp_t * (wp_t * tj(rp))[:, None])
                if "tel" in lay:                        # plane rows couple tel and landmarks
                    Jtel = -np.einsum("ni,nij->nj", n, Rk) / self.sigma_plane
                    o = lay["tel"][0]
                    Hth[o:o + 3, o:o + 3] += (Jtel * wp[:, None]).T @ Jtel
                    gth[o:o + 3] += Jtel.T @ (wp * rp)
                    for k in range(3):
                        F[o + k].index_add_(0, ids_t, (wp_t * tj(Jtel[:, k]))[:, None] * Jp_t)
        else:
            Hth, gth, HXX, gX, F = normal_equations_host(pb, st, self.grid, lay, M, self.c_px,
                                                         self.planes if self.has_planes() else None,
                                                         self.plane_res, self.sigma_plane, self.c_plane)
        th = self.pack(st)
        if self.Rreg is not None:
            Hth += (self.Rreg.T @ self.Rreg).toarray()
            gth += self.Rreg.T @ (self.Rreg @ th)
        pr, pv, prh = self.prior_rows(st)
        for c_, v_, h_ in zip(pr, pv, prh):
            Hth[c_, c_] += v_ * v_
            gth[c_] += v_ * h_
        ar = self.aspect_row(st)
        if ar is not None:                              # one row: (fx - fy) / sigma
            w_, r_ = ar
            i0 = lay["intr"][0]
            Hth[i0, i0] += w_ * w_
            Hth[i0 + 1, i0 + 1] += w_ * w_
            Hth[i0, i0 + 1] -= w_ * w_
            Hth[i0 + 1, i0] -= w_ * w_
            gth[i0] += w_ * r_
            gth[i0 + 1] -= w_ * r_
        pp = self.pp_rows(st)
        if pp is not None:                              # two rows: (cx, cy) - centre
            w_, r_ = pp
            i0 = lay["intr"][0] + 2
            for k in (0, 1):
                Hth[i0 + k, i0 + k] += w_ * w_
                gth[i0 + k] += w_ * r_[k]
        if self.matches is not None:                      # camera-only rows: no landmark coupling
            rm, Jm = match_residuals(self.matches, st, self.grid, lay, pb.model, pb.H, self.sigma_match)
            wm = np.repeat(cauchy_w(rm[0::2] ** 2 + rm[1::2] ** 2, self.c_match), 2)
            Wm = sp.diags(wm)
            Hth += (Jm.T @ Wm @ Jm).toarray()
            gth += Jm.T @ (wm * rm)
        return Linearization(Hth, gth, HXX, gX, F, self.on_dev)

    def step(self, lin: "Linearization", lam: float):
        """The damped step (dtheta, dX), every landmark eliminated."""
        Hd = lin.damped(lam)
        if not lin.dev:
            return schur_step_host(Hd, lin.gth, lin.HXX, lin.gX, lin.F, lam)
        B = landmark_blocks_inv(lin.HXX, lam)
        if Hd.shape[0] == 0:
            return np.zeros(0), torch.einsum("nij,nj->ni", B, -lin.gX).cpu().numpy()
        S, rhs = schur_reduce(Hd, lin.gth, B, lin.gX, lin.F)
        dth = torch.linalg.solve(S, rhs)
        return dth.cpu().numpy(), back_substitute(B, lin.gX, lin.F, dth).cpu().numpy()


@dataclass
class Linearization:
    Hth: np.ndarray                   # (P,P) host
    gth: np.ndarray                   # (P,)
    HXX: object                       # (M,3,3)  device tensor (dev) or ndarray
    gX: object                        # (M,3)
    F: object                         # (P,M,3) device tensor (dev), or csr (P,3M)
    dev: bool

    def damped(self, lam: float) -> np.ndarray:
        return self.Hth + lam * np.diag(np.maximum(np.diag(self.Hth), 1e-9))


def lm_solve(pb: Problem, st: State, grid, active: dict, iters: int = 15, verbose: bool = True, **kw):
    """Levenberg-Marquardt with Schur elimination of landmarks. Returns (state, cost).
    Keyword arguments are CamLM's: planes, priors, matches, tie_aspect, pp_prior, ..."""
    cam = CamLM(pb, st, grid, active, **kw)
    lam = 1e-3
    cost = cam.cost(st)
    for it in range(iters):
        lin = cam.linearize(st)
        while True:                                     # damped steps until the cost goes down
            try:
                dth, dX = cam.step(lin, lam)
            except (np.linalg.LinAlgError, torch.linalg.LinAlgError):
                lam *= 10
                continue
            cand = cam.apply(st, dth, dX)
            c_new = cam.cost(cand)
            if c_new < cost:
                st, cost = cand, c_new
                lam = max(lam / 3, 1e-7)
                break
            lam *= 5
            if lam > 1e6:
                break
        del lin                                          # device memory for the next linearisation
        if verbose:
            print(f"    it {it:2d}  cost {cost:.6e}  lambda {lam:.1e}")
        if lam > 1e6:
            break
    if DEV.type == "cuda":
        torch.cuda.empty_cache()                         # hand the cached blocks back to other processes
    return st, cost


SHARED_BLOCKS = ("tel", "rel")


def joint_lm_solve(cams: list[CamLM], states: list[State], glob: list[np.ndarray] | None = None,
                   n_glob: int = 0, iters: int = 15, verbose: bool = True):
    """Levenberg-Marquardt over several cameras at once. Returns (states, total cost).

    Shared between the cameras: the blocks in SHARED_BLOCKS (the LiDAR -> ego, or
    motion-frame, offset; every camera must have the same ones active and start from
    the same values), and the landmarks with glob[c][j] = g >= 0, which are landmark g
    of n_glob landmarks seen by more than one camera (each camera keeps a copy in its
    state, kept equal; a camera may hold several copies of one). The step is the exact
    Gauss-Newton step of the joint problem: every landmark, shared or not, is
    eliminated through its 3x3 block, which leaves one dense system over all cameras'
    parameters with the shared blocks counted once.
    """
    glob = glob if glob is not None else [np.full(c.pb.n_lm, -1) for c in cams]
    names = [n for n in SHARED_BLOCKS if n in cams[0].lay]
    for c in cams:
        if [n for n in SHARED_BLOCKS if n in c.lay] != names:
            raise ValueError("every camera needs the same shared blocks active")
        if not c.on_dev:
            raise ValueError("joint_lm_solve needs every camera on the device path")
    ns = 3 * len(names)
    info, off = [], ns
    for c, g in zip(cams, glob):
        P = c.lay["P"]
        sidx = np.concatenate([c.lay[n][0] + np.arange(3) for n in names]).astype(int) if names else np.zeros(0, int)
        m = np.full(P, -1)
        m[sidx] = np.arange(ns)
        priv = np.setdiff1d(np.arange(P), sidx)
        m[priv] = off + np.arange(len(priv))
        off += len(priv)
        gl = np.flatnonzero(g >= 0)
        ids, inv = np.unique(g[gl], return_inverse=True)           # this camera's shared landmarks
        t = lambda a: torch.as_tensor(a, device=DEV)
        info.append({"map": t(m), "gl": t(gl), "loc": t(np.flatnonzero(g < 0)), "ids": t(ids), "inv": t(inv)})

    def total(sts):
        return sum(c.cost(s) for c, s in zip(cams, sts))

    lam = 1e-3
    cost = total(states)
    for it in range(iters):
        lins = []
        for c, s in zip(cams, states):
            lin = c.linearize(s)
            # C cameras' F fit neither on the device nor, in float64, comfortably in host
            # memory; float32 there changes the step a little, never the solution (the
            # reduced right-hand side vanishes with the float64 gradients)
            lin.F = lin.F.to("cpu", torch.float32)
            lins.append(lin)
        while True:
            try:
                steps = _joint_step(cams, lins, info, off, n_glob, lam)
            except (np.linalg.LinAlgError, torch.linalg.LinAlgError):
                lam *= 10
                continue
            cand = [c.apply(s, dth, dX) for c, s, (dth, dX) in zip(cams, states, steps)]
            c_new = total(cand)
            if c_new < cost:
                states, cost = cand, c_new
                lam = max(lam / 3, 1e-7)
                break
            lam *= 5
            if lam > 1e6:
                break
        del lins
        if verbose:
            print(f"    joint it {it:2d}  cost {cost:.6e}  lambda {lam:.1e}")
        if lam > 1e6:
            break
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return states, cost


def _joint_step(cams, lins, info, N: int, ng: int, lam: float):
    S = torch.zeros(N, N, dtype=DT, device=DEV)
    r = torch.zeros(N, dtype=DT, device=DEV)
    Hg = torch.zeros(ng, 3, 3, dtype=DT, device=DEV)
    gg = torch.zeros(ng, 3, dtype=DT, device=DEV)
    Bs, Fs = [], []
    for c, lin, d in zip(cams, lins, info):
        P, m = c.lay["P"], d["map"]
        B = landmark_blocks_inv(lin.HXX, lam, d["loc"])
        Sc, rc = schur_reduce(lin.damped(lam), lin.gth, B, lin.gX, lin.F, d["loc"])
        S.index_put_((m[:, None], m[None, :]), Sc, accumulate=True)
        r.index_put_((m,), rc, accumulate=True)
        Bs.append(B)
        if len(d["gl"]):
            gid = d["ids"][d["inv"]]
            Hg.index_add_(0, gid, lin.HXX[d["gl"]])
            gg.index_add_(0, gid, lin.gX[d["gl"]])
            # the camera's coupling to each of its shared landmarks (copies summed)
            Fl = lin.F[:, d["gl"].cpu(), :].to(DEV, DT)
            Fsum = torch.zeros(P, len(d["ids"]), 3, dtype=DT, device=DEV).index_add_(1, d["inv"], Fl)
            Fs.append(Fsum)
        else:
            Fs.append(None)
    if ng:
        I3 = torch.eye(3, dtype=DT, device=DEV)
        Bg = torch.linalg.inv(Hg + lam * torch.diagonal(Hg, dim1=1, dim2=2)[:, :, None] * I3 + 1e-9 * I3)
        pos = torch.full((len(cams), ng), -1, dtype=torch.long, device=DEV)
        for k, d in enumerate(info):
            pos[k, d["ids"]] = torch.arange(len(d["ids"]), device=DEV)
        for k1, d1 in enumerate(info):
            if Fs[k1] is None:
                continue
            P1 = Fs[k1].shape[0]
            FB1 = torch.einsum("pni,nij->pnj", Fs[k1], Bg[d1["ids"]])
            r.index_put_((d1["map"],), FB1.reshape(P1, -1) @ gg[d1["ids"]].reshape(-1), accumulate=True)
            for k2, d2 in enumerate(info):
                if Fs[k2] is None:
                    continue
                both = (pos[k1] >= 0) & (pos[k2] >= 0)
                if not bool(both.any()):
                    continue
                i1, i2 = pos[k1][both], pos[k2][both]
                blk = FB1[:, i1].reshape(P1, -1) @ Fs[k2][:, i2].reshape(Fs[k2].shape[0], -1).T
                S.index_put_((d1["map"][:, None], d2["map"][None, :]), -blk, accumulate=True)
    dtheta = torch.linalg.solve(S, r)
    steps, acc = [], torch.zeros(ng, 3, dtype=DT, device=DEV)
    dths = [dtheta[d["map"]] for d in info]
    for d, Fsum, dth in zip(info, Fs, dths):
        if Fsum is not None:
            acc.index_add_(0, d["ids"], torch.einsum("pni,p->ni", Fsum, dth))
    dXg = torch.einsum("nij,nj->ni", Bg, -gg - acc) if ng else None
    for c, lin, d, B, dth in zip(cams, lins, info, Bs, dths):
        dX = back_substitute(B, lin.gX, lin.F, dth, d["loc"])
        if len(d["gl"]):
            dX[d["gl"]] = dXg[d["ids"][d["inv"]]]
        steps.append((dth.cpu().numpy(), dX.cpu().numpy()))
    return steps


# The device path holds F (P x M x 3, 8 bytes each: 1.4 GB for P = 620 camera parameters
# and M = 96k landmarks) on the device, up to 4 GB, which leaves room for the streamed
# chunks on an 8 GB card only if nothing else uses it. Larger problems use the sparse
# host path (slow; thin the data with --frame-step instead).
DEVICE_F_BYTES = 4.0e9
_ROWS = 1 << 19                                     # rows per chunk for the (rows, K, 3) temporaries
_OBS = 1 << 19                                      # observations per linearisation chunk
_LMS = 1 << 14                                      # landmarks per Schur chunk


def _obs_chunks(n: int):
    return [slice(s, min(s + _OBS, n)) for s in range(0, n, _OBS)]


def normal_equations_device(pb: Problem, st: State, grid, lay: dict, M: int, c_px: float):
    """Reprojection part of the Gauss-Newton system, Cauchy-weighted, on the device, in
    chunks of observations. Returns H_theta (P,P) and g_theta (P,) on the host; H_XX
    (M,3,3), g_X (M,3) and F = J_theta^T W J_X as (P, M, 3) on the device."""
    P = lay["P"]
    H = torch.zeros(P, P, dtype=DT, device=DEV)
    g = torch.zeros(P, dtype=DT, device=DEV)
    HXX = torch.zeros(M, 3, 3, dtype=DT, device=DEV)
    gX = torch.zeros(M, 3, dtype=DT, device=DEV)
    F = torch.zeros(P, M, 3, dtype=DT, device=DEV)
    X = torch.as_tensor(st.X, dtype=DT, device=DEV)
    for sl in _obs_chunks(len(pb.obs_f)):
        r, jac, JX, _ = residuals_and_jacobians(pb, st, grid, lay, dense=True, idx=sl, X=X)
        _accumulate(H, g, HXX, gX, F, r, jac, JX, pb.obs_j[sl], M, c_px)
        del r, jac, JX
    return H.cpu().numpy(), g.cpu().numpy(), HXX, gX, F


def _accumulate(H, g, HXX, gX, F, r, jac: RowJacobian, JX, obs_j, M: int, c_px: float):
    P = jac.P
    r2 = r[0::2] ** 2 + r[1::2] ** 2
    w = (1.0 / (1.0 + r2 / (c_px * c_px))).repeat_interleave(2)
    wr = w * r
    oj = obs_j.repeat_interleave(2)
    JX = JX.reshape(-1, 3)
    for s in range(0, len(w), _ROWS):
        sl = slice(s, s + _ROWS)
        HXX.index_add_(0, oj[sl], JX[sl, :, None] * JX[sl, None, :] * w[sl, None, None])
    gX.index_add_(0, oj, JX * wr[:, None])
    cg, Jg, Cs, Vs = jac.cg, jac.Jg, jac.Cs, jac.Vs
    Kg, Ks = Jg.shape[1], Cs.shape[1]
    if Kg:
        WJg = Jg * w[:, None]
        H[cg[:, None], cg[None, :]] += Jg.T @ WJg
        g[cg] += Jg.T @ wr
        Fg = torch.zeros(M, Kg, 3, dtype=DT, device=DEV)
        for s in range(0, len(w), _ROWS):
            sl = slice(s, s + _ROWS)
            Fg.index_add_(0, oj[sl], WJg[sl, :, None] * JX[sl, None, :])
        F[cg] += Fg.permute(1, 0, 2)
        del Fg
    if Ks:
        Hf, Ff = H.view(-1), F.view(-1, 3)
        Hsg = torch.zeros(P, Kg, dtype=DT, device=DEV)
        for a in range(Ks):
            ca, wa = Cs[:, a], w * Vs[:, a]
            g.index_add_(0, ca, wa * r)
            Ff.index_add_(0, ca * M + oj, wa[:, None] * JX)
            if Kg:
                Hsg.index_add_(0, ca, wa[:, None] * Jg)
            for b in range(a, Ks):                       # upper triangle of the per-row block
                v = wa * Vs[:, b]
                Hf.index_add_(0, ca * P + Cs[:, b], v)
                if b != a:
                    Hf.index_add_(0, Cs[:, b] * P + ca, v)
        if Kg:
            H[:, cg] += Hsg
            H[cg, :] += Hsg.T


def landmark_blocks_inv(HXX, lam: float, sel=None):
    """Inverses of the damped 3x3 landmark blocks (all, or those in sel; others 0)."""
    I3 = torch.eye(3, dtype=DT, device=DEV)
    HXd = HXX + lam * torch.diagonal(HXX, dim1=1, dim2=2)[:, :, None] * I3 + 1e-9 * I3
    if sel is None:
        return torch.linalg.inv(HXd)
    B = torch.zeros_like(HXd)
    B[sel] = torch.linalg.inv(HXd[sel])
    return B


def _lm_chunks(M: int, sel=None):
    ids = torch.arange(M, device=DEV) if sel is None else sel
    return [ids[s:s + _LMS] for s in range(0, len(ids), _LMS)]


def schur_reduce(Hd: np.ndarray, gth: np.ndarray, B, gX, F, sel=None):
    """Eliminate the landmarks (all, or those in sel) from the damped system:
    S = Hd - sum F_j B_j F_j^T, rhs = -g_theta + sum F_j B_j g_Xj. F may live on the
    host (joint solves); it is brought over one chunk of landmarks at a time."""
    P = Hd.shape[0]
    S = torch.as_tensor(Hd, dtype=DT, device=DEV).clone()
    rhs = -torch.as_tensor(gth, dtype=DT, device=DEV)
    for ch in _lm_chunks(B.shape[0], sel):
        Fc = (F[:, ch.cpu(), :] if F.device.type == "cpu" else F[:, ch, :]).to(DEV, DT)
        FB = torch.einsum("pmi,mij->pmj", Fc, B[ch]).reshape(P, -1)
        S -= FB @ Fc.reshape(P, -1).T
        rhs += FB @ gX[ch].reshape(-1)
    return S, rhs


def back_substitute(B, gX, F, dth, sel=None):
    """dX = B (-g_X - F^T dtheta) for the eliminated landmarks (others 0)."""
    dX = torch.zeros_like(gX)
    for ch in _lm_chunks(B.shape[0], sel):
        Fc = (F[:, ch.cpu(), :] if F.device.type == "cpu" else F[:, ch, :]).to(DEV, DT)
        dX[ch] = torch.einsum("nij,nj->ni", B[ch], -gX[ch] - torch.einsum("pmi,p->mi", Fc, dth))
    return dX


def normal_equations_host(pb: Problem, st: State, grid, lay: dict, M: int, c_px: float, planes,
                          plane_res, sigma_plane: float, c_plane: float):
    """As normal_equations_device, with scipy sparse matrices (any P, slow)."""
    r, Jt, JX, _ = residuals_and_jacobians(pb, st, grid, lay)
    N = len(r) // 2
    r2 = r[0::2] ** 2 + r[1::2] ** 2
    w = np.repeat(cauchy_w(r2, c_px), 2)
    oj = pb.obs_j.cpu().numpy()
    JXf = JX.reshape(2 * N, 3)
    HXX = np.zeros((M, 3, 3))
    np.add.at(HXX, np.repeat(oj, 2), JXf[:, :, None] * JXf[:, None, :] * w[:, None, None])
    gX = np.zeros((M, 3))
    np.add.at(gX, np.repeat(oj, 2), JXf * (w * r)[:, None])
    Wd = sp.diags(w)
    Hth = (Jt.T @ Wd @ Jt).toarray()
    gth = Jt.T @ (w * r)
    A = sp.csr_matrix((JXf.ravel(), (np.repeat(np.arange(2 * N), 3), (np.repeat(oj, 2)[:, None] * 3 + np.arange(3)).ravel())),
                      shape=(2 * N, 3 * M))
    F = (Jt.T @ (Wd @ A)).tocsr()                     # (P, 3M)
    if planes is not None and len(planes[0]):
        ids, n, q, Rk = planes
        rp = plane_res(st)
        wp = cauchy_w(rp ** 2, c_plane)
        Jp = n / sigma_plane
        np.add.at(HXX, ids, Jp[:, :, None] * Jp[:, None, :] * wp[:, None, None])
        np.add.at(gX, ids, Jp * (wp * rp)[:, None])
        if "tel" in lay:                                # plane rows couple tel and landmarks
            Np = len(ids)
            A_p = sp.csr_matrix((Jp.ravel(), (np.repeat(np.arange(Np), 3), (ids[:, None] * 3 + np.arange(3)).ravel())),
                                shape=(Np, 3 * M))
            Jtel = -np.einsum("ni,nij->nj", n, Rk) / sigma_plane
            Jt_p = sp.csr_matrix((Jtel.ravel(), (np.repeat(np.arange(Np), 3),
                                                 np.tile(lay["tel"][0] + np.arange(3), Np))),
                                 shape=(Np, lay["P"]))
            Wp = sp.diags(wp)
            Hth += (Jt_p.T @ Wp @ Jt_p).toarray()
            gth += Jt_p.T @ (wp * rp)
            F = (F + Jt_p.T @ (Wp @ A_p)).tocsr()
    return Hth, gth, HXX, gX, F


def schur_step_host(Hd: np.ndarray, gth: np.ndarray, HXX, gX, F, lam: float):
    M, P = HXX.shape[0], Hd.shape[0]
    HXd = HXX + lam * np.einsum("nii->ni", HXX)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3)
    B = np.linalg.inv(HXd)
    Bmat = sp.csr_matrix((B.ravel(), (np.repeat(np.arange(3 * M), 3),
                                      (np.arange(M)[:, None, None] * 3 + np.arange(3)[None, None, :]).repeat(3, axis=1).ravel())),
                         shape=(3 * M, 3 * M))
    FB = F @ Bmat
    S = Hd - (FB @ F.T).toarray()
    rhs = -gth + FB @ gX.ravel()
    dth = np.linalg.solve(S, rhs) if P else np.zeros(0)
    dX = np.einsum("nij,nj->ni", B, -gX - (F.T @ dth).reshape(M, 3))
    return dth, dX


def reprojection_stats(pb: Problem, st: State, grid) -> dict:
    X = torch.as_tensor(st.X, dtype=DT, device=DEV)
    G = torch.as_tensor(st.G, dtype=DT, device=DEV) if st.G is not None else None
    e = []
    for sl in _obs_chunks(len(pb.obs_f)):
        Xc, *_ = camera_points(pb, st, X, sl)
        pix = project(Xc, torch.as_tensor(st.intr, dtype=DT, device=DEV), pb.model, grid, G)
        e.append((pix - pb.obs_uv[sl]).norm(dim=-1).cpu().numpy())
    e = np.concatenate(e) if e else np.zeros(0)
    return {"n_obs": int(len(e)), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
            "rms_inlier_px": float(np.sqrt(np.mean(e[e < 3] ** 2))), "inlier_frac": float(np.mean(e < 3)), "err": e}
