#!/usr/bin/env python3
"""Continuous-time multi-sweep point-to-plane refinement of a LiDAR odometry.

    python online_calib/lo_refine.py CLIP KISS.npz --out REFINED.npz [--iters 5]

The same method and numbers as /hdd/DM_calib/lidar_odo/lo_refine.py (the DM rig's
GNSS-free calibration), reimplemented to run several times faster; the result agrees
with it to rounding. KISS.npz: /hdd/DM_calib/lidar_odo/lo_kiss.py on CLIP (knot times
tau = sweep header + 0.1 s, poses T_w_L). The output feeds lo_traj.py.

Every sweep is registered point-to-plane to the sweeps 1, 2, 3, 5, 8, 13 and 20 away,
in both roles, on thin planar patches, with the trajectory continuous in time: knot
T_k at tau_k, a point of sweep k at header_k + t lies between knots k-1 and k (SLERP /
linear), so moving a knot moves the end of one sweep and the start of the next. Six
unknowns per knot (rotation about the knot's own centre, world axes, and a
translation); Gauss-Newton on the dense normal equations, Cauchy weights,
re-association every iteration; knot 1 held by a strong prior (gauge) and a weak
constant-velocity prior on every knot.

Where the time goes, and what changed from the original:
  * poses are evaluated once per firing time (a sweep has ~1000), not per point;
  * each sweep pair touches only ~4 knots, so its normal equations are one small
    dense product scattered into H, not 16 per-point scatters;
  * sweeps load and pairs register on all cores.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from traj import quat_mul, quat_to_matrix, rotvec_to_quat

OFFSETS = (1, 2, 3, 5, 8, 13, 20)
SWEEP_NS = 100_000_000


class LOTraj:
    """Knots T_w_L(tau_k); SLERP / linear in between. Same file format as lotraj.LOTraj."""

    def __init__(self, tau: np.ndarray, T: np.ndarray):
        self.tau = np.asarray(tau, np.int64)
        self.T = np.asarray(T, np.float64)

    @classmethod
    def load(cls, path: Path) -> "LOTraj":
        z = np.load(path)
        tau, T = z["tau"].astype(np.int64), z["T_w_L"].astype(np.float64)
        if "ext" in z.files and bool(z["ext"]):
            return cls(tau, T)
        o = np.argsort(tau)
        tau, T = tau[o], T[o]
        # a knot before the first, so sweep 0 can be deskewed too
        d = np.linalg.inv(T[0]) @ T[1]
        return cls(np.concatenate([[tau[0] - (tau[1] - tau[0])], tau]),
                   np.concatenate([[T[0] @ np.linalg.inv(d)], T]))

    def save(self, path: Path, **extra) -> None:
        np.savez(path, tau=self.tau, T_w_L=self.T, ext=True, **extra)

    def seg(self, t):
        t = np.atleast_1d(np.asarray(t, np.float64))
        i = np.clip(np.searchsorted(self.tau, t) - 1, 0, len(self.tau) - 2)
        return i, (t - self.tau[i]) / (self.tau[i + 1] - self.tau[i])


def load_sweep(path: Path, rmin: float, rmax: float):
    s = np.load(path)
    r = np.sqrt(s["x"].astype(np.float64) ** 2 + s["y"] ** 2 + s["z"] ** 2)
    return s[(r > rmin) & (r < rmax)]


def voxel_down(p: np.ndarray, voxel: float, *extra):
    k = np.floor(p / voxel).astype(np.int64) + (1 << 20)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    _, idx = np.unique(key, return_index=True)
    idx.sort()
    return (p[idx], *(e[idx] for e in extra))


class Sweep:
    """Points of one sweep in the body frame, grouped by firing time: `u` indexes the
    distinct times, each with its knot pair (ui, ui + 1) and weight ua."""

    def __init__(self, p: np.ndarray, t: np.ndarray, tr: LOTraj):
        self.p, self.t = p, t
        ut, self.u = np.unique(t, return_inverse=True)
        self.ui, self.ua = tr.seg(ut)
        self.i, self.a = self.ui[self.u], self.ua[self.u]

    def subset(self, keep: np.ndarray) -> "Sweep":
        s = Sweep.__new__(Sweep)
        s.p, s.t, s.u, s.ui, s.ua = self.p[keep], self.t[keep], self.u[keep], self.ui, self.ua
        s.i, s.a = self.i[keep], self.a[keep]
        return s


def knot_quats(tr: LOTraj):
    q = Rot.from_matrix(tr.T[:, :3, :3])
    return q.as_quat(), (q[:-1].inv() * q[1:]).as_rotvec()


def world(sw: Sweep, tr: LOTraj, qk, dk) -> np.ndarray:
    """Body points -> world, each at its own time (the pose once per firing time)."""
    i, a = sw.ui, sw.ua
    R = quat_to_matrix(quat_mul(qk[i], rotvec_to_quat(dk[i] * a[:, None])))
    c = tr.T[i, :3, 3] * (1 - a)[:, None] + tr.T[i + 1, :3, 3] * a[:, None]
    return np.einsum("nij,nj->ni", R[sw.u], sw.p) + c[sw.u]


def patch_planes(nb: np.ndarray):
    """(centroid, eigenvalues ascending, eigenvectors) of each (n, k, 3) neighbourhood."""
    mu = nb.mean(1)
    d = nb - mu[:, None]
    C = np.matmul(d.transpose(0, 2, 1), d) / nb.shape[1]
    w, v = np.linalg.eigh(C)
    return mu, w, v


def prepare(f: Path, hdr: int, tr: LOTraj, a):
    s = load_sweep(f, 3.0, 80.0)
    p = np.stack([s["x"], s["y"], s["z"]], -1).astype(np.float64)
    t = hdr + s["t"].astype(np.float64) * 1e9
    tp, tt = voxel_down(p, a.tgt_voxel, t)
    tgt = Sweep(tp, tt, tr)
    sp_, st_ = voxel_down(p, a.src_voxel, t)
    src = Sweep(sp_, st_, tr)
    # source points that are planar within their own sweep
    d, nn = cKDTree(tgt.p).query(src.p, k=10, distance_upper_bound=0.5)
    ok = np.isfinite(d).all(1)
    nb = tgt.p[nn[ok]]
    mu = nb.mean(1)
    dd = nb - mu[:, None]
    w_ = np.linalg.eigvalsh(np.matmul(dd.transpose(0, 2, 1), dd) / 10)
    pl = np.zeros(len(src.p), bool)
    pl[ok] = (np.sqrt(np.maximum(w_[:, 0], 0)) < 0.03) & ((w_[:, 1] - w_[:, 0]) / np.maximum(w_[:, 2], 1e-12) > 0.6)
    return tgt, src, np.flatnonzero(pl)


def pair_system(i, j, Ws, Wt, trees, src, tgt, knot_c, scale):
    """Normal equations of one (source sweep i, target sweep j) pair on its few knots."""
    P = Ws[i]
    d, nn = trees[j].query(P, k=8, distance_upper_bound=0.6)
    ok = np.isfinite(d).all(1)
    if ok.sum() < 30:
        return None
    nb = Wt[j][nn[ok]]
    mu, w_, v_ = patch_planes(nb)
    thick = np.sqrt(np.maximum(w_[:, 0], 0))
    planar = (w_[:, 1] - w_[:, 0]) / np.maximum(w_[:, 2], 1e-12)
    good = (thick < 0.03) & (planar > 0.6)
    if good.sum() < 30:
        return None
    idx = np.flatnonzero(ok)[good]
    n = v_[good, :, 0]
    q = mu[good]
    p = P[idx]
    r = np.einsum("ni,ni->n", n, p - q)
    gate = np.abs(r) < 0.3
    idx, n, q, p, r = idx[gate], n[gate], q[gate], p[gate], r[gate]
    nnj = nn[ok][good][gate]
    si, sa = src[i].i[idx], src[i].a[idx]
    ti_ = tgt[j].i[nnj[:, 0]]
    ta = tgt[j].a[nnj].mean(1)
    wgt = 1.0 / (1.0 + (r / scale) ** 2)
    # dr/d(knot) = +-(weight) [((x - c) x n)^T, n^T]; the pair's knots in a local dense block
    blocks = ((si, 1 - sa, p, 1.0), (si + 1, sa, p, 1.0), (ti_, 1 - ta, q, -1.0), (ti_ + 1, ta, q, -1.0))
    knots = np.unique(np.concatenate([b[0] for b in blocks]))
    L = len(knots)
    m = len(r)
    J = np.zeros((m, 6 * L))
    rows = np.arange(m)[:, None]
    for kn, wt, pt, sgn in blocks:
        c = knot_c[kn]
        Jb = np.concatenate([np.cross(pt - c, n), n], 1) * (sgn * wt)[:, None]
        col = 6 * np.searchsorted(knots, kn)[:, None] + np.arange(6)
        np.add.at(J, (np.broadcast_to(rows, col.shape), col), Jb)
    Jw = J * wgt[:, None]
    return (knots, Jw.T @ J, Jw.T @ r, float(np.sum(np.log1p((r / scale) ** 2))), r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("seg", type=Path)
    ap.add_argument("init", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--src-voxel", type=float, default=0.3)
    ap.add_argument("--tgt-voxel", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=0.03, help="Cauchy scale (m)")
    ap.add_argument("--cv-sigma", type=float, default=0.05, help="const-velocity prior (m, rad*10 m)")
    ap.add_argument("--max-src", type=int, default=3000)
    ap.add_argument("--offsets", type=int, nargs="+", default=list(OFFSETS))
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    t_start = time.time()
    tr = LOTraj.load(a.init)
    files = sorted((a.seg / "lidar").glob("*.npy"))
    hdr = np.array([int(f.stem) for f in files], np.int64)
    keep = (hdr >= tr.tau[0]) & (hdr + SWEEP_NS <= tr.tau[-1])
    files, hdr = [f for f, k in zip(files, keep) if k], hdr[keep]
    pool = ThreadPoolExecutor(a.threads)
    prep = list(pool.map(lambda fh: prepare(fh[0], fh[1], tr, a), zip(files, hdr)))
    tgt = [x[0] for x in prep]
    rng = np.random.default_rng(0)                  # the same draws, in the same order, as the original
    src = []
    for _, sw, keep_ in prep:
        if len(keep_) > a.max_src:
            keep_ = np.sort(rng.choice(keep_, a.max_src, replace=False))
        src.append(sw.subset(keep_))
    del prep
    K, N = len(tr.tau), len(files)
    print(f"{a.seg.name}: {N} sweeps, {K} knots, src {np.mean([len(s.p) for s in src]):.0f} "
          f"tgt {np.mean([len(s.p) for s in tgt]):.0f} pts/sweep; load {time.time() - t_start:.1f} s", flush=True)
    pairs = [(i, i + o) for o in a.offsets for i in range(N - o)]
    pairs += [(j, i) for (i, j) in pairs]
    for it in range(a.iters):
        t_it = time.time()
        qk, dk = knot_quats(tr)
        Wt = list(pool.map(lambda s: world(s, tr, qk, dk), tgt))
        Ws = list(pool.map(lambda s: world(s, tr, qk, dk), src))
        trees = list(pool.map(cKDTree, Wt))
        knot_c = tr.T[:, :3, 3].copy()
        H = np.zeros((6 * K, 6 * K))
        g = np.zeros(6 * K)
        cost, allr = 0.0, []
        for res in pool.map(lambda ij: pair_system(*ij, Ws, Wt, trees, src, tgt, knot_c, a.scale), pairs):
            if res is None:
                continue
            knots, Hl, gl, c_, r = res
            ix = (6 * knots[:, None] + np.arange(6)).ravel()
            H[np.ix_(ix, ix)] += Hl
            g[ix] += gl
            cost += c_
            allr.append(r)
        # priors: gauge on knot 1, constant velocity (second difference) everywhere
        H[6:12, 6:12] += np.eye(6) * 1e8
        sig = a.cv_sigma
        cf = np.array([1.0, -2.0, 1.0])
        blk = np.kron(np.outer(cf, cf), np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]) / sig ** 2)
        for k in range(1, K - 1):
            H[6 * (k - 1):6 * (k + 2), 6 * (k - 1):6 * (k + 2)] += blk
        pos = tr.T[:, :3, 3]
        e = pos[:-2] - 2 * pos[1:-1] + pos[2:]
        Rk = tr.T[:, :3, :3]
        rm = Rot.from_matrix(np.einsum("nji,njk->nik", Rk[1:-1], Rk[:-2])).as_rotvec()
        rp = Rot.from_matrix(np.einsum("nji,njk->nik", Rk[1:-1], Rk[2:])).as_rotvec()
        er = np.einsum("nij,nj->ni", Rk[1:-1], rm + rp)
        for off, c in zip(range(3), cf):
            gv = g.reshape(K, 6)
            gv[off:off + K - 2, 3:] += c * e / sig ** 2
            gv[off:off + K - 2, :3] += c * er * 100 / sig ** 2
        H[np.diag_indices_from(H)] += 1e-6
        dx = -np.linalg.solve(H, g).reshape(K, 6)
        T = tr.T.copy()
        T[:, :3, :3] = np.einsum("nij,njk->nik", Rot.from_rotvec(dx[:, :3]).as_matrix(), T[:, :3, :3])
        T[:, :3, 3] += dx[:, 3:]
        tr.T = T
        r = np.concatenate(allr)
        mad = 1.4826 * np.median(np.abs(r))
        print(f"  it {it}: {len(r)} residuals, robust std {1e3 * mad:.2f} mm, cost {cost:.0f}, "
              f"step max rot {np.degrees(np.abs(dx[:, :3]).max()):.4f} deg trans {1e3 * np.abs(dx[:, 3:]).max():.1f} mm "
              f"({time.time() - t_it:.0f} s)", flush=True)
    tr.save(a.out, wall_s=time.time() - t_start)
    print(f"  done in {time.time() - t_start:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
