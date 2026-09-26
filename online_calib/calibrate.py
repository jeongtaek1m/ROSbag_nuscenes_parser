#!/usr/bin/env python3
"""Targetless online calibration of one camera against LIDAR_TOP and the INS.

    python online_calib/calibrate.py CHANNEL --clips CLIP [CLIP ...] --ins INS.npz \
        --lidar-ego lidar_ego.json --tracks DIR --out DIR [--model kb|radtan] [--grid 12 7]

Stages (each re-triangulates the landmarks and drops outlier observations):
  1  lens + extrinsic from tracks alone
  2  + LiDAR plane constraints (landmarks must lie on the LiDAR map)
  3  + time offset and rolling-shutter readout
  4  + windshield field (--grid; 0 0 to skip)
Writes DIR/calib_<CHANNEL>_<model>[_grid].json.

Final stage (--final-from CALIB.json --matches MATCHES.npz): starting from the
calibration refine_pnp.py produced, one joint problem holds the feature tracks
(lens, windshield field, camera on the car, timing), the LiDAR-image matches
(camera relative to the LiDAR) and the LiDAR's position on the car, so the two
kinds of evidence are reconciled instead of one overriding the other. Writes
DIR/final_<CHANNEL>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ba import (DEV, LidarMatches, Problem, State, associate_planes, build_map, build_problem,
                lm_solve, match_residuals, param_layout, reprojection_stats, triangulate)
from camera_model import N_INTR, Grid
from init_cam import init_focal_and_rotation
from traj import Trajectory


def prune(pb: Problem, keep_obs: np.ndarray) -> Problem:
    """Keep the given observations and the landmarks that still have >= 3."""
    k = torch.as_tensor(keep_obs, device=DEV)
    pb.obs_f, pb.obs_j, pb.obs_uv = pb.obs_f[k], pb.obs_j[k], pb.obs_uv[k]
    cnt = torch.bincount(pb.obs_j, minlength=pb.n_lm)
    k2 = cnt[pb.obs_j] >= 3
    pb.obs_f, pb.obs_j, pb.obs_uv = pb.obs_f[k2], pb.obs_j[k2], pb.obs_uv[k2]
    return pb


def compact(pb: Problem, st: State) -> tuple[Problem, State]:
    """Renumber landmarks so only observed ones remain."""
    used, new_j = torch.unique(pb.obs_j, return_inverse=True)
    u = used.cpu().numpy()
    pb.obs_j = new_j
    pb.n_lm = len(u)
    pb.lm_clip = pb.lm_clip[u]
    if pb.lm_track is not None:
        pb.lm_track = pb.lm_track[u]
    st.X = st.X[u]
    return pb, st


def stage_report(name, pb, st, grid, planes, t0):
    s = reprojection_stats(pb, st, grid)
    ext = np.linalg.inv(st.T_cl)
    rep = {"stage": name, "n_landmarks": pb.n_lm, "n_planes": int(len(planes[0])) if planes else 0,
           "reproj_median_px": s["median_px"], "reproj_rms_inlier_px": s["rms_inlier_px"],
           "inlier_frac": s["inlier_frac"], "intr": st.intr.tolist(), "dt_ms": st.dt * 1e3,
           "rs_ms": st.rs * 1e3, "cam_in_lidar_t": ext[:3, 3].tolist(), "tel": st.tel.tolist(),
           "seconds": time.time() - t0}
    print(f"  [{name}] lm {pb.n_lm}  planes {rep['n_planes']}  reproj median {s['median_px']:.3f}px  "
          f"rms(<3px) {s['rms_inlier_px']:.3f}px  inliers {100 * s['inlier_frac']:.1f}%  "
          f"dt {st.dt * 1e3:.2f}ms rs {st.rs * 1e3:.2f}ms  t_lc {np.round(ext[:3, 3], 3)}  "
          f"tel {np.round(st.tel, 3)}  "
          f"intr {np.round(st.intr[:4], 1)} {np.round(st.intr[4:], 4)}")
    return rep, s["err"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("channel")
    p.add_argument("--clips", type=Path, nargs="+", required=True)
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--lidar-ego", type=Path, required=True)
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=("kb", "radtan", "rational"), default="kb")
    p.add_argument("--grid", type=int, nargs=2, default=(12, 7))
    p.add_argument("--near", action=argparse.BooleanOptionalAction, default=True,
                   help="also fit the depth-dependent (1/range) windshield grid")
    p.add_argument("--no-time", action="store_true", help="keep time offset and rolling shutter at 0")
    p.add_argument("--rs", action=argparse.BooleanOptionalAction, default=True,
                   help="estimate the rolling-shutter readout (--no-rs holds it at 0: a global-"
                        "shutter camera, e.g. every camera on the DM rig)")
    p.add_argument("--frame-step", type=int, default=2,
                   help="use every Nth tracked frame (default 2); raise it to trade a little "
                        "precision for memory and time on long clips")
    p.add_argument("--fix-intr", action="store_true",
                   help="hold the lens model at whatever --init-from / --final-from supplies and "
                        "estimate only the pose and timing (a warm start from a target-based "
                        "calibration, whose intrinsics are usually far better determined)")
    p.add_argument("--pp-prior", type=float, default=None, metavar="SIGMA_PX",
                   help="Gaussian prior of this width pulling the principal point towards the "
                        "image centre. Needed for the same reason as --square-pixels: cy and the "
                        "camera's pitch are near-degenerate when the car only yaws.")
    p.add_argument("--square-pixels", type=float, nargs="?", const=0.05, default=None,
                   metavar="SIGMA_PX",
                   help="tie fy to fx with a Gaussian prior of this width in pixels (default "
                        "0.05 when the flag is given without a value). Needed wherever the "
                        "camera looks along the driving direction and the car only yaws: see "
                        "ba.lm_solve's tie_aspect.")
    p.add_argument("--init-f", type=float, default=None,
                   help="start from this focal length instead of the one init_cam scans for "
                        "(a basin-of-attraction check; the fit is still free to move it)")
    p.add_argument("--init-from", type=Path, default=None, help="start from a previous calib json")
    p.add_argument("--final-from", type=Path, default=None,
                   help="run only the final joint stage, starting from this calibration")
    p.add_argument("--matches", type=Path, default=None, help="LiDAR-image matches (refine_pnp.py)")
    p.add_argument("--tag", default="", help="suffix for the final stage's output file")
    p.add_argument("--fixed-lidar-ego", type=Path, default=None,
                   help="final stage: hold the LiDAR on the car at this json's T_ego_lidar "
                        "(e.g. the fusion of all cameras) instead of estimating it")
    p.add_argument("--free-motion-frame", action="store_true",
                   help="final stage on LiDAR odometry: estimate a small rigid offset between the "
                        "frame the camera's tracks move with and the LiDAR frame of the matches, as "
                        "the LiDAR -> ego transform is estimated on the INS. The two disagree by "
                        "~0.15 deg / ~5 cm on the 2026-09-23 drives (the INS runs show the same), "
                        "and without it the lens absorbs the conflict (cx moves by ~100 px); with it "
                        "the held-out LiDAR -> image error matches the INS calibration's")
    a = p.parse_args()
    t0 = time.time()

    traj = Trajectory(a.ins)
    T_el = np.array(json.loads(a.lidar_ego.read_text())["T_ego_lidar"])
    if traj.lidar_frame and not np.allclose(T_el, np.eye(4)):
        raise SystemExit(f"{a.ins} is LiDAR odometry (ego = LiDAR frame): --lidar-ego must be the identity")
    el = not traj.lidar_frame                   # estimate the LiDAR's place on the car?
    tracks = [dict(np.load(a.tracks / f"tracks_{c.name}_{a.channel}.npz")) for c in a.clips]
    W, H = (int(v) for v in tracks[0]["image_wh"])
    if a.final_from:
        final_stage(a, traj, tracks, W, H, t0)
        return

    # ------------------------------------------------------------- init
    if a.init_from:
        prev = json.loads(a.init_from.read_text())
        intr = np.array(prev["intr"]) if prev["model"] == a.model else None
        if intr is None and a.fix_intr:
            raise SystemExit(f"--fix-intr needs --init-from with model {a.model}, "
                             f"but that file is {prev['model']}")
        T_cl = np.array(prev["T_cam_lidar"])
    else:
        intr = None
    if intr is None or not a.init_from:
        ini = init_focal_and_rotation(tracks[-1], traj)
        f = ini["f"] if a.init_f is None else a.init_f
        intr = np.array([f, f, W / 2, H / 2] + [0.0] * (N_INTR[a.model] - 4), dtype=float)
        if not a.init_from:
            T_ec = np.eye(4)
            T_ec[:3, :3] = ini["R_ec"]
            T_ec[:3, 3] = T_el[:3, 3]                   # camera starts at the LiDAR's position
            T_cl = np.linalg.inv(T_ec) @ T_el
        print(f"init: f {f:.0f}px (scan said {ini['f']:.0f}), optical axis in ego "
              f"{np.round(ini['R_ec'][:, 2], 3)}, "
              f"hand-eye residual {np.degrees(ini['handeye_residual_median']):.2f} deg")
    st = State(intr=intr, T_cl=T_cl)

    pb = build_problem(tracks, traj, T_el, a.model, frame_step=a.frame_step)
    print(f"{a.channel}: {len(pb.obs_f)} observations, {pb.n_lm} tracks")
    print("building LiDAR maps ...")
    maps = []
    for c in a.clips:                       # the map is per clip, shared by every camera
        cache = a.out / f"map_{c.name}.npz"
        if cache.exists():
            z = np.load(cache)
            from scipy.spatial import cKDTree
            maps.append((z["P"], cKDTree(z["P"]), z["sweep"], z["R"]))
        else:
            maps.append(build_map(c, traj, T_el))
            a.out.mkdir(parents=True, exist_ok=True)
            np.savez(cache, P=maps[-1][0], sweep=maps[-1][2], R=maps[-1][3])
    pb.maps = maps
    print("  map points:", [len(m[0]) for m in pb.maps])

    reports = []
    grid = None

    def retri(st, pb, grid):
        X, ok = triangulate(pb, st, grid)
        st.X = X
        keep = ok[pb.obs_j.cpu().numpy()]
        pb = prune(pb, keep)
        pb, st = compact(pb, st)
        return st, pb

    def drop_outliers(st, pb, grid, thresh):
        s = reprojection_stats(pb, st, grid)
        pb = prune(pb, s["err"] < thresh)
        pb, st = compact(pb, st)
        return st, pb

    # --------------------------------------------------------- stage 1
    st, pb = retri(st, pb, grid)
    st, _ = lm_solve(pb, st, grid, {"intr": not a.fix_intr, "xi": True}, iters=12,
                     tie_aspect=a.square_pixels, pp_prior=a.pp_prior)
    st, pb = drop_outliers(st, pb, grid, 8.0)
    st, pb = retri(st, pb, grid)
    st, _ = lm_solve(pb, st, grid, {"intr": not a.fix_intr, "xi": True}, iters=12,
                     tie_aspect=a.square_pixels, pp_prior=a.pp_prior)
    rep, _ = stage_report("1 lens+extrinsic", pb, st, grid, None, t0)
    reports.append(rep)

    # --------------------------------------------------------- stage 2
    planes = None
    for gate in (0.5, 0.25, 0.15):
        st, pb = drop_outliers(st, pb, grid, 4.0)
        planes = associate_planes(pb, st.X, np.ones(pb.n_lm, bool), gate)
        st, _ = lm_solve(pb, st, grid, {"intr": not a.fix_intr, "xi": True, "tel": el}, planes=planes, iters=12,
                         priors={"tel": 0.5}, tie_aspect=a.square_pixels, pp_prior=a.pp_prior)
    rep, _ = stage_report("2 +LiDAR planes", pb, st, grid, planes, t0)
    reports.append(rep)

    # --------------------------------------------------------- stage 3
    if not a.no_time:
        planes = associate_planes(pb, st.X, np.ones(pb.n_lm, bool), 0.15)
        st, _ = lm_solve(pb, st, grid, {"intr": not a.fix_intr, "xi": True, "dt": True, "rs": a.rs, "tel": el},
                         planes=planes, iters=15, priors={"dt": 0.05, "rs": 0.05, "tel": 0.5}, tie_aspect=a.square_pixels, pp_prior=a.pp_prior)
        rep, _ = stage_report("3 +time offset/rolling shutter", pb, st, grid, planes, t0)
        reports.append(rep)

    # --------------------------------------------------------- stage 4
    gx, gy = a.grid
    if gx > 0:
        grid = Grid(W, H, gx, gy)
        st.G = np.zeros((4, grid.n))
        act = {"intr": not a.fix_intr, "xi": True, "Ginf": True, "Gnear": a.near, "tel": el}
        if not a.no_time:
            act.update(dt=True, rs=a.rs)
        for _ in range(2):
            st, pb = drop_outliers(st, pb, grid, 3.0)
            st, pb = retri(st, pb, grid)
            planes = associate_planes(pb, st.X, np.ones(pb.n_lm, bool), 0.15)
            st, _ = lm_solve(pb, st, grid, act, planes=planes, iters=15,
                             priors={"dt": 0.05, "rs": 0.05, "tel": 0.5}, tie_aspect=a.square_pixels, pp_prior=a.pp_prior)
        rep, _ = stage_report("4 +windshield field", pb, st, grid, planes, t0)
        reports.append(rep)

    tag = f"{a.channel}_{a.model}" + (f"_g{gx}x{gy}" + ("" if a.near else "_inf") if gx > 0 else "")
    out = {
        "channel": a.channel, "model": a.model, "width": W, "height": H,
        "intr": st.intr.tolist(), "T_cam_lidar": st.T_cl.tolist(),
        "T_lidar_cam": np.linalg.inv(st.T_cl).tolist(),
        "T_ego_lidar": (lambda T: (T.__setitem__((slice(0, 3), 3), T[:3, 3] + st.tel), T)[1])(T_el.copy()).tolist(),
        "T_ego_lidar_prior": T_el.tolist(), "tel_correction_m": st.tel.tolist(),
        "dt_s": st.dt, "rs_s": st.rs, "square_pixels_sigma_px": a.square_pixels, "fix_intr": a.fix_intr, "pp_prior_px": a.pp_prior, "rs_estimated": bool(a.rs and not a.no_time),
        "grid": None if grid is None else {"gx": gx, "gy": gy, "G": st.G.tolist(), "near": a.near},
        "clips": [c.name for c in a.clips], "stages": reports,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"calib_{tag}.json").write_text(json.dumps(out))
    print(f"-> {a.out / f'calib_{tag}.json'}  ({time.time() - t0:.0f}s)")


def final_stage(a, traj, tracks, W, H, t0) -> None:
    prev = json.loads(a.final_from.read_text())
    T_el = np.array(prev["T_ego_lidar"])
    if a.fixed_lidar_ego:
        T_el = np.array(json.loads(a.fixed_lidar_ego.read_text())["T_ego_lidar"])
    g = prev.get("grid")
    grid = Grid(W, H, g["gx"], g["gy"]) if g else None
    st = State(intr=np.array(prev["intr"]), T_cl=np.array(prev["T_cam_lidar"]),
               dt=prev.get("dt_s", 0.0), rs=prev.get("rs_s", 0.0),
               G=np.array(g["G"]) if g else None)
    matches = LidarMatches.load(a.matches)
    if len(matches.uv) == 0:
        raise SystemExit(
            f"{a.matches} holds no LiDAR-image matches, so there is nothing for the final "
            "stage to reconcile. That means refine_pnp.py found no consistent set: check "
            "its log, and check that the calibration it started from is not already so far "
            "out that the rendering it matches against is meaningless.")
    pb = build_problem(tracks, traj, T_el, prev["model"], frame_step=getattr(a, "frame_step", 2))
    print(f"{a.channel} final: {len(pb.obs_f)} track observations, {len(matches.uv)} LiDAR matches")
    # LiDAR odometry: the trajectory is the LiDAR's own, so there is nothing to estimate,
    # unless --free-motion-frame lets the frame the tracks move in sit a little off the
    # LiDAR frame the matches live in (see its help)
    free_el = a.fixed_lidar_ego is None and (not traj.lidar_frame or a.free_motion_frame)
    act = {"intr": not getattr(a, "fix_intr", False), "xi": True, "dt": True, "rs": getattr(a, "rs", True),
           "tel": free_el, "rel": free_el}
    if grid is not None:
        act.update(Ginf=True, Gnear=g.get("near", True))
    priors = {"dt": 0.05, "rs": 0.05, "tel": 0.5, "rel": np.radians(3.0)}
    reports = []
    for k, thresh in enumerate((6.0, 3.0, 3.0)):
        X, ok = triangulate(pb, st, grid)
        st.X = X
        pb = prune(pb, ok[pb.obs_j.cpu().numpy()])
        pb, st = compact(pb, st)
        s_ = reprojection_stats(pb, st, grid)
        pb = prune(pb, s_["err"] < thresh)
        pb, st = compact(pb, st)
        st, _ = lm_solve(pb, st, grid, act, iters=15, priors=priors, matches=matches, verbose=False,
                         tie_aspect=getattr(a, "square_pixels", None),
                         pp_prior=getattr(a, "pp_prior", None))
        lay = param_layout(pb, st, act)
        rm, _ = match_residuals(matches, st, grid, lay, pb.model, pb.H, 1.0)
        em = np.hypot(rm[0::2], rm[1::2])
        rep_, _ = stage_report(f"final {k}", pb, st, grid, None, t0)
        rep_.update(match_median_px=float(np.median(em)), match_inlier_frac=float(np.mean(em < 4)),
                    rel_deg=np.degrees(st.rel).tolist())
        print(f"    LiDAR-ego correction: t {np.round(st.tel, 3)} m, rot {np.round(np.degrees(st.rel), 3)} deg")
        print(f"    LiDAR matches: median {np.median(em):.2f}px, <4px {100 * np.mean(em < 4):.0f}%")
        reports.append(rep_)
    from ba import T_el_of
    T_el_new = T_el_of(T_el, st.tel, st.rel)
    out = dict(prev)
    out.update({"intr": st.intr.tolist(), "T_cam_lidar": st.T_cl.tolist(),
                "T_lidar_cam": np.linalg.inv(st.T_cl).tolist(), "T_ego_lidar": T_el_new.tolist(),
                "dt_s": st.dt, "rs_s": st.rs, "final_stages": reports,
                "grid": None if grid is None else {**g, "G": st.G.tolist()}})
    name = f"final_{a.channel}{a.tag}.json"
    (a.out / name).write_text(json.dumps(out))
    print(f"-> {a.out / name}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
