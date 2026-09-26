#!/usr/bin/env python3
"""Final stage for all cameras at once: one LiDAR -> ego (or motion-frame) offset shared
by every camera, and optionally landmarks shared between cameras.

    python online_calib/rig_final.py --channels CAM_FRONT ... --clips CLIP [...] --ins TRAJ \
        --lidar-ego START.json --tracks DIR --from DIR --out DIR [--links LINKS.npz]

The per-camera final stage (calibrate.py --final-from) estimates the LiDAR's place on
the car (on LiDAR odometry: the offset between the frame the tracks move with and the
LiDAR frame, --free-motion-frame) once per camera, and the seven answers are fused
afterwards. It is one physical quantity, so here it is one unknown: the Gauss-Newton
step of the joint problem is taken exactly (ba.joint_lm_solve: every camera eliminates
its landmarks and then its own parameters; the shared offset, and the shared
landmarks, are solved together).

Inputs per camera: --from DIR/<prefix><CHANNEL>.json (a calibration to start from,
e.g. refine_pnp.py's output) and DIR/<prefix><CHANNEL>.matches.npz (its LiDAR-image
matches). --lidar-ego: the offset every camera starts from (identity on LiDAR
odometry). --links (cross_tracks.py): new tracks carried from one camera into its
neighbour, linked to the tracks they came from; each linked group becomes one landmark
seen by all of its cameras.

Writes DIR/final_<CHANNEL>.json (calibrate.py's format; T_ego_lidar is the shared one)
and DIR/rig_final.json (the shared offset and the per-round statistics).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ba import (CamLM, LidarMatches, State, T_el_of, build_problem, joint_lm_solve, match_residuals,
                param_layout, reprojection_stats, triangulate)
from calibrate import compact, prune
from camera_model import Grid
from traj import Trajectory


def load_links(path: Path, clips: list[Path], channels: list[str], max_groups: int):
    """Linked tracks -> groups: list of {(camera, clip, track id), ...}, largest first."""
    z = np.load(path, allow_pickle=True)
    clip_ix = {c.name: i for i, c in enumerate(clips)}
    cam_ix = {c: i for i, c in enumerate(channels)}
    parent: dict = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for clip, ca, ta, cb, tb in zip(z["clip"], z["cam_a"], z["track_a"], z["cam_b"], z["track_b"]):
        if clip not in clip_ix or ca not in cam_ix or cb not in cam_ix:
            continue
        a = (cam_ix[ca], clip_ix[clip], int(ta))
        b = (cam_ix[cb], clip_ix[clip], int(tb))
        parent[find(a)] = find(b)
    groups: dict = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    out = [g for g in groups.values() if len({n[0] for n in g}) >= 2]
    if max_groups and len(out) > max_groups:           # an even sample, not just the largest
        out = [out[k] for k in np.random.default_rng(0).choice(len(out), max_groups, replace=False)]
    return out


def add_cross_tracks(tr: dict, extra, clip: str, ch: str) -> dict:
    """Append cross_tracks.py's new tracks of this camera and clip to its KLT tracks."""
    if extra is None:
        return tr
    m = (extra["x_clip"] == clip) & (extra["x_cam"] == ch)
    if m.any():
        tr["obs_frame"] = np.concatenate([tr["obs_frame"], extra["x_frame"][m]])
        tr["obs_track"] = np.concatenate([tr["obs_track"], extra["x_track"][m]])
        tr["obs_xy"] = np.concatenate([tr["obs_xy"], extra["x_xy"][m].astype(tr["obs_xy"].dtype)])
    return tr


def glob_index(pbs, groups) -> list[np.ndarray]:
    """Per camera: its landmarks' shared-landmark id, or -1."""
    look = [{(int(c), int(t)): j for j, (c, t) in enumerate(zip(pb.lm_clip, pb.lm_track))} for pb in pbs]
    glob = [np.full(pb.n_lm, -1) for pb in pbs]
    for g, grp in enumerate(groups):
        for cam, clip, track in grp:
            j = look[cam].get((clip, track))
            if j is not None:
                glob[cam][j] = g
    return glob


def share_landmarks(sts, glob, pbs, n_glob: int):
    """Give every copy of a shared landmark the same position: the observation-weighted
    mean of the copies' own triangulations."""
    acc = np.zeros((n_glob, 3))
    wsum = np.zeros(n_glob)
    for st, g, pb in zip(sts, glob, pbs):
        cnt = np.bincount(pb.obs_j.cpu().numpy(), minlength=pb.n_lm)
        m = g >= 0
        np.add.at(acc, g[m], st.X[m] * cnt[m, None])
        np.add.at(wsum, g[m], cnt[m])
    mean = acc / np.maximum(wsum, 1)[:, None]
    for st, g in zip(sts, glob):
        m = g >= 0
        st.X[m] = mean[g[m]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--channels", nargs="+", required=True)
    p.add_argument("--clips", type=Path, nargs="+", required=True)
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--lidar-ego", type=Path, required=True, help="the shared offset to start from")
    p.add_argument("--tracks", type=Path, required=True)
    p.add_argument("--from", dest="src", type=Path, required=True)
    p.add_argument("--prefix", default="pnp2_", help="file name prefix of the inputs in --from")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--links", type=Path, default=None)
    p.add_argument("--max-shared-landmarks", type=int, default=None)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--iters", type=int, default=8, help="LM iterations per round (it starts from the per-camera finals)")
    a = p.parse_args()
    t0 = time.time()
    traj = Trajectory(a.ins)
    T_el0 = np.array(json.loads(a.lidar_ego.read_text())["T_ego_lidar"])
    C = len(a.channels)

    groups = load_links(a.links, a.clips, a.channels, a.max_shared_landmarks) if a.links else []
    extra = None
    if a.links:
        z = np.load(a.links)
        # only the new tracks of the groups kept (their ids are unique across the file)
        used = np.array(sorted({tr for g in groups for _, _, tr in g}), np.int64)
        m = np.isin(z["x_track"], used)
        extra = {k: z[k][m] for k in ("x_clip", "x_cam", "x_track", "x_frame", "x_xy")}
    prevs, pbs, sts, grids, matches = [], [], [], [], []
    for ch in a.channels:
        prev = json.loads((a.src / f"{a.prefix}{ch}.json").read_text())
        tracks = [add_cross_tracks(dict(np.load(a.tracks / f"tracks_{c.name}_{ch}.npz")), extra, c.name, ch)
                  for c in a.clips]
        W, H = (int(v) for v in tracks[0]["image_wh"])
        g = prev.get("grid")
        grids.append(Grid(W, H, g["gx"], g["gy"]) if g else None)
        sts.append(State(intr=np.array(prev["intr"]), T_cl=np.array(prev["T_cam_lidar"]),
                         dt=prev.get("dt_s", 0.0), rs=prev.get("rs_s", 0.0), G=np.array(g["G"]) if g else None))
        pbs.append(build_problem(tracks, traj, T_el0, prev["model"], frame_step=a.frame_step))
        matches.append(LidarMatches.load(a.src / f"{a.prefix}{ch}.matches.npz"))
        prevs.append(prev)
        print(f"{ch}: {len(pbs[-1].obs_f)} track observations, {len(matches[-1].uv)} LiDAR matches", flush=True)

    n_glob = len(groups)
    glob = glob_index(pbs, groups)
    if a.links:
        print(f"shared landmarks: {n_glob} groups, "
              f"{sum(int((g >= 0).sum()) for g in glob)} camera copies", flush=True)

    rounds = []
    for k, thresh in enumerate((6.0, 3.0, 3.0)):
        for c in range(C):
            X, ok = triangulate(pbs[c], sts[c], grids[c])
            sts[c].X = X
            keep = ok[pbs[c].obs_j.cpu().numpy()]
            pbs[c], sts[c], glob[c] = _prune(pbs[c], sts[c], glob[c], keep)
        share_landmarks(sts, glob, pbs, n_glob)
        for c in range(C):
            err = reprojection_stats(pbs[c], sts[c], grids[c])["err"]
            pbs[c], sts[c], glob[c] = _prune(pbs[c], sts[c], glob[c], err < thresh)
        cams = []
        for c in range(C):
            g = prevs[c].get("grid")
            act = {"intr": True, "xi": True, "dt": True, "rs": True, "tel": True, "rel": True}
            if g:
                act.update(Ginf=True, Gnear=g.get("near", True))
            priors = {"dt": 0.05, "rs": 0.05}
            if c == 0:                                  # the shared offset's prior, once
                priors.update(tel=0.5, rel=np.radians(3.0))
            cams.append(CamLM(pbs[c], sts[c], grids[c], act, priors=priors, matches=matches[c]))
        sts, cost = joint_lm_solve(cams, sts, glob, n_glob, iters=a.iters)
        rep = {"round": k, "cost": cost, "tel_m": sts[0].tel.tolist(), "rel_deg": np.degrees(sts[0].rel).tolist(),
               "cameras": {}}
        for c, ch in enumerate(a.channels):
            s = reprojection_stats(pbs[c], sts[c], grids[c])
            rm, _ = match_residuals(matches[c], sts[c], grids[c], {"P": 0}, pbs[c].model, pbs[c].H, 1.0)
            em = np.hypot(rm[0::2], rm[1::2])
            rep["cameras"][ch] = {"n_landmarks": pbs[c].n_lm, "n_shared": int((glob[c] >= 0).sum()),
                                  "reproj_median_px": s["median_px"], "reproj_rms_inlier_px": s["rms_inlier_px"],
                                  "inlier_frac": s["inlier_frac"], "match_median_px": float(np.median(em)),
                                  "match_inlier_frac": float(np.mean(em < 4))}
            print(f"  [{k}] {ch:16s} lm {pbs[c].n_lm:6d} shared {rep['cameras'][ch]['n_shared']:5d}  "
                  f"reproj median {s['median_px']:.3f}px  matches median {np.median(em):.2f}px", flush=True)
        print(f"  [{k}] shared offset: t {np.round(sts[0].tel, 3)} m, rot {np.round(np.degrees(sts[0].rel), 3)} deg "
              f"({time.time() - t0:.0f}s)", flush=True)
        rounds.append(rep)

    T_el = T_el_of(T_el0, sts[0].tel, sts[0].rel)
    a.out.mkdir(parents=True, exist_ok=True)
    for c, ch in enumerate(a.channels):
        st, g = sts[c], prevs[c].get("grid")
        out = dict(prevs[c])
        out.update({"intr": st.intr.tolist(), "T_cam_lidar": st.T_cl.tolist(),
                    "T_lidar_cam": np.linalg.inv(st.T_cl).tolist(), "T_ego_lidar": T_el.tolist(),
                    "dt_s": st.dt, "rs_s": st.rs,
                    "final_stages": [dict(r["cameras"][ch], stage=f"rig {r['round']}") for r in rounds],
                    "grid": None if g is None else {**g, "G": st.G.tolist()},
                    "rig_final": {"channels": a.channels, "links": str(a.links) if a.links else None}})
        (a.out / f"final_{ch}.json").write_text(json.dumps(out))
    (a.out / "rig_final.json").write_text(json.dumps({
        "T_ego_lidar": T_el.tolist(), "T_ego_lidar_start": T_el0.tolist(), "channels": a.channels,
        "n_shared_landmarks": n_glob, "rounds": rounds, "seconds": time.time() - t0}, indent=1))
    print(f"-> {a.out}  ({time.time() - t0:.0f}s)")


def _prune(pb, st, g, keep):
    """prune + compact, carrying the shared-landmark ids along."""
    pb = prune(pb, keep)
    import torch
    used = torch.unique(pb.obs_j).cpu().numpy()
    pb, st = compact(pb, st)
    return pb, st, g[used]


if __name__ == "__main__":
    main()
