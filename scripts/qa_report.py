"""Auto-generated QA report for the produced NuScenes-formatted dataset.

Sections:
  1. Per-channel sync drop (from the source bag's raw timestamps)
  2. Per-scene sync survival (from nuscenes samples)
  3. Ego pose anomalies (large translation/rotation jumps between samples)
  4. Camera frame drops (per-channel long gaps in the source bag)
  5. Per-scene stats (samples, span, ego trajectory length, lidar point counts)

Usage:
    python scripts/qa_report.py /data/tcar_nuscenes [--bag /path/to.bag]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import collect_header_timestamps, make_typestore  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent


def section(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 60 - len(title)))


def per_channel_sync_drops(ts_by_ch: dict, sync_ms: float = 25.0) -> None:
    """ts_by_ch: channel -> sorted header timestamps (ns), from the source bag."""
    cam_channels = sorted(c for c in ts_by_ch if c != "LIDAR_TOP")
    lidar_ts = ts_by_ch.get("LIDAR_TOP")
    if lidar_ts is None or not len(cam_channels):
        print("  [skip] need both lidar and camera timestamps")
        return

    sync_ns = int(sync_ms * 1e6)
    print(f"  Lidar anchor: {len(lidar_ts)} frames")
    print(f"  Tolerance:    {sync_ms} ms")
    print(f"  {'channel':18} {'p50':>7} {'p99':>7} {'>tol':>7}  {'pct in tol':>11}")
    worst = np.zeros_like(lidar_ts)
    for ch in cam_channels:
        cam_ts = ts_by_ch[ch]
        idx = np.searchsorted(cam_ts, lidar_ts)
        idx_l = np.clip(idx - 1, 0, len(cam_ts) - 1)
        idx_r = np.clip(idx, 0, len(cam_ts) - 1)
        d = np.minimum(np.abs(cam_ts[idx_l] - lidar_ts), np.abs(cam_ts[idx_r] - lidar_ts))
        worst = np.maximum(worst, d)
        p50 = np.median(d) / 1e6
        p99 = np.percentile(d, 99) / 1e6
        n_over = int(np.sum(d > sync_ns))
        pct_in = 100 - 100 * n_over / len(d)
        print(f"  {ch:18} {p50:7.2f} {p99:7.2f} {n_over:7d}  {pct_in:10.2f}%")

    n_drop = int(np.sum(worst > sync_ns))
    print(f"  Worst-channel ≤ {sync_ms}ms:  {len(lidar_ts) - n_drop}/{len(lidar_ts)} "
          f"({100*(len(lidar_ts)-n_drop)/len(lidar_ts):.2f}%)")


def per_scene_sync(nusc: NuScenes) -> None:
    """For each scene, count how many of the expected 2Hz keyframes survived."""
    print(f"  {'scene':12} {'samples':>8} {'span(s)':>8} {'expected':>9} {'kept%':>7}")
    for sc in nusc.scene:
        first = nusc.get('sample', sc['first_sample_token'])
        last = nusc.get('sample', sc['last_sample_token'])
        span_s = (last['timestamp'] - first['timestamp']) / 1e6
        expected = round(span_s * 2) + 1  # 2 Hz
        kept_pct = 100 * sc['nbr_samples'] / max(expected, 1)
        flag = " <" if kept_pct < 90 else ""
        print(f"  {sc['name']:12} {sc['nbr_samples']:>8} {span_s:>8.2f} "
              f"{expected:>9} {kept_pct:>6.1f}%{flag}")


def ego_pose_anomalies(nusc: NuScenes,
                       lin_thresh: float = 30.0,  # m/s
                       ang_thresh: float = 90.0   # deg/s
                       ) -> None:
    """Per scene, walk consecutive samples, flag big jumps."""
    print(f"  thresholds: |Δt|/Δt > {lin_thresh} m/s OR rotation > {ang_thresh} deg/s")
    n_flagged = 0
    flagged_scenes: dict[str, int] = {}
    for sc in nusc.scene:
        prev_pose = None
        prev_ts = None
        tok = sc['first_sample_token']
        while tok:
            s = nusc.get('sample', tok)
            sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
            ego = nusc.get('ego_pose', sd['ego_pose_token'])
            if prev_pose is not None:
                dt = (ego['timestamp'] - prev_ts) / 1e6
                if dt > 0:
                    dxyz = np.array(ego['translation']) - np.array(prev_pose['translation'])
                    lin = np.linalg.norm(dxyz) / dt
                    q1 = ego['rotation']  # wxyz
                    q0 = prev_pose['rotation']
                    R1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                    R0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
                    ang = (R0.inv() * R1).magnitude() * 180 / np.pi / dt
                    if lin > lin_thresh or ang > ang_thresh:
                        n_flagged += 1
                        flagged_scenes[sc['name']] = flagged_scenes.get(sc['name'], 0) + 1
            prev_pose, prev_ts = ego, ego['timestamp']
            tok = s['next']
    if n_flagged == 0:
        print(f"  no anomalies (good)")
    else:
        print(f"  flagged {n_flagged} sample boundaries")
        for sname, cnt in sorted(flagged_scenes.items()):
            print(f"    {sname}: {cnt}")


def camera_frame_drops(ts_by_ch: dict) -> None:
    """Long gaps mean frames were dropped before they ever reached the bag."""
    print(f"  {'channel':18} {'frames':>7} {'med dt':>8} {'>3x med':>8} {'longest':>10}")
    for ch in sorted(c for c in ts_by_ch if c != "LIDAR_TOP"):
        ts = ts_by_ch[ch]
        if len(ts) < 2:
            continue
        dts = np.diff(ts) / 1e6
        med = float(np.median(dts))
        print(f"  {ch:18} {len(ts):>7} {med:7.2f}ms "
              f"{int(np.sum(dts > 3 * med)):>8} {float(dts.max()):8.1f}ms")


def per_scene_stats(nusc: NuScenes, dataroot: Path) -> None:
    print(f"  {'scene':12} {'samples':>8} {'span(s)':>8} {'traj(m)':>8} "
          f"{'avg_v':>7} {'lidar_pts(med)':>14}")
    for sc in nusc.scene:
        xs, ys, zs, lidar_pt_counts = [], [], [], []
        tok = sc['first_sample_token']
        while tok:
            s = nusc.get('sample', tok)
            sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
            ego = nusc.get('ego_pose', sd['ego_pose_token'])
            xs.append(ego['translation'][0])
            ys.append(ego['translation'][1])
            zs.append(ego['translation'][2])
            # cheap lidar point count via raw .bin file size
            f = dataroot / sd['filename']
            if f.exists():
                size = f.stat().st_size
                # 5 floats * 4 bytes = 20 bytes/point
                lidar_pt_counts.append(size // 20)
            tok = s['next']
        if not xs:
            continue
        xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
        traj = float(np.sum(np.linalg.norm(np.diff(np.stack([xs, ys, zs], -1), axis=0), axis=1)))
        first = nusc.get('sample', sc['first_sample_token'])
        last = nusc.get('sample', sc['last_sample_token'])
        span = (last['timestamp'] - first['timestamp']) / 1e6
        avg_v = traj / span if span > 0 else 0.0
        med_pts = int(np.median(lidar_pt_counts)) if lidar_pt_counts else 0
        print(f"  {sc['name']:12} {sc['nbr_samples']:>8} {span:>8.2f} {traj:>8.1f} "
              f"{avg_v:>6.2f}m/s {med_pts:>14}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataroot", type=Path, help="NuScenes dataroot, e.g. /data/tcar_nuscenes")
    p.add_argument("--bag", type=Path, default=None,
                   help="Source .bag, for the raw-timestamp sections [1] and [2]. "
                        "Those are skipped when it is not given.")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--sync-ms", type=float, default=25.0)
    args = p.parse_args()

    print(f"NuScenes: {args.dataroot} ({args.version})")
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    print(f"  {len(nusc.log)} log(s), {len(nusc.scene)} scenes, {len(nusc.sample)} samples, "
          f"{len(nusc.sample_data)} sample_data")

    if args.bag:
        # Sections [1] and [2] measure what the bag contained, which is the only
        # place frames that never made it into the dataset can still be counted.
        typestore = make_typestore((_ROOT / "packet_decoder" / "src"
                                    / "rslidar_msg" / "msg", "rslidar_msg"))
        ts_by_ch = collect_header_timestamps(args.bag, typestore)
        section(f"[1] Per-channel sync (source bag)  — {args.bag.name}")
        per_channel_sync_drops(ts_by_ch, args.sync_ms)
        section(f"[2] Camera frame drops (source bag)  — {args.bag.name}")
        camera_frame_drops(ts_by_ch)
    else:
        section("[1,2] Raw-timestamp sections skipped (pass --bag to enable)")

    for log in nusc.log:
        print(f"  log: {log['logfile']}  (date={log['date_captured']})")

    section("[3] Per-scene sync survival (kept vs expected at 2Hz)")
    per_scene_sync(nusc)

    section("[4] Ego pose anomalies (jumps between consecutive samples)")
    ego_pose_anomalies(nusc)

    section("[5] Per-scene stats")
    per_scene_stats(nusc, args.dataroot)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
