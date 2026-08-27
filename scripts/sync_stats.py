"""Sync analysis from a stage-1 intermediate folder.

For each lidar timestamp (keyframe anchor), computes |cam_ts - lidar_ts| per
channel via nearest-neighbor (both sides). Reports per-channel distribution and
'all channels must pass' acceptance rate per tolerance.

Usage:
    python scripts/sync_stats.py /data/intermediate/<bag_name>
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("intermediate", type=Path,
                   help="Stage-1 intermediate dir, e.g. /data/intermediate/<bag>")
    args = p.parse_args()

    cam_root = args.intermediate / "cameras"
    lidar_root = args.intermediate / "lidar"
    if not cam_root.exists():
        raise SystemExit(f"missing: {cam_root}")
    if not lidar_root.exists():
        raise SystemExit(f"missing: {lidar_root}")

    cam_channels = sorted([d.name for d in cam_root.iterdir() if d.is_dir()])
    print(f"channels: {cam_channels}")

    ts_by_ch: dict = {}
    for ch in cam_channels:
        ts = np.array(sorted(int(f.stem) for f in (cam_root / ch).glob("*.jpg")),
                      dtype=np.int64)
        ts_by_ch[ch] = ts
    ts_by_ch["LIDAR_TOP"] = np.array(
        sorted(int(f.name.split(".", 1)[0]) for f in lidar_root.glob("*.bin.zst")),
        dtype=np.int64,
    )

    for ch in cam_channels + ["LIDAR_TOP"]:
        ts = ts_by_ch[ch]
        if len(ts) < 2:
            print(f"  {ch:18}  n={len(ts):6}  (not enough samples)")
            continue
        print(f"  {ch:18}  n={len(ts):6}  span={ts[-1]-ts[0]:>15} ns "
              f"({(ts[-1]-ts[0])/1e9:.1f}s)  median dt={np.median(np.diff(ts))/1e6:.2f}ms")

    lidar_ts = ts_by_ch["LIDAR_TOP"]
    print("\n=== Per-channel |cam_ts - lidar_ts| stats (ms) ===")
    print(f"{'channel':18} {'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}  pct≤25ms  pct≤50ms  pct≤100ms")

    worst_per_kf = np.zeros_like(lidar_ts, dtype=np.int64)
    for ch in cam_channels:
        cam_ts = ts_by_ch[ch]
        idx = np.searchsorted(cam_ts, lidar_ts)
        idx_left = np.clip(idx - 1, 0, len(cam_ts) - 1)
        idx_right = np.clip(idx, 0, len(cam_ts) - 1)
        d_left = np.abs(cam_ts[idx_left] - lidar_ts)
        d_right = np.abs(cam_ts[idx_right] - lidar_ts)
        diffs = np.minimum(d_left, d_right)
        worst_per_kf = np.maximum(worst_per_kf, diffs)
        p50, p90, p95, p99 = np.percentile(diffs, [50, 90, 95, 99])
        pct_25 = 100 * np.mean(diffs <= 25_000_000)
        pct_50 = 100 * np.mean(diffs <= 50_000_000)
        pct_100 = 100 * np.mean(diffs <= 100_000_000)
        print(f"{ch:18} {p50/1e6:7.2f} {p90/1e6:7.2f} {p95/1e6:7.2f} {p99/1e6:7.2f} {diffs.max()/1e6:7.1f}  "
              f"{pct_25:6.2f}%   {pct_50:6.2f}%   {pct_100:6.2f}%")

    print(f"\n=== Per-keyframe worst-channel sync (all {len(cam_channels)} cams must pass) ===")
    print(f"{'tolerance':>10}  {'kept keyframes':>16}  {'%':>7}  {'dropped':>8}")
    for tol_ms in [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
        tol_ns = tol_ms * 1_000_000
        kept = int(np.sum(worst_per_kf <= tol_ns))
        pct = 100 * kept / len(lidar_ts)
        print(f"{tol_ms:>7} ms  {kept:>16}  {pct:6.2f}%  {len(lidar_ts) - kept:>8}")

    print("\n=== Camera frame rate (median Δt) ===")
    for ch in cam_channels:
        dt_med = np.median(np.diff(ts_by_ch[ch])) / 1e6
        print(f"  {ch:18}  {dt_med:.2f} ms (~{1000/dt_med:.1f} Hz)  "
              f"→ theoretical min sync error ~{dt_med/2:.1f} ms")
    dt_lidar = np.median(np.diff(lidar_ts)) / 1e6
    print(f"  LIDAR_TOP         {dt_lidar:.2f} ms (~{1000/dt_lidar:.1f} Hz)")

    n_2hz = len(lidar_ts) // 5
    print(f"\nIf 2Hz keyframes: ~{n_2hz} samples (every 5th lidar)")


if __name__ == "__main__":
    main()
