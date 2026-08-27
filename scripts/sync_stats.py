"""Sync analysis straight from a rosbag — run it before converting.

For each lidar timestamp (the keyframe anchor), computes |cam_ts - lidar_ts| per
channel by two-sided nearest neighbour. Reports the per-channel distribution and
the 'all channels must pass' acceptance rate at each candidate tolerance, which
is how --sync-ms gets chosen.

Usage:
    python scripts/sync_stats.py /path/to.bag
    python scripts/sync_stats.py /path/to.bag --max-seconds 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import collect_header_timestamps, make_typestore  # noqa: E402


def main():
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path, help="Path to a .bag file.")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Only inspect the first N seconds (0 = whole bag).")
    p.add_argument("--packet-msg-dir", type=Path,
                   default=here / "packet_decoder" / "src" / "rslidar_msg" / "msg")
    args = p.parse_args()

    typestore = make_typestore((args.packet_msg_dir, "rslidar_msg"))
    ts_by_ch = collect_header_timestamps(args.bag, typestore, args.max_seconds)
    if "LIDAR_TOP" not in ts_by_ch:
        raise SystemExit(f"no lidar topic in {args.bag}")
    cam_channels = sorted(c for c in ts_by_ch if c != "LIDAR_TOP")
    print(f"channels: {cam_channels}")

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
