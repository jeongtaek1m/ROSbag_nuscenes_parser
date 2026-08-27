"""Verify a stage-1 intermediate dir is consistent before running stage 2.

Checks:
  - All required artifacts exist (calib.json, meta.json, meta_lidar.json,
    odom.parquet, cameras/CH/, lidar/).
  - Camera frame counts match meta.json.
  - Lidar frame count matches meta_lidar.json.
  - Odom samples match meta.json count.
  - All channels in cam dir map to a topic in meta.json's topic_to_channel.
  - All standard nuscenes cam channels are present in calib.
  - Camera/lidar/odom timestamp range overlaps bag_start_ns..bag_end_ns.
  - No large gaps in odom (>2 sec).

Usage:
    python scripts/preflight_check.py /data/intermediate/<bag>
Exits 0 on pass, 1 on warning, 2 on fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


NUSCENES_CAMS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("intermediate", type=Path)
    args = p.parse_args()

    root = args.intermediate
    fails: list[str] = []
    warns: list[str] = []
    oks: list[str] = []

    def fail(msg): fails.append(msg)
    def warn(msg): warns.append(msg)
    def ok(msg):   oks.append(msg)

    # 1. Required files
    required = ["calib.json", "meta.json", "meta_lidar.json", "odom.parquet"]
    for r in required:
        if not (root / r).exists():
            fail(f"missing: {r}")
    if not (root / "cameras").exists():
        fail("missing: cameras/")
    if not (root / "lidar").exists():
        fail("missing: lidar/")
    if fails:
        for f in fails:
            print(f"FAIL  {f}")
        sys.exit(2)

    meta = json.loads((root / "meta.json").read_text())
    meta_lidar = json.loads((root / "meta_lidar.json").read_text())
    calib = json.loads((root / "calib.json").read_text())

    # 2. Camera frame counts vs meta
    expected_cams = {ch for ch in meta["topic_to_channel"].values()
                     if ch not in ("EGO", "ANNOTATION")
                     and "LIDAR_TOP" not in ch}
    cam_dirs = {d.name for d in (root / "cameras").iterdir() if d.is_dir()}
    if expected_cams != cam_dirs:
        warn(f"camera dir mismatch: expected {sorted(expected_cams)}, "
             f"found {sorted(cam_dirs)}")
    total_cam_frames = 0
    cam_counts: dict[str, int] = {}
    for ch in cam_dirs:
        n = sum(1 for _ in (root / "cameras" / ch).glob("*.jpg"))
        cam_counts[ch] = n
        total_cam_frames += n
    if total_cam_frames != meta["n_camera_frames"]:
        warn(f"camera frame count drift: meta={meta['n_camera_frames']}, "
             f"found={total_cam_frames}")
    else:
        ok(f"camera frames: {total_cam_frames} (matches meta)")

    # 3. Lidar frame count vs meta_lidar
    n_lidar = sum(1 for _ in (root / "lidar").glob("*.bin.zst"))
    if n_lidar != meta_lidar["n_lidar_frames"]:
        warn(f"lidar frame count drift: meta_lidar={meta_lidar['n_lidar_frames']}, "
             f"found={n_lidar}")
    else:
        ok(f"lidar frames: {n_lidar} (matches meta_lidar)")

    # 4. Odom samples vs meta
    odom_tbl = pq.read_table(root / "odom.parquet")
    n_odom = len(odom_tbl)
    if n_odom != meta["n_odom_samples"]:
        warn(f"odom samples drift: meta={meta['n_odom_samples']}, found={n_odom}")
    else:
        ok(f"odom samples: {n_odom} (matches meta)")

    # 5. Required calib channels
    for ch in NUSCENES_CAMS + ["LIDAR_TOP"]:
        if ch not in calib:
            fail(f"calib missing: {ch}")
    if not fails:
        ok(f"calib has all 6 std cams + LIDAR_TOP")

    # 6. Timestamp range coverage
    bag_start, bag_end = meta["bag_start_ns"], meta["bag_end_ns"]
    bag_dur_s = (bag_end - bag_start) / 1e9

    odom_ts = np.asarray(odom_tbl.column("ts_ns")).astype(np.int64)
    if odom_ts.min() < bag_start or odom_ts.max() > bag_end:
        warn("odom timestamps outside bag range")
    else:
        ok(f"odom ts in bag range ({bag_dur_s:.1f}s)")

    lidar_ts = np.array(
        sorted(int(f.name.split(".", 1)[0]) for f in (root / "lidar").glob("*.bin.zst")),
        dtype=np.int64,
    )
    if len(lidar_ts) > 0:
        # Lidar uses bag receive time so range should match bag span
        coverage = (lidar_ts.max() - lidar_ts.min()) / 1e9
        if coverage < 0.95 * bag_dur_s:
            warn(f"lidar coverage only {coverage:.1f}s of {bag_dur_s:.1f}s bag")
        else:
            ok(f"lidar coverage {coverage:.1f}s of {bag_dur_s:.1f}s bag")

    # 7. Per-channel cam frame rate sanity
    for ch in cam_dirs:
        ts = np.array(sorted(int(f.stem) for f in (root / "cameras" / ch).glob("*.jpg")),
                      dtype=np.int64)
        if len(ts) < 2:
            warn(f"{ch}: <2 frames")
            continue
        dts = np.diff(ts) / 1e6  # ms
        med = np.median(dts)
        gaps = int(np.sum(dts > 3 * med))  # >3x median = real drop
        if gaps > 0:
            warn(f"{ch}: {gaps} long gaps (>3× median dt of {med:.1f}ms)")
        else:
            ok(f"{ch}: {len(ts)} frames, median dt {med:.1f}ms")

    # 8. Odom gaps
    odom_dts = np.diff(odom_ts) / 1e6  # ms
    odom_gaps = int(np.sum(odom_dts > 2000))
    if odom_gaps > 0:
        warn(f"odom: {odom_gaps} gaps > 2s")
    else:
        med_odom = np.median(odom_dts)
        ok(f"odom: median dt {med_odom:.1f}ms, no gaps")

    # 9. Lidar decoder calibration ready
    if not meta_lidar.get("lidar_decoder", {}).get("calibration_ready", False):
        fail("lidar decoder calibration not ready (DIFOP missing?)")
    skipped = meta_lidar.get("lidar_decoder", {}).get("msop_skipped_before_calib", 0)
    if skipped > 1000:
        warn(f"lidar: {skipped} MSOP packets skipped before DIFOP arrived "
             f"(~{skipped/1000:.1f}s of data lost at start)")

    # ----- summary -----
    print()
    for o in oks:
        print(f"OK    {o}")
    for w in warns:
        print(f"WARN  {w}")
    for f in fails:
        print(f"FAIL  {f}")
    print()
    print(f"Result: {len(oks)} OK, {len(warns)} WARN, {len(fails)} FAIL")
    if fails:
        sys.exit(2)
    if warns:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
