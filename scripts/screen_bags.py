"""Screen rosbags for conversion suitability, by measured camera delivery rate.

The cameras reach the ROS1 bag through /ros_bridge. A ROS1 bag does NOT record
the DDS QoS the bridge subscribed with (ConnectionExtRosbag1 carries only
callerid and latching), so the setting cannot be read off the file. What can be
measured is its consequence: a best_effort bridge silently drops frames under
load, a reliable one does not.

Method, per camera topic:
  - modal inter-arrival gap  -> the source cadence (loss does not move the mode)
  - delivered / expected     -> delivery ratio at that cadence
  - longest gap              -> worst single dropout

A bag whose channels all sit near 100% was recorded reliable-like and is safe to
convert. Mixed per-channel ratios mean frames were dropped on the bridge; those
frames are gone and no amount of downstream sync work brings them back.

Usage:
    python scripts/screen_bags.py /path/*.bag
    python scripts/screen_bags.py /mnt/ssd/bags/ --min-delivery 0.99
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    LIDAR_PACKETS_TOPIC,
    LIDAR_POINTS_TOPIC,
    ODOM_TOPIC,
)

CAM_RE = re.compile(r"^/camera_\d+/compressed$")
LIDAR_TOPICS = (LIDAR_POINTS_TOPIC, LIDAR_PACKETS_TOPIC)


def collect(bag: Path, max_seconds: float) -> tuple[float, dict[str, np.ndarray], dict[str, str]]:
    """Arrival times per topic of interest. No deserialization — we only need
    when each message landed, so the JPEG payloads are never touched."""
    times: dict[str, list[int]] = defaultdict(list)
    callerid: dict[str, str] = {}
    with AnyReader([bag]) as reader:
        dur = (reader.end_time - reader.start_time) / 1e9
        wanted = [c for c in reader.connections
                  if CAM_RE.match(c.topic) or c.topic in LIDAR_TOPICS
                  or c.topic == ODOM_TOPIC]
        if not wanted:
            return dur, {}, {}
        for c in wanted:
            callerid[c.topic] = getattr(c.ext, "callerid", "?")
        t_end = reader.start_time + int(max_seconds * 1e9) if max_seconds else None
        for conn, bag_ns, _raw in reader.messages(connections=wanted):
            if t_end and bag_ns > t_end:
                break
            times[conn.topic].append(bag_ns)
    span = min(dur, max_seconds) if max_seconds else dur
    return span, {k: np.asarray(v, dtype=np.int64) for k, v in times.items()}, callerid


def rate_stats(ts: np.ndarray, span_s: float, nominal_hz: float = 0.0) -> dict:
    if len(ts) < 10:
        return {}
    dt = np.diff(ts) / 1e9
    modal = float(np.median(dt))          # loss adds long gaps, mode is unmoved
    inferred_hz = 1.0 / modal
    # Bursty publishers (two messages back to back) drag the modal gap far below
    # the true cadence and would fake a huge loss. Only trust the inference when
    # it stays near the observed average rate.
    avg_hz = len(ts) / span_s
    unreliable = inferred_hz > 1.3 * avg_hz
    nominal_hz = nominal_hz or inferred_hz
    expected = span_s * nominal_hz
    return {
        "n": len(ts),
        "hz": len(ts) / span_s,
        "nominal_hz": nominal_hz,
        "delivery": len(ts) / expected,
        "max_gap_ms": float(dt.max()) * 1e3,
        "n_gaps": int(np.sum(dt > 1.5 * modal)),
        "unreliable": unreliable,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", type=Path, nargs="+",
                   help="Bag files, or directories to scan for *.bag")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="Only inspect the first N seconds (0 = whole bag).")
    p.add_argument("--nominal-hz", type=float, default=0.0,
                   help="Known camera source rate (e.g. 30). Default: infer from "
                        "the modal inter-arrival gap.")
    p.add_argument("--min-delivery", type=float, default=0.99,
                   help="Per-channel delivery ratio required to pass (default 0.99).")
    args = p.parse_args()

    bags: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            bags.extend(sorted(path.rglob("*.bag")))
        elif path.suffix == ".bag":
            bags.append(path)
    if not bags:
        raise SystemExit("no .bag files found")

    verdicts: list[tuple[Path, str, float]] = []
    for bag in bags:
        print(f"\n{'=' * 78}\n{bag}")
        try:
            span, times, callerid = collect(bag, args.max_seconds)
        except Exception as exc:
            print(f"  !! unreadable: {exc}")
            verdicts.append((bag, "UNREADABLE", 0.0))
            continue
        print(f"  span {span:.1f} s")
        if not times:
            print("  !! none of the pipeline topics present")
            verdicts.append((bag, "NO TOPICS", 0.0))
            continue

        cams = sorted(t for t in times if CAM_RE.match(t))
        print(f"  {'topic':30} {'callerid':22} {'n':>7} {'src Hz':>8} "
              f"{'deliv':>7} {'gaps':>6} {'max gap':>9}")
        worst = 1.0
        for topic in cams + [t for t in times if t not in cams]:
            is_cam = bool(CAM_RE.match(topic))
            s = rate_stats(times[topic], span,
                           args.nominal_hz if is_cam else 0.0)
            if not s:
                print(f"  {topic:30} {callerid.get(topic, '?'):22} "
                      f"{len(times[topic]):>7}  (too few messages)")
                continue
            flag = ""
            if s["unreliable"]:
                flag = "  ~ bursty arrivals, delivery estimate unreliable"
            elif is_cam:
                worst = min(worst, s["delivery"])
                flag = "  <" if s["delivery"] < args.min_delivery else ""
            print(f"  {topic:30} {callerid.get(topic, '?'):22} {s['n']:>7} "
                  f"{s['nominal_hz']:>8.2f} {s['delivery'] * 100:>6.1f}% "
                  f"{s['n_gaps']:>6} {s['max_gap_ms']:>8.0f}ms{flag}")

        if not cams:
            verdict = "NO CAMERAS"
        elif worst >= args.min_delivery:
            verdict = "PASS"
        elif worst >= 0.90:
            verdict = "MARGINAL"
        else:
            verdict = "DROP"
        print(f"  --> {verdict}  (worst camera delivery {worst * 100:.1f}%)")
        verdicts.append((bag, verdict, worst))

    print(f"\n{'=' * 78}\nSUMMARY  (pass threshold {args.min_delivery * 100:.0f}% per channel)")
    for bag, verdict, worst in verdicts:
        print(f"  {verdict:11} {worst * 100:>6.1f}%  {bag.name}")
    n_pass = sum(1 for _, v, _ in verdicts if v == "PASS")
    print(f"\n  {n_pass}/{len(verdicts)} bags usable as-is.")
    if n_pass < len(verdicts):
        print("  Dropped frames are lost at record time — they cannot be recovered")
        print("  downstream. Re-record with the bridge on reliable QoS.")


if __name__ == "__main__":
    main()
