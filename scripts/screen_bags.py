#!/usr/bin/env python3
"""Screen rosbags before converting: what bag2nuscenes will do with each one.

Everything here is measured on header timestamps — the clock the converter
synchronizes on — with the converter's own rules (`coverage_window`,
`nearest_ts`, six standard cameras gate a keyframe, CAM_TRAFFIC is best-effort),
so the numbers are the numbers the conversion will produce.

Per bag:
  - per-stream delivery, gaps and start/end offsets (cameras, LiDAR, odom).
    Camera delivery is the record-time question: the cameras reach the ROS1 bag
    through /ros_bridge, the bag does not record the QoS, and a best_effort
    bridge silently drops frames under load — the delivery ratio is its footprint.
  - the coverage window: the interval in which every required stream is live,
    and how much the converter will cut at head and tail
  - odom gaps: ego poses inside one are interpolated straight across it, which
    nothing downstream can detect
  - INS solution status from INSPVA: the driver keeps publishing odom while the
    INS is still aligning or has lost GNSS, and the converter does not look
  - keyframe acceptance per sync tolerance, to choose --sync-ms

Verdict PASS / MARGINAL / DROP with every reason listed; thresholds are flags.

Usage:
    python scripts/screen_bags.py /path/*.bag
    python scripts/screen_bags.py /mnt/ssd/bags/ --nominal-hz 30 --sync-ms 25
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from common import (  # noqa: E402
    INSPVA_TOPIC,
    LIDAR_PACKETS_TOPIC,
    LIDAR_POINTS_TOPIC,
    NUSCENES_CAMS,
    ODOM_TOPIC,
    TOPIC_TO_CAM_CHANNEL,
    make_typestore,
    stamp_to_ns,
)
from nuscenes_writer import coverage_window, nearest_ts  # noqa: E402

INS_GOOD = 3  # novatel_oem7_msgs/InertialSolutionStatus.INS_SOLUTION_GOOD
_INS_NAMES = {
    0: "INS_INACTIVE", 1: "INS_ALIGNING", 2: "INS_HIGH_VARIANCE",
    3: "INS_SOLUTION_GOOD", 6: "INS_SOLUTION_FREE", 7: "INS_ALIGNMENT_COMPLETE",
    8: "DETERMINING_ORIENTATION", 9: "WAITING_INITIAL_POS", 10: "WAITING_AZIMUTH",
    11: "INITIALIZING_BIASES", 12: "MOTION_DETECT",
}
TOLERANCES_MS = (10, 15, 20, 25, 30, 50)
# A header.stamp further than this from the bag's receive clock is not wall
# time at all (zero, or seconds since boot); the converter cannot sync on it.
OFF_CLOCK_NS = 86_400 * 10**9
REQUIRED = ["LIDAR_TOP", *NUSCENES_CAMS, "ODOM"]


def _stream_name(topic: str) -> str:
    if topic in TOPIC_TO_CAM_CHANNEL:
        return TOPIC_TO_CAM_CHANNEL[topic]
    return {LIDAR_POINTS_TOPIC: "LIDAR_TOP", LIDAR_PACKETS_TOPIC: "LIDAR_TOP",
            ODOM_TOPIC: "ODOM", INSPVA_TOPIC: "INSPVA"}[topic]


@dataclass
class BagStreams:
    span_s: float
    bag_start_ns: int                      # bag receive clock at the first message
    stamps: dict[str, np.ndarray]          # stream -> sorted header stamps (ns)
    callerid: dict[str, str]
    lidar_from_packets: bool = False
    ins_ts: np.ndarray | None = None       # INSPVA header stamps
    ins_status: np.ndarray | None = None   # INSPVA status per sample
    ins_names: dict[int, str] = field(default_factory=lambda: dict(_INS_NAMES))


def collect(bag: Path, max_seconds: float, typestore) -> BagStreams:
    """Header stamps per stream, plus INSPVA status. Deserialization is cheap:
    rosbags hands back views, so JPEG and point-cloud payloads are never copied."""
    stamps: dict[str, list[int]] = defaultdict(list)
    callerid: dict[str, str] = {}
    ins_ts: list[int] = []
    ins_status: list[int] = []
    names = dict(_INS_NAMES)
    packets = False
    with AnyReader([bag], default_typestore=typestore) as reader:
        dur = (reader.end_time - reader.start_time) / 1e9
        bag_start_ns = reader.start_time
        wanted = (set(TOPIC_TO_CAM_CHANNEL)
                  | {ODOM_TOPIC, INSPVA_TOPIC, LIDAR_POINTS_TOPIC, LIDAR_PACKETS_TOPIC})
        conns = [c for c in reader.connections if c.topic in wanted]
        for c in conns:
            callerid[_stream_name(c.topic)] = getattr(c.ext, "callerid", "?")
        t_end = reader.start_time + int(max_seconds * 1e9) if max_seconds else None
        for conn, bag_ns, raw in reader.messages(connections=conns):
            if t_end and bag_ns > t_end:
                break
            topic = conn.topic
            if topic == LIDAR_PACKETS_TOPIC:
                # No per-frame stamp before decoding; the converter uses the
                # bag receive clock on this path, so do the same.
                packets = True
                stamps["LIDAR_TOP"].append(int(bag_ns))
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            if topic == INSPVA_TOPIC:
                ins_ts.append(stamp_to_ns(msg.header.stamp))
                ins_status.append(int(msg.status.status))
                if len(ins_ts) == 1:
                    names.update({int(v): k for k, v in vars(msg.status).items()
                                  if k.isupper() and isinstance(v, int)})
                continue
            stamps[_stream_name(topic)].append(stamp_to_ns(msg.header.stamp))
    span = min(dur, max_seconds) if max_seconds else dur
    out = BagStreams(
        span_s=span, bag_start_ns=int(bag_start_ns),
        stamps={k: np.array(sorted(v), dtype=np.int64) for k, v in stamps.items()},
        callerid=callerid, lidar_from_packets=packets, ins_names=names,
    )
    if ins_ts:
        order = np.argsort(ins_ts)
        out.ins_ts = np.array(ins_ts, dtype=np.int64)[order]
        out.ins_status = np.array(ins_status, dtype=np.int64)[order]
    return out


def rate_stats(ts: np.ndarray, span_s: float, nominal_hz: float = 0.0) -> dict:
    if len(ts) < 10:
        return {}
    dt = np.diff(ts) / 1e9
    modal = float(np.median(dt))          # loss adds long gaps, the mode is unmoved
    inferred_hz = 1.0 / modal
    # Bursty publishers (two messages back to back) drag the modal gap far below
    # the true cadence and would fake a huge loss. Only trust the inference when
    # it stays near the observed average rate.
    avg_hz = len(ts) / span_s
    unreliable = inferred_hz > 1.3 * avg_hz
    nominal_hz = nominal_hz or inferred_hz
    return {
        "n": len(ts),
        "nominal_hz": nominal_hz,
        "delivery": len(ts) / (span_s * nominal_hz),
        "max_gap_s": float(dt.max()),
        "n_gaps": int(np.sum(dt > 1.5 * modal)),
        "unreliable": unreliable,
    }


def long_gaps(ts: np.ndarray, min_gap_s: float) -> list[tuple[int, float]]:
    """(start_ns, gap_s) for every gap longer than min_gap_s."""
    if len(ts) < 2:
        return []
    dt = np.diff(ts)
    idx = np.where(dt > int(min_gap_s * 1e9))[0]
    return [(int(ts[i]), float(dt[i]) / 1e9) for i in idx]


def status_runs(ts: np.ndarray, status: np.ndarray, names: dict[int, str]
                ) -> list[tuple[str, int, int]]:
    """Consecutive runs where the INS status is not SOLUTION_GOOD:
    (dominant status name, start_ns, end_ns)."""
    bad = status != INS_GOOD
    if not bad.any():
        return []
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs = []
    for a, b in zip(starts, ends):
        vals, counts = np.unique(status[a:b], return_counts=True)
        dominant = int(vals[np.argmax(counts)])
        runs.append((names.get(dominant, str(dominant)), int(ts[a]), int(ts[b - 1])))
    return runs


def keyframe_acceptance(bs: BagStreams, window: dict, stride: int,
                        usable: list[str]) -> list[dict]:
    """Keyframe survival per tolerance, using the converter's rule: anchors are
    every `stride`th lidar frame inside the window, the six standard cameras
    must all be within tolerance, CAM_TRAFFIC is attached when it happens to be."""
    lidar = bs.stamps["LIDAR_TOP"]
    anchors = lidar[::stride]
    anchors = anchors[(anchors >= window["start_ns"]) & (anchors <= window["end_ns"])]
    required = [c for c in NUSCENES_CAMS if c in usable]
    if not len(anchors) or not required:
        return []
    diffs = {c: nearest_ts(anchors, bs.stamps[c])[1] for c in required}
    worst = np.max(np.stack([diffs[c] for c in required]), axis=0)
    traffic = nearest_ts(anchors, bs.stamps["CAM_TRAFFIC"])[1] if "CAM_TRAFFIC" in usable else None
    rows = []
    for tol_ms in TOLERANCES_MS:
        tol = tol_ms * 1_000_000
        kept = worst <= tol
        rows.append({
            "tol_ms": tol_ms, "n": len(anchors), "kept": int(kept.sum()),
            "traffic": (float(np.mean(traffic[kept] <= tol)) if traffic is not None and kept.any() else None),
        })
    return rows


def _fmt_t(ns: int, t0: int) -> str:
    return f"{(ns - t0) / 1e9:8.1f}s"


def screen(bag: Path, args, typestore) -> tuple[str, list[str], float]:
    bs = collect(bag, args.max_seconds, typestore)
    print(f"  span {bs.span_s:.1f} s" + ("   (lidar from packets: frame stamps are bag receive times)"
                                          if bs.lidar_from_packets else ""))
    if not bs.stamps:
        print("  !! none of the pipeline topics present")
        return "NO TOPICS", ["no pipeline topics"], 0.0

    reasons: list[str] = []
    level = 0  # 0 PASS, 1 MARGINAL, 2 DROP

    def flag(lvl: int, why: str) -> None:
        nonlocal level
        level = max(level, lvl)
        reasons.append(why)

    # ---------------------------------------------------------------- streams
    off_clock = [s for s in bs.stamps
                 if abs(int(bs.stamps[s][0]) - bs.bag_start_ns) > OFF_CLOCK_NS]
    if off_clock:
        flag(2, f"header stamps are not wall time on {', '.join(off_clock)} "
                f"(first stamp ≈ {bs.stamps[off_clock[0]][0] / 1e9:.0f} s vs bag clock "
                f"{bs.bag_start_ns / 1e9:.0f} s) — cannot be synchronized")
    present = [s for s in REQUIRED if s in bs.stamps and s not in off_clock]
    missing = [s for s in REQUIRED if s not in bs.stamps]
    for s in missing:
        flag(2, f"missing {s}" + (" (bag2nuscenes refuses without odom)" if s == "ODOM" else ""))
    extra = [s for s in bs.stamps if s not in REQUIRED and s not in off_clock]
    window = (coverage_window({s: bs.stamps[s] for s in present}, int(args.sync_ms * 1e6))
              if present else coverage_window({"(none)": np.array([0, 0])}, 0))
    # Offsets for the table are shown for every stream, relative to the same
    # origin as the window; only the required streams define the window itself.
    offsets = coverage_window({s: bs.stamps[s] for s in present + extra}, 0)["streams"]
    if window["earliest_ns"] == window["latest_ns"]:   # degenerate: nothing usable
        offsets = {}

    print(f"  {'stream':16} {'callerid':22} {'n':>7} {'src Hz':>7} {'deliv':>7} "
          f"{'gaps':>5} {'max gap':>9} {'starts':>9} {'ends':>9}")
    worst_cam = 1.0
    for s in present + extra + off_clock:
        ts = bs.stamps[s]
        w = offsets.get(s)
        offs = (f"{w['start_offset_s']:+8.3f}s {w['end_offset_s']:+8.3f}s" if w
                else f"{'off-clock':>9} {'':>9}")
        if s == "LIDAR_TOP" and bs.lidar_from_packets:
            print(f"  {s:16} {bs.callerid.get(s, '?'):22} {len(ts):>7} {'(pkts)':>7} {'':>7} {'':>5} {'':>9} {offs}")
            continue
        is_cam = s in TOPIC_TO_CAM_CHANNEL.values()
        st = rate_stats(ts, bs.span_s, args.nominal_hz if is_cam else 0.0)
        if not st:
            print(f"  {s:16} {bs.callerid.get(s, '?'):22} {len(ts):>7}  (too few messages)")
            continue
        note = ""
        if st["unreliable"]:
            note = "  ~ bursty, delivery estimate unreliable"
        elif s in NUSCENES_CAMS:
            worst_cam = min(worst_cam, st["delivery"])
            note = "  <" if st["delivery"] < args.min_delivery else ""
        print(f"  {s:16} {bs.callerid.get(s, '?'):22} {st['n']:>7} {st['nominal_hz']:>7.2f} "
              f"{st['delivery'] * 100:>6.1f}% {st['n_gaps']:>5} {st['max_gap_s'] * 1e3:>7.0f}ms "
              f"{offs}{note}")
        if s == "LIDAR_TOP" and st["max_gap_s"] > args.max_lidar_gap:
            flag(1, f"lidar gap {st['max_gap_s']:.2f}s (no keyframes there)")
    if worst_cam < args.min_delivery:
        flag(1 if worst_cam >= 0.90 else 2,
             f"worst standard-camera delivery {worst_cam * 100:.1f}% < {args.min_delivery * 100:.0f}% "
             "(frames dropped at record time)")

    # ------------------------------------------------------------- coverage
    t0 = window["earliest_ns"]
    if missing:
        print(f"  coverage window (without {', '.join(missing)}):", end=" ")
    else:
        print("  coverage window:", end=" ")
    if window["end_ns"] <= window["start_ns"]:
        print("EMPTY — required streams do not overlap")
        flag(2, "required streams do not overlap in time")
    else:
        print(f"{_fmt_t(window['start_ns'], t0).strip()} .. {_fmt_t(window['end_ns'], t0).strip()}  "
              f"-> head cut {window['head_cut_s']:.2f}s ({window['last_start']} starts last), "
              f"tail cut {window['tail_cut_s']:.2f}s ({window['first_end']} ends first)")
        cut = window["head_cut_s"] + window["tail_cut_s"]
        if cut > args.max_cut:
            flag(1, f"coverage window cuts {cut:.1f}s of the bag")

    # ------------------------------------------------------------ odom gaps
    if "ODOM" in bs.stamps:
        gaps = long_gaps(bs.stamps["ODOM"], args.max_odom_gap)
        if gaps:
            print(f"  odom gaps > {args.max_odom_gap:.2f}s: "
                  + ", ".join(f"{_fmt_t(s_, t0).strip()} ({g:.2f}s)" for s_, g in gaps[:8])
                  + (" ..." if len(gaps) > 8 else ""))
            worst = max(g for _, g in gaps)
            flag(2, f"odom gap {worst:.2f}s (ego pose interpolated across it)")
        else:
            print(f"  odom gaps > {args.max_odom_gap:.2f}s: none")

    # ------------------------------------------------------------ INS status
    if bs.ins_ts is None:
        print("  INS status (INSPVA): absent — INS solution quality unknown")
    else:
        bad_frac = float(np.mean(bs.ins_status != INS_GOOD))
        runs = status_runs(bs.ins_ts, bs.ins_status, bs.ins_names)
        print(f"  INS status (INSPVA): SOLUTION_GOOD {100 * (1 - bad_frac):.1f}% of {len(bs.ins_ts)} samples",
              end="")
        if runs:
            print("; other: " + ", ".join(
                f"{name} {_fmt_t(a, t0).strip()}..{_fmt_t(b, t0).strip()} ({(b - a) / 1e9:.1f}s)"
                for name, a, b in runs[:6]) + (" ..." if len(runs) > 6 else ""))
        else:
            print()
        in_window = [(n, a, b) for n, a, b in runs
                     if b >= window["start_ns"] and a <= window["end_ns"]]
        longest = max(((b - a) / 1e9 for _, a, b in in_window), default=0.0)
        if longest >= args.max_ins_bad_run:
            flag(2, f"INS not SOLUTION_GOOD for {longest:.1f}s inside the window")
        elif bad_frac > args.max_ins_bad_frac:
            flag(1, f"INS not SOLUTION_GOOD for {100 * bad_frac:.1f}% of samples")

    # ------------------------------------------------------ sync acceptance
    if "LIDAR_TOP" in present and not bs.lidar_from_packets and window["end_ns"] > window["start_ns"]:
        rows = keyframe_acceptance(bs, window, args.keyframe_stride, present + extra)
        if rows:
            print(f"  keyframe acceptance (every {args.keyframe_stride}th lidar frame in window, "
                  f"{sum(c in present for c in NUSCENES_CAMS)} standard cams gate; "
                  f"{rows[0]['n']} candidates):")
            print(f"    {'tol':>6} {'kept':>7} {'CAM_TRAFFIC attached':>22}")
            for r in rows:
                mark = "  <- --sync-ms" if r["tol_ms"] == args.sync_ms else ""
                tr = f"{100 * r['traffic']:.1f}%" if r["traffic"] is not None else "n/a"
                print(f"    {r['tol_ms']:>4}ms {100 * r['kept'] / r['n']:>6.1f}% {tr:>22}{mark}")
            at = next((r for r in rows if r["tol_ms"] == args.sync_ms), None)
            if at and at["kept"] < 0.9 * at["n"]:
                flag(1, f"only {100 * at['kept'] / at['n']:.0f}% of keyframes survive --sync-ms {args.sync_ms:g}")

    verdict = ("PASS", "MARGINAL", "DROP")[level]
    print(f"  --> {verdict}" + (": " + "; ".join(reasons) if reasons else ""))
    return verdict, reasons, worst_cam


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", type=Path, nargs="+", help="Bag files, or directories to scan for *.bag")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Only inspect the first N seconds (0 = whole bag).")
    p.add_argument("--nominal-hz", type=float, default=0.0,
                   help="Known camera source rate (e.g. 30). Default: infer from the modal inter-arrival gap.")
    p.add_argument("--sync-ms", type=float, default=25.0, help="Sync tolerance the conversion will use (default 25).")
    p.add_argument("--keyframe-stride", type=int, default=5, help="Keyframe stride the conversion will use (default 5).")
    p.add_argument("--min-delivery", type=float, default=0.99, help="Per-camera delivery ratio required to pass (default 0.99).")
    p.add_argument("--max-odom-gap", type=float, default=0.5, help="Odom gap (s) that fails a bag (default 0.5).")
    p.add_argument("--max-lidar-gap", type=float, default=0.3, help="Lidar gap (s) that marks a bag marginal (default 0.3).")
    p.add_argument("--max-ins-bad-run", type=float, default=2.0,
                   help="INS not SOLUTION_GOOD for this long inside the window fails a bag (s, default 2).")
    p.add_argument("--max-ins-bad-frac", type=float, default=0.01,
                   help="Fraction of INSPVA samples not SOLUTION_GOOD that marks a bag marginal (default 0.01).")
    p.add_argument("--max-cut", type=float, default=5.0, help="Head+tail cut (s) that marks a bag marginal (default 5).")
    p.add_argument("--packet-msg-dir", type=Path,
                   default=_ROOT / "packet_decoder" / "src" / "rslidar_msg" / "msg")
    args = p.parse_args()

    bags: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            bags.extend(sorted(path.rglob("*.bag")))
        elif path.suffix == ".bag":
            bags.append(path)
    if not bags:
        raise SystemExit("no .bag files found")
    typestore = make_typestore((args.packet_msg_dir, "rslidar_msg"))

    verdicts: list[tuple[Path, str, list[str], float]] = []
    for bag in bags:
        print(f"\n{'=' * 78}\n{bag}")
        try:
            verdict, reasons, worst = screen(bag, args, typestore)
        except Exception as exc:  # unreadable index, truncated file, ...
            print(f"  !! unreadable: {exc}")
            verdict, reasons, worst = "UNREADABLE", [str(exc)], 0.0
        verdicts.append((bag, verdict, reasons, worst))

    print(f"\n{'=' * 78}\nSUMMARY")
    for bag, verdict, reasons, worst in verdicts:
        print(f"  {verdict:10} cam {worst * 100:>5.1f}%  {bag.name}"
              + (f"  — {reasons[0]}" + (f" (+{len(reasons) - 1})" if len(reasons) > 1 else "") if reasons else ""))
    n_pass = sum(1 for _, v, _, _ in verdicts if v == "PASS")
    print(f"\n  {n_pass}/{len(verdicts)} bags usable as-is.")


if __name__ == "__main__":
    main()
