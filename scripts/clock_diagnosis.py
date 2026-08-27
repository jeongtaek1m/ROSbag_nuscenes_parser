"""Diagnose sensor clocks in a rosbag before deciding the sync time base.

Answers three questions without touching the conversion pipeline:

  1. Does the Robosense MSOP packet carry a usable embedded timestamp, and at
     what byte offset? (auto-detected and self-validated: consecutive MSOP
     packets must be ~166.7 us apart = 3 blocks x 55.56 us)
  2. How does that lidar clock relate to the bag receive clock? Reports the
     linear fit (offset + skew in ppm) and, more importantly, the *residual
     jitter* — that residual is exactly what switching to MSOP timestamps
     would remove from per-point timing.
  3. Do the camera / odom drivers emit a real capture timestamp, or just
     ros::Time::now() at the callback? If header.stamp == bag receive time to
     within microseconds, it is the latter and the exposure->stamp latency is
     an unknown constant.

Also reports the frame-split azimuth (is_frame_begin packets), which tells you
whether the 0-degree seam sits inside CAM_FRONT's field of view.

Usage:
    python scripts/clock_diagnosis.py /path/to.bag
    python scripts/clock_diagnosis.py /path/to.bag --max-seconds 120 --out clock_diag/
"""
from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    INSPVA_TOPIC,
    LIDAR_PACKETS_TOPIC,
    make_typestore,
    stamp_to_ns,
)

MSOP_ID = b"\x55\xAA\x05\x5A"
HEADER_SIZE = 80
BLOCK_DURATION_US = 55.56
BLOCKS_PER_PKT = 3
PKT_PERIOD_US = BLOCKS_PER_PKT * BLOCK_DURATION_US  # 166.68 us

GPS_EPOCH_UNIX = 315964800  # 1980-01-06T00:00:00Z
LEAP_SECONDS = 18           # GPS-UTC as of 2026; bump if a new leap second lands


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))


# ---------------------------------------------------------------- MSOP probe
def parse_utc_us(buf: bytes, off: int) -> int:
    """RSTimestampUTC: 6-byte big-endian seconds + 4-byte big-endian micros."""
    sec = int.from_bytes(buf[off:off + 6], "big")
    us = int.from_bytes(buf[off + 6:off + 10], "big")
    return sec * 1_000_000 + us


def parse_ymd_us(buf: bytes, off: int) -> int | None:
    """RSTimestampYMD: YY MM DD hh mm ss + ms(2) + us(2)."""
    import calendar
    y, mo, d, h, mi, s = buf[off:off + 6]
    ms = struct.unpack_from(">H", buf, off + 6)[0]
    us = struct.unpack_from(">H", buf, off + 8)[0]
    if not (1 <= mo <= 12 and 1 <= d <= 31 and h < 24 and mi < 60 and s < 62):
        return None
    if ms > 999 or us > 999:
        return None
    try:
        sec = calendar.timegm((y + 2000, mo, d, h, mi, s, 0, 0, -1))
    except Exception:
        return None
    return sec * 1_000_000 + ms * 1000 + us


def probe_msop_offset(packets: list[bytes]) -> tuple[str, int, float]:
    """Try every plausible byte offset; the correct one yields consecutive
    deltas near PKT_PERIOD_US. Returns (kind, offset, score)."""
    best = ("none", -1, 0.0)
    for kind, fn in [("utc", parse_utc_us), ("ymd", parse_ymd_us)]:
        for off in range(0, HEADER_SIZE - 10 + 1):
            vals = []
            ok = True
            for p in packets:
                v = fn(p, off)
                if v is None:
                    ok = False
                    break
                vals.append(v)
            if not ok or len(vals) < 3:
                continue
            d = np.diff(np.asarray(vals, dtype=np.int64))
            if np.any(d <= 0):
                continue
            score = float(np.mean(np.abs(d - PKT_PERIOD_US) < 60))
            if score > best[2]:
                best = (kind, off, score)
    return best


# ------------------------------------------------------------------- report
def fit_clock(t_ref_s: np.ndarray, delta_s: np.ndarray) -> dict:
    """Linear-fit delta against reference time. Returns offset/skew/residual and
    whether the residual contains a step (a step is not removable by a fit)."""
    t0 = t_ref_s - t_ref_s[0]
    slope, intercept = np.polyfit(t0, delta_s, 1)
    resid = delta_s - (slope * t0 + intercept)
    n_bin = min(400, max(10, len(resid) // 50))
    med = np.array([np.median(b) for b in np.array_split(resid, n_bin)])
    steps = np.abs(np.diff(med))
    typical = float(np.median(steps)) if len(steps) else 0.0
    max_step = float(steps.max()) if len(steps) else 0.0
    return {
        "offset_ms": intercept * 1e3,
        "skew_ppm": slope * 1e6,
        "span_s": float(t0[-1]),
        "resid_std_us": float(np.std(resid)) * 1e6,
        "resid_p99_us": float(np.percentile(np.abs(resid), 99)) * 1e6,
        "max_step_ms": max_step * 1e3,
        "typical_step_ms": typical * 1e3,
        "has_step": bool(max_step > 1e-3 and max_step > 4 * typical),
        "n": len(delta_s),
    }


def fit_and_report(name: str, t_ref_s: np.ndarray, delta_s: np.ndarray) -> dict:
    """Linear-fit delta against reference time; report offset, skew, residual."""
    f = fit_clock(t_ref_s, delta_s)
    t0 = t_ref_s - t_ref_s[0]
    slope, intercept = np.polyfit(t0, delta_s, 1)
    resid = delta_s - (slope * t0 + intercept)
    print(f"  {name}")
    print(f"    offset at bag start : {intercept * 1e3:+10.3f} ms")
    print(f"    skew                : {slope * 1e6:+10.2f} ppm "
          f"({slope * (t0[-1]) * 1e3:+.1f} ms drift over {t0[-1]:.0f} s)")
    print(f"    residual after fit  : std {np.std(resid) * 1e6:8.1f} us   "
          f"p99 {np.percentile(np.abs(resid), 99) * 1e6:8.1f} us   "
          f"max {np.max(np.abs(resid)) * 1e6:8.1f} us")
    # step detection on a coarse grid so per-packet noise does not trigger it
    n_bin = min(400, max(10, len(resid) // 50))
    binned = np.array_split(resid, n_bin)
    med = np.array([np.median(b) for b in binned])
    steps = np.abs(np.diff(med))
    # A real clock step stands out against the ordinary bin-to-bin wobble;
    # comparing to the median step keeps heavy transport jitter from tripping it.
    typical = float(np.median(steps)) if len(steps) else 0.0
    if len(steps) and steps.max() > 1e-3 and steps.max() > 4 * typical:
        print(f"    !! discontinuity    : max step {steps.max() * 1e3:.2f} ms "
              f"(typical {typical * 1e3:.2f} ms) -> clock re-disciplined or wrapping")
    else:
        print(f"    continuity          : ok (max step "
              f"{steps.max() * 1e6 if len(steps) else 0:.1f} us)")
    return f


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path)
    p.add_argument("--packet-msg-dir", type=Path,
                   default=here / "packet_decoder" / "src" / "rslidar_msg" / "msg")
    p.add_argument("--msg-dir", type=Path, default=here / "msg")
    p.add_argument("--max-seconds", type=float, default=180.0,
                   help="Only read the first N seconds of the bag (default 180).")
    p.add_argument("--out", type=Path, default=None,
                   help="If set, write diagnostic plots here.")
    args = p.parse_args()

    typestore = make_typestore((args.packet_msg_dir, "rslidar_msg"),
                               (args.msg_dir, "data_processing"))

    msop_raw: list[bytes] = []        # first N packet payloads, for offset probe
    msop_bag_s: list[float] = []      # bag receive time per MSOP packet
    msop_all: list[bytes] = []        # payloads kept for embedded-ts extraction
    frame_begin_az: list[int] = []    # azimuth of first block on frame-begin pkts
    frame_begin_bag_s: list[float] = []
    header_vs_bag: dict[str, list[tuple[float, float]]] = defaultdict(list)
    gps_vs_bag: list[tuple[float, float]] = []
    counts: dict[str, int] = defaultdict(int)

    with AnyReader([args.bag], default_typestore=typestore) as reader:
        t0 = reader.start_time
        t_end = t0 + int(args.max_seconds * 1e9)

        section("Topics")
        for c in sorted(reader.connections, key=lambda c: c.topic):
            print(f"  {c.topic:34} {c.msgtype:44} {c.msgcount:>9}")

        for conn, bag_ns, raw in reader.messages():
            if bag_ns > t_end:
                break
            topic = conn.topic
            counts[topic] += 1
            bag_s = bag_ns / 1e9

            if topic == LIDAR_PACKETS_TOPIC:
                msg = reader.deserialize(raw, conn.msgtype)
                if msg.is_difop:
                    continue
                data = bytes(msg.data)
                if data[:4] != MSOP_ID:
                    continue
                if len(msop_raw) < 40:
                    msop_raw.append(data)
                msop_all.append(data[:HEADER_SIZE])
                msop_bag_s.append(bag_s)
                if msg.is_frame_begin:
                    frame_begin_az.append(
                        struct.unpack_from(">H", data, HEADER_SIZE + 2)[0])
                    frame_begin_bag_s.append(bag_s)
                # lidar packet header.stamp too
                header_vs_bag[topic + " (header.stamp)"].append(
                    (bag_s, stamp_to_ns(msg.header.stamp) / 1e9 - bag_s))
            elif ("compressed" in topic or "odom" in topic or "fusion" in topic
                  or "rslidar_points" in topic or "inspva" in topic):
                msg = reader.deserialize(raw, conn.msgtype)
                hdr = getattr(msg, "header", None)
                if hdr is None:
                    continue
                header_vs_bag[topic].append(
                    (bag_s, stamp_to_ns(hdr.stamp) / 1e9 - bag_s))
                nh = getattr(msg, "nov_header", None)
                if topic == INSPVA_TOPIC and nh is not None:
                    gps_s = (GPS_EPOCH_UNIX + int(nh.gps_week_number) * 604800
                             + int(nh.gps_week_milliseconds) / 1e3 - LEAP_SECONDS)
                    gps_vs_bag.append((bag_s, gps_s - bag_s))

    # ------------------------------------------------ 1. MSOP embedded stamp
    section("1. MSOP embedded timestamp")
    if not msop_raw:
        print(f"  no MSOP packets on {LIDAR_PACKETS_TOPIC} in the first "
              f"{args.max_seconds}s — skipping lidar clock analysis")
        kind = "none"
    else:
        kind, off, score = probe_msop_offset(msop_raw)
        if kind == "none" or score < 0.8:
            print(f"  NO valid embedded timestamp found "
                  f"(best score {score:.2f} at {kind}/off={off})")
            print(f"  -> the lidar is not writing a usable stamp; bag receive "
                  f"time is the only option")
        else:
            print(f"  format={kind}  byte_offset={off}  "
                  f"self-check score={score:.3f}")
            print(f"  (score = fraction of consecutive packets whose delta is "
                  f"within 60us of {PKT_PERIOD_US:.1f}us)")
            fn = parse_utc_us if kind == "utc" else parse_ymd_us
            msop_us = np.array([fn(b, off) for b in msop_all], dtype=np.float64)
            bag_s = np.asarray(msop_bag_s)
            msop_s = msop_us / 1e6
            import datetime as _dt
            print(f"  first embedded stamp: "
                  f"{_dt.datetime.utcfromtimestamp(msop_s[0]).isoformat()}Z")
            print(f"  first bag stamp     : "
                  f"{_dt.datetime.utcfromtimestamp(bag_s[0]).isoformat()}Z")
            print(f"  absolute difference : {(msop_s[0] - bag_s[0]):+.3f} s")

            section("2. Lidar clock vs bag receive clock")
            fit_and_report("msop_embedded - bag_receive", bag_s, msop_s - bag_s)
            print()
            print(f"  INTERPRETATION")
            print(f"    The 'residual after fit' is the receive jitter that is "
                  f"currently baked")
            print(f"    into every per-point timestamp (decode_lidar.py uses "
                  f"bag receive time).")
            print(f"    Switching intra-frame timing to the embedded stamp "
                  f"removes exactly that.")

    # ------------------------------------------------ 3. driver stamp check
    section("3. header.stamp vs bag receive time (is it a real capture time?)")
    print(f"  {'topic':38} {'n':>7} {'median':>10} {'p1':>9} {'p99':>9}  verdict")
    for topic in sorted(header_vs_bag):
        arr = np.asarray(header_vs_bag[topic])
        if len(arr) < 10:
            continue
        d = arr[:, 1] * 1e3  # ms
        med, p1, p99 = np.median(d), np.percentile(d, 1), np.percentile(d, 99)
        spread = p99 - p1
        if abs(med) < 0.05 and spread < 0.2:
            verdict = "ros::Time::now() at callback — NOT a capture time"
        elif med < -0.5:
            verdict = "stamped before receive — plausible capture/DMA time"
        else:
            verdict = "check manually"
        print(f"  {topic:38} {len(arr):>7} {med:>9.3f}ms {p1:>8.3f} "
              f"{p99:>8.3f}  {verdict}")

    # A driver stamping from the sensor's own oscillator shows a steady skew
    # against bag receive time. A driver stamping from the host clock does not.
    section("3b. Skew of each driver stamp against the bag (host) clock")
    print("  A non-zero, steady ppm means that stamp comes from a separate,")
    print("  undisciplined oscillator — it cannot be compared to other sensors")
    print("  without first fitting offset+skew onto a common axis.")
    for topic in sorted(header_vs_bag):
        arr = np.asarray(header_vs_bag[topic])
        if len(arr) < 100 or (arr[-1, 0] - arr[0, 0]) < 20:
            continue
        fit_and_report(topic, arr[:, 0], arr[:, 1])

    # ------------------------------------------------ 3c. host clock vs GPS
    section("3c. Host (bag) clock vs true GPS time, from INSPVA")
    if gps_vs_bag:
        arr = np.asarray(gps_vs_bag)
        print(f"  n={len(arr)}  (GPS epoch 1980-01-06, {LEAP_SECONDS} leap seconds)")
        fit_and_report("gps_time - bag_receive", arr[:, 0], arr[:, 1])
        print()
        print("  This is the only absolute reference in the bag. If the offset is")
        print("  small and the skew is ~0, the recording host clock is disciplined")
        print("  and is a legitimate common time base for every other sensor.")
    else:
        print("  no /novatel/oem7/inspva — cannot anchor to GPS")

    # ------------------------------------------------ 4. frame split azimuth
    section("4. Frame split azimuth (0-degree seam location)")
    if frame_begin_az:
        az = np.asarray(frame_begin_az) / 100.0  # centi-degrees -> degrees
        print(f"  frame_begin packets: {len(az)}")
        print(f"  azimuth at split   : median {np.median(az):.2f} deg  "
              f"(p1 {np.percentile(az, 1):.2f}, p99 {np.percentile(az, 99):.2f})")
        fb = np.asarray(frame_begin_bag_s)
        if len(fb) > 2:
            print(f"  frame period       : median "
                  f"{np.median(np.diff(fb)) * 1e3:.2f} ms")
        m = float(np.median(az))
        if m < 20 or m > 340:
            print(f"  !! the seam sits at vehicle FRONT — CAM_FRONT's FOV "
                  f"contains points from")
            print(f"     both ends of the sweep (up to one full frame period "
                  f"apart). Rotate the")
            print(f"     split angle to the rear, or motion-compensate using "
                  f"per-point timestamps.")
        else:
            print(f"  seam is away from the vehicle front — ok")
    else:
        print("  no is_frame_begin packets seen")

    # ------------------------------------------------ optional plots
    if args.out and kind != "none" and msop_raw:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        args.out.mkdir(parents=True, exist_ok=True)
        fn = parse_utc_us if kind == "utc" else parse_ymd_us
        msop_s = np.array([fn(b, off) for b in msop_all], dtype=np.float64) / 1e6
        bag_s = np.asarray(msop_bag_s)
        t0v = bag_s - bag_s[0]
        delta = msop_s - bag_s
        slope, intercept = np.polyfit(t0v, delta, 1)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(t0v, (delta - delta[0]) * 1e3, lw=0.5)
        ax[0].set_ylabel("msop - bag  (ms, zeroed)")
        ax[0].set_title(f"lidar clock vs bag clock — skew {slope*1e6:+.1f} ppm")
        ax[0].grid(alpha=0.3)
        ax[1].plot(t0v, (delta - (slope * t0v + intercept)) * 1e6, lw=0.4)
        ax[1].set_ylabel("residual (us)")
        ax[1].set_xlabel("time in bag (s)")
        ax[1].set_title("residual after linear fit = receive jitter removed by "
                        "using embedded stamps")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "lidar_clock.png", dpi=110)
        print(f"\n  plot -> {args.out / 'lidar_clock.png'}")


if __name__ == "__main__":
    main()
