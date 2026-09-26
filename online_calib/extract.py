#!/usr/bin/env python3
"""Pull calibration clips out of a bag: raw camera JPEGs, LiDAR sweeps with
per-point times, and the INS streams.

    python online_calib/extract.py BAG --window 20 50 --window 70 110 --out WORK

Writes, per window, WORK/<bag stem>_<t0>_<t1>/
    cam/<CHANNEL>/<header ns>.jpg        every frame, bytes as recorded
    lidar/<header ns>.npy                structured: x y z intensity (f4), ring (u2),
                                         t (f4, s after the header stamp)
and once per bag WORK/<bag stem>_ins.npz with odom (t, pos, quat wxyz, vel),
CORRIMU (t, count, rates, accels) and INSPVA (t, status, and the full solution:
lat lon height, velocity north east up, roll pitch azimuth).

Only reads the bag. Times are header stamps, as in the converter, except that
INSPVA and CORRIMU also get their measurement time from the receiver's GPS time
(pva_t, imu_tg), on the header stamps' clock (UTC). The header stamps of the
NovAtel topics are arrival times, 1-10 ms late and jittery; the 2026-09-23 odom
also holds its position for a second at a time on some drives (BESTPOS as the
position source), which the 100 Hz INSPVA does not.

    python online_calib/extract.py BAG --ins-only --out WORK    # just the INS file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    CORRIMU_TOPIC,
    INSPVA_TOPIC,
    LIDAR_POINTS_TOPIC,
    ODOM_TOPIC,
    TOPIC_TO_CAM_CHANNEL,
    make_typestore,
    stamp_to_ns,
)
from converter import _pointcloud2_dtype  # noqa: E402

GPS_EPOCH_UNIX_S = 315964800          # 1980-01-06 in Unix time
GPS_MINUS_UTC_S = 18                   # leap seconds since 2017-01-01

SWEEP_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                        ("ring", "<u2"), ("t", "<f4")])


def sweep_from_msg(msg, header_ns: int) -> np.ndarray:
    dt = _pointcloud2_dtype(msg)
    arr = np.frombuffer(msg.data, dtype=dt, count=len(msg.data) // dt.itemsize)
    keep = np.isfinite(arr["x"]) & np.isfinite(arr["y"]) & np.isfinite(arr["z"])
    a = arr[keep]
    out = np.empty(len(a), dtype=SWEEP_DTYPE)
    for k in ("x", "y", "z", "intensity", "ring"):
        out[k] = a[k]
    out["t"] = (a["timestamp"] - header_ns / 1e9).astype(np.float32)
    return out


def gps_ns(nov_header) -> int:
    """Receiver GPS time of a NovAtel message, as UTC ns (the header stamps' clock)."""
    ms = int(nov_header.gps_week_number) * 604_800_000 + int(nov_header.gps_week_milliseconds)
    return (ms + (GPS_EPOCH_UNIX_S - GPS_MINUS_UTC_S) * 1000) * 1_000_000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path)
    p.add_argument("--window", type=float, nargs=2, action="append", default=[],
                   metavar=("T0", "T1"), help="seconds from bag start; repeatable")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cameras", nargs="*", default=None,
                   help="channels to keep (default: all seven)")
    p.add_argument("--cam-stride", type=int, default=1,
                   help="keep every Nth frame of each camera (default 1 = all 30 fps)")
    p.add_argument("--ins-only", action="store_true", help="write only the INS file")
    args = p.parse_args()
    if not args.window and not args.ins_only:
        p.error("give --window at least once, or --ins-only")

    stem = args.bag.stem
    cams = set(args.cameras) if args.cameras else set(TOPIC_TO_CAM_CHANNEL.values())
    typestore = make_typestore()
    odom, imu, ins, pva = [], [], [], []
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        t_start = reader.start_time
        wins = [(int(t_start + a * 1e9), int(t_start + b * 1e9), f"{stem}_{int(a):03d}_{int(b):03d}")
                for a, b in args.window]
        for _, _, name in wins:
            for ch in cams:
                (args.out / name / "cam" / ch).mkdir(parents=True, exist_ok=True)
            (args.out / name / "lidar").mkdir(parents=True, exist_ok=True)

        def window_of(ns: int):
            for a, b, name in wins:
                if a <= ns <= b:
                    return name
            return None

        # INS streams: whole bag (small)
        conns = [c for c in reader.connections if c.topic in (ODOM_TOPIC, CORRIMU_TOPIC, INSPVA_TOPIC)]
        for c, _, raw in tqdm(reader.messages(connections=conns),
                              total=sum(c.msgcount for c in conns), desc="ins", unit="msg"):
            m = reader.deserialize(raw, c.msgtype)
            t = stamp_to_ns(m.header.stamp)
            if c.topic == ODOM_TOPIC:
                pp, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
                odom.append((t, pp.x, pp.y, pp.z, q.w, q.x, q.y, q.z, v.x, v.y, v.z))
            elif c.topic == CORRIMU_TOPIC:
                imu.append((t, gps_ns(m.nov_header), m.imu_data_count, m.pitch_rate, m.roll_rate,
                            m.yaw_rate, m.lateral_acc, m.longitudinal_acc, m.vertical_acc))
            else:
                ins.append((t, int(m.status.status)))
                pva.append((gps_ns(m.nov_header), t, m.latitude, m.longitude, m.height, m.north_velocity,
                            m.east_velocity, m.up_velocity, m.roll, m.pitch, m.azimuth, int(m.status.status)))
        odom_a = np.array(sorted(odom), dtype=np.float64)
        imu.sort()
        pva.sort()
        late = np.array([r[1] - r[0] for r in pva]) / 1e6
        print(f"INSPVA arrival - GPS time: median {np.median(late):.2f} ms, min {late.min():.2f} ms")
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez(args.out / f"{stem}_ins.npz",
                 odom_t=odom_a[:, 0].astype(np.int64), odom=odom_a[:, 1:],
                 imu_t=np.array([r[0] for r in imu], dtype=np.int64),
                 imu_tg=np.array([r[1] for r in imu], dtype=np.int64),
                 imu=np.array([r[2:] for r in imu], dtype=np.float64),
                 ins_t=np.array([r[0] for r in sorted(ins)], dtype=np.int64),
                 ins_status=np.array([r[1] for r in sorted(ins)], dtype=np.int64),
                 pva_t=np.array([r[0] for r in pva], dtype=np.int64),
                 pva_th=np.array([r[1] for r in pva], dtype=np.int64),
                 pva=np.array([r[2:] for r in pva], dtype=np.float64))
        if args.ins_only:
            return

        # Sensor payloads, only inside the windows
        topics = {t for t, ch in TOPIC_TO_CAM_CHANNEL.items() if ch in cams} | {LIDAR_POINTS_TOPIC}
        conns = [c for c in reader.connections if c.topic in topics]
        n = 0
        seen: dict[str, int] = {}
        for a, b, name in wins:
            for c, bag_ns, raw in tqdm(reader.messages(connections=conns, start=a - int(2e8), stop=b + int(2e8)),
                                       desc=name, unit="msg"):
                m = reader.deserialize(raw, c.msgtype)
                t = stamp_to_ns(m.header.stamp)
                if window_of(t) != name:
                    continue
                if c.topic == LIDAR_POINTS_TOPIC:
                    np.save(args.out / name / "lidar" / f"{t}.npy", sweep_from_msg(m, t))
                else:
                    ch = TOPIC_TO_CAM_CHANNEL[c.topic]
                    seen[ch] = seen.get(ch, -1) + 1
                    if seen[ch] % args.cam_stride:
                        continue
                    (args.out / name / "cam" / ch / f"{t}.jpg").write_bytes(bytes(m.data))
                n += 1
    print(f"wrote {n} payloads under {args.out}")


if __name__ == "__main__":
    main()
