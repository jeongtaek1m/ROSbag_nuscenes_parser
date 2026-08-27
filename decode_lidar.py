#!/usr/bin/env python3
"""Stage 1 (lidar only): rosbag -> intermediate/<bag>/lidar/*.bin.zst.

Decodes Robosense MSOP/DIFOP packets to (x, y, z, intensity, ring) per point.
Isolated from bag2raw.py because packet decoding is the slow part of stage 1
(~40 min for 7M packets) — splitting lets re-runs of bag2raw.py (cameras/odom)
skip this work.

Output:
    <out>/<bag_basename>/
        lidar/<frame_ts_ns>.bin.zst   # 5 float32 packed, zstd lv3
        meta_lidar.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zstandard as zstd
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from common import (
    LIDAR_PACKETS_TOPIC,
    LIDAR_POINTS_TOPIC,
    make_typestore,
    stamp_to_ns,
)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "packet_decoder" / "scripts"))
from modules import RSP128Decoder  # noqa: E402

_PF_DTYPE = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


class _Stamp:
    """Shim for rospy.Time — RSP128Decoder only calls .to_sec()."""
    __slots__ = ("_s",)
    def __init__(self, secs: float):
        self._s = secs
    def to_sec(self) -> float:
        return self._s


def pointcloud2_to_xyzir(msg) -> np.ndarray:
    point_step = int(msg.point_step)
    data = bytes(msg.data)
    n_pts = len(data) // point_step
    sorted_fields = sorted(msg.fields, key=lambda f: f.offset)
    parts: list = []
    cursor = 0
    for f in sorted_fields:
        if f.offset > cursor:
            parts.append((f"_pad{cursor}", f"V{f.offset - cursor}"))
        np_type = _PF_DTYPE[f.datatype]
        parts.append((f.name, np_type))
        cursor = f.offset + np.dtype(np_type).itemsize
    if cursor < point_step:
        parts.append(("_pad_end", f"V{point_step - cursor}"))
    arr = np.frombuffer(data, dtype=np.dtype(parts), count=n_pts)
    needed = ["x", "y", "z", "intensity", "ring"]
    out = np.empty((n_pts, 5), dtype=np.float32)
    for i, name in enumerate(needed):
        out[:, i] = arr[name].astype(np.float32, copy=False)
    return out


def _save_lidar_frame(points, frame_ts_s, out_dir, cctx) -> int | None:
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return None
    arr = arr[:, :5]  # decoder emits 6 cols (xyzir + per-point ts); drop ts
    frame_ts_ns = int(frame_ts_s * 1e9)
    rel = f"lidar/{frame_ts_ns}.bin.zst"
    (out_dir / rel).write_bytes(cctx.compress(arr.tobytes()))
    return frame_ts_ns


def process(bag_path: Path, out_root: Path, packet_msg_dir: Path) -> Path:
    out_dir = out_root / bag_path.stem
    (out_dir / "lidar").mkdir(parents=True, exist_ok=True)

    typestore = make_typestore((packet_msg_dir, "rslidar_msg"))
    cctx = zstd.ZstdCompressor(level=3)
    decoder = RSP128Decoder()
    n_frames = 0
    n_msop_skipped = 0
    topics_used: set[str] = set()

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        wanted = {LIDAR_PACKETS_TOPIC, LIDAR_POINTS_TOPIC}
        conns = [c for c in reader.connections if c.topic in wanted]
        if not conns:
            raise SystemExit(f"No lidar topics in {bag_path}")
        total = sum(c.msgcount for c in conns)

        for connection, bag_ts, rawdata in tqdm(
            reader.messages(connections=conns), total=total, unit="msg"
        ):
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            topics_used.add(topic)

            if topic == LIDAR_PACKETS_TOPIC:
                pkt_data = bytes(msg.data)
                if msg.is_difop:
                    if not decoder.calibration_ready:
                        decoder.decode_difop(pkt_data)
                    continue
                if not decoder.calibration_ready:
                    n_msop_skipped += 1
                    continue
                # lidar header.stamp is on a separate clock; use bag receive time
                pkt_ts_s = bag_ts / 1e9
                completed = decoder.decode_msop(
                    pkt_data, _Stamp(pkt_ts_s), bool(msg.is_frame_begin)
                )
                for points, frame_ts in completed:
                    if _save_lidar_frame(points, frame_ts, out_dir, cctx) is not None:
                        n_frames += 1

            elif topic == LIDAR_POINTS_TOPIC:
                ts_ns = stamp_to_ns(msg.header.stamp)
                xyzir = pointcloud2_to_xyzir(msg)
                rel = f"lidar/{ts_ns}.bin.zst"
                (out_dir / rel).write_bytes(cctx.compress(xyzir.tobytes()))
                n_frames += 1

        for points, frame_ts in decoder.flush():
            if _save_lidar_frame(points, frame_ts, out_dir, cctx) is not None:
                n_frames += 1

        bag_start_ns = int(reader.start_time)
        bag_end_ns = int(reader.end_time)

    meta = {
        "source_bag": str(bag_path.resolve()),
        "lidar_topic": sorted(topics_used),
        "time_base": (
            "bag_receive"
            if LIDAR_PACKETS_TOPIC in topics_used
            else "lidar_header_stamp"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag_start_ns": bag_start_ns,
        "bag_end_ns": bag_end_ns,
        "n_lidar_frames": n_frames,
        "lidar_decoder": {
            "type": "RSP128",
            "calibration_ready": decoder.calibration_ready,
            "msop_skipped_before_calib": n_msop_skipped,
        },
        "decode_lidar_version": "0.1",
    }
    (out_dir / "meta_lidar.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main():
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path, help="Path to a single .bag file.")
    p.add_argument("--out", type=Path, default=Path("/mnt/hdd/intermediate"),
                   help="Intermediate root (default: /mnt/hdd/intermediate).")
    p.add_argument("--packet-msg-dir", type=Path,
                   default=here / "packet_decoder" / "src" / "rslidar_msg" / "msg",
                   help="rslidar_msg .msg directory.")
    args = p.parse_args()

    for path, label in [(args.bag, "bag"), (args.packet_msg_dir, "packet msg dir")]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_dir = process(args.bag, args.out, args.packet_msg_dir)
    print(f"Done. Lidar: {out_dir}/lidar/")


if __name__ == "__main__":
    main()
