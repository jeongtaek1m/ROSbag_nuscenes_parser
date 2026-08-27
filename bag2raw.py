#!/usr/bin/env python3
"""Stage 1 (cameras + odom + annotations): rosbag -> intermediate.

Lidar is handled separately by `decode_lidar.py` (slow packet decoding).
This script is fast (~5 min for 1183 sec bag) — re-run any time camera
mapping or annotation handling changes, without touching lidar artifacts.

Output:
    <out>/<bag_basename>/
        cameras/CAM_*/<header_ns>.jpg     # JPEG bytes copied as-is
        odom.parquet                       # ts_ns, tx,ty,tz, qw,qx,qy,qz
        annotations.parquet                # ts_ns + ObjectFusion fields, flattened
        calib.json                         # snapshot of calibration used
        meta.json                          # source bag, mapping, timestamps
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rosbags.highlevel import AnyReader
from tqdm import tqdm

from common import (
    ANNOTATION_TOPIC,
    ODOM_TOPIC,
    TOPIC_TO_CAM_CHANNEL,
    load_calib,
    make_typestore,
    stamp_to_ns,
)


def process(bag_path: Path, out_root: Path, calib_dir: Path, msg_dir: Path) -> Path:
    out_dir = out_root / bag_path.stem
    cam_dirs = {ch: out_dir / "cameras" / ch for ch in TOPIC_TO_CAM_CHANNEL.values()}
    for d in cam_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    typestore = make_typestore((msg_dir, "data_processing"))

    odom_rows: list[tuple] = []
    ann_rows: list[tuple] = []
    n_camera_frames = 0

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        wanted = set(TOPIC_TO_CAM_CHANNEL) | {ODOM_TOPIC, ANNOTATION_TOPIC}
        conns = [c for c in reader.connections if c.topic in wanted]
        total = sum(c.msgcount for c in conns)

        for connection, _bag_ts, rawdata in tqdm(
            reader.messages(connections=conns), total=total, unit="msg"
        ):
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            if topic in TOPIC_TO_CAM_CHANNEL:
                ts_ns = stamp_to_ns(msg.header.stamp)
                channel = TOPIC_TO_CAM_CHANNEL[topic]
                rel = f"cameras/{channel}/{ts_ns}.jpg"
                (out_dir / rel).write_bytes(bytes(msg.data))
                n_camera_frames += 1

            elif topic == ODOM_TOPIC:
                ts_ns = stamp_to_ns(msg.header.stamp)
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odom_rows.append((ts_ns, p.x, p.y, p.z, q.w, q.x, q.y, q.z))

            elif topic == ANNOTATION_TOPIC:
                ts_ns = stamp_to_ns(msg.header.stamp)
                for o in msg.object_list:
                    ann_rows.append((
                        ts_ns,
                        int(o.track_id), int(o.type),
                        float(o.box_center_base.x),
                        float(o.box_center_base.y),
                        float(o.box_center_base.z),
                        float(o.box_size.width),
                        float(o.box_size.length),
                        float(o.box_size.height),
                        float(o.yaw_base),
                        float(o.velocity_base.x),
                        float(o.velocity_base.y),
                        float(o.velocity_base.z),
                        float(o.yaw_rate),
                        int(o.track_state), int(o.track_age),
                    ))

        bag_start_ns = int(reader.start_time)
        bag_end_ns = int(reader.end_time)

    if odom_rows:
        cols = list(zip(*odom_rows))
        pq.write_table(
            pa.table({
                "ts_ns": pa.array(cols[0], type=pa.uint64()),
                "tx": pa.array(cols[1], type=pa.float64()),
                "ty": pa.array(cols[2], type=pa.float64()),
                "tz": pa.array(cols[3], type=pa.float64()),
                "qw": pa.array(cols[4], type=pa.float64()),
                "qx": pa.array(cols[5], type=pa.float64()),
                "qy": pa.array(cols[6], type=pa.float64()),
                "qz": pa.array(cols[7], type=pa.float64()),
            }),
            out_dir / "odom.parquet",
        )

    if ann_rows:
        cols = list(zip(*ann_rows))
        pq.write_table(
            pa.table({
                "ts_ns":       pa.array(cols[0],  type=pa.uint64()),
                "track_id":    pa.array(cols[1],  type=pa.uint32()),
                "type":        pa.array(cols[2],  type=pa.uint8()),
                "tx":          pa.array(cols[3],  type=pa.float32()),
                "ty":          pa.array(cols[4],  type=pa.float32()),
                "tz":          pa.array(cols[5],  type=pa.float32()),
                "w":           pa.array(cols[6],  type=pa.float32()),
                "l":           pa.array(cols[7],  type=pa.float32()),
                "h":           pa.array(cols[8],  type=pa.float32()),
                "yaw":         pa.array(cols[9],  type=pa.float32()),
                "vx":          pa.array(cols[10], type=pa.float32()),
                "vy":          pa.array(cols[11], type=pa.float32()),
                "vz":          pa.array(cols[12], type=pa.float32()),
                "yaw_rate":    pa.array(cols[13], type=pa.float32()),
                "track_state": pa.array(cols[14], type=pa.uint8()),
                "track_age":   pa.array(cols[15], type=pa.uint32()),
            }),
            out_dir / "annotations.parquet",
        )

    (out_dir / "calib.json").write_text(json.dumps(load_calib(calib_dir), indent=2))

    meta = {
        "source_bag": str(bag_path.resolve()),
        "topic_to_channel": {
            **TOPIC_TO_CAM_CHANNEL,
            ODOM_TOPIC: "EGO",
            ANNOTATION_TOPIC: "ANNOTATION",
        },
        "calib_source": str(calib_dir.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bag_start_ns": bag_start_ns,
        "bag_end_ns": bag_end_ns,
        "n_camera_frames": n_camera_frames,
        "n_odom_samples": len(odom_rows),
        "n_annotation_objects": len(ann_rows),
        "bag2raw_version": "0.3",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main():
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path, help="Path to a single .bag file.")
    p.add_argument("--out", type=Path, default=Path("/mnt/hdd/intermediate"),
                   help="Intermediate root (default: /mnt/hdd/intermediate).")
    p.add_argument("--calib", type=Path, default=here / "calib" / "2025_6_27",
                   help="Calibration directory (default: calib/2025_6_27).")
    p.add_argument("--msg-dir", type=Path, default=here / "msg",
                   help="Custom .msg directory (default: msg/).")
    args = p.parse_args()

    for path, label in [(args.bag, "bag"), (args.calib, "calib"), (args.msg_dir, "msg dir")]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_dir = process(args.bag, args.out, args.calib, args.msg_dir)
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
