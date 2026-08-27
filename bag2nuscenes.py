#!/usr/bin/env python3
"""Convert a ROS1 rosbag into a NuScenes v1.0-trainval dataset in one pass.

    python bag2nuscenes.py /path/to.bag --out /data/tcar_nuscenes --calib calib/2025_8_19

The bag is read exactly once. Sensor payloads stream straight into a staging
area inside the output root; once synchronization and scene partitioning have
decided which frames become samples and sweeps, the staged files are *renamed*
into place, which is a metadata operation on the same filesystem. Nothing is
copied twice and no intermediate dump survives the run.

LiDAR comes from whichever topic the bag carries: `/middle/rslidar_points`
(PointCloud2, read directly) or `/middle/rslidar_packets` (raw MSOP/DIFOP,
decoded here). Only the packet path is slow.

Running it again on another bag appends: scene numbering continues, sensor and
category tokens are reused, and re-importing the same bag is refused.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from common import (
    ANNOTATION_TOPIC,
    LIDAR_PACKETS_TOPIC,
    LIDAR_POINTS_TOPIC,
    NUSCENES_CAMS,
    ODOM_TOPIC,
    TOPIC_TO_CAM_CHANNEL,
    load_calib,
    make_typestore,
    stamp_to_ns,
)
from nuscenes_writer import (
    SensorData,
    build_tables,
    new_token,
    load_existing_tables,
    merge_tables,
    partition_scenes,
    sync_keyframes,
    validate_with_devkit,
    write_tables,
)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "packet_decoder" / "scripts"))
from modules import RSP128Decoder  # noqa: E402

STAGING_DIRNAME = ".staging"

# PointField.datatype -> numpy dtype
_PF_DTYPE = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


class _Stamp:
    """Shim for rospy.Time — RSP128Decoder only ever calls .to_sec()."""
    __slots__ = ("_s",)

    def __init__(self, secs: float):
        self._s = secs

    def to_sec(self) -> float:
        return self._s


def pointcloud2_to_xyzir(msg) -> np.ndarray:
    """PointCloud2 -> (N, 5) float32 of x, y, z, intensity, ring.

    Field offsets are honoured rather than assumed, because the padding between
    fields differs between Robosense models.
    """
    point_step = int(msg.point_step)
    data = bytes(msg.data)
    n_pts = len(data) // point_step
    parts: list = []
    cursor = 0
    for f in sorted(msg.fields, key=lambda f: f.offset):
        if f.offset > cursor:
            parts.append((f"_pad{cursor}", f"V{f.offset - cursor}"))
        np_type = _PF_DTYPE[f.datatype]
        parts.append((f.name, np_type))
        cursor = f.offset + np.dtype(np_type).itemsize
    if cursor < point_step:
        parts.append(("_pad_end", f"V{point_step - cursor}"))
    arr = np.frombuffer(data, dtype=np.dtype(parts), count=n_pts)
    out = np.empty((n_pts, 5), dtype=np.float32)
    for i, name in enumerate(["x", "y", "z", "intensity", "ring"]):
        out[:, i] = arr[name].astype(np.float32, copy=False)
    return out


def _write_lidar_frame(points, frame_ts_ns: int, staging: Path) -> int:
    """Write one sweep as NuScenes .pcd.bin (5 x float32 per point).

    Robosense publishes an organized cloud, so no-return directions arrive as
    NaN placeholders — around 42% of a 128x1800 sweep. NuScenes point clouds are
    unorganized and devkit consumers do not expect NaN, so they are dropped
    here. Returns the number of points kept.
    """
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return 0
    # The packet decoder emits a 6th column of per-point timestamps; .pcd.bin is
    # 5 floats per point, so it is dropped here.
    arr = arr[:, :5]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if not len(arr):
        return 0
    (staging / "lidar" / f"{frame_ts_ns}.pcd.bin").write_bytes(arr.tobytes())
    return len(arr)


def read_bag(bag_path: Path, staging: Path, calib_dir: Path,
             msg_dir: Path, packet_msg_dir: Path) -> tuple[SensorData, dict]:
    """Single pass over the bag: stage sensor payloads, collect timestamps.

    Camera JPEGs and LiDAR frames land in `staging` named by timestamp; odom and
    annotations are small enough to hold in memory until the tables are built.
    """
    for ch in TOPIC_TO_CAM_CHANNEL.values():
        (staging / "cameras" / ch).mkdir(parents=True, exist_ok=True)
    (staging / "lidar").mkdir(parents=True, exist_ok=True)

    typestore = make_typestore((msg_dir, "data_processing"),
                               (packet_msg_dir, "rslidar_msg"))

    cam_ts: dict[str, list[int]] = {ch: [] for ch in TOPIC_TO_CAM_CHANNEL.values()}
    cam_size: dict[str, tuple[int, int]] = {}
    lidar_ts: list[int] = []
    odom_rows: list[tuple] = []
    ann_rows: list[tuple] = []
    decoder = RSP128Decoder()
    n_msop_skipped = 0
    lidar_topics_seen: set[str] = set()

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        wanted_topics = (set(TOPIC_TO_CAM_CHANNEL)
                         | {ODOM_TOPIC, ANNOTATION_TOPIC,
                            LIDAR_POINTS_TOPIC, LIDAR_PACKETS_TOPIC})
        conns = [c for c in reader.connections if c.topic in wanted_topics]
        if not conns:
            raise SystemExit(f"no pipeline topics in {bag_path}")
        present = sorted({c.topic for c in conns})
        print(f"  topics: {len(present)} of {len(wanted_topics)} expected")
        for t in sorted(wanted_topics - set(present)):
            print(f"    [missing] {t}")

        for connection, bag_ns, rawdata in tqdm(
            reader.messages(connections=conns),
            total=sum(c.msgcount for c in conns), unit="msg",
        ):
            topic = connection.topic
            msg = reader.deserialize(rawdata, connection.msgtype)

            if topic in TOPIC_TO_CAM_CHANNEL:
                ch = TOPIC_TO_CAM_CHANNEL[topic]
                ts_ns = stamp_to_ns(msg.header.stamp)
                (staging / "cameras" / ch / f"{ts_ns}.jpg").write_bytes(bytes(msg.data))
                cam_ts[ch].append(ts_ns)

            elif topic == LIDAR_POINTS_TOPIC:
                lidar_topics_seen.add(topic)
                ts_ns = stamp_to_ns(msg.header.stamp)
                if _write_lidar_frame(pointcloud2_to_xyzir(msg), ts_ns, staging):
                    lidar_ts.append(ts_ns)

            elif topic == LIDAR_PACKETS_TOPIC:
                lidar_topics_seen.add(topic)
                pkt = bytes(msg.data)
                if msg.is_difop:
                    if not decoder.calibration_ready:
                        decoder.decode_difop(pkt)
                    continue
                if not decoder.calibration_ready:
                    n_msop_skipped += 1
                    continue
                # The lidar's own clock is not disciplined (no PTP), so frames
                # are placed on the recording host clock instead.
                for points, frame_ts in decoder.decode_msop(
                    pkt, _Stamp(bag_ns / 1e9), bool(msg.is_frame_begin)
                ):
                    ts_ns = int(frame_ts * 1e9)
                    if _write_lidar_frame(points, ts_ns, staging):
                        lidar_ts.append(ts_ns)

            elif topic == ODOM_TOPIC:
                p_, q_ = msg.pose.pose.position, msg.pose.pose.orientation
                odom_rows.append((stamp_to_ns(msg.header.stamp),
                                  p_.x, p_.y, p_.z, q_.w, q_.x, q_.y, q_.z))

            elif topic == ANNOTATION_TOPIC:
                ts_ns = stamp_to_ns(msg.header.stamp)
                for o in msg.object_list:
                    ann_rows.append((
                        ts_ns, int(o.track_id), int(o.type),
                        float(o.box_center_base.x), float(o.box_center_base.y),
                        float(o.box_center_base.z), float(o.box_size.width),
                        float(o.box_size.length), float(o.box_size.height),
                        float(o.yaw_base), float(o.velocity_base.x),
                        float(o.velocity_base.y), float(o.velocity_base.z),
                        float(o.yaw_rate), int(o.track_state), int(o.track_age),
                    ))

        if LIDAR_PACKETS_TOPIC in lidar_topics_seen:
            for points, frame_ts in decoder.flush():
                ts_ns = int(frame_ts * 1e9)
                if _write_lidar_frame(points, ts_ns, staging):
                    lidar_ts.append(ts_ns)

        bag_start_ns, bag_end_ns = int(reader.start_time), int(reader.end_time)

    if not odom_rows:
        raise SystemExit(
            f"no {ODOM_TOPIC} in {bag_path.name} — ego_pose cannot be built. "
            "This bag is not convertible."
        )
    if not lidar_ts:
        raise SystemExit(f"no lidar frames decoded from {bag_path.name}")

    # Read one JPEG per channel for the height/width that sample_data records.
    for ch, ts in cam_ts.items():
        if ts:
            img = cv2.imread(str(staging / "cameras" / ch / f"{ts[0]}.jpg"))
            if img is None:
                raise SystemExit(f"unreadable JPEG for {ch} at {ts[0]}")
            cam_size[ch] = (img.shape[0], img.shape[1])

    odom = np.array(sorted(odom_rows), dtype=np.float64)
    data = SensorData(
        calib=load_calib(calib_dir),
        cam_ts={ch: np.array(sorted(v), dtype=np.int64)
                for ch, v in cam_ts.items() if v},
        lidar_ts=np.array(sorted(lidar_ts), dtype=np.int64),
        odom_ts=odom[:, 0].astype(np.int64),
        odom_t=odom[:, 1:4],
        odom_R=Rotation.from_quat(odom[:, [5, 6, 7, 4]]),  # wxyz -> xyzw
        cam_size=cam_size,
        bag_start_ns=bag_start_ns,
        bag_end_ns=bag_end_ns,
    )
    stats = {
        "source_bag": str(bag_path.resolve()),
        "calib_source": str(calib_dir.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lidar_topic": sorted(lidar_topics_seen),
        "lidar_time_base": ("bag_receive"
                            if LIDAR_PACKETS_TOPIC in lidar_topics_seen
                            else "lidar_header_stamp"),
        "msop_skipped_before_calib": n_msop_skipped,
        "n_camera_frames": sum(len(v) for v in cam_ts.values()),
        "n_lidar_frames": len(lidar_ts),
        "n_odom_samples": len(odom_rows),
        "n_annotation_objects": len(ann_rows),
    }
    return data, stats


def _ensure_placeholder_map(out_root: Path) -> None:
    """devkit's render_sample loads a map raster; give it a tiny valid PNG."""
    map_path = out_root / "maps" / "placeholder.png"
    if map_path.exists():
        return
    map_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(map_path), np.full((100, 100), 255, dtype=np.uint8))


def materialize(plan: list[tuple[int, str, str]], staging: Path,
                out_root: Path) -> None:
    """Move each staged frame to its NuScenes path.

    A rename, not a copy: staging lives inside out_root so this is the same
    filesystem, and the data is never written twice.
    """
    _ensure_placeholder_map(out_root)
    n_moved = n_missing = 0
    for ts_ns, channel, rel_target in plan:
        target = out_root / rel_target
        if target.exists():
            continue
        if channel == "LIDAR_TOP":
            src = staging / "lidar" / f"{ts_ns}.pcd.bin"
        else:
            src = staging / "cameras" / channel / f"{ts_ns}.jpg"
        if not src.exists():
            n_missing += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        src.rename(target)
        n_moved += 1
    print(f"  moved {n_moved} files into place"
          + (f"   [!] {n_missing} staged files missing" if n_missing else ""))


def main() -> None:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path, help="Path to a single .bag file.")
    p.add_argument("--out", type=Path, default=Path("/data/tcar_nuscenes"),
                   help="NuScenes dataroot (default: /data/tcar_nuscenes).")
    p.add_argument("--calib", type=Path, default=here / "calib" / "2025_6_27",
                   help="Calibration snapshot directory.")
    p.add_argument("--version", default="v1.0-trainval",
                   help="NuScenes version subdirectory (default: v1.0-trainval).")
    p.add_argument("--keyframe-stride", type=int, default=5,
                   help="Use every Kth lidar frame as a keyframe (5 = 2Hz from 10Hz).")
    p.add_argument("--sync-ms", type=float, default=25.0,
                   help="Per-channel camera sync tolerance in ms (default 25).")
    p.add_argument("--scene-dur", type=float, default=20.0,
                   help="Scene length in seconds (default 20). A trailing scene "
                        "shorter than 90%% of this is dropped.")
    p.add_argument("--include-traffic-cam", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include CAM_TRAFFIC as a 7th channel.")
    p.add_argument("--msg-dir", type=Path, default=here / "msg")
    p.add_argument("--packet-msg-dir", type=Path,
                   default=here / "packet_decoder" / "src" / "rslidar_msg" / "msg")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the NuScenes(...) load check at the end.")
    p.add_argument("--keep-staging", action="store_true",
                   help="Leave the staging directory for debugging.")
    args = p.parse_args()

    for path, label in [(args.bag, "bag"), (args.calib, "calib"),
                        (args.msg_dir, "msg dir")]:
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    log_name = args.bag.stem
    json_dir = args.out / args.version
    existing = load_existing_tables(json_dir)
    if existing and any(r.get("logfile") == log_name
                        for r in existing.get("log.json", [])):
        raise SystemExit(
            f"log '{log_name}' is already in {json_dir}/log.json — "
            "remove that log entry first if you mean to re-import it."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    staging = args.out / STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging)

    try:
        print(f"[1/5] Reading {args.bag.name} ...")
        data, stats = read_bag(args.bag, staging, args.calib,
                               args.msg_dir, args.packet_msg_dir)
        print(f"  cameras: {sum(len(v) for v in data.cam_ts.values())} frames "
              f"over {len(data.cam_ts)} channels")
        print(f"  lidar:   {len(data.lidar_ts)} frames "
              f"({stats['lidar_time_base']})")
        print(f"  odom:    {len(data.odom_ts)} samples")

        channels = [c for c in NUSCENES_CAMS if c in data.cam_ts]
        missing = sorted(set(NUSCENES_CAMS) - set(channels))
        if missing:
            raise SystemExit(f"missing required camera channels: {missing}")
        if args.include_traffic_cam and "CAM_TRAFFIC" in data.cam_ts:
            channels.append("CAM_TRAFFIC")

        # Only the six standard channels gate a keyframe. CAM_TRAFFIC rides
        # along when it is in tolerance; it has placeholder calibration and must
        # not be able to throw away otherwise-good samples.
        required = [c for c in channels if c in NUSCENES_CAMS]
        print(f"\n[2/5] Sync ({args.sync_ms} ms)...")
        keyframes = sync_keyframes(data, channels, int(args.sync_ms * 1e6),
                                   args.keyframe_stride, required=required)

        print(f"\n[3/5] Scene partitioning ({args.scene_dur}s windows)...")
        scenes = partition_scenes(keyframes, args.scene_dur)
        if not scenes:
            raise SystemExit("no scenes survived partitioning")

        print(f"\n[4/5] Building tables ({len(scenes)} scenes)...")
        if existing:
            print(f"  append mode: {len(existing['scene.json'])} existing scenes")
        tables, plan = build_tables(data, scenes, channels,
                                    log_token=new_token(), log_name=log_name,
                                    existing=existing)
        for k, v in tables.items():
            print(f"  + {k:28} {len(v):>8}")

        print(f"\n[5/5] Materializing under {args.out} ...")
        materialize(plan, staging, args.out)
        if existing:
            tables = merge_tables(existing, tables)
        write_tables(tables, args.out, args.version)
        (args.out / f"{log_name}.import.json").write_text(json.dumps(stats, indent=2))
        print(f"  tables -> {json_dir}")
    finally:
        if staging.exists() and not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)

    if args.no_validate:
        print("\n(validation skipped)")
    else:
        print("\nValidating with nuscenes-devkit...")
        validate_with_devkit(args.out, args.version)
    print("\nDone.")


if __name__ == "__main__":
    main()
