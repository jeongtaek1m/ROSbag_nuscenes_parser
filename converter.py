"""The conversion pipeline behind bag2nuscenes.py and bag2nuscenes_full.py.

Both tools do the same thing — read the bag once, pick frames, cut scenes, write
a NuScenes dataset — and differ only in which data they carry (a `Profile`):

  standard  LIDAR_TOP + the six standard cameras, GNSS/INS as CAN bus files.
            Shaped like the real nuScenes, for the devkit and nuScenes-based
            training code.
  full      the same scenes and frames, plus CAM_TRAFFIC, the four bottom
            LiDARs, the front radar, per-point LiDAR times, and every other
            topic in the bag as per-scene JSON under ext/.

Frame selection and scene cutting are identical, so a scene name in one output
covers the same interval as in the other when both are built from the same bag
with the same split.

The bag is read exactly once. Sensor payloads stream into a staging directory
inside the output root (camera JPEGs are rectified on the way, on a thread
pool); once frame selection and scene cutting have decided which frames become
samples and sweeps, the staged files are renamed into place.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from canbus import CAN_MARGIN_NS, InsData, corrimu_to_ego, scene_messages, write_scene
from common import (
    CAM_CHANNEL_TO_TOPIC,
    CORRIMU_TOPIC,
    EXTRA_CAMS,
    EXTRA_LIDAR_TOPIC_TO_CHANNEL,
    INSPVA_TOPIC,
    LIDAR_PACKETS_TOPIC,
    LIDAR_POINTS_TOPIC,
    NUSCENES_CAMS,
    ODOM_TOPIC,
    RADAR_TOPIC_TO_CHANNEL,
    TOPIC_TO_CAM_CHANNEL,
    default_intrinsic,
    make_typestore,
    quat_wxyz_to_R,
    resolve_calib,
    stamp_to_ns,
)
from nuscenes_writer import (
    SensorData,
    build_tables,
    coverage_window,
    format_coverage,
    format_plan,
    load_existing_tables,
    merge_tables,
    new_token,
    official_scene_names,
    partition_scenes,
    plan_frames,
    required_streams,
    validate_with_devkit,
    write_tables,
)
from rectify import Rectifier

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "packet_decoder" / "scripts"))
from modules import RSP128Decoder  # noqa: E402

STAGING_DIRNAME = ".staging"
LOCK_NAME = ".convert.lock"
# Staged file suffix per modality. Radar frames are staged raw and turned into
# NuScenes PCD at materialization, when the ego motion they are compensated
# with is fully known.
STAGE_EXT = {"camera": ".jpg", "lidar": ".pcd.bin", "radar": ".radar.bin"}
POINT_TIME_EXT = ".time.bin"

# Not exported as sidecars: payloads that already are sample_data, the
# camera_info topics (a placeholder K/D, not a calibration), the NovAtel raw
# binary log, and the ARS548 DetectionList (every detection slot as a nested
# message: prohibitively slow to deserialize, and /radar/PointCloudDetection
# already carries its valid detections as RADAR_FRONT).
SIDECAR_SKIP_TOPICS = (
    {f"/camera_{i}/camera_info" for i in range(7)}
    | {"/novatel/oem7/oem7raw", "/radar/DetectionList", LIDAR_PACKETS_TOPIC}
)


@dataclass(frozen=True)
class Profile:
    name: str
    cameras: tuple[str, ...]            # camera channels emitted
    extra_lidars: dict                  # topic -> channel, besides LIDAR_TOP
    radars: dict                        # topic -> channel
    point_time: bool                    # per-point time next to every LiDAR frame
    sidecars: bool                      # every other topic -> ext/<scene>/<topic>.json
    default_out: Path
    description: str


STANDARD = Profile(
    name="standard", cameras=tuple(NUSCENES_CAMS), extra_lidars={}, radars={},
    point_time=False, sidecars=False, default_out=Path("/data/tcar_nuscenes"),
    description="LIDAR_TOP + six standard cameras (rectified) + GNSS/INS as CAN bus",
)
FULL = Profile(
    name="full", cameras=tuple(NUSCENES_CAMS + EXTRA_CAMS),
    extra_lidars=dict(EXTRA_LIDAR_TOPIC_TO_CHANNEL), radars=dict(RADAR_TOPIC_TO_CHANNEL),
    point_time=True, sidecars=True, default_out=Path("/data/tcar_nuscenes_full"),
    description="standard + CAM_TRAFFIC, bottom LiDARs, front radar, per-point LiDAR "
                "time and every other topic as per-scene JSON",
)

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


_PCD_COLUMNS = ("x", "y", "z", "intensity", "ring")
_PC2_DTYPES: dict[tuple, np.dtype] = {}


def _pointcloud2_dtype(msg) -> np.dtype:
    """Structured dtype that views a PointCloud2 buffer in place.

    Field offsets and point_step are honoured rather than assumed, because the
    padding between fields differs between Robosense models. Cached per field
    layout, which is constant for a topic.
    """
    key = (tuple((f.name, int(f.offset), int(f.datatype), int(f.count))
                 for f in msg.fields),
           int(msg.point_step), bool(msg.is_bigendian))
    dt = _PC2_DTYPES.get(key)
    if dt is None:
        formats = []
        for f in msg.fields:
            t = np.dtype(_PF_DTYPE[f.datatype])
            if msg.is_bigendian:
                t = t.newbyteorder(">")
            formats.append(t if int(f.count) == 1 else (t, int(f.count)))
        dt = np.dtype({"names": [f.name for f in msg.fields],
                       "formats": formats,
                       "offsets": [int(f.offset) for f in msg.fields],
                       "itemsize": int(msg.point_step)})
        _PC2_DTYPES[key] = dt
    return dt


def pointcloud2_to_pcdbin(msg, frame_ts_ns: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray | None]:
    """PointCloud2 -> ((M, 5) float32 of x, y, z, intensity, ring, NaN dropped, times).

    Robosense publishes an organized cloud, so no-return directions arrive as
    NaN placeholders — around 42% of a 128x1800 sweep. NuScenes point clouds are
    unorganized and devkit consumers do not expect NaN, so they are dropped.

    With frame_ts_ns, also returns each kept point's acquisition time as float32
    seconds relative to it (the `timestamp` field; 0..0.1 s for a sweep, since
    the header stamp is the start of the sweep), else None.

    The buffer is viewed in place through a structured dtype and only the
    surviving points are copied, once per column.
    """
    dt = _pointcloud2_dtype(msg)
    missing = [c for c in _PCD_COLUMNS if c not in dt.names]
    if missing:
        raise SystemExit(f"PointCloud2 lacks field(s) {missing}; "
                         f"has {list(dt.names)}")
    arr = np.frombuffer(msg.data, dtype=dt, count=len(msg.data) // dt.itemsize)
    # ring is an integer field and always finite, so these four checks are
    # exactly an all-five-columns test.
    keep = np.isfinite(arr["x"])
    for name in ("y", "z", "intensity"):
        keep &= np.isfinite(arr[name])
    out = np.empty((int(keep.sum()), 5), dtype=np.float32)
    if len(out):
        for k, name in enumerate(_PCD_COLUMNS):
            out[:, k] = arr[name][keep]
    times = None
    if frame_ts_ns is not None and "timestamp" in dt.names:
        times = (arr["timestamp"][keep] - frame_ts_ns / 1e9).astype(np.float32)
    return out, times


def lidar_points_to_pcdbin(points, frame_ts_ns: int | None = None
                           ) -> tuple[np.ndarray, np.ndarray | None]:
    """Packet-decoder points (N, 6: x, y, z, intensity, ring, time) -> as above."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return np.empty((0, 5), dtype=np.float32), None
    arr = arr[np.isfinite(arr[:, :5]).all(axis=1)]
    times = None
    if frame_ts_ns is not None and arr.shape[1] >= 6:
        times = (arr[:, 5] - frame_ts_ns / 1e9).astype(np.float32)
    return np.ascontiguousarray(arr[:, :5], dtype=np.float32), times


_RADAR_FIELDS = [
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("dyn_prop", "i1"), ("id", "<i2"),
    ("rcs", "<f4"), ("vx", "<f4"), ("vy", "<f4"), ("vx_comp", "<f4"), ("vy_comp", "<f4"),
    ("is_quality_valid", "i1"), ("ambig_state", "i1"), ("x_rms", "i1"), ("y_rms", "i1"),
    ("invalid_state", "i1"), ("pdh0", "i1"), ("vx_rms", "i1"), ("vy_rms", "i1"),
]
_RADAR_DTYPE = np.dtype(_RADAR_FIELDS)   # packed, 43 bytes, as in nuScenes


def radar_to_pcd(det: np.ndarray, v_sensor: np.ndarray) -> bytes:
    """ARS548 detections (N, 5: x, y, z, radial velocity, RCS; radar frame) ->
    NuScenes radar .pcd bytes.

    ARS548 detections carry only the radial velocity, so vx, vy are that
    velocity resolved along the line of sight, and vx_comp, vy_comp the same
    after adding back the sensor's own motion (`v_sensor`, radar frame):
    a static target gets ~0. Fields the ARS548 does not report take the values
    that pass the devkit's default filters: dyn_prop 4 (unknown), ambig_state 3
    (unambiguous), invalid_state 0 (valid), is_quality_valid 1, pdh0 and the
    *_rms 0. A frame without detections is one NaN point, the nuScenes
    convention for an empty sweep.
    """
    n = len(det)
    arr = np.zeros(max(n, 1), dtype=_RADAR_DTYPE)
    if n:
        xyz = det[:, :3].astype(np.float64)
        rng = np.linalg.norm(xyz, axis=1)
        u = np.divide(xyz, rng[:, None], out=np.zeros_like(xyz), where=rng[:, None] > 0)
        v_r = det[:, 3].astype(np.float64)
        v_comp = v_r + u @ np.asarray(v_sensor, dtype=np.float64)
        arr["x"], arr["y"], arr["z"] = det[:, 0], det[:, 1], det[:, 2]
        arr["dyn_prop"] = 4
        arr["id"] = np.arange(n)
        arr["rcs"] = det[:, 4]
        arr["vx"], arr["vy"] = v_r * u[:, 0], v_r * u[:, 1]
        arr["vx_comp"], arr["vy_comp"] = v_comp * u[:, 0], v_comp * u[:, 1]
        arr["is_quality_valid"] = 1
        arr["ambig_state"] = 3
    else:
        for name in ("x", "y", "z", "rcs", "vx", "vy", "vx_comp", "vy_comp"):
            arr[name] = np.nan
    width = len(arr)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(n_ for n_, _ in _RADAR_FIELDS)}\n"
        "SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\n"
        "TYPE F F F I I F F F F F I I I I I I I I\n"
        f"COUNT {' '.join(['1'] * len(_RADAR_FIELDS))}\n"
        f"WIDTH {width}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {width}\n"
        "DATA binary\n"
    )
    # The devkit's reader asserts every field ends strictly before the end of
    # the data, so the last byte of the last point needs something after it.
    return header.encode("ascii") + arr.tobytes() + b"\n"


# ------------------------------------------------------------ message -> JSON
_FIELD_CACHE: dict[type, list[str]] = {}


def _msg_fields(cls) -> list[str]:
    names = _FIELD_CACHE.get(cls)
    if names is None:
        ann = getattr(cls, "__annotations__", {})
        names = [f.name for f in dataclasses.fields(cls)
                 if not str(ann.get(f.name, "")).startswith("ClassVar")]
        _FIELD_CACHE[cls] = names
    return names


def msg_to_json(m):
    """A deserialized rosbags message -> plain JSON-able Python.

    Constants are left out; builtin_interfaces Time and Duration become integer
    nanoseconds; arrays become lists.
    """
    if dataclasses.is_dataclass(m):
        mt = getattr(m, "__msgtype__", "")
        if mt in ("builtin_interfaces/msg/Time", "builtin_interfaces/msg/Duration"):
            return stamp_to_ns(m)
        return {name: msg_to_json(getattr(m, name)) for name in _msg_fields(type(m))}
    if isinstance(m, np.ndarray):
        return m.tolist()
    if isinstance(m, (list, tuple)):
        return [msg_to_json(x) for x in m]
    if isinstance(m, (bytes, bytearray, memoryview)):
        return list(bytes(m))
    if isinstance(m, np.generic):
        return m.item()
    return m


def _topic_slug(topic: str) -> str:
    return topic.strip("/").replace("/", "__")


class _StagingWriter:
    """Write staged payloads on a thread pool.

    A job is either a buffer or a callable producing one; camera rectification
    runs as such a callable, so decode/remap/encode happens on the workers.
    OpenCV and file I/O release the GIL, so the work overlaps with the main
    thread's deserialization. At most `depth` jobs are in flight, which bounds
    memory. Completed writes are reaped on every submit and the first failure
    is re-raised on the main thread.
    """

    def __init__(self, workers: int = 4, depth: int = 64):
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="staging")
        self._slots = threading.BoundedSemaphore(depth)
        self._pending: deque[Future] = deque()

    def submit(self, path: Path, job) -> None:
        """Queue `job` (bytes, a C-contiguous array, or a callable returning one)."""
        self._reap(block=False)
        self._slots.acquire()
        try:
            self._pending.append(self._pool.submit(self._write, path, job))
        except BaseException:
            self._slots.release()
            raise

    def _write(self, path: Path, job) -> None:
        try:
            path.write_bytes(job() if callable(job) else job)
        finally:
            self._slots.release()

    def _reap(self, block: bool) -> None:
        while self._pending and (block or self._pending[0].done()):
            self._pending.popleft().result()  # re-raises a failed write

    def close(self) -> None:
        """Wait for every queued write; raises if any of them failed."""
        try:
            self._reap(block=True)
        finally:
            self._pool.shutdown(wait=True)

    def __enter__(self) -> _StagingWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            # Don't mask the original error with a write failure; just drain.
            self._pool.shutdown(wait=True)


# ------------------------------------------------------------------ reading
@dataclass
class BagContents:
    data: SensorData
    ins: InsData
    rectifiers: dict[str, Rectifier]
    sidecar_topics: dict[str, dict]      # slug -> {"topic", "msgtype", "count"}
    stats: dict


def read_bag(bag_path: Path, staging: Path, profile: Profile, channels: list[str],
             calib: dict, packet_msg_dir: Path, balance: float, jpeg_quality: int,
             workers: int) -> BagContents:
    """Single pass over the bag: stage sensor payloads, collect timestamps.

    Camera JPEGs (rectified), LiDAR frames and raw radar frames land in
    `staging/<channel>/<timestamp><ext>`; sidecar topics stream to
    `staging/_ext/<topic>.jsonl`. Odom, INSPVA and CORRIMU are small enough to
    hold in memory until the tables and CAN bus files are built.
    """
    modality = {"LIDAR_TOP": "lidar"}
    topic_channel: dict[str, str] = {}
    for ch in channels:
        if ch.startswith("CAM_"):
            modality[ch] = "camera"
            topic_channel[CAM_CHANNEL_TO_TOPIC[ch]] = ch
    for topic, ch in profile.extra_lidars.items():
        if ch in channels:
            modality[ch] = "lidar"
            topic_channel[topic] = ch
    for topic, ch in profile.radars.items():
        if ch in channels:
            modality[ch] = "radar"
            topic_channel[topic] = ch
    for ch in channels:
        (staging / ch).mkdir(parents=True, exist_ok=True)
    ext_dir = staging / "_ext"
    ext_dir.mkdir(parents=True, exist_ok=True)

    typestore = make_typestore((packet_msg_dir, "rslidar_msg"))
    frames: dict[str, list[int]] = {ch: [] for ch in channels}
    rectifiers: dict[str, Rectifier] = {}
    cam_size: dict[str, tuple[int, int]] = {}
    odom_rows: list[tuple] = []
    ins_rows: list[tuple] = []
    imu_rows: list[tuple] = []
    decoder = RSP128Decoder()
    n_msop_skipped = 0
    lidar_topics_seen: set[str] = set()
    sidecar_files: dict[str, object] = {}
    sidecar_topics: dict[str, dict] = {}
    payload_topics = (set(TOPIC_TO_CAM_CHANNEL) | set(EXTRA_LIDAR_TOPIC_TO_CHANNEL)
                      | set(RADAR_TOPIC_TO_CHANNEL) | {LIDAR_POINTS_TOPIC})

    def stage_lidar(ch: str, ts_ns: int, pts: np.ndarray, times) -> None:
        if not len(pts):
            return
        writer.submit(staging / ch / f"{ts_ns}{STAGE_EXT['lidar']}", pts)
        if profile.point_time and times is not None:
            writer.submit(staging / ch / f"{ts_ns}{POINT_TIME_EXT}", times)
        frames[ch].append(ts_ns)

    try:
        with _StagingWriter(workers=workers, depth=4 * workers) as writer, \
                AnyReader([bag_path], default_typestore=typestore) as reader:
            wanted = (set(topic_channel)
                      | {ODOM_TOPIC, INSPVA_TOPIC, CORRIMU_TOPIC,
                         LIDAR_POINTS_TOPIC, LIDAR_PACKETS_TOPIC})
            if profile.sidecars:
                wanted |= {c.topic for c in reader.connections
                           if c.topic not in payload_topics
                           and c.topic not in SIDECAR_SKIP_TOPICS}
            conns = [c for c in reader.connections if c.topic in wanted]
            if not conns:
                raise SystemExit(f"no pipeline topics in {bag_path}")
            present = {c.topic for c in conns}
            expected = set(topic_channel) | {ODOM_TOPIC, INSPVA_TOPIC, CORRIMU_TOPIC}
            print(f"  topics: {len(present)} read"
                  + (f" ({len(present - expected - {LIDAR_POINTS_TOPIC, LIDAR_PACKETS_TOPIC})}"
                     " as sidecars)" if profile.sidecars else ""))
            for t in sorted(expected - present):
                print(f"    [missing] {t}")

            for connection, bag_ns, rawdata in tqdm(
                reader.messages(connections=conns),
                total=sum(c.msgcount for c in conns), unit="msg",
            ):
                topic = connection.topic
                msg = reader.deserialize(rawdata, connection.msgtype)

                if profile.sidecars and topic not in payload_topics \
                        and topic not in SIDECAR_SKIP_TOPICS:
                    slug = _topic_slug(topic)
                    f = sidecar_files.get(slug)
                    if f is None:
                        f = sidecar_files[slug] = open(ext_dir / f"{slug}.jsonl", "w")
                        sidecar_topics[slug] = {"topic": topic,
                                                "msgtype": connection.msgtype, "count": 0}
                    header = getattr(msg, "header", None)
                    t_ns = stamp_to_ns(header.stamp) if header is not None else 0
                    if not t_ns:
                        t_ns = int(bag_ns)
                    f.write(json.dumps({"utime": t_ns // 1000, "bag_utime": int(bag_ns) // 1000,
                                        "msg": msg_to_json(msg)}, separators=(",", ":")))
                    f.write("\n")
                    sidecar_topics[slug]["count"] += 1

                ch = topic_channel.get(topic)
                if ch is not None and modality[ch] == "camera":
                    ts_ns = stamp_to_ns(msg.header.stamp)
                    raw = bytes(msg.data)
                    rect = rectifiers.get(ch)
                    if rect is None:
                        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            raise SystemExit(f"unreadable JPEG for {ch} at {ts_ns}")
                        cam_size[ch] = (img.shape[0], img.shape[1])
                        c = calib[ch]
                        if c["intrinsic"] is None:   # no calibration: default K
                            c["intrinsic"] = default_intrinsic(img.shape[1], img.shape[0])
                        rect = rectifiers[ch] = Rectifier(
                            c["intrinsic"], c["distortion"], c["model"],
                            (img.shape[1], img.shape[0]), balance, jpeg_quality)
                    writer.submit(staging / ch / f"{ts_ns}{STAGE_EXT['camera']}",
                                  (lambda r=rect, b=raw: r(b)))
                    frames[ch].append(ts_ns)

                elif topic == LIDAR_POINTS_TOPIC:
                    lidar_topics_seen.add(topic)
                    ts_ns = stamp_to_ns(msg.header.stamp)
                    pts, times = pointcloud2_to_pcdbin(
                        msg, ts_ns if profile.point_time else None)
                    stage_lidar("LIDAR_TOP", ts_ns, pts, times)

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
                    # The lidar's own clock is not disciplined on the old
                    # packet bags, so frames are placed on the recording host
                    # clock instead.
                    for points, frame_ts in decoder.decode_msop(
                        pkt, _Stamp(bag_ns / 1e9), bool(msg.is_frame_begin)
                    ):
                        ts_ns = int(frame_ts * 1e9)
                        pts, times = lidar_points_to_pcdbin(
                            points, ts_ns if profile.point_time else None)
                        stage_lidar("LIDAR_TOP", ts_ns, pts, times)

                elif ch is not None and modality[ch] == "lidar":
                    ts_ns = stamp_to_ns(msg.header.stamp)
                    pts, times = pointcloud2_to_pcdbin(
                        msg, ts_ns if profile.point_time else None)
                    stage_lidar(ch, ts_ns, pts, times)

                elif ch is not None and modality[ch] == "radar":
                    ts_ns = stamp_to_ns(msg.header.stamp)
                    dt = _pointcloud2_dtype(msg)
                    arr = np.frombuffer(msg.data, dtype=dt, count=len(msg.data) // dt.itemsize)
                    det = np.empty((len(arr), 5), dtype=np.float32)
                    for k, name in enumerate(("x", "y", "z", "v", "RCS")):
                        det[:, k] = arr[name]
                    det = det[np.isfinite(det).all(axis=1)]
                    writer.submit(staging / ch / f"{ts_ns}{STAGE_EXT['radar']}", det)
                    frames[ch].append(ts_ns)

                elif topic == ODOM_TOPIC:
                    p_, q_ = msg.pose.pose.position, msg.pose.pose.orientation
                    v_ = msg.twist.twist.linear
                    odom_rows.append((stamp_to_ns(msg.header.stamp),
                                      p_.x, p_.y, p_.z, q_.w, q_.x, q_.y, q_.z,
                                      v_.x, v_.y, v_.z))

                elif topic == INSPVA_TOPIC:
                    ins_rows.append((stamp_to_ns(msg.header.stamp), msg.latitude,
                                     msg.longitude, msg.height, int(msg.status.status)))

                elif topic == CORRIMU_TOPIC:
                    imu_rows.append((stamp_to_ns(msg.header.stamp), msg.imu_data_count,
                                     msg.pitch_rate, msg.roll_rate, msg.yaw_rate,
                                     msg.lateral_acc, msg.longitudinal_acc, msg.vertical_acc))

            if LIDAR_PACKETS_TOPIC in lidar_topics_seen:
                for points, frame_ts in decoder.flush():
                    ts_ns = int(frame_ts * 1e9)
                    pts, times = lidar_points_to_pcdbin(
                        points, ts_ns if profile.point_time else None)
                    stage_lidar("LIDAR_TOP", ts_ns, pts, times)

            bag_start_ns, bag_end_ns = int(reader.start_time), int(reader.end_time)
    finally:
        for f in sidecar_files.values():
            f.close()

    if not odom_rows:
        raise SystemExit(f"no {ODOM_TOPIC} in {bag_path.name} — ego_pose cannot be built. "
                         "This bag is not convertible.")
    if not imu_rows:
        raise SystemExit(f"no {CORRIMU_TOPIC} in {bag_path.name} — the CAN bus pose "
                         "(accel, rotation_rate) cannot be built.")
    if not frames["LIDAR_TOP"]:
        raise SystemExit(f"no lidar frames decoded from {bag_path.name}")

    odom = np.array(sorted(odom_rows), dtype=np.float64)
    odom_ts = np.array(sorted(r[0] for r in odom_rows), dtype=np.int64)
    # Ego poses are interpolated between odom samples, so a gap in odom is
    # invisible in the output; report it here and let screen_bags.py judge it.
    odom_max_gap_ms = float(np.diff(odom_ts).max() / 1e6) if len(odom_ts) > 1 else 0.0
    imu = np.array(sorted(imu_rows), dtype=np.float64)
    imu_ts, imu_accel, imu_rate, imu_hz = corrimu_to_ego(
        np.array(sorted(r[0] for r in imu_rows), dtype=np.int64), imu[:, 1], imu[:, 2],
        imu[:, 3], imu[:, 4], imu[:, 5], imu[:, 6], imu[:, 7])
    gnss = np.array(sorted(ins_rows), dtype=np.float64).reshape(-1, 5)
    ins = InsData(
        odom_ts=odom_ts, odom_t=odom[:, 1:4], odom_q=odom[:, 4:8], odom_vel=odom[:, 8:11],
        imu_ts=imu_ts, imu_accel=imu_accel, imu_rate=imu_rate, imu_hz=imu_hz,
        gnss_ts=np.array(sorted(r[0] for r in ins_rows), dtype=np.int64),
        gnss_llh=gnss[:, 1:4], gnss_status=gnss[:, 4].astype(np.int64),
    )
    data = SensorData(
        calib=calib,
        frames={ch: np.array(sorted(v), dtype=np.int64) for ch, v in frames.items()},
        modality=modality,
        intrinsic={ch: r.K_new.tolist() for ch, r in rectifiers.items()},
        cam_size=cam_size,
        odom_ts=odom_ts,
        odom_t=odom[:, 1:4],
        odom_R=Rotation.from_quat(odom[:, [5, 6, 7, 4]]),  # wxyz -> xyzw
        bag_start_ns=bag_start_ns,
        bag_end_ns=bag_end_ns,
    )
    stats = {
        "lidar_topic": sorted(lidar_topics_seen),
        "lidar_time_base": ("bag_receive" if LIDAR_PACKETS_TOPIC in lidar_topics_seen
                            else "lidar_header_stamp (sweep start)"),
        "msop_skipped_before_calib": n_msop_skipped,
        "n_frames": {ch: len(v) for ch, v in frames.items()},
        "n_odom_samples": len(odom_rows),
        "odom_max_gap_ms": odom_max_gap_ms,
        "n_inspva_samples": len(ins_rows),
        "imu_hz": imu_hz,
    }
    return BagContents(data, ins, rectifiers, sidecar_topics, stats)


# ------------------------------------------------------------ materializing
def _ensure_placeholder_map(out_root: Path) -> None:
    """devkit's render_sample loads a map raster; give it a tiny valid PNG."""
    map_path = out_root / "maps" / "placeholder.png"
    if map_path.exists():
        return
    map_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(map_path), np.full((100, 100), 255, dtype=np.uint8))


def _radar_sensor_velocity(ins: InsData, calib: dict, t_ns: int) -> np.ndarray:
    """Velocity of the radar itself at t_ns, in the radar frame."""
    R_radar_ego = quat_wxyz_to_R(calib["rotation"])       # OpenCV extrinsic: P_r = R P_e + t
    p_radar_ego = -R_radar_ego.T @ np.asarray(calib["translation"], dtype=np.float64)
    v, w = ins.ego_motion_at(t_ns)
    return R_radar_ego @ (v + np.cross(w, p_radar_ego))


def materialize(plan: list[tuple[int, str, str]], contents: BagContents,
                staging: Path, out_root: Path) -> dict:
    """Move each staged frame to its NuScenes path (radar: convert to PCD).

    A rename, not a copy: staging lives inside out_root so this is the same
    filesystem, and the data is never written twice.
    """
    _ensure_placeholder_map(out_root)
    modality = contents.data.modality
    n_moved = n_missing = n_time = 0
    for ts_ns, channel, rel_target in plan:
        target = out_root / rel_target
        if target.exists():
            continue
        mod = modality[channel]
        src = staging / channel / f"{ts_ns}{STAGE_EXT[mod]}"
        if not src.exists():
            n_missing += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if mod == "radar":
            det = np.fromfile(src, dtype=np.float32).reshape(-1, 5)
            v_s = _radar_sensor_velocity(contents.ins, contents.data.calib[channel], ts_ns)
            target.write_bytes(radar_to_pcd(det, v_s))
            src.unlink()
        else:
            src.rename(target)
            if mod == "lidar":
                t_src = staging / channel / f"{ts_ns}{POINT_TIME_EXT}"
                if t_src.exists():
                    t_src.rename(target.with_name(
                        target.name.removesuffix(STAGE_EXT["lidar"]) + POINT_TIME_EXT))
                    n_time += 1
        n_moved += 1
    print(f"  moved {n_moved} files into place"
          + (f" (+{n_time} per-point time files)" if n_time else "")
          + (f"   [!] {n_missing} staged files missing" if n_missing else ""))
    return {"n_files": n_moved, "n_point_time_files": n_time, "n_missing": n_missing}


def export_sidecars(staging: Path, out_root: Path, spans: list[tuple[str, int, int]],
                    topics: dict[str, dict]) -> int:
    """Split every staged sidecar topic into ext/<scene>/<topic>.json.

    A scene gets the messages whose utime lies within its span plus
    CAN_MARGIN_NS, as JSON arrays of {"utime", "bag_utime", "msg"}.
    """
    starts = np.array([s - CAN_MARGIN_NS for _, s, _ in spans], dtype=np.int64)
    ends = np.array([e + CAN_MARGIN_NS for _, _, e in spans], dtype=np.int64)
    n_written = 0
    for slug in sorted(topics):
        # Streamed straight into one open file per scene: a raw CAN topic runs
        # to millions of lines on a long bag.
        outs: dict[int, object] = {}
        try:
            with open(staging / "_ext" / f"{slug}.jsonl") as f:
                for line in f:
                    # Lines start with {"utime":<int>, — read it without a full parse.
                    t_ns = int(line[9:line.index(",", 9)]) * 1000
                    for k in np.flatnonzero((starts <= t_ns) & (ends >= t_ns)):
                        o = outs.get(int(k))
                        if o is None:
                            d = out_root / "ext" / spans[k][0]
                            d.mkdir(parents=True, exist_ok=True)
                            o = outs[int(k)] = open(d / f"{slug}.json", "w")
                            o.write("[")
                        else:
                            o.write(",")
                        o.write(line.rstrip("\n"))
        finally:
            for o in outs.values():
                o.write("]")
                o.close()
        n_written += len(outs)
    for name, _, _ in spans:
        d = out_root / "ext" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "_index.json").write_text(json.dumps(
            {slug: {"topic": i["topic"], "msgtype": i["msgtype"]}
             for slug, i in sorted(topics.items())}, indent=1))
    return n_written


# ---------------------------------------------------------------------- CLI
def _parser(profile: Profile, doc: str) -> argparse.ArgumentParser:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=doc.split("\n")[0])
    p.add_argument("bags", type=Path, nargs="+", metavar="BAG",
                   help=".bag files, or directories: every *.bag under them, in name "
                        "order. Bags already in the dataset are skipped.")
    p.add_argument("--out", type=Path, default=profile.default_out,
                   help=f"NuScenes dataroot (default: {profile.default_out}).")
    p.add_argument("--calib", type=Path, default=None,
                   help="Calibration snapshot directory (not shipped with this repo; "
                        "see README 'Calibration'). Channels it lacks — all of them "
                        "when omitted — get defaults: identity extrinsic, 90° pinhole, "
                        "no distortion.")
    p.add_argument("--split", choices=("train", "val"), default="train",
                   help="Official nuScenes split this bag's scenes are named into "
                        "(default: train). Keep all scenes of a bag in one split.")
    p.add_argument("--version", default="v1.0-trainval",
                   help="NuScenes version subdirectory (default: v1.0-trainval).")
    p.add_argument("--keyframe-stride", type=int, default=5,
                   help="Every Kth LiDAR frame is a sample (5 = 2 Hz from 10 Hz).")
    p.add_argument("--sync-ms", type=float, default=25.0,
                   help="Max |camera - LiDAR| for the camera set of a frame (default 25).")
    p.add_argument("--scene-dur", type=float, default=20.0,
                   help="Scene length in seconds (default 20). Every scene is "
                        "exactly this long; shorter remainders are not used.")
    p.add_argument("--camera-group-ms", type=float, default=5.0,
                   help="Camera stamps closer than this belong to one capture "
                        "instant (default 5; the cameras share a trigger).")
    p.add_argument("--rectify-balance", type=float, default=0.0,
                   help="0 = crop to valid pixels, 1 = keep the whole field of "
                        "view with black corners (default 0).")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="JPEG quality of rectified images (default 90, about the "
                        "size of the camera's own JPEGs).")
    p.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) - 2),
                   help="Threads for rectification and file writes.")
    p.add_argument("--packet-msg-dir", type=Path,
                   default=here / "packet_decoder" / "src" / "rslidar_msg" / "msg")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the NuScenes(...) / NuScenesCanBus load check at the end.")
    p.add_argument("--keep-staging", action="store_true",
                   help="Leave the staging directory for debugging.")
    return p


def _expand_bags(paths: list[Path]) -> list[Path]:
    """Files as given, directories as every *.bag under them (sorted), no repeats."""
    out: list[Path] = []
    for path in paths:
        found = sorted(path.rglob("*.bag")) if path.is_dir() else [path]
        if path.is_dir() and not found:
            print(f"  [!] no .bag files under {path}")
        out.extend(f for f in found if f not in out)
    return out


def _imported_logs(json_dir: Path) -> set[str]:
    f = json_dir / "log.json"
    return {r["logfile"] for r in json.loads(f.read_text())} if f.exists() else set()


def run(profile: Profile, doc: str, argv: list[str] | None = None) -> None:
    """Parse the CLI and convert each bag into args.out, appending to what is there.

    Bags are converted one after another; one that is already in the dataset
    (same file name) is skipped, and one that fails is reported and the rest
    still run. The devkit check runs once, at the end.

    Runs into one dataroot take turns: the tables are read at the start and
    rewritten at the end of every bag, and the staging directory is shared, so
    a second process on the same root is refused while the first holds the lock
    file.
    """
    args = _parser(profile, doc).parse_args(argv)
    for path, label in [*((b, "bag") for b in args.bags), (args.calib, "calib")]:
        if path is not None and not path.exists():
            raise SystemExit(f"{label} not found: {path}")
    bags = _expand_bags(args.bags)
    if not bags:
        raise SystemExit("no .bag files to convert")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"{lock} exists: another conversion is writing to {args.out} "
            f"({lock.read_text().strip() or 'unknown'}). Runs into one dataroot must "
            "take turns. If none is running — a killed run leaves the file behind — "
            "delete it.") from None
    with os.fdopen(fd, "w") as f:
        f.write(f"pid {os.getpid()} converting {', '.join(map(str, args.bags))}\n")

    results: list[tuple[Path, str, str]] = []    # (bag, status, detail)
    first_scene = None
    try:
        for i, bag in enumerate(bags, 1):
            if len(bags) > 1:
                print(f"\n{'=' * 78}\n[{i}/{len(bags)}] {bag}")
            if bag.stem in _imported_logs(args.out / args.version):
                print(f"  already in the dataset (log '{bag.stem}') — skipped")
                results.append((bag, "skipped", "already imported"))
                continue
            try:
                names = _convert(profile, args, bag)
            except (Exception, SystemExit) as exc:
                if len(bags) == 1:
                    raise
                detail = str(exc) or type(exc).__name__
                print(f"  !! {bag.name} failed: {detail}")
                results.append((bag, "failed", detail))
                continue
            first_scene = first_scene or names[0]
            results.append((bag, "converted", f"{len(names)} scenes "
                                               f"({names[0]} .. {names[-1]})"))
        if args.no_validate or first_scene is None:
            print("\n(validation skipped)")
        else:
            print("\nValidating with nuscenes-devkit...")
            validate_with_devkit(args.out, args.version)
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus
            pose = NuScenesCanBus(dataroot=str(args.out)).get_messages(first_scene, "pose")
            print(f"  ✓ NuScenesCanBus: {first_scene} pose has {len(pose)} messages")
    finally:
        lock.unlink(missing_ok=True)

    if len(bags) > 1:
        print(f"\n{'=' * 78}\nSUMMARY  -> {args.out}")
        for bag, status, detail in results:
            print(f"  {status:10} {bag.name}  {detail}")
    print("\nDone.")
    if any(status == "failed" for _, status, _ in results):
        raise SystemExit(1)


def _convert(profile: Profile, args: argparse.Namespace, bag: Path) -> list[str]:
    """Convert one bag into args.out; returns the new scene names."""
    cv2.setNumThreads(1)   # parallelism comes from the staging pool

    log_name = bag.stem
    json_dir = args.out / args.version
    existing = load_existing_tables(json_dir)
    if existing and any(r.get("logfile") == log_name
                        for r in existing.get("log.json", [])):
        raise SystemExit(
            f"log '{log_name}' is already in {json_dir}/log.json — "
            "remove that log entry first if you mean to re-import it."
        )

    # ------------------------------------------------ channels and calibration
    gating = list(NUSCENES_CAMS)
    optional = ([c for c in profile.cameras if c not in gating]
                + list(profile.extra_lidars.values()) + list(profile.radars.values()))
    channels = ["LIDAR_TOP", *gating, *optional]
    calib, defaulted = resolve_calib(args.calib, channels)
    if defaulted:
        print(f"  [!] no calibration{'' if args.calib is None else f' in {args.calib}'} for "
              f"{', '.join(defaulted)} — using defaults: identity extrinsic, 90° pinhole K, "
              "no distortion (not for geometry; these cameras are not rectified)")

    staging = args.out / STAGING_DIRNAME
    if staging.exists():
        shutil.rmtree(staging)

    try:
        print(f"[1/5] Reading {bag.name} ({profile.name}: {profile.description}) ...")
        contents = read_bag(bag, staging, profile, channels, calib,
                            args.packet_msg_dir, args.rectify_balance,
                            args.jpeg_quality, args.workers)
        data, stats = contents.data, contents.stats
        for ch in channels:
            print(f"  {ch:20} {len(data.frames[ch]):>7} frames")
        print(f"  odom: {len(data.odom_ts)} samples, max gap {stats['odom_max_gap_ms']:.0f} ms"
              + ("   [!] ego pose is interpolated across gaps — run "
                 "scripts/screen_bags.py" if stats["odom_max_gap_ms"] > 500 else ""))
        print(f"  IMU: {stats['imu_hz']:.1f} Hz (CORRIMU), INSPVA: {stats['n_inspva_samples']} samples")
        for ch in gating:
            if not len(data.frames[ch]):
                raise SystemExit(f"no frames on required camera channel {ch}")
        absent = [ch for ch in optional if ch in channels and not len(data.frames[ch])]
        if absent:
            print(f"  [!] not in this bag, left out: {absent}")
            channels = [c for c in channels if c not in absent]

        # ---------------------------------------------------- frame selection
        sync_ns = int(args.sync_ms * 1e6)
        lidar_ts = data.frames["LIDAR_TOP"]
        lidar_period = float(np.median(np.diff(lidar_ts)))
        print(f"\n[2/5] Frame selection (camera set within {args.sync_ms:g} ms of LIDAR_TOP) ...")
        window = coverage_window(
            required_streams(lidar_ts, data.frames, gating, data.odom_ts), sync_ns)
        for line in format_coverage(window):
            print("  " + line)
        if window["end_ns"] <= window["start_ns"]:
            raise SystemExit("required streams (lidar, standard cameras, odom) "
                             "do not overlap in time — nothing to convert")
        plan = plan_frames(lidar_ts, data.frames, gating, sync_ns, window,
                           group_ns=int(args.camera_group_ms * 1e6),
                           max_gap_ns=int(1.5 * lidar_period))
        for line in format_plan(plan):
            print("  " + line)

        print(f"\n[3/5] Scenes ({args.scene_dur:g} s each) ...")
        frames_per_scene = int(round(args.scene_dur * 1e9 / lidar_period))
        cut, part_stats = partition_scenes(plan["segments"], frames_per_scene)
        if not cut:
            raise SystemExit(f"no unbroken run of {args.scene_dur:g} s — no scenes")
        used = {s["name"] for s in (existing or {}).get("scene.json", [])}
        names = official_scene_names(args.split, used, len(cut))
        scenes = []
        for name, (seg_k, idx) in zip(names, cut):
            kf_idx = idx[::args.keyframe_stride]
            t0 = (lidar_ts[idx[0]] - data.bag_start_ns) / 1e9
            scenes.append({
                "name": name,
                "description": f"{log_name} segment {seg_k}, {t0:.1f}-"
                               f"{t0 + args.scene_dur:.1f} s into the bag",
                "keyframes": [{"lidar_ts": int(lidar_ts[i]),
                               "cam_ts": {ch: int(plan["cam"][ch][i]) for ch in gating}}
                              for i in kf_idx],
            })
        print(f"  {len(scenes)} scenes, {len(scenes[0]['keyframes'])} samples each "
              f"({names[0]} .. {names[-1]}, official '{args.split}' list); "
              f"{part_stats['frames_unused'] * lidar_period / 1e9:.1f} s of usable "
              f"frames left over in segment remainders")

        print("\n[4/5] Building tables ...")
        if existing:
            print(f"  append mode: {len(existing['scene.json'])} existing scenes")
        kf_tol_ns = {}
        for ch in optional:
            if ch not in channels:
                continue
            if data.modality[ch] == "camera":
                kf_tol_ns[ch] = sync_ns
            else:
                f_ts = data.frames[ch]
                kf_tol_ns[ch] = int(np.median(np.diff(f_ts)) / 2) if len(f_ts) > 1 else 0
        tables, file_plan, attach = build_tables(
            data, scenes, channels, gating, kf_tol_ns,
            log_token=new_token(), log_name=log_name, existing=existing)
        for k, v in tables.items():
            print(f"  + {k:28} {len(v):>8}")
        n_samples = sum(len(s["keyframes"]) for s in scenes)
        for ch, n in attach.items():
            print(f"  {ch}: in {n}/{n_samples} samples")

        print(f"\n[5/5] Writing under {args.out} ...")
        mat_stats = materialize(file_plan, contents, staging, args.out)
        spans = [(s["name"], s["keyframes"][0]["lidar_ts"], s["keyframes"][-1]["lidar_ts"])
                 for s in scenes]
        for name, a, b in spans:
            write_scene(args.out / "can_bus", name, scene_messages(contents.ins, a, b))
        print(f"  can_bus: pose, ms_imu, meta for {len(spans)} scenes")
        n_sidecar = 0
        if profile.sidecars:
            n_sidecar = export_sidecars(staging, args.out, spans, contents.sidecar_topics)
            print(f"  ext: {len(contents.sidecar_topics)} topics -> {n_sidecar} per-scene files")
        if existing:
            tables = merge_tables(existing, tables)
        write_tables(tables, args.out, args.version)
        import_record = {
            "tool": profile.name,
            "source_bag": str(bag.resolve()),
            "calib_source": str(args.calib.resolve()) if args.calib else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "scenes": [{"name": s["name"], "description": s["description"]} for s in scenes],
            "channels": channels,
            "calib_defaulted": defaulted,
            "coverage_window": window,
            "frame_plan": plan["stats"],
            "scene_partition": part_stats,
            "best_effort_attached": attach,
            "rectification": {ch: r.describe() for ch, r in contents.rectifiers.items()},
            "calib": calib,
            "files": mat_stats,
            "sidecar_topics": contents.sidecar_topics if profile.sidecars else {},
            **stats,
        }
        (args.out / f"{log_name}.import.json").write_text(json.dumps(import_record, indent=2))
        print(f"  tables -> {json_dir}")
    finally:
        if staging.exists() and not args.keep_staging:
            shutil.rmtree(staging, ignore_errors=True)

    return [sc["name"] for sc in scenes]
