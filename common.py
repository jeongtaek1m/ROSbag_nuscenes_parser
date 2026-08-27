"""Definitions shared by the conversion stages and the diagnostic scripts.

Kept deliberately flat: this project is an offline batch converter that is run
as `python <stage>.py`, not an installed library, so there is no package to
import from. Scripts under `scripts/` reach these helpers with

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import stamp_to_ns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------- topic layout
# Verified against scripts/extract_cam_viz.py output (see docs/pipeline_overview.md);
# the channel names are the NuScenes ones, the topics are what the vehicle publishes.
TOPIC_TO_CAM_CHANNEL = {
    "/camera_4/compressed": "CAM_FRONT",
    "/camera_1/compressed": "CAM_FRONT_RIGHT",
    "/camera_6/compressed": "CAM_FRONT_LEFT",
    "/camera_2/compressed": "CAM_BACK",
    "/camera_0/compressed": "CAM_BACK_RIGHT",
    "/camera_5/compressed": "CAM_BACK_LEFT",
    "/camera_3/compressed": "CAM_TRAFFIC",
}
CAM_CHANNEL_TO_TOPIC = {v: k for k, v in TOPIC_TO_CAM_CHANNEL.items()}

# The six channels a standard NuScenes consumer expects, plus our extra one.
NUSCENES_CAMS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
EXTRA_CAMS = ["CAM_TRAFFIC"]

ODOM_TOPIC = "/novatel/oem7/odom"
# Carries GPS week/ms internally, so it is the only absolute time reference in
# a bag — used by scripts/clock_diagnosis.py to anchor the host clock.
INSPVA_TOPIC = "/novatel/oem7/inspva"
ANNOTATION_TOPIC = "/post_fusion_object"
LIDAR_PACKETS_TOPIC = "/middle/rslidar_packets"
LIDAR_POINTS_TOPIC = "/middle/rslidar_points"


# ------------------------------------------------------------------ rosbag io
def stamp_to_ns(stamp) -> int:
    """std_msgs/Header stamp -> integer nanoseconds, for ROS1 and ROS2 field names."""
    nsec = getattr(stamp, "nanosec", None)
    if nsec is None:
        nsec = getattr(stamp, "nsec", 0)
    return int(stamp.sec) * 1_000_000_000 + int(nsec)


def make_typestore(*msg_dirs: tuple[Path, str]):
    """Build a ROS1 typestore with custom .msg definitions registered.

    Each argument is a (directory, package_name) pair, e.g.
    (Path("msg"), "data_processing") or (.../rslidar_msg/msg, "rslidar_msg").
    Missing directories are skipped so callers can pass optional ones.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)
    add_types: dict = {}
    for msg_dir, package in msg_dirs:
        if msg_dir is None or not Path(msg_dir).exists():
            continue
        for msg_path in sorted(Path(msg_dir).glob("*.msg")):
            add_types.update(
                get_types_from_msg(msg_path.read_text(),
                                   f"{package}/msg/{msg_path.stem}")
            )
    typestore.register(add_types)
    return typestore


# ----------------------------------------------------------------- geometry
def quat_wxyz_to_R(q) -> np.ndarray:
    """Quaternion [w, x, y, z] -> 3x3 rotation matrix (scipy wants [x, y, z, w])."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def R_to_quat_wxyz(R) -> list[float]:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


def opencv_ext_to_nuscenes_pose(R_cam_ego, t_cam_ego) -> tuple[list[float], list[float]]:
    """OpenCV extrinsic (P_cam = R @ P_ego + t) -> NuScenes sensor-in-ego pose.

    NuScenes stores where the sensor sits in the ego frame, which is the inverse
    of the projection extrinsic our calibration files hold.
    """
    R_ego_cam = np.asarray(R_cam_ego).T
    t_ego_cam = (-R_ego_cam @ np.asarray(t_cam_ego)).tolist()
    return R_to_quat_wxyz(R_ego_cam), t_ego_cam


# -------------------------------------------------------------- calibration
def load_calib(calib_dir: Path) -> dict:
    """Read a calibration snapshot directory into the dict stored as calib.json.

    Per camera: intrinsic.txt (3x3 K), distortion.txt, quat_r.txt (w,x,y,z) and
    t.txt, both in the OpenCV extrinsic convention. The number of distortion
    coefficients selects the projection model: 4 -> OpenCV fisheye (equidistant),
    5 -> plumb_bob pinhole. LIDAR_TOP carries r.txt (also a w,x,y,z quaternion)
    and t.txt.
    """
    calib_dir = Path(calib_dir)
    out: dict = {}
    for cam in NUSCENES_CAMS:
        d = calib_dir / cam
        distortion = np.loadtxt(d / "distortion.txt").reshape(-1).tolist()
        if len(distortion) == 4:
            model = "fisheye"
        elif len(distortion) == 5:
            model = "pinhole"
        else:
            model = "unknown"
        out[cam] = {
            "intrinsic": np.loadtxt(d / "intrinsic.txt").tolist(),
            "distortion": distortion,
            "model": model,
            "rotation": np.loadtxt(d / "quat_r.txt").reshape(-1).tolist(),
            "translation": np.loadtxt(d / "t.txt").reshape(-1).tolist(),
        }
    lidar_d = calib_dir / "LIDAR_TOP"
    out["LIDAR_TOP"] = {
        "rotation": np.loadtxt(lidar_d / "r.txt").reshape(-1).tolist(),
        "translation": np.loadtxt(lidar_d / "t.txt").reshape(-1).tolist(),
    }
    return out


# ------------------------------------------------------ timestamp collection
def collect_header_timestamps(bag_path: Path, typestore, max_seconds: float = 0.0
                              ) -> dict[str, np.ndarray]:
    """Per-channel header.stamp arrays (ns) for the cameras and LIDAR_TOP.

    Reads header stamps, not bag arrival times: those are the timestamps the
    converter synchronizes on. Returns channel -> sorted int64 array.
    """
    from rosbags.highlevel import AnyReader  # local: keeps the import optional

    lidar_topics = (LIDAR_POINTS_TOPIC, LIDAR_PACKETS_TOPIC)
    out: dict[str, list[int]] = {}
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        wanted = set(TOPIC_TO_CAM_CHANNEL) | set(lidar_topics)
        conns = [c for c in reader.connections if c.topic in wanted]
        if not conns:
            raise SystemExit(f"no camera or lidar topics in {bag_path}")
        t_end = (reader.start_time + int(max_seconds * 1e9)) if max_seconds else None
        for conn, bag_ns, raw in reader.messages(connections=conns):
            if t_end and bag_ns > t_end:
                break
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic in TOPIC_TO_CAM_CHANNEL:
                ch = TOPIC_TO_CAM_CHANNEL[conn.topic]
                out.setdefault(ch, []).append(stamp_to_ns(msg.header.stamp))
            elif conn.topic == LIDAR_POINTS_TOPIC:
                out.setdefault("LIDAR_TOP", []).append(stamp_to_ns(msg.header.stamp))
            else:
                # Packet bags have no per-frame stamp until the sweep is decoded;
                # bag arrival time is what the converter uses there too.
                out.setdefault("LIDAR_TOP", []).append(int(bag_ns))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


# ------------------------------------------------------------ label taxonomy
# Derived from msg/ObjectType.msg so the perception enum and the labelling
# taxonomy cannot drift apart. The names follow the NuScenes dotted convention
# (family.thing) because downstream tooling splits on the first component.
# The integer key is the `type` field carried in ObjectFusion.
OBJECT_TYPE_TO_CATEGORY = {
    0:  ("unknown",                         "Unclassified object."),
    1:  ("vehicle.car",                     "Passenger car, SUV, van."),
    2:  ("vehicle.bus",                     "Bus or coach."),
    3:  ("vehicle.truck",                   "Truck or lorry."),
    4:  ("vehicle.construction",            "Construction vehicle."),
    5:  ("vehicle.bicycle",                 "Bicycle, with or without rider."),
    6:  ("vehicle.tricycle",                "Three-wheeled vehicle."),
    7:  ("human.pedestrian",                "Pedestrian."),
    8:  ("movable_object.trafficcone",      "Traffic cone."),
    9:  ("movable_object.barrow",           "Hand cart or barrow."),
    10: ("animal",                          "Animal other than a bird."),
    11: ("movable_object.warning_triangle", "Roadside warning triangle."),
    12: ("animal.bird",                     "Bird."),
    13: ("movable_object.water_barrier",    "Water-filled barrier."),
    14: ("static_object.lamp_post",         "Lamp or utility post."),
    15: ("static_object.traffic_sign",      "Traffic sign."),
    16: ("static_object.warning_post",      "Warning post or delineator."),
    17: ("movable_object.traffic_barrel",   "Traffic barrel."),
    18: ("vehicle.articulated_head",        "Tractor unit of an articulated vehicle."),
    19: ("vehicle.articulated_body",        "Trailer of an articulated vehicle."),
    20: ("vision_obstacle",                 "Obstacle detected by vision only."),
    50: ("static_object.unknown",           "Unclassified static object."),
}

# Derived from msg/MotionType.msg.
MOTION_TYPE_TO_ATTRIBUTE = {
    0: ("motion.unknown",       "Motion state could not be determined."),
    1: ("motion.stationary",    "Never observed to move."),
    2: ("motion.stopped",       "Temporarily stopped but able to move."),
    3: ("motion.moving_slowly", "Moving at or below 5 km/h."),
    4: ("motion.moving",        "Moving above 5 km/h."),
}

# NuScenes' four standard visibility bins (fraction of the object visible across
# all camera channels). Kept verbatim so devkit filters written for NuScenes work.
VISIBILITY_LEVELS = [
    ("v0-40",   "Between 0 and 40% visible."),
    ("v40-60",  "Between 40 and 60% visible."),
    ("v60-80",  "Between 60 and 80% visible."),
    ("v80-100", "Between 80 and 100% visible."),
]
