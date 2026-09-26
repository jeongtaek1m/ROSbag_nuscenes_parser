"""GNSS/INS written as nuScenes CAN bus expansion files.

nuscenes-devkit reads these with

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    NuScenesCanBus(dataroot=...).get_messages("scene-0001", "pose")

which looks for `<dataroot>/can_bus/<scene name>_<message>.json` and only
accepts the nuScenes message names. Two of them carry what the INS measures:

  pose    one record per INS pose sample (INSPVA, 100 Hz; nuScenes has 50 Hz)
    utime          microseconds, same clock as sample_data
    pos            global frame, identical to ego_pose (common.global_frame_id: local
                   east-north-up at the location's fixed origin, metres)
    orientation    w, x, y, z — identical to ego_pose
    vel            ego frame, m/s                      (INSPVA velocity)
    accel          ego frame, m/s^2, gravity removed   (CORRIMU)
    rotation_rate  ego frame, rad/s                    (CORRIMU)
    lat, lon, height, ins_status   extra keys from INSPVA (WGS84 degrees,
                   ellipsoidal metres, InertialSolutionStatus; 3 = SOLUTION_GOOD)
  ms_imu  one record per CORRIMU sample (100 Hz)
    utime, linear_accel (m/s^2), rotation_rate (rad/s) — ego-aligned axes
    q              w, x, y, z: IMU (ego) -> global, i.e. the ego_pose orientation

Times are the receiver's GPS measurement times (common.novatel_gps_ns), not the
header stamps, which are arrival times.

Unlike nuScenes' ms_imu, linear_accel has gravity removed: CORRIMU is
gravity-compensated by the INS and nothing is added back. `meta` records this.

Ego frame: x forward, y left, z up (base_link, the frame ego_pose describes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common import (
    GEOID_UNDULATION,
    GLOBAL_ORIGINS,
    UTM_ZONE,
    geodetic_to_enu,
    global_frame_id,
    novatel_attitude,
    novatel_gps_ns,
    stamp_to_ns,
    utm_to_geodetic,
)

# Records whose utime lies this far outside a scene are still written with it,
# so interpolation at the first and last sample has data on both sides.
CAN_MARGIN_NS = 500_000_000


def corrimu_to_ego(t_ns: np.ndarray, count: np.ndarray, pitch_rate: np.ndarray,
                   roll_rate: np.ndarray, yaw_rate: np.ndarray, lateral_acc: np.ndarray,
                   longitudinal_acc: np.ndarray, vertical_acc: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """NovAtel CORRIMU -> (t_ns, accel_ego (M,3) m/s^2, rate_ego (M,3) rad/s, imu_hz).

    CORRIMU carries increments summed over `imu_data_count` IMU samples, in
    NovAtel's vehicle frame (x right, y forward, z up); rates are value / count
    * IMU rate. The IMU rate is not in the message, so it is measured: samples
    accumulated per second over the log (125 Hz on the T-Car's IMU, logged at
    100 Hz, hence counts of 1 and 2). Checked on the 2026-09-23 bags against
    /bsw/vehicle_can yaw rate and the derivative of odom speed.
    """
    ok = count > 0
    t = t_ns[ok]
    if len(t) < 2:
        raise ValueError("CORRIMU: fewer than two usable samples")
    imu_hz = float(count[ok][1:].sum() / ((t[-1] - t[0]) / 1e9))
    k = (imu_hz / count[ok])[:, None]
    accel = np.stack([longitudinal_acc, -lateral_acc, vertical_acc], axis=1)[ok] * k
    rate = np.stack([roll_rate, -pitch_rate, yaw_rate], axis=1)[ok] * k
    return t, accel, rate, imu_hz


def pose_from_inspva(lat, lon, height, v_north, v_east, v_up, roll, pitch, azimuth, origin
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INSPVA samples -> (global position (N,3), orientation w,x,y,z (N,4),
    velocity in the ego frame (N,3)). `origin` is the global frame's
    (lat, lon, height); INSPVA height is ellipsoidal, as the frame needs."""
    pos = geodetic_to_enu(lat, lon, height, origin)
    rot = novatel_attitude(roll, pitch, azimuth)
    vel = rot.inv().apply(np.stack([v_east, v_north, v_up], axis=1))
    return pos, rot.as_quat()[:, [3, 0, 1, 2]], vel


def pose_from_odom(utm_xyz: np.ndarray, q_wxyz: np.ndarray, vel_ego: np.ndarray, origin,
                   zone: int, undulation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """/novatel/oem7/odom samples -> the same as pose_from_inspva, for bags without
    INSPVA. odom holds UTM positions with sea-level height (ellipsoidal minus the
    geoid `undulation`) and an orientation already in true-north ENU. Its position
    can freeze for a second at a time (a 1 Hz source), so this is a fallback only."""
    lat, lon = utm_to_geodetic(utm_xyz[:, 0], utm_xyz[:, 1], zone)
    return geodetic_to_enu(lat, lon, utm_xyz[:, 2] + undulation, origin), q_wxyz, vel_ego


@dataclass
class InsData:
    pose_ts: np.ndarray        # int64 ns, sorted
    pose_t: np.ndarray         # (N, 3) global position
    pose_q: np.ndarray         # (N, 4) w, x, y, z
    pose_vel: np.ndarray       # (N, 3) ego frame
    pose_source: str           # "inspva" | "odom"
    imu_ts: np.ndarray         # int64 ns, sorted
    imu_accel: np.ndarray      # (M, 3) ego frame
    imu_rate: np.ndarray       # (M, 3) ego frame
    imu_hz: float
    gnss_ts: np.ndarray        # int64 ns (INSPVA), sorted; may be empty
    gnss_llh: np.ndarray       # (K, 3) lat, lon, height
    gnss_status: np.ndarray    # (K,) InertialSolutionStatus
    global_frame: str = ""     # common.global_frame_id
    _slerp: Slerp | None = field(default=None, repr=False)

    def orientation_at(self, t_ns: np.ndarray) -> np.ndarray:
        """(n, 4) w, x, y, z at t_ns, which must lie inside pose coverage."""
        if self._slerp is None:
            self._slerp = Slerp(self.pose_ts.astype(np.float64),
                                Rotation.from_quat(self.pose_q[:, [1, 2, 3, 0]]))
        xyzw = self._slerp(np.asarray(t_ns, dtype=np.float64)).as_quat()
        return xyzw[:, [3, 0, 1, 2]]

    def ego_motion_at(self, t_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """(velocity, angular rate) of the ego frame at t_ns, both in the ego frame."""
        v = np.array([np.interp(t_ns, self.pose_ts, self.pose_vel[:, i]) for i in range(3)])
        w = np.array([np.interp(t_ns, self.imu_ts, self.imu_rate[:, i]) for i in range(3)])
        return v, w


# ------------------------------------------------ rows from the bag's messages
def inspva_row(msg) -> tuple:
    """(GPS time ns, header ns, lat, lon, height, v north, v east, v up, roll, pitch,
    azimuth, status) of a novatel_oem7_msgs/INSPVA."""
    return (novatel_gps_ns(msg.nov_header), stamp_to_ns(msg.header.stamp),
            msg.latitude, msg.longitude, msg.height,
            msg.north_velocity, msg.east_velocity, msg.up_velocity,
            msg.roll, msg.pitch, msg.azimuth, int(msg.status.status))


def odom_row(msg) -> tuple:
    """(header ns, UTM x, y, sea-level z, q w, x, y, z, ego velocity x, y, z) of the odom."""
    p_, q_, v_ = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
    return (stamp_to_ns(msg.header.stamp), p_.x, p_.y, p_.z, q_.w, q_.x, q_.y, q_.z,
            v_.x, v_.y, v_.z)


def corrimu_row(msg) -> tuple:
    """(GPS time ns, count, pitch rate, roll rate, yaw rate, lateral, longitudinal,
    vertical acc) of a novatel_oem7_msgs/CORRIMU."""
    return (novatel_gps_ns(msg.nov_header), msg.imu_data_count, msg.pitch_rate, msg.roll_rate,
            msg.yaw_rate, msg.lateral_acc, msg.longitudinal_acc, msg.vertical_acc)


def _unique_sorted(rows: list[tuple]) -> list[tuple]:
    """One row per time (a message recorded twice is dropped); SLERP needs the
    times strictly increasing."""
    return sorted({r[0]: r for r in rows}.values())


def build_ins(ins_rows: list[tuple], odom_rows: list[tuple], imu_rows: list[tuple],
              location: str) -> tuple[InsData, dict]:
    """The INS streams of one bag -> (InsData, stats). The ego pose comes from INSPVA
    at GPS time; odom is used only when a bag has no INSPVA."""
    if not ins_rows and not odom_rows:
        raise ValueError("no INSPVA and no odom: ego_pose cannot be built")
    if not imu_rows:
        raise ValueError("no CORRIMU: the CAN bus pose (accel, rotation_rate) cannot be built")
    origin, frame_id = GLOBAL_ORIGINS[location], global_frame_id(location)
    ins_rows, odom_rows, imu_rows = map(_unique_sorted, (ins_rows, odom_rows, imu_rows))
    gnss = np.array([r[1:] for r in ins_rows], dtype=np.float64).reshape(-1, 11)
    gnss_ts = np.array([r[0] for r in ins_rows], dtype=np.int64)
    if ins_rows:
        source, pose_ts = "inspva", gnss_ts
        pose_t, pose_q, pose_vel = pose_from_inspva(
            gnss[:, 1], gnss[:, 2], gnss[:, 3], gnss[:, 4], gnss[:, 5], gnss[:, 6],
            gnss[:, 7], gnss[:, 8], gnss[:, 9], origin)
        lag = (gnss[:, 0] - gnss_ts) / 1e6
        lag_ms = {"median": float(np.median(lag)), "p99": float(np.percentile(lag, 99))}
    else:
        odom = np.array([r[1:] for r in odom_rows], dtype=np.float64)
        source = "odom"
        pose_ts = np.array([r[0] for r in odom_rows], dtype=np.int64)
        pose_t, pose_q, pose_vel = pose_from_odom(odom[:, 0:3], odom[:, 3:7], odom[:, 7:10], origin,
                                                  UTM_ZONE, GEOID_UNDULATION[location])
        lag_ms = None
    imu = np.array([r[1:] for r in imu_rows], dtype=np.float64)
    imu_ts, imu_accel, imu_rate, imu_hz = corrimu_to_ego(
        np.array([r[0] for r in imu_rows], dtype=np.int64), imu[:, 0], imu[:, 1],
        imu[:, 2], imu[:, 3], imu[:, 4], imu[:, 5], imu[:, 6])
    ins = InsData(
        pose_ts=pose_ts, pose_t=pose_t, pose_q=pose_q, pose_vel=pose_vel, pose_source=source,
        imu_ts=imu_ts, imu_accel=imu_accel, imu_rate=imu_rate, imu_hz=imu_hz,
        gnss_ts=gnss_ts, gnss_llh=gnss[:, 1:4], gnss_status=gnss[:, 10].astype(np.int64),
        global_frame=frame_id)
    stats = {
        "pose_source": source,
        "global_frame": frame_id,
        "n_pose_samples": len(pose_ts),
        # Ego poses are interpolated between INS samples, so a gap in them is
        # invisible in the output; report it and let screen_bags.py judge it.
        "pose_max_gap_ms": float(np.diff(pose_ts).max() / 1e6) if len(pose_ts) > 1 else 0.0,
        "ins_header_lag_ms": lag_ms,
        "n_odom_samples": len(odom_rows),
        "n_inspva_samples": len(ins_rows),
        "imu_hz": imu_hz,
    }
    return ins, stats


def _vec(rows: np.ndarray, i: int) -> list[float]:
    return [float(x) for x in rows[i]]


def scene_messages(ins: InsData, start_ns: int, end_ns: int) -> dict:
    """pose, ms_imu and meta for the interval [start_ns, end_ns] plus CAN_MARGIN_NS."""
    lo, hi = start_ns - CAN_MARGIN_NS, end_ns + CAN_MARGIN_NS
    o = np.flatnonzero((ins.pose_ts >= lo) & (ins.pose_ts <= hi))
    ts = ins.pose_ts[o]
    acc = np.stack([np.interp(ts, ins.imu_ts, ins.imu_accel[:, i]) for i in range(3)], axis=1)
    rot = np.stack([np.interp(ts, ins.imu_ts, ins.imu_rate[:, i]) for i in range(3)], axis=1)
    have_gnss = len(ins.gnss_ts) > 0
    if have_gnss:
        llh = np.stack([np.interp(ts, ins.gnss_ts, ins.gnss_llh[:, i]) for i in range(3)], axis=1)
        near = np.clip(np.searchsorted(ins.gnss_ts, ts), 0, len(ins.gnss_ts) - 1)
        status = ins.gnss_status[near]
    pose = []
    for j, i in enumerate(o):
        rec = {
            "utime": int(ts[j] // 1000),
            "pos": _vec(ins.pose_t, i),
            "orientation": _vec(ins.pose_q, i),
            "vel": _vec(ins.pose_vel, i),
            "accel": _vec(acc, j),
            "rotation_rate": _vec(rot, j),
        }
        if have_gnss:
            rec.update(lat=float(llh[j, 0]), lon=float(llh[j, 1]), height=float(llh[j, 2]),
                       ins_status=int(status[j]))
        pose.append(rec)

    # IMU samples outside pose coverage have no orientation to report.
    m = np.flatnonzero((ins.imu_ts >= max(lo, ins.pose_ts[0]))
                       & (ins.imu_ts <= min(hi, ins.pose_ts[-1])))
    q = ins.orientation_at(ins.imu_ts[m]) if len(m) else np.zeros((0, 4))
    ms_imu = [{
        "utime": int(ins.imu_ts[i] // 1000),
        "linear_accel": _vec(ins.imu_accel, i),
        "rotation_rate": _vec(ins.imu_rate, i),
        "q": _vec(q, j),
    } for j, i in enumerate(m)]

    def rate(recs: list) -> float:
        if len(recs) < 2:
            return 0.0
        return (len(recs) - 1) / ((recs[-1]["utime"] - recs[0]["utime"]) / 1e6)

    src = {"inspva": "/novatel/oem7/inspva", "odom": "/novatel/oem7/odom"}[ins.pose_source]
    meta = {
        "pose": {"message_count": len(pose), "message_freq": rate(pose),
                 "source": f"{src} (pos, orientation, vel), "
                           "/novatel/oem7/corrimu (accel, rotation_rate), "
                           "/novatel/oem7/inspva (lat, lon, height, ins_status)",
                 "time": "receiver GPS measurement time as UTC" if ins.pose_source == "inspva"
                         else "odom header stamps (arrival time)",
                 "frames": f"pos/orientation global ({ins.global_frame}: east-north-up tangent plane); "
                           "vel, accel, rotation_rate ego (x forward, y left, z up); "
                           "accel has gravity removed"},
        "ms_imu": {"message_count": len(ms_imu), "message_freq": rate(ms_imu),
                   "source": f"/novatel/oem7/corrimu, q from {src}",
                   "frames": "ego-aligned axes; linear_accel has gravity removed "
                             "(nuScenes ms_imu includes it)",
                   "imu_hz": ins.imu_hz},
    }
    return {"pose": pose, "ms_imu": ms_imu, "meta": meta}


def write_scene(can_dir: Path, scene_name: str, messages: dict) -> None:
    can_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in messages.items():
        (can_dir / f"{scene_name}_{name}.json").write_text(
            json.dumps(payload, separators=(",", ":")))
