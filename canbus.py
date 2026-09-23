"""GNSS/INS written as nuScenes CAN bus expansion files.

nuscenes-devkit reads these with

    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    NuScenesCanBus(dataroot=...).get_messages("scene-0001", "pose")

which looks for `<dataroot>/can_bus/<scene name>_<message>.json` and only
accepts the nuScenes message names. Two of them carry what the INS measures:

  pose    one record per odom sample (100 Hz; nuScenes has 50 Hz)
    utime          microseconds, same clock as sample_data
    pos            global frame, identical to ego_pose (UTM 52N, metres)
    orientation    w, x, y, z — identical to ego_pose
    vel            ego frame, m/s                      (odom twist)
    accel          ego frame, m/s^2, gravity removed   (CORRIMU)
    rotation_rate  ego frame, rad/s                    (CORRIMU)
    lat, lon, height, ins_status   extra keys from INSPVA (WGS84 degrees,
                   ellipsoidal metres, InertialSolutionStatus; 3 = SOLUTION_GOOD)
  ms_imu  one record per CORRIMU sample (100 Hz)
    utime, linear_accel (m/s^2), rotation_rate (rad/s) — ego-aligned axes
    q              w, x, y, z: IMU (ego) -> global, i.e. the odom orientation

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


@dataclass
class InsData:
    odom_ts: np.ndarray        # int64 ns, sorted
    odom_t: np.ndarray         # (N, 3) global position
    odom_q: np.ndarray         # (N, 4) w, x, y, z
    odom_vel: np.ndarray       # (N, 3) ego frame
    imu_ts: np.ndarray         # int64 ns, sorted
    imu_accel: np.ndarray      # (M, 3) ego frame
    imu_rate: np.ndarray       # (M, 3) ego frame
    imu_hz: float
    gnss_ts: np.ndarray        # int64 ns (INSPVA), sorted; may be empty
    gnss_llh: np.ndarray       # (K, 3) lat, lon, height
    gnss_status: np.ndarray    # (K,) InertialSolutionStatus
    _slerp: Slerp | None = field(default=None, repr=False)

    def orientation_at(self, t_ns: np.ndarray) -> np.ndarray:
        """(n, 4) w, x, y, z at t_ns, which must lie inside odom coverage."""
        if self._slerp is None:
            self._slerp = Slerp(self.odom_ts.astype(np.float64),
                                Rotation.from_quat(self.odom_q[:, [1, 2, 3, 0]]))
        xyzw = self._slerp(np.asarray(t_ns, dtype=np.float64)).as_quat()
        return xyzw[:, [3, 0, 1, 2]]

    def ego_motion_at(self, t_ns: int) -> tuple[np.ndarray, np.ndarray]:
        """(velocity, angular rate) of the ego frame at t_ns, both in the ego frame."""
        v = np.array([np.interp(t_ns, self.odom_ts, self.odom_vel[:, i]) for i in range(3)])
        w = np.array([np.interp(t_ns, self.imu_ts, self.imu_rate[:, i]) for i in range(3)])
        return v, w


def _vec(rows: np.ndarray, i: int) -> list[float]:
    return [float(x) for x in rows[i]]


def scene_messages(ins: InsData, start_ns: int, end_ns: int) -> dict:
    """pose, ms_imu and meta for the interval [start_ns, end_ns] plus CAN_MARGIN_NS."""
    lo, hi = start_ns - CAN_MARGIN_NS, end_ns + CAN_MARGIN_NS
    o = np.flatnonzero((ins.odom_ts >= lo) & (ins.odom_ts <= hi))
    ts = ins.odom_ts[o]
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
            "pos": _vec(ins.odom_t, i),
            "orientation": _vec(ins.odom_q, i),
            "vel": _vec(ins.odom_vel, i),
            "accel": _vec(acc, j),
            "rotation_rate": _vec(rot, j),
        }
        if have_gnss:
            rec.update(lat=float(llh[j, 0]), lon=float(llh[j, 1]), height=float(llh[j, 2]),
                       ins_status=int(status[j]))
        pose.append(rec)

    # IMU samples outside odom coverage have no orientation to report.
    m = np.flatnonzero((ins.imu_ts >= max(lo, ins.odom_ts[0]))
                       & (ins.imu_ts <= min(hi, ins.odom_ts[-1])))
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

    meta = {
        "pose": {"message_count": len(pose), "message_freq": rate(pose),
                 "source": "/novatel/oem7/odom (pos, orientation, vel), "
                           "/novatel/oem7/corrimu (accel, rotation_rate), "
                           "/novatel/oem7/inspva (lat, lon, height, ins_status)",
                 "frames": "pos/orientation global (UTM 52N); vel, accel, rotation_rate ego "
                           "(x forward, y left, z up); accel has gravity removed"},
        "ms_imu": {"message_count": len(ms_imu), "message_freq": rate(ms_imu),
                   "source": "/novatel/oem7/corrimu, q from /novatel/oem7/odom",
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
