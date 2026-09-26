"""INS trajectory, LiDAR sweeps and SE(3) helpers shared by the calibration stages.

Poses come from INSPVA (100 Hz) at the receiver's GPS time: world frame = local
east-north-up at the first sample, ego frame = base_link (x forward, y left,
z up) at the INS reference point. Files extracted before INSPVA was recorded in
full fall back to /novatel/oem7/odom at its header stamps (UTM) — which on some
2026-09-23 drives holds its position for a second at a time and is stamped on
arrival, so do not calibrate from those.

A third source, "dm", is the second vehicle's GNSS/INS (`/gps/fix` + `/gps/vel`
+ `/imu/data`), already reduced to local ENU by `online/extract_dm.py`; its npz
carries a `dm` key, which is what picks it. Nothing else in the pipeline needs
to know which vehicle it is looking at.

A fourth, "lo", is LiDAR odometry written by `lo_traj.py` in the same form as
"dm" plus an `lo` key: there the "ego" frame IS the LiDAR frame, so the
LiDAR -> ego transform is the identity and must not be estimated
(`Trajectory.lidar_frame`). No GNSS enters it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from canbus import corrimu_to_ego  # noqa: E402


# ------------------------------------------------------------------ geodesy
WGS84_A, WGS84_F = 6378137.0, 1 / 298.257223563
# NovAtel vehicle frame (x right, y forward, z up) -> base_link (x forward, y left, z up)
R_NOVATEL_BASE = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def geodetic_to_ecef(lat_deg, lon_deg, h) -> np.ndarray:
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - e2 * np.sin(la) ** 2)
    return np.stack([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - e2) + h) * np.sin(la)], axis=-1)


def geodetic_to_enu(lat_deg, lon_deg, h, lat0: float, lon0: float, h0: float) -> np.ndarray:
    la, lo = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(lo), np.cos(lo), 0.0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return (geodetic_to_ecef(lat_deg, lon_deg, h) - geodetic_to_ecef(lat0, lon0, h0)) @ R.T


def novatel_attitude(roll_deg, pitch_deg, azimuth_deg) -> Rotation:
    """INSPVA angles -> ENU-from-base_link rotation. NovAtel: roll right-handed
    about y (forward), pitch right-handed about x (right), azimuth left-handed
    about z from north, applied z-x-y."""
    R_enu_vehicle = Rotation.from_euler("ZXY", np.stack([-np.asarray(azimuth_deg), pitch_deg, roll_deg], -1),
                                        degrees=True)
    return R_enu_vehicle * Rotation.from_matrix(R_NOVATEL_BASE)


# ------------------------------------------------------------------ SE(3)
def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return se3(R.T, -R.T @ t)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """xi = (rotation vector, translation) -> 4x4, first-order translation coupling."""
    return se3(Rotation.from_rotvec(xi[:3]).as_matrix(), xi[3:6])


def rotvec_to_quat(v: np.ndarray) -> np.ndarray:
    """(n, 3) rotation vectors -> (n, 4) quaternions x y z w."""
    th = np.sqrt(np.einsum("ni,ni->n", v, v))
    h = 0.5 * th
    s = np.where(th < 1e-8, 0.5 - th * th / 48, np.sin(h) / np.where(th < 1e-8, 1.0, th))
    return np.concatenate([v * s[:, None], np.cos(h)[:, None]], axis=1)


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of (n, 4) quaternions x y z w."""
    px, py, pz, pw = p.T
    qx, qy, qz, qw = q.T
    return np.stack([pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw,
                     pw * qw - px * qx - py * qy - pz * qz], axis=1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(n, 4) unit quaternions x y z w -> (n, 3, 3)."""
    x, y, z, w = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


# ------------------------------------------------------------- trajectory
_DEDUP_MIN = 20000        # pose queries longer than this are evaluated once per distinct time


class Trajectory:
    """Ego poses at arbitrary times from the 100 Hz odom, IMU rates from CORRIMU."""

    def __init__(self, ins_npz: Path, source: str = "auto"):
        z = np.load(ins_npz)
        if source == "auto":
            self.source = "lo" if "lo" in z else "dm" if "dm" in z else "inspva" if "pva_t" in z else "odom"
        else:
            self.source = source
        self.lidar_frame = self.source == "lo"
        if self.source in ("dm", "lo"):
            # Second vehicle (DM rig), written by online/extract_dm.py: position
            # already in local ENU metres, attitude from /imu/data (full 3D, ENU
            # world / FLU body), body rates from the same message. No CORRIMU.
            self.t = z["t"].astype(np.int64)
            self.origin = z["origin"].copy()
            self.pos = z["pos"]
            self.rot = Rotation.from_quat(z["quat"][:, [1, 2, 3, 0]])
            self.vel_ego = z["vel_ego"]
            self._slerp = Slerp(self.t.astype(np.float64), self.rot)
            self.imu_t = z["imu_t"].astype(np.int64)
            self.imu_rate = z["imu_rate"]
            self.imu_hz = float(z["imu_hz"])
            return
        if self.source == "inspva":
            self.t = z["pva_t"].astype(np.int64)
            a = z["pva"]                      # lat lon h, v north east up, roll pitch azimuth, status
            self.origin = a[0, :3].copy()
            self.pos = geodetic_to_enu(a[:, 0], a[:, 1], a[:, 2], *self.origin)
            self.rot = novatel_attitude(a[:, 6], a[:, 7], a[:, 8])
            self.vel_ego = self.rot.inv().apply(a[:, [4, 3, 5]])
            imu_t = z["imu_tg"]
        else:
            self.t = z["odom_t"].astype(np.int64)
            od = z["odom"]
            self.origin = od[0, :3].copy()
            self.pos = od[:, :3] - self.origin
            self.rot = Rotation.from_quat(od[:, [4, 5, 6, 3]])      # wxyz -> xyzw
            self.vel_ego = od[:, 7:10]
            imu_t = z["imu_t"]
        self._slerp = Slerp(self.t.astype(np.float64), self.rot)
        imu = z["imu"]
        self.imu_t, _, self.imu_rate, self.imu_hz = corrimu_to_ego(
            imu_t, imu[:, 0], imu[:, 1], imu[:, 2], imu[:, 3], imu[:, 4], imu[:, 5], imu[:, 6])

    def R(self, t_ns) -> np.ndarray:
        """(n, 3, 3) world-from-ego rotation: SLERP between samples, as scipy's
        Slerp (and like it, ValueError outside the samples), but elementwise on
        quaternions — it is called for every LiDAR point of a render window."""
        if not hasattr(self, "_qk"):
            self._tk = self.t.astype(np.float64)
            self._qk = self.rot.as_quat()                                   # x y z w
            self._dk = (self.rot[:-1].inv() * self.rot[1:]).as_rotvec()
        t = np.atleast_1d(t_ns).astype(np.float64)
        if len(t) > _DEDUP_MIN:                  # LiDAR points share their firing times
            tu, inv = np.unique(t, return_inverse=True)
            if len(tu) < len(t) // 2:
                return self.R(tu)[inv]
        tk = self._tk
        if len(t) and (t.min() < tk[0] or t.max() > tk[-1]):
            raise ValueError("Interpolation times must be within the range "
                             f"[{tk[0]:.0f}, {tk[-1]:.0f}], both inclusive.")
        i = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, len(tk) - 2)
        a = (t - tk[i]) / (tk[i + 1] - tk[i])
        return quat_to_matrix(quat_mul(self._qk[i], rotvec_to_quat(self._dk[i] * a[:, None])))

    def covers(self, t0_ns, t1_ns=None) -> np.ndarray:
        """Whether [t0, t1] (default: the instant t0) lies inside the samples. A LiDAR
        odometry spans only its clip's sweeps, where the INS spans the whole bag."""
        t0 = np.atleast_1d(t0_ns).astype(np.float64)
        t1 = t0 if t1_ns is None else np.atleast_1d(t1_ns).astype(np.float64)
        return (t0 >= self.t[0]) & (t1 <= self.t[-1])

    def p(self, t_ns) -> np.ndarray:
        t = np.atleast_1d(t_ns).astype(np.float64)
        if len(t) > _DEDUP_MIN:
            tu, inv = np.unique(t, return_inverse=True)
            if len(tu) < len(t) // 2:
                return self.p(tu)[inv]
        return np.stack([np.interp(t, self.t, self.pos[:, i]) for i in range(3)], axis=-1)

    def T(self, t_ns) -> np.ndarray:
        R, p = self.R(t_ns), self.p(t_ns)
        T = np.tile(np.eye(4), (len(R), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T

    def omega_ego(self, t_ns) -> np.ndarray:
        """(n, 3) angular rate in the ego frame (rad/s)."""
        t = np.atleast_1d(t_ns).astype(np.float64)
        return np.stack([np.interp(t, self.imu_t, self.imu_rate[:, i]) for i in range(3)], axis=-1)

    def vel_world(self, t_ns) -> np.ndarray:
        t = np.atleast_1d(t_ns).astype(np.float64)
        v = np.stack([np.interp(t, self.t, self.vel_ego[:, i]) for i in range(3)], axis=-1)
        return np.einsum("nij,nj->ni", self.R(t_ns), v)

    def speed(self, t_ns) -> np.ndarray:
        t = np.atleast_1d(t_ns).astype(np.float64)
        return np.hypot(np.interp(t, self.t, self.vel_ego[:, 0]), np.interp(t, self.t, self.vel_ego[:, 1]))


# ------------------------------------------------------------------ sweeps
def load_sweep(path: Path, rmin: float = 2.5, rmax: float = 80.0) -> np.ndarray:
    """Structured sweep (see extract.py), without the car itself and far points."""
    s = np.load(path)
    r = np.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2)
    return s[(r > rmin) & (r < rmax)]


def sweep_xyz(s: np.ndarray) -> np.ndarray:
    return np.stack([s["x"], s["y"], s["z"]], axis=-1).astype(np.float64)


def deskew_to_world(s: np.ndarray, header_ns: int, traj: Trajectory, T_el: np.ndarray) -> np.ndarray:
    """Every point to the world frame at its own acquisition time."""
    ut, inv = np.unique(s["t"], return_inverse=True)          # ~1000 firing times per sweep
    t = header_ns + ut.astype(np.float64) * 1e9
    p_e = transform(T_el, sweep_xyz(s))
    return np.einsum("nij,nj->ni", traj.R(t)[inv], p_e) + traj.p(t)[inv]


def voxel_down(p: np.ndarray, voxel: float, *extra: np.ndarray):
    """One point per voxel (the first); extra arrays are subsampled alongside."""
    key = np.floor(p / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return (p[idx], *(e[idx] for e in extra)) if extra else p[idx]


def normals_pca(points: np.ndarray, k: int = 12, tree: cKDTree | None = None):
    """(normals, planarity) from the k nearest neighbours of every point."""
    tree = tree or cKDTree(points)
    _, nn = tree.query(points, k=k)
    nb = points[nn] - points[nn].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov)
    normals = v[:, :, 0]
    planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return normals, planarity
