"""The T-Car calibration (tcar_calib_<date>/) as calibrated_sensor values.

A calibration folder holds intrinsic/<CAM>.json (Kannala-Brandt), windshield/<CAM>.json (refraction field)
and extrinsic/<CAM>.json + extrinsic/LIDAR_TOP.json (T_ego_cam, T_ego_lidar; ego = the NovAtel output point,
x forward, y left, z up — the frame ego_pose places). Only the full set carries it: the full converter copies
nothing itself, scripts/apply_calibration.py puts a folder into <dataroot>/calibration/, and from then on the
converter gives every recording it appends the same values (apply_values on the new records).

The images stay as recorded (not rectified): K and the distortion describe the raw JPEGs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DIRNAME = "calibration"                # <dataroot>/calibration/
APPLIED = "applied.json"               # its record of what was applied


def quat_wxyz(R) -> list[float]:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def load_values(calib: Path) -> dict[str, dict]:
    """{channel: calibrated_sensor fields} from a calibration folder. Raises ValueError on a bad file."""
    calib = Path(calib)
    out = {}
    for p in sorted((calib / "extrinsic").glob("*.json")):
        ch = p.stem
        e = json.loads(p.read_text())
        T = np.asarray(e["T_ego_lidar" if ch.startswith("LIDAR") else "T_ego_cam"], dtype=np.float64)
        if T.shape != (4, 4) or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-4:
            raise ValueError(f"{p}: not a rigid 4x4 transform")
        rec = {"translation": T[:3, 3].tolist(), "rotation": quat_wxyz(T[:3, :3]), "camera_intrinsic": []}
        if ch.startswith("CAM"):
            i = json.loads((calib / "intrinsic" / f"{ch}.json").read_text())
            if i.get("model") != "kannala_brandt":
                raise ValueError(f"{ch}: intrinsic model {i.get('model')!r}, expected kannala_brandt")
            if not (calib / "windshield" / f"{ch}.json").is_file():
                raise ValueError(f"{ch}: windshield/{ch}.json missing")
            rec.update({
                "camera_intrinsic": [[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]],
                "camera_model": "kannala_brandt",
                "camera_distortion": [i["k1"], i["k2"], i["k3"], i["k4"]],
                "image_size": [i["image_width"], i["image_height"]],
                "time_offset_s": i["time_offset_s"],
                "rolling_shutter_readout_s": i["rolling_shutter_readout_s"],
                "windshield": f"{DIRNAME}/windshield/{ch}.json",
            })
        out[ch] = rec
    if not out:
        raise ValueError(f"{calib}: no extrinsic/*.json")
    return out


def apply_values(records: list[dict], channel_of: dict[str, str], values: dict[str, dict],
                 name: str) -> dict[str, int]:
    """Set the calibration on calibrated_sensor records in place; channel_of maps sensor token -> channel.

    A channel the calibration covers takes its values and `calibration` = name; any other gets
    `calibration` = None and keeps the converter's identity placeholder. Returns records per channel."""
    n: dict[str, int] = {}
    for r in records:
        ch = channel_of[r["sensor_token"]]
        if ch in values:
            keep = {"token": r["token"], "sensor_token": r["sensor_token"]}
            r.clear()
            r.update(keep, **values[ch], calibration=name)
            n[ch] = n.get(ch, 0) + 1
        else:
            r["calibration"] = None
    return n


def dataset_calibration(root: Path) -> tuple[str, dict[str, dict]] | None:
    """(name, values) of the calibration a dataset carries (<root>/calibration/), or None."""
    d = Path(root) / DIRNAME
    if not (d / APPLIED).is_file():
        return None
    return json.loads((d / APPLIED).read_text())["calibration"], load_values(d)
