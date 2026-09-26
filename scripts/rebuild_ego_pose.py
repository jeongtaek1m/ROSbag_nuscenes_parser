#!/usr/bin/env python3
"""Rewrite a converted dataset's ego poses from INSPVA, without touching its sensor files.

    python scripts/rebuild_ego_pose.py DATAROOT --bags /path/to/bags [...] --out PATCH_DIR
    python scripts/rebuild_ego_pose.py DATAROOT --bags /path/to/bags [...] --in-place

For datasets converted before 2026-09-24, whose ego_pose came from
/novatel/oem7/odom in UTM: on the 2026-09-23 bags that position holds for a second
at a time for 55-80 % of the moving samples (ego poses off by 0.8-2.6 m on median
per log, up to 15 m), and the stamps
are arrival times. This recomputes every ego_pose record of every log with the
converter's own code (canbus.build_ins, nuscenes_writer.interp_pose): INSPVA at
GPS time, in the location's global frame (common.global_frame_id). The same
records keep their tokens and timestamps; only translation and rotation change.
It also rewrites the CAN bus pose / ms_imu / meta files of every scene and sets
log.json's `global_frame`, so bag2nuscenes can append to the dataset again.

Every log's bag must be found under --bags (by name: <logfile>.bag). Nothing else
in the dataset changes: sample_data, calibrated_sensor and the sensor files stay.

--out writes the changed files under PATCH_DIR with the dataset's layout
(<version>/ego_pose.json, <version>/log.json, can_bus/*.json) for review and
copying; --in-place replaces them in DATAROOT, holding the converter's lock.
Either way PATCH_DIR or DATAROOT gets rebuild_ego_pose.report.json with how far
each log's poses moved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from canbus import build_ins, corrimu_row, inspva_row, odom_row, scene_messages  # noqa: E402
from common import (  # noqa: E402
    CORRIMU_TOPIC,
    GEOID_UNDULATION,
    GLOBAL_ORIGINS,
    INSPVA_TOPIC,
    LOCATION,
    ODOM_TOPIC,
    UTM_ZONE,
    geodetic_to_enu,
    global_frame_id,
    make_typestore,
    utm_to_geodetic,
)
from nuscenes_writer import interp_pose  # noqa: E402

LOCK_NAME = ".convert.lock"          # the converter's; one writer per dataroot


def find_bag(logfile: str, roots: list[Path]) -> Path | None:
    name = f"{logfile}.bag"
    for r in roots:
        if r.is_file() and r.name == name:
            return r
        if r.is_dir():
            hit = next(iter(sorted(r.rglob(name))), None)
            if hit:
                return hit
    return None


def read_ins(bag: Path) -> tuple[list, list, list]:
    topics = {INSPVA_TOPIC, ODOM_TOPIC, CORRIMU_TOPIC}
    ins, odom, imu = [], [], []
    with AnyReader([bag], default_typestore=make_typestore()) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for c, _, raw in tqdm(reader.messages(connections=conns), total=sum(c.msgcount for c in conns),
                              desc=bag.name, unit="msg"):
            msg = reader.deserialize(raw, c.msgtype)
            if c.topic == INSPVA_TOPIC:
                ins.append(inspva_row(msg))
            elif c.topic == ODOM_TOPIC:
                odom.append(odom_row(msg))
            else:
                imu.append(corrimu_row(msg))
    return ins, odom, imu


def old_translation_in_frame(t: np.ndarray, old_frame: str | None, location: str) -> np.ndarray:
    """Old ego_pose translations in the new frame, to report how far they moved."""
    if old_frame == global_frame_id(location):
        return t
    lat, lon = utm_to_geodetic(t[:, 0], t[:, 1], UTM_ZONE)       # odom: UTM, sea-level height
    return geodetic_to_enu(lat, lon, t[:, 2] + GEOID_UNDULATION[location], GLOBAL_ORIGINS[location])


def write_json(path: Path, obj, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent) if indent else json.dumps(obj, separators=(",", ":")))
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataroot", type=Path)
    p.add_argument("--bags", type=Path, nargs="+", required=True, help="bag files or directories holding them")
    p.add_argument("--version", default="v1.0-trainval")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--out", type=Path, help="write the changed files here (the dataset is not modified)")
    out.add_argument("--in-place", action="store_true", help="replace the files in DATAROOT")
    a = p.parse_args()

    json_dir = a.dataroot / a.version
    tables = {n: json.loads((json_dir / f"{n}.json").read_text())
              for n in ("log", "scene", "sample", "sample_data", "ego_pose")}
    frame_id = global_frame_id(LOCATION)

    bags = {lg["logfile"]: find_bag(lg["logfile"], a.bags) for lg in tables["log"]}
    missing = [k for k, v in bags.items() if v is None]
    if missing:
        raise SystemExit(f"bags not found under {', '.join(map(str, a.bags))}: "
                         + ", ".join(f"{m}.bag" for m in missing)
                         + " — every log must be rebuilt, or the dataset would mix frames")

    scene_log = {s["token"]: s["log_token"] for s in tables["scene"]}
    sample_log = {s["token"]: scene_log[s["scene_token"]] for s in tables["sample"]}
    pose_log = {sd["ego_pose_token"]: sample_log[sd["sample_token"]] for sd in tables["sample_data"]}
    poses_by_log: dict[str, list[dict]] = {}
    for ep in tables["ego_pose"]:
        poses_by_log.setdefault(pose_log[ep["token"]], []).append(ep)
    samples_by_scene: dict[str, list[int]] = {}
    for s in tables["sample"]:
        samples_by_scene.setdefault(s["scene_token"], []).append(s["timestamp"])

    target = a.dataroot if a.in_place else a.out
    lock = a.dataroot / LOCK_NAME
    if a.in_place:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"{lock} exists: a conversion is writing to {a.dataroot}") from None
        os.write(fd, f"rebuild_ego_pose pid {os.getpid()}".encode())
        os.close(fd)
    try:
        report = {"created_at": datetime.now(timezone.utc).isoformat(), "global_frame": frame_id, "logs": {}}
        can_msgs: dict[str, dict] = {}
        for lg in tables["log"]:
            bag = bags[lg["logfile"]]
            print(f"{lg['logfile']}: reading INS from {bag}")
            ins, stats = build_ins(*read_ins(bag), LOCATION)
            recs = poses_by_log.get(lg["token"], [])
            q_ns = np.array([r["timestamp"] * 1000 for r in recs], dtype=np.int64)
            view = SimpleNamespace(pose_ts=ins.pose_ts, pose_t=ins.pose_t,
                                   pose_R=Rotation.from_quat(ins.pose_q[:, [1, 2, 3, 0]]), _slerp=None)
            trans, quats = interp_pose(q_ns, view)
            old_t = old_translation_in_frame(np.array([r["translation"] for r in recs], dtype=np.float64),
                                             lg.get("global_frame"), LOCATION)
            moved = np.linalg.norm((old_t - trans)[:, :2], axis=1)
            for r, t, q in zip(recs, trans, quats):
                r["translation"] = [float(x) for x in t]
                r["rotation"] = [float(x) for x in q]
            report["logs"][lg["logfile"]] = {
                "bag": str(bag), "n_ego_pose": len(recs), "old_frame": lg.get("global_frame") or "odom/UTM",
                "horizontal_move_m": {"median": float(np.median(moved)), "p95": float(np.percentile(moved, 95)),
                                      "max": float(moved.max())} if len(moved) else None,
                **stats,
            }
            print(f"  {len(recs)} ego poses from {stats['pose_source']}; moved (horizontal) median "
                  f"{np.median(moved):.3f} m, p95 {np.percentile(moved, 95):.3f} m, max {moved.max():.3f} m")
            lg["location"] = lg.get("location") or LOCATION
            lg["global_frame"] = frame_id
            for sc in tables["scene"]:
                if sc["log_token"] == lg["token"]:
                    ts = samples_by_scene[sc["token"]]
                    can_msgs[sc["name"]] = scene_messages(ins, min(ts) * 1000, max(ts) * 1000)

        for name, msgs in can_msgs.items():
            for kind, payload in msgs.items():
                write_json(target / "can_bus" / f"{name}_{kind}.json", payload, indent=None)
        write_json(target / a.version / "ego_pose.json", tables["ego_pose"])
        write_json(target / a.version / "log.json", tables["log"])
        write_json(target / "rebuild_ego_pose.report.json", report)
        print(f"-> {target}: {a.version}/ego_pose.json, {a.version}/log.json, "
              f"can_bus/ ({len(can_msgs)} scenes), rebuild_ego_pose.report.json")
    finally:
        if a.in_place:
            lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
