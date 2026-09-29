#!/usr/bin/env python3
"""Take one recording (log) out of a converted dataset.

    python remove_log.py <dataroot> <log name> [...] [--version v1.0-trainval]

Removes the log's scenes, samples, sample_data, ego_poses, annotations and
calibrated_sensors from the tables, its map row, its sensor files, its CAN bus and
ext/ files and its <log>.import.json. Sensors and the taxonomy stay. Holds the
converter's lock while it works; the tables are replaced atomically. Move the bag
itself out of the input too, or the next run of bag2nuscenes.py converts it again.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from converter import LOCK_NAME  # noqa: E402
from nuscenes_writer import write_tables  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataroot", type=Path)
    p.add_argument("logs", nargs="+")
    p.add_argument("--version", default="v1.0-trainval")
    a = p.parse_args()
    jd = a.dataroot / a.version
    lock = a.dataroot / LOCK_NAME
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"pid {os.getpid()} remove_log {' '.join(a.logs)}".encode())
    os.close(fd)
    try:
        t = {f.name: json.loads(f.read_text()) for f in jd.glob("*.json")}
        logs = {r["token"] for r in t["log.json"] if r["logfile"] in a.logs}
        missing = set(a.logs) - {r["logfile"] for r in t["log.json"] if r["token"] in logs}
        if missing:
            raise SystemExit(f"not in {jd}/log.json: {sorted(missing)}")
        scenes = [s for s in t["scene.json"] if s["log_token"] in logs]
        sc_tok = {s["token"] for s in scenes}
        smp = {s["token"] for s in t["sample.json"] if s["scene_token"] in sc_tok}
        sds = [d for d in t["sample_data.json"] if d["sample_token"] in smp]
        ep = {d["ego_pose_token"] for d in sds}
        cs_used = {d["calibrated_sensor_token"] for d in t["sample_data.json"] if d["sample_token"] not in smp}
        cs_drop = {d["calibrated_sensor_token"] for d in sds} - cs_used
        anns = t.get("sample_annotation.json", [])
        ann_keep = [x for x in anns if x["sample_token"] not in smp]
        inst_used = {x["instance_token"] for x in ann_keep}
        inst_drop = {x["instance_token"] for x in anns if x["sample_token"] in smp} - inst_used
        for d in sds:
            (a.dataroot / d["filename"]).unlink(missing_ok=True)
        for s in scenes:
            for f in (a.dataroot / "can_bus").glob(f"{s['name']}_*.json"):
                f.unlink()
            shutil.rmtree(a.dataroot / "ext" / s["name"], ignore_errors=True)
        for name in a.logs:
            (a.dataroot / f"{name}.import.json").unlink(missing_ok=True)
        t["log.json"] = [r for r in t["log.json"] if r["token"] not in logs]
        t["scene.json"] = [s for s in t["scene.json"] if s["token"] not in sc_tok]
        t["sample.json"] = [s for s in t["sample.json"] if s["token"] not in smp]
        t["sample_data.json"] = [d for d in t["sample_data.json"] if d["sample_token"] not in smp]
        t["ego_pose.json"] = [e for e in t["ego_pose.json"] if e["token"] not in ep]
        t["calibrated_sensor.json"] = [c for c in t["calibrated_sensor.json"] if c["token"] not in cs_drop]
        if "sample_annotation.json" in t:
            t["sample_annotation.json"] = ann_keep
        if "instance.json" in t:
            t["instance.json"] = [i for i in t["instance.json"] if i["token"] not in inst_drop]
        if "map.json" in t:
            # one map row per log here; a shared row only loses the log's token
            for m in t["map.json"]:
                m["log_tokens"] = [x for x in m["log_tokens"] if x not in logs]
            t["map.json"] = [m for m in t["map.json"] if m["log_tokens"]]
        write_tables(t, a.dataroot, a.version)
        print(f"removed {len(logs)} log(s): {len(scenes)} scenes ({', '.join(s['name'] for s in scenes)}), "
              f"{len(smp)} samples, {len(sds)} sample_data and their files")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
