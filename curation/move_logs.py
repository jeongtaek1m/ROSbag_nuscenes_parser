#!/usr/bin/env python3
"""Move whole recordings (logs) from a later volume into an earlier one, in the standard and the full set alike, so
the scene names a 최종 확정 freed are taken by the next recordings.

    python curation/move_logs.py --standard <root> --full <root> --fill               # plan only
    python curation/move_logs.py --standard <root> --full <root> --fill --apply
    python curation/move_logs.py --standard <root> --full <root> --into 1 --from 2 --logs H-4_...,H-5_... --apply

A volume holds at most the official nuScenes train names (685 without the CAN-bus blacklist); the recordings after
that went into tcar_nuscenes_2, _3, ... (volumes.py). When curation takes scenes out (apply_selection.py --finalize
renumbers the rest, so the free names are the last ones), --fill moves the logs of the later volumes up into the
room: oldest first, whole logs only (a log never spans two volumes), passing over a log that does not fit for a
later one that does, never a log published from its volume (selection/hf_release.json). The full set's volumes
are the standard set's twins: the same logs move there under the same new names.

What goes with a log: its rows of every table (its scenes take the next free names of the target; its
calibrated_sensor rows are re-pointed at the target's sensor rows by channel; its map row), its sensor files (the
full set's .time.bin too), can_bus and ext/ files under the new names, <log>.import.json (the names in it renamed)
and its curation state: decisions (by name, renamed), locks (a scene a curation round locked stays locked:
selection/locked.json `moved_in`), detections, LiDAR diff, sizes, object counts. result.json is not carried: run
the rules again on both volumes.

So that an interruption never loses data: files are hard-linked into the target (one file system; a set's own
files, never the other set's), the target's tables are written (the log is then in both volumes), the source's
tables are written without it, then the source's names of the files are removed. <target>/_moved/<stamp>.json
records each step; the same command run again finishes a move that was interrupted. A volume every log left is
set aside as <name>.emptied-<stamp>. Holds the converter's lock of every volume it writes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import volumes  # noqa: E402
from nuscenes_writer import CAN_BUS_BLACKLIST, official_scene_names  # noqa: E402

VERSION = "v1.0-trainval"
LOCK_NAME = ".convert.lock"            # converter.LOCK_NAME
BIG = ("sample_data.json", "ego_pose.json")
CAN_RE = re.compile(r"(scene-\d{4})_(.+\.json)")


# ------------------------------------------------------------------ tables, streamed
def iter_rows(path: Path, block: int = 1 << 26):
    """The objects of a JSON array file one at a time (a 2 GB table never in memory at once)."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        buf, i, started = "", 0, False
        while True:
            more = f.read(block)
            buf = buf[i:] + more
            i = 0
            while True:
                while i < len(buf) and buf[i] in " \r\n\t,":
                    i += 1
                if not started:
                    if i < len(buf) and buf[i] == "[":
                        started, i = True, i + 1
                        continue
                if i >= len(buf) or buf[i] == "]":
                    break
                try:
                    obj, j = dec.raw_decode(buf, i)
                except json.JSONDecodeError:
                    break
                yield obj
                i = j
            if not more:
                return


def _row(obj) -> str:
    return "  " + json.dumps(obj, indent=2).replace("\n", "\n  ")


def write_rows(path: Path, rows) -> None:
    """A JSON array as json.dumps(rows, indent=2) writes it (nuscenes_writer.write_tables), row by row."""
    with open(path, "w", encoding="utf-8") as f:
        first = True
        for r in rows:
            f.write("[\n" if first else ",\n")
            f.write(_row(r))
            first = False
        f.write("[]" if first else "\n]")


def append_rows(src: Path, rows, out: Path) -> None:
    """src's array with rows added at its end, into out; src's bytes are copied as they are."""
    size = src.stat().st_size
    with open(src, "rb") as f:
        f.seek(max(0, size - 65536))
        tail = f.read()
    t = tail.rstrip()
    if not t.endswith(b"]"):
        raise SystemExit(f"{src}: not a JSON array")
    before = t[:-1].rstrip()
    empty = before.endswith(b"[")
    cut = size - len(tail) + len(before)                     # src's bytes up to its last row (or its "[")
    with open(src, "rb") as f, open(out, "wb") as o:
        left = cut
        while left:
            b = f.read(min(left, 1 << 24))
            o.write(b)
            left -= len(b)
        n = 0
        for r in rows:
            o.write((b"\n" if empty and not n else b",\n") + _row(r).encode())
            n += 1
        o.write(b"]" if empty and not n else b"\n]")


def small_tables(root: Path) -> dict:
    return {f.name: json.loads(f.read_text()) for f in sorted((root / VERSION).glob("*.json")) if f.name not in BIG}


def official_train() -> list[str]:
    from nuscenes.utils.splits import create_splits_scenes
    return [n for n in sorted(create_splits_scenes()["train"]) if int(n[-4:]) not in CAN_BUS_BLACKLIST]


def free_names(T: dict) -> int:
    """Free official names; MOVE_LOGS_MAX_SCENES caps a volume lower (for tests)."""
    used = {s["name"] for s in T.get("scene.json", [])}
    free = sum(n not in used for n in official_train())
    cap = os.environ.get("MOVE_LOGS_MAX_SCENES")
    return min(free, int(cap) - len(used)) if cap else free


def rename_in(obj, mapping):
    if isinstance(obj, dict):
        return {mapping.get(k, k): rename_in(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename_in(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj


def log_order(T: dict) -> list[tuple[str, int, int]]:
    """[(logfile, scenes, first sample µs)] of a volume, oldest first."""
    smp = {s["token"]: s["timestamp"] for s in T["sample.json"]}
    first, count = {}, {}
    for sc in T["scene.json"]:
        t = smp.get(sc["first_sample_token"], 0)
        first[sc["log_token"]] = min(first.get(sc["log_token"], t), t)
        count[sc["log_token"]] = count.get(sc["log_token"], 0) + 1
    name = {l["token"]: l["logfile"] for l in T["log.json"]}
    return sorted(((name[t], count[t], first[t]) for t in count), key=lambda x: (x[2], x[0]))


def _jread(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _jwrite(path: Path, obj, indent=1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
    tmp.replace(path)


def published_logs(root: Path) -> set[str]:
    """Logs a release published from this volume (scripts/hf_release.py): they keep their names."""
    return set((_jread(root / "selection" / "hf_release.json", {}) or {}).get("logs") or {})


# ------------------------------------------------------------------ plan
def plan(dst: Path, src: Path, logs: list[str], room: int | None = None) -> dict:
    Td, Ts = small_tables(dst), small_tables(src)
    by_name = {l["logfile"]: l for l in Ts["log.json"]}
    unknown = [n for n in logs if n not in by_name]
    if unknown:
        raise SystemExit(f"not in {src}: {unknown}")
    if {l["logfile"] for l in Td["log.json"]} & set(logs):
        raise SystemExit(f"already in {dst}: {sorted({l['logfile'] for l in Td['log.json']} & set(logs))}")
    pub = published_logs(src) & set(logs)
    if pub:
        raise SystemExit(f"published from {src.name} under their names, not moved: {sorted(pub)}")
    log_tok = {by_name[n]["token"] for n in logs}
    first = {s["token"]: s["timestamp"] for s in Ts["sample.json"]}
    order = {n: i for i, (n, _, _) in enumerate(log_order(Ts))}
    lname = {l["token"]: l["logfile"] for l in Ts["log.json"]}
    scenes = sorted((s for s in Ts["scene.json"] if s["log_token"] in log_tok),
                    key=lambda s: (order[lname[s["log_token"]]], first.get(s["first_sample_token"], 0)))
    official = set(official_train())
    bad = [s["name"] for s in scenes if s["name"] not in official]
    if bad:
        raise SystemExit(f"scenes outside the official train list: {bad[:5]}")
    free = free_names(Td) if room is None else room
    if len(scenes) > free:
        raise SystemExit(f"{len(scenes)} scenes do not fit into {dst.name}'s {free} free names")
    new = official_scene_names("train", {s["name"] for s in Td["scene.json"]}, len(scenes))
    sc_tok = {s["token"] for s in scenes}
    samples = {s["token"] for s in Ts["sample.json"] if s["scene_token"] in sc_tok}
    if any(a["sample_token"] in samples for a in Ts.get("sample_annotation.json", [])):
        raise SystemExit("these logs have annotations; annotated logs are not moved")
    return {"src": str(src), "dst": str(dst), "logs": logs, "log_tokens": sorted(log_tok),
            "mapping": {s["name"]: n for s, n in zip(scenes, new)}, "scene_tokens": sorted(sc_tok),
            "n_samples": len(samples)}


def twin_plan(p: dict, dst: Path, src: Path) -> dict:
    """The same move in the twin set: the same logs, the twins of the same scenes under the same new names."""
    Ts, Td = small_tables(src), small_tables(dst)
    by_name = {l["logfile"]: l for l in Ts["log.json"]}
    missing = [n for n in p["logs"] if n not in by_name]
    if missing:
        raise SystemExit(f"{src}: the twin set lacks {missing}: convert them into it first")
    log_tok = {by_name[n]["token"] for n in p["logs"]}
    names = sorted(s["name"] for s in Ts["scene.json"] if s["log_token"] in log_tok)
    if names != sorted(p["mapping"]):
        raise SystemExit(f"{src.name}: these logs' scenes are not the twins' ({len(names)} vs {len(p['mapping'])} "
                         "or other names): bring the full set in step first (curation/mirror_full.py)")
    taken = {s["name"] for s in Td["scene.json"]} & set(p["mapping"].values())
    if taken:
        raise SystemExit(f"{dst.name} already has {sorted(taken)[:3]}: the two sets' volumes are not in step")
    q = plan(dst, src, p["logs"], room=len(p["mapping"]) + free_names(Td))
    q["mapping"] = p["mapping"]
    return q


def fill_plans(std: Path, full: Path | None, log=print) -> list[tuple[dict, dict | None]]:
    """--fill for the first volume with room: the logs of the later volumes that fit, oldest first, in both sets.
    (Run again after it: the next volume's room is known once this move is done.)"""
    fam = [v for v in volumes.volumes(std) if (v / VERSION / "scene.json").exists()]
    ffam = [v for v in volumes.volumes(full) if (v / VERSION / "scene.json").exists()] if full else []
    if full and len(ffam) != len(fam):
        raise SystemExit(f"the standard set has {len(fam)} volumes, the full set {len(ffam)}: not in step")
    for i, dst in enumerate(fam):
        room = free_names(small_tables(dst))
        if not room or i == len(fam) - 1:
            continue
        log(f"{dst.name}: free scene names {room}")
        out = []
        for j in range(i + 1, len(fam)):
            src = fam[j]
            Ts = small_tables(src)
            pub = published_logs(src)
            pick = []
            for name, n, _ in log_order(Ts):
                if name in pub:
                    log(f"  {name}: published from {src.name}, not moved")
                    continue
                if n <= room:
                    pick.append(name)
                    room -= n
            if pick:
                p = plan(dst, src, pick)
                out.append((p, twin_plan(p, ffam[i], ffam[j]) if full else None))
            if not room:
                break
        if out:
            return out
    return []


# ------------------------------------------------------------------ steps
def _same_small_file(a: Path, b: Path, limit: int = 1 << 20) -> bool:
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return sa.st_size == sb.st_size and sa.st_size <= limit and a.read_bytes() == b.read_bytes()


def _link(a: Path, b: Path) -> None:
    """b as a second name of a (a symlink stays a symlink); a small file already there with the same bytes is fine."""
    if os.path.lexists(b):
        if os.path.samestat(os.lstat(a), os.lstat(b)) or _same_small_file(a, b):
            return
        raise SystemExit(f"conflict: {b} exists and is not {a}")
    b.parent.mkdir(parents=True, exist_ok=True)
    os.link(a, b, follow_symlinks=False)


def _files(filename: str) -> list[str]:
    """A sample_data file and the full set's per-point time file next to a LiDAR frame."""
    return [filename] + ([filename[:-len(".pcd.bin")] + ".time.bin"] if filename.endswith(".pcd.bin") else [])


def _samples(root: Path, sc_tok: set) -> set:
    return {s["token"] for s in json.loads((root / VERSION / "sample.json").read_text()) if s["scene_token"] in sc_tok}


def step_target(p: dict, log=print) -> None:
    """Links of the sensor files, then the target's tables with the logs added."""
    src, dst = Path(p["src"]), Path(p["dst"])
    Ts, Td = small_tables(src), small_tables(dst)
    log_tok, sc_tok = set(p["log_tokens"]), set(p["scene_tokens"])
    samples = {s["token"] for s in Ts["sample.json"] if s["scene_token"] in sc_tok}
    ego, cs, n = set(), set(), 0
    for d in iter_rows(src / VERSION / "sample_data.json"):
        if d["sample_token"] not in samples:
            continue
        ego.add(d["ego_pose_token"])
        cs.add(d["calibrated_sensor_token"])
        for f in _files(d["filename"]):
            if os.path.lexists(src / f) or f == d["filename"]:
                _link(src / f, dst / f)
                n += 1
    log(f"  {n} sensor files linked into {dst.name}")
    chan_dst = {s["channel"]: s for s in Td["sensor.json"]}
    sensor_map = {}
    for s in Ts["sensor.json"]:
        have = chan_dst.get(s["channel"])
        if have is None:
            Td["sensor.json"].append(s)
            chan_dst[s["channel"]] = have = s
        sensor_map[s["token"]] = have["token"]
    have_cs = {c["token"] for c in Td["calibrated_sensor.json"]}
    Td["calibrated_sensor.json"] += [dict(c, sensor_token=sensor_map[c["sensor_token"]])
                                     for c in Ts["calibrated_sensor.json"] if c["token"] in cs and c["token"] not in have_cs]
    Td["log.json"] += [l for l in Ts["log.json"] if l["token"] in log_tok]
    Td["scene.json"] += [dict(s, name=p["mapping"][s["name"]]) for s in Ts["scene.json"] if s["token"] in sc_tok]
    Td["sample.json"] += [s for s in Ts["sample.json"] if s["token"] in samples]
    maps = Td.setdefault("map.json", [])
    for m in Ts.get("map.json", []):
        mine = [t for t in m["log_tokens"] if t in log_tok]
        if not mine:
            continue
        same = next((x for x in maps if x["token"] == m["token"] or x["filename"] == m["filename"]), None)
        if same:
            same["log_tokens"] += [t for t in mine if t not in same["log_tokens"]]
        else:
            maps.append(dict(m, log_tokens=mine))
        if m.get("filename") and (src / m["filename"]).exists() and not (dst / m["filename"]).exists():
            (dst / m["filename"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / m["filename"], dst / m["filename"])
    jd = dst / VERSION
    tmp = {}
    for name, rows in Td.items():
        tmp[name] = jd / f".{name}.move"
        tmp[name].write_text(json.dumps(rows, indent=2))
    tmp["sample_data.json"] = jd / ".sample_data.json.move"
    append_rows(jd / "sample_data.json", (d for d in iter_rows(src / VERSION / "sample_data.json")
                                          if d["sample_token"] in samples), tmp["sample_data.json"])
    tmp["ego_pose.json"] = jd / ".ego_pose.json.move"
    append_rows(jd / "ego_pose.json", (e for e in iter_rows(src / VERSION / "ego_pose.json") if e["token"] in ego),
                tmp["ego_pose.json"])
    for name, t in tmp.items():
        t.replace(jd / name)
    log(f"  {dst.name}: + {len(log_tok)} logs, {len(sc_tok)} scenes, {len(samples)} samples, {len(ego)} ego poses")


def step_side_files(p: dict, log=print) -> None:
    """can_bus and ext/ under the new names, the import records (links or rewritten copies; the source keeps its)."""
    src, dst, mapping = Path(p["src"]), Path(p["dst"]), p["mapping"]
    n = 0
    for f in sorted((src / "can_bus").glob("*.json")) if (src / "can_bus").exists() else []:
        m = CAN_RE.fullmatch(f.name)
        if m and m.group(1) in mapping:
            _link(f, dst / "can_bus" / f"{mapping[m.group(1)]}_{m.group(2)}")
            n += 1
    for old, new in mapping.items():
        e = src / "ext" / old
        if e.is_dir():
            for f in e.rglob("*"):
                if f.is_file():
                    _link(f, dst / "ext" / new / f.relative_to(e))
    for lg in p["logs"]:
        f = src / f"{lg}.import.json"
        if f.exists():
            (dst / f.name).write_text(json.dumps(rename_in(json.loads(f.read_text()), mapping), indent=2))
    log(f"  can_bus {n} files, ext and import records in {dst.name}")


def step_curation(p: dict, log=print) -> None:
    """The logs' curation state into the target's selection/ (the source keeps its until step_unlink)."""
    import numpy as np
    src, dst = Path(p["src"]) / "selection", Path(p["dst"]) / "selection"
    if not src.is_dir():
        return
    mapping, toks = p["mapping"], set(p["scene_tokens"])
    moved = []
    for name, key in (("human_decisions.json", "decisions"), ("human_labels.json", "labels")):
        s = _jread(src / name, None)
        if not s or not isinstance(s.get(key), dict):
            continue
        d = _jread(dst / name, None) or {k: v for k, v in s.items() if k != key}
        d.setdefault(key, {})
        for old, new in mapping.items():
            if old in s[key]:
                d[key][new] = s[key][old]
        _jwrite(dst / name, d)
        moved.append(name)
    lk = _jread(src / "locked.json", {}) or {}
    held = sorted({t for r in (lk.get("rounds") or []) + (lk.get("moved_in") or []) for t in (r or {}).get("tokens") or []} & toks)
    if held:
        d = _jread(dst / "locked.json", None) or {"version": 1, "rounds": []}
        rounds = [r.get("stamp") for r in lk.get("rounds") or [] if set(r.get("tokens") or []) & toks]
        d.setdefault("moved_in", []).append({"stamp": p["stamp"], "from": Path(p["src"]).name, "rounds": rounds,
                                             "n": len(held), "logs": p["logs"], "tokens": held})
        _jwrite(dst / "locked.json", d)
        moved.append(f"locked.json ({len(held)} locked scenes)")
    for name in ("lidar_diff.json", "scene_bytes.json"):               # {"scenes": {token: {... name}}}
        s = _jread(src / name, None)
        if not s or not isinstance(s.get("scenes"), dict):
            continue
        d = _jread(dst / name, None) or {k: v for k, v in s.items() if k != "scenes"}
        d.setdefault("scenes", {})
        for t, v in s["scenes"].items():
            if t in toks:
                d["scenes"][t] = dict(v, name=mapping.get(v.get("name"), v.get("name"))) if isinstance(v, dict) else v
        _jwrite(dst / name, d)
        moved.append(name)
    s = _jread(src / "object_counts.json", None)                        # {"scenes": {name: {token, ...}}}
    if s and isinstance(s.get("scenes"), dict):
        d = _jread(dst / "object_counts.json", None) or {k: v for k, v in s.items() if k != "scenes"}
        d.setdefault("scenes", {})
        for old, v in s["scenes"].items():
            if old in mapping and (v or {}).get("token") in toks:
                d["scenes"][mapping[old]] = v
        _jwrite(dst / "object_counts.json", d)
        moved.append("object_counts.json")
    for f in sorted(src.glob("*.npz")):                                 # keyed by token (+ '.field')
        with np.load(f) as z:
            mine = {k: z[k] for k in z.files if re.split(r"[.|]", k, 1)[0] in toks}
        if not mine:
            continue
        have = {}
        if (dst / f.name).exists():
            with np.load(dst / f.name) as z:
                have = {k: z[k] for k in z.files}
        have.update(mine)
        dst.mkdir(parents=True, exist_ok=True)
        tmp = dst / (f.name[:-4] + ".tmp.npz")
        np.savez(tmp, **have)
        tmp.replace(dst / f.name)
        moved.append(f"{f.name} ({len(mine)} keys)")
    if moved:
        log("  curation state: " + ", ".join(moved))


def step_source(p: dict, log=print) -> None:
    """The source's tables without the logs (its files stay until step_unlink)."""
    src = Path(p["src"])
    T = small_tables(src)
    log_tok, sc_tok = set(p["log_tokens"]), set(p["scene_tokens"])
    samples = {s["token"] for s in T["sample.json"] if s["scene_token"] in sc_tok}
    gone_ego, cs_gone, cs_kept = set(), set(), set()
    for d in iter_rows(src / VERSION / "sample_data.json"):
        if d["sample_token"] in samples:
            gone_ego.add(d["ego_pose_token"])
            cs_gone.add(d["calibrated_sensor_token"])
        else:
            cs_kept.add(d["calibrated_sensor_token"])
    T["log.json"] = [l for l in T["log.json"] if l["token"] not in log_tok]
    T["scene.json"] = [s for s in T["scene.json"] if s["token"] not in sc_tok]
    T["sample.json"] = [s for s in T["sample.json"] if s["token"] not in samples]
    T["calibrated_sensor.json"] = [c for c in T["calibrated_sensor.json"] if c["token"] not in cs_gone - cs_kept]
    for m in T.get("map.json", []):
        m["log_tokens"] = [t for t in m["log_tokens"] if t not in log_tok]
    if "map.json" in T:
        T["map.json"] = [m for m in T["map.json"] if m["log_tokens"]]
    jd = src / VERSION
    tmp = {}
    for name, rows in T.items():
        tmp[name] = jd / f".{name}.move"
        tmp[name].write_text(json.dumps(rows, indent=2))
    tmp["sample_data.json"] = jd / ".sample_data.json.move"
    write_rows(tmp["sample_data.json"], (d for d in iter_rows(jd / "sample_data.json") if d["sample_token"] not in samples))
    tmp["ego_pose.json"] = jd / ".ego_pose.json.move"
    write_rows(tmp["ego_pose.json"], (e for e in iter_rows(jd / "ego_pose.json") if e["token"] not in gone_ego))
    for name, t in tmp.items():
        t.replace(jd / name)
    log(f"  {src.name}: - {len(log_tok)} logs, {len(sc_tok)} scenes, {len(samples)} samples")


def step_unlink(p: dict, log=print) -> None:
    """The source's names of what the target now has: sensor files (the target's rows say which), can_bus, ext,
    import records, and the logs' entries in the source's curation state."""
    import numpy as np
    src, dst, mapping = Path(p["src"]), Path(p["dst"]), p["mapping"]
    sc_tok = set(p["scene_tokens"])
    samples = _samples(dst, sc_tok)
    n = 0
    for d in iter_rows(dst / VERSION / "sample_data.json"):
        if d["sample_token"] not in samples:
            continue
        for f in _files(d["filename"]):
            a, b = src / f, dst / f
            if os.path.lexists(a):
                if not os.path.lexists(b) or not os.path.samestat(os.lstat(a), os.lstat(b)):
                    raise SystemExit(f"{b} is not the same file as {a}: not removing it")
                a.unlink()
                n += 1
    for f in sorted((src / "can_bus").glob("*.json")) if (src / "can_bus").exists() else []:
        m = CAN_RE.fullmatch(f.name)
        if m and m.group(1) in mapping and (dst / "can_bus" / f"{mapping[m.group(1)]}_{m.group(2)}").exists():
            f.unlink()
    for old, new in mapping.items():
        if (src / "ext" / old).is_dir() and (dst / "ext" / new).is_dir():
            shutil.rmtree(src / "ext" / old)
    for lg in p["logs"]:
        if (dst / f"{lg}.import.json").exists():
            (src / f"{lg}.import.json").unlink(missing_ok=True)
    sel = src / "selection"
    for name, key in (("human_decisions.json", "decisions"), ("human_labels.json", "labels")):
        j = _jread(sel / name, None)
        if j and isinstance(j.get(key), dict):
            j[key] = {k: v for k, v in j[key].items() if k not in mapping}
            _jwrite(sel / name, j)
    for name in ("lidar_diff.json", "scene_bytes.json"):
        j = _jread(sel / name, None)
        if j and isinstance(j.get("scenes"), dict):
            j["scenes"] = {t: v for t, v in j["scenes"].items() if t not in sc_tok}
            _jwrite(sel / name, j)
    j = _jread(sel / "object_counts.json", None)
    if j and isinstance(j.get("scenes"), dict):
        j["scenes"] = {k: v for k, v in j["scenes"].items() if (v or {}).get("token") not in sc_tok}
        _jwrite(sel / "object_counts.json", j)
    for f in sorted(sel.glob("*.npz")) if sel.is_dir() else []:
        with np.load(f) as z:
            keep = {k: z[k] for k in z.files if re.split(r"[.|]", k, 1)[0] not in sc_tok}
            changed = len(keep) != len(z.files)
        if changed:
            tmp = sel / (f.name[:-4] + ".tmp.npz")
            np.savez(tmp, **keep)
            tmp.replace(f)
    for d in (src / "samples", src / "sweeps"):
        for sub in d.iterdir() if d.is_dir() else []:
            try:
                sub.rmdir()
            except OSError:
                pass
    log(f"  {n} sensor files and the logs' other files removed from {src.name}")


def _confirmed_lists(root: Path, note: str) -> None:
    """keep.txt / drop.txt and decisions_log.txt of a volume after a move."""
    import apply_selection
    sel = root / "selection"
    if not sel.is_dir():
        return
    logs = {l["token"]: l["logfile"] for l in json.loads((root / VERSION / "log.json").read_text())}
    apply_selection.write_confirmed(str(sel), [{"name": s["name"], "log": logs.get(s["log_token"], ""), "token": s["token"]}
                                               for s in json.loads((root / VERSION / "scene.json").read_text())])
    with open(sel / "decisions_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === {note} ===\n")


STEPS = ("planned", "target", "side", "curation", "source", "unlink", "done")


def run_move(p: dict, log=print) -> None:
    dst, src = Path(p["dst"]), Path(p["src"])
    man = dst / "_moved" / f"{p['stamp']}.json"

    def mark(step):
        p["step"] = step
        _jwrite(man, p)
    if p.get("step") not in STEPS:
        mark("planned")
    log(f"{src.name} -> {dst.name}: {', '.join(p['logs'])} ({len(p['mapping'])} scenes, "
        f"{min(p['mapping'].values())} .. {max(p['mapping'].values())})")
    todo = STEPS[STEPS.index(p["step"]):]
    if "planned" in todo:
        step_target(p, log)
        mark("target")
    if "target" in todo:
        step_side_files(p, log)
        mark("side")
    if "side" in todo:
        step_curation(p, log)
        mark("curation")
    if "curation" in todo:
        step_source(p, log)
        mark("source")
    if "source" in todo:
        step_unlink(p, log)
        mark("unlink")
    if "unlink" in todo:
        _confirmed_lists(dst, f"{src.name}에서 녹화 {len(p['logs'])}개 (씬 {len(p['mapping'])}개) 옮겨 옴: {', '.join(p['logs'])}")
        _confirmed_lists(src, f"녹화 {len(p['logs'])}개 (씬 {len(p['mapping'])}개)를 {dst.name}로 옮김: {', '.join(p['logs'])}")
        mark("done")


def set_aside_if_empty(root: Path, log=print) -> None:
    """A volume every log moved out of: renamed to <name>.emptied-<stamp> (its _removed/ backups and curation
    history stay in it), so no tool sees an empty volume and the converter makes it again when it needs it."""
    if not root.is_dir() or json.loads((root / VERSION / "log.json").read_text()):
        return
    (root / LOCK_NAME).unlink(missing_ok=True)
    to = root.with_name(f"{root.name}.emptied-{time.strftime('%Y%m%d-%H%M%S')}")
    root.rename(to)
    log(f"  {root.name} has no log left: set aside as {to.name}")


class Locks:
    def __init__(self, roots):
        self.paths = [Path(r) / LOCK_NAME for r in dict.fromkeys(map(str, roots))]
        self.held = []

    def __enter__(self):
        for path in self.paths:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self.__exit__()
                raise SystemExit(f"{path} exists: another job writes that volume; run again when it is done")
            os.write(fd, f"pid {os.getpid()} move_logs".encode())
            os.close(fd)
            self.held.append(path)
        return self

    def __exit__(self, *exc):
        for path in self.held:
            path.unlink(missing_ok=True)
        self.held = []


def unfinished(std: Path, full: Path | None) -> list[dict]:
    out = []
    for fam in (volumes.volumes(std),) + ((volumes.volumes(full),) if full else ()):
        for v in fam:
            for f in sorted((v / "_moved").glob("*.json")) if (v / "_moved").is_dir() else []:
                p = _jread(f, {})
                if p.get("step") != "done":
                    out.append(p)
    return out


def main() -> None:
    data = os.environ.get("TCAR_DATA_ROOT", "/data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard", type=Path, default=Path(data) / "parsed" / "tcar_nuscenes")
    ap.add_argument("--full", type=Path, default=Path(data) / "parsed" / "tcar_nuscenes_full")
    ap.add_argument("--no-full", action="store_true", help="the standard set only")
    ap.add_argument("--fill", action="store_true", help="the oldest logs of the later volumes that fit, volume by volume")
    ap.add_argument("--into", type=int, help="with --from and --logs: the volume to move them into (1, 2, ...)")
    ap.add_argument("--from", dest="src", type=int, help="the volume they are in")
    ap.add_argument("--logs", default="", help="comma separated log names")
    ap.add_argument("--apply", action="store_true", help="do it (default: plan only)")
    a = ap.parse_args()
    std = volumes.base_of(a.standard)
    full = None if a.no_full or not (volumes.base_of(a.full) / VERSION).is_dir() else volumes.base_of(a.full)
    left = unfinished(std, full)
    if left:
        print("an interrupted move: " + "; ".join(f"{Path(p['src']).name} -> {Path(p['dst']).name} {p['logs']} at '{p['step']}'"
                                               for p in left))
        if not a.apply:
            print("plan only; --apply finishes it first")
            return
        with Locks([Path(p["dst"]) for p in left] + [Path(p["src"]) for p in left]):
            for p in left:
                run_move(p)
        for p in left:
            set_aside_if_empty(Path(p["src"]))
    moved = 0
    while True:
        if a.fill:
            plans = fill_plans(std, full)
        elif a.into and a.src and a.logs:
            p = plan(volumes.volume(std, a.into), volumes.volume(std, a.src), [x for x in a.logs.split(",") if x])
            plans = [(p, twin_plan(p, volumes.volume(full, a.into), volumes.volume(full, a.src)) if full else None)]
        else:
            raise SystemExit("give --fill, or --into, --from and --logs")
        if not plans:
            print("nothing to move" if not moved else f"done: {moved} moves")
            return
        for p, q in plans:
            m = p["mapping"]
            print(f"{Path(p['src']).name} -> {Path(p['dst']).name}: {len(p['logs'])} logs, {len(m)} scenes "
                  f"({min(m.values())} .. {max(m.values())})" + (f"; full {Path(q['src']).name} -> {Path(q['dst']).name}" if q else ""))
            for lg in p["logs"]:
                print(f"    {lg}")
        if not a.apply:
            print("plan only - nothing changed; add --apply (a later volume's room is planned after this move)")
            return
        roots = [Path(x[k]) for x in [p for p, _ in plans] + [q for _, q in plans if q] for k in ("src", "dst")]
        with Locks(roots):
            for p, q in plans:
                p["stamp"] = time.strftime("%Y%m%d-%H%M%S")
                run_move(p)
                if q:
                    q["stamp"] = p["stamp"]
                    run_move(q)
                moved += 1
                time.sleep(1)                                 # one manifest per second at most
        for p, q in plans:
            set_aside_if_empty(Path(p["src"]))
            if q:
                set_aside_if_empty(Path(q["src"]))
        if not a.fill:
            print(f"done: {moved} moves")
            return


if __name__ == "__main__":
    main()
