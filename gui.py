#!/usr/bin/env python3
"""Korean desktop workflow for T-Car ROS 1 bags → nuScenes, and the dataset's curation.

Run: python gui.py [--data <data root>]   (default /data)
Data layout: raw/<course>/<bag>, raw/_excluded/<course>/<bag>, parsed/tcar_nuscenes, logs/.
A course is the letter of a bag's name (A-1_2026-09-28-14-40-58.bag → A); every course
feeds the one dataset. Curation (curation/: detect, rule, review, apply) works on that
dataset; its review viewer runs inside the window. Conversion and curation remain in
the existing subprocess CLIs.
"""
from __future__ import annotations

import argparse
import codecs
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtNetwork, QtWidgets
    Signal = QtCore.Signal
except ImportError:
    from PyQt5 import QtCore, QtGui, QtNetwork, QtWidgets
    Signal = QtCore.pyqtSignal
try:                            # the review viewer inside the window; without it, a browser tab
    if QtCore.__name__.startswith("PySide6"):
        from PySide6.QtWebEngineWidgets import QWebEngineView
    else:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None

HERE = Path(__file__).resolve().parent
Qt = QtCore.Qt
SETTINGS = QtCore.QSettings("IRCV", "TCAR Parser")
COURSE_RE = re.compile(r"[A-Z][A-Z0-9]*")
BAG_RE = re.compile(r"(?P<course>[A-Z][A-Z0-9]*)-(?P<num>\d+)_(?P<ts>\d{4}(?:-\d\d){5})$")
SIDECARS = ("_integrity.csv", "_recorder.log")      # written next to every bag by the recorder
DATASET = "tcar_nuscenes"                           # parsed/<DATASET>: every course, one dataset
VERSION = "v1.0-trainval"
DEFAULT_ROOT = Path("/data")
ASSETS = HERE / "assets"
CURATION = HERE / "curation"
FONT = "Pretendard"
C = {
    "bg": "#0c0c0e", "side": "#111113", "surface": "#161618", "hover": "#1c1c1f",
    "line": "#232327", "line2": "#303036", "text": "#ececef", "sub": "#8e8e96",
    "faint": "#5f5f67", "accent": "#8b95ff", "ok": "#4cc38a", "warn": "#e8a94f",
    "bad": "#f07068", "muted": "#5f5f67",
}
QSS = """
* { font-family: '%(font)s', 'Noto Sans CJK KR', 'Malgun Gothic', sans-serif; font-size: 13px; color: %(text)s; }
QMainWindow, QDialog, QWidget#root, QScrollArea, QWidget#content { background: %(bg)s; }
QWidget#sidebar { background: %(side)s; border-right: 1px solid %(line)s; }
QLabel { background: transparent; }
QLabel#brand { font-size: 15px; font-weight: 600; }
QLabel#title { font-size: 22px; font-weight: 600; }
QLabel#actionTitle { font-size: 15px; font-weight: 600; }
QLabel#sub, QLabel#path { color: %(sub)s; }
QLabel#path { font-size: 12px; }
QLabel#faint { color: %(faint)s; font-size: 12px; }
QLabel#section { color: %(sub)s; font-size: 12px; font-weight: 500; }
QLabel#eyebrow { color: %(sub)s; font-size: 12px; font-weight: 500; }
QFrame#actionCard { background: %(surface)s; border: 1px solid %(line)s; border-radius: 10px; }
QFrame#emptyCard { background: transparent; border: 1px dashed %(line2)s; border-radius: 10px; }
QPushButton { background: %(surface)s; border: 1px solid %(line2)s; border-radius: 7px; padding: 7px 14px; font-weight: 500; }
QPushButton:hover { background: %(hover)s; border-color: #3c3c43; }
QPushButton:pressed { background: #202024; }
QPushButton:disabled { color: %(faint)s; background: transparent; border-color: %(line)s; }
QPushButton#primary { background: %(text)s; color: %(bg)s; border: none; font-weight: 600; padding: 8px 16px; }
QPushButton#primary:hover { background: #ffffff; }
QPushButton#primary:disabled { background: %(hover)s; color: %(faint)s; }
QPushButton#danger { color: %(bad)s; }
QPushButton#danger:hover { background: rgba(240, 112, 104, 20); border-color: rgba(240, 112, 104, 90); }
QPushButton#danger:disabled { color: %(faint)s; background: transparent; border-color: %(line)s; }
QPushButton#quiet { background: transparent; border: none; color: %(sub)s; padding: 6px 8px; }
QPushButton#quiet:hover { color: %(text)s; background: %(hover)s; }
QPushButton#quiet:disabled { color: %(faint)s; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { border: none; border-radius: 7px; margin: 1px 0; }
QListWidget::item:hover { background: %(hover)s; }
QListWidget::item:selected { background: #1f1f23; }
QTableWidget { background: transparent; border: none; outline: none; gridline-color: transparent;
               selection-background-color: transparent; }
QTableWidget::item { border-bottom: 1px solid %(line)s; }
QTableWidget::item:hover { background: %(surface)s; }
QTableWidget::item:selected { background: #1a1a1e; }
QHeaderView { background: transparent; }
QHeaderView::section { background: transparent; color: %(faint)s; border: none; border-bottom: 1px solid %(line)s;
                       padding: 8px 16px; font-size: 12px; font-weight: 500; }
QTableCornerButton::section { background: transparent; border: none; }
QLineEdit, QComboBox { background: %(surface)s; border: 1px solid %(line2)s; border-radius: 7px; padding: 6px 10px;
                       selection-background-color: #34385e; }
QLineEdit:focus, QComboBox:focus { border-color: %(accent)s; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: url(%(chevron)s); width: 14px; height: 14px; }
QComboBox QAbstractItemView { background: %(surface)s; border: 1px solid %(line2)s; outline: none; padding: 4px;
                              selection-background-color: %(hover)s; }
QCheckBox { spacing: 8px; color: %(sub)s; }
QCheckBox:hover { color: %(text)s; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px; border: 1px solid #45454d; background: %(surface)s; }
QCheckBox::indicator:hover { border-color: %(sub)s; }
QCheckBox::indicator:checked { background: %(text)s; border-color: %(text)s; image: url(%(check)s); }
QCheckBox::indicator:disabled { border-color: %(line)s; }
QFrame#jobBar { background: %(side)s; border-top: 1px solid %(line)s; }
QFrame#segment { background: %(surface)s; border: 1px solid %(line2)s; border-radius: 8px; }
QPushButton#seg { background: transparent; border: none; border-radius: 6px; padding: 6px 10px; color: %(sub)s; }
QPushButton#seg:hover { color: %(text)s; }
QPushButton#seg:checked { background: %(line2)s; color: %(text)s; font-weight: 600; }
QPushButton#seg:disabled { color: %(faint)s; }
QProgressBar { background: transparent; border: none; }
QProgressBar::chunk { background: %(accent)s; }
QPlainTextEdit { background: %(bg)s; border: 1px solid %(line)s; border-radius: 8px; padding: 8px;
                 font-family: 'JetBrains Mono', 'D2Coding', 'NanumGothicCoding', monospace; font-size: 12px; color: #b4b4bc; }
QScrollArea { border: none; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #2c2c32; border-radius: 3px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3a3a41; }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page { height: 0; background: none; }
QToolTip { background: #1f1f23; color: %(text)s; border: 1px solid %(line2)s; padding: 6px 8px; border-radius: 6px; }
QMessageBox QLabel { color: %(text)s; }
""" % dict(C, font=FONT, check=(ASSETS / "check.svg").as_posix(), chevron=(ASSETS / "chevron.svg").as_posix())


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return "—"


def is_recording(p: Path) -> bool:
    return p.is_file() and p.suffix == ".bag" and not p.name.startswith(".")


def route_key(name: str) -> tuple:
    """A-2 before A-10: the course letter, then the route number, then the name."""
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    return (m.group(1), int(m.group(2)), name) if m else ("~", 0, name)


def detect_route(p: Path) -> str | None:
    """The course of a bag: A for A-1_2026-09-28-14-40-58.bag."""
    m = BAG_RE.match(p.stem)
    return m["course"] if m else None


def _bag_time(stem: str) -> datetime | None:
    m = BAG_RE.match(stem)
    return datetime.strptime(m["ts"], "%Y-%m-%d-%H-%M-%S") if m else None


def companions(bag: Path) -> list[Path]:
    """The recorder's files for a bag. Their stamp can be a second off the bag's
    (A-6_..-15-02-16_recorder.log next to A-6_..-15-02-17.bag)."""
    m, when = BAG_RE.match(bag.stem), _bag_time(bag.stem)
    if not m:
        return []
    out = []
    for f in bag.parent.glob(f"{m['course']}-{m['num']}_*"):
        for suffix in SIDECARS:
            if f.name.endswith(suffix) and f.is_file():
                other = _bag_time(f.name[:-len(suffix)])
                if other and abs((other - when).total_seconds()) <= 5:
                    out.append(f)
    return sorted(out)


_U32 = struct.Struct("<I")


def _fields(header: bytes) -> dict[str, bytes]:
    out, i = {}, 0
    while i + 4 <= len(header):
        n = _U32.unpack_from(header, i)[0]
        name, _, value = header[i + 4:i + 4 + n].partition(b"=")
        out[name.decode(errors="replace")] = value
        i += 4 + n
    return out


_SPAN_CACHE: dict[tuple, float | None] = {}


def bag_span(bag: Path) -> float | None:
    """Recorded seconds, from the chunk infos in the bag's index (a few MB at its end)."""
    try:
        st = bag.stat()
    except OSError:
        return None
    key = (str(bag), st.st_size, st.st_mtime_ns)
    if key not in _SPAN_CACHE:
        span = None
        try:
            with bag.open("rb") as f:
                if f.readline() != b"#ROSBAG V2.0\n":
                    raise ValueError("not a ROS 1 bag")
                hlen = _U32.unpack(f.read(4))[0]
                index_pos = struct.unpack("<Q", _fields(f.read(hlen))["index_pos"])[0]
                if not index_pos:
                    raise ValueError("unindexed bag")
                f.seek(index_pos)
                buf, i, lo, hi = f.read(), 0, None, None
            while i + 4 <= len(buf):
                hlen = _U32.unpack_from(buf, i)[0]
                fields = _fields(buf[i + 4:i + 4 + hlen])
                i += 8 + hlen + _U32.unpack_from(buf, i + 4 + hlen)[0]
                if fields.get("op") == b"\x06":                         # chunk info
                    s, sn, e, en = struct.unpack("<IIII", fields["start_time"] + fields["end_time"])
                    lo = min(lo, s * 10**9 + sn) if lo is not None else s * 10**9 + sn
                    hi = max(hi, e * 10**9 + en) if hi is not None else e * 10**9 + en
            span = (hi - lo) / 1e9 if lo is not None else None
        except (OSError, ValueError, KeyError, struct.error):
            span = None
        _SPAN_CACHE[key] = span
    return _SPAN_CACHE[key]


def recording_info(p: Path) -> dict:
    m, when = BAG_RE.match(p.stem), _bag_time(p.stem)
    files = [p, *companions(p)]
    size = 0
    for f in files:
        try:
            size += f.stat().st_size
        except OSError:
            pass
    return {"name": p.stem, "path": str(p), "files": [str(f) for f in files],
            "label": f"{m['course']}-{m['num']}" if m else "",
            "date": f"{when:%Y-%m-%d}" if when else "", "when": f"{when:%Y-%m-%d %H:%M}" if when else "",
            "duration": bag_span(p), "size": size, "route": detect_route(p), "raw": True}


def parsed_status(dataroot: Path) -> dict:
    version = dataroot / VERSION
    empty = {"logs": {}, "samples_by_log": {}, "scenes": 0, "samples": 0, "names": [], "names_by_log": {},
             "imported": set(), "error": ""}
    if not version.exists():
        return empty
    try:
        logs = json.loads((version / "log.json").read_text())
        scenes = json.loads((version / "scene.json").read_text())
        by_token = {row["token"]: row["logfile"] for row in logs}
        per = {row["logfile"]: 0 for row in logs}
        per_samples = {row["logfile"]: 0 for row in logs}
        names_by_log = {row["logfile"]: [] for row in logs}
        samples = 0
        for scene in scenes:
            per[by_token[scene["log_token"]]] += 1
            per_samples[by_token[scene["log_token"]]] += scene["nbr_samples"]
            names_by_log[by_token[scene["log_token"]]].append(scene["name"])
            samples += scene["nbr_samples"]
        # parsed logs curation took every scene out of: gone from log.json, import record kept
        imported = set(per) | {f.name[:-len(".import.json")] for f in dataroot.glob("*.import.json")}
        return {"logs": per, "samples_by_log": per_samples, "scenes": len(scenes), "samples": samples,
                "names": sorted(sc["name"] for sc in scenes), "names_by_log": names_by_log,
                "imported": imported, "error": ""}
    except (OSError, ValueError, KeyError, TypeError):
        return dict(empty, error="데이터셋 목록을 읽을 수 없습니다. 검증 로그를 확인하세요.")


def course_status(status: dict, course: str) -> dict:
    """The dataset-wide status narrowed to one course's logs."""
    logs = {n: k for n, k in status["logs"].items() if detect_route(Path(n)) == course}
    return {"logs": logs, "scenes": sum(logs.values()),
            "samples": sum(status["samples_by_log"].get(n, 0) for n in logs),
            "imported": {n for n in status["imported"] if detect_route(Path(n)) == course},
            "error": status["error"]}


_IMPORT_CACHE: dict[tuple, dict | None] = {}


def import_stats(dataroot: Path, name: str) -> dict | None:
    """What a parsed log's <log>.import.json says about its time: the recording's span,
    the seconds that became scenes at parse time, and where the rest went."""
    path = dataroot / f"{name}.import.json"
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _IMPORT_CACHE:
        out = None
        try:
            data = json.loads(path.read_text())
            cw, plan, part = data["coverage_window"], data["frame_plan"], data["scene_partition"]
            lidar = cw["streams"]["LIDAR_TOP"]
            period = (lidar["last_ns"] - lidar["first_ns"]) / 1e9 / max(data["n_frames"]["LIDAR_TOP"] - 1, 1)
            orig = (cw["latest_ns"] - cw["earliest_ns"]) / 1e9
            used = part["frames_in_scenes"] * period
            short = part["frames_unused"] * period
            camera = plan["n_no_camera_set"] * period
            out = {"orig_s": orig, "parsed_s": used, "short_s": short, "camera_s": camera,
                   "other_s": max(orig - used - short - camera, 0.0),
                   "n_scenes": part["n_scenes"], "scene_s": part["frames_per_scene"] * period,
                   "lidar_gaps": plan.get("n_lidar_gaps", 0), "source": data.get("source_bag") or ""}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            out = None
        _IMPORT_CACHE[key] = out
    return _IMPORT_CACHE[key]


def log_time(dataroot: Path, name: str, scenes_now: int) -> dict | None:
    """import_stats with the dataset as it is now: scenes curation took out count as lost."""
    st = import_stats(dataroot, name)
    if not st:
        return None
    curated = max(st["n_scenes"] - scenes_now, 0)
    return dict(st, used_s=scenes_now * st["scene_s"], curated_n=curated, curated_s=curated * st["scene_s"])


def loss_text(st: dict) -> str:
    """One recording's time, from the original span down to the dataset."""
    parts = [f"원본 {st['orig_s']:.1f}초 중 데이터셋 {st['used_s']:.1f}초 ({100 * st['used_s'] / st['orig_s']:.1f}%)"]
    if st.get("curated_n"):
        parts.append(f"큐레이션으로 뺀 씬 {st['curated_n']}개 {st['curated_s']:.1f}초")
    for key, name in (("short_s", "20초 미만 자투리"), ("camera_s", "카메라 누락·동기 벗어남"),
                      ("other_s", "센서 시작·종료 구간")):
        if st[key] >= 0.05:
            parts.append(f"{name} {st[key]:.1f}초")
    if st["lidar_gaps"]:
        parts.append(f"LiDAR 프레임 끊김 {st['lidar_gaps']}회")
    return "\n".join(parts)


def dataset_only_info(dataroot: Path, name: str) -> dict:
    """A log in the dataset whose bag is not under raw/: what its import record says."""
    rec = {"name": name, "path": "", "files": [], "label": "", "date": "", "when": "",
           "duration": None, "size": None, "route": detect_route(Path(name)), "raw": False}
    m, when = BAG_RE.match(name), _bag_time(name)
    if m:
        rec.update(label=f"{m['course']}-{m['num']}", date=f"{when:%Y-%m-%d}", when=f"{when:%Y-%m-%d %H:%M}")
    st = import_stats(dataroot, name)
    if st:
        rec["duration"], rec["path"] = st["orig_s"], st["source"]
    return rec


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def decisions_dir() -> str:
    """Where the confirmed lists (human_decisions.json, keep.txt, drop.txt, deleted.txt) are copied besides the
    dataset's selection/: the saved choice, else this checkout's curation/ when it is writable (not in the
    AppImage). The review server and the jobs get it as $TCAR_DECISIONS_DIR."""
    saved = str(SETTINGS.value("decisions_dir", ""))
    if not saved and os.access(CURATION, os.W_OK):
        saved = str(CURATION)
        SETTINGS.setValue("decisions_dir", saved)
    return saved


def curation_data(dataroot: Path) -> dict:
    """curation/'s state for the whole dataset. A scene a filter caught is a candidate (후보, per filter: 객체 부족,
    정지 중복); the rest passed the filters (필터 통과). A person confirms in the review (남기기 확정 / 버리기 확정,
    human_decisions.json); the final deletion takes the confirmed 버리기 scenes out (삭제 완료, _removed/<stamp>/)."""
    sel = dataroot / "selection"

    def read(name):
        try:
            data = json.loads((sel / name).read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}
    decisions = read("human_decisions.json").get("decisions") or {}
    result = read("result.json")
    rule = result.get("scenes") or {}
    counts = read("object_counts.json")
    backups, deleted, deleted_logs = [], 0, []
    for m in sorted((dataroot / "_removed").glob("*/manifest.json")):
        if (m.parent / "UNDONE").exists():
            continue
        backups.append(m.parent)
        try:
            man = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        if man.get("done"):
            deleted += len(man.get("deleted") or [])
            deleted_logs += [x.get("log") for x in man.get("scenes") or []]   # for their time (older backups: none)
    return {"detected": set(counts.get("scenes") or {}), "ruled": set(rule),
            # locked by an earlier round's 최종 확정 (apply_selection.py --finalize): never deleted, never candidates
            "locked": {n for n, v in rule.items() if (v or {}).get("locked")},
            "rounds": read("locked.json").get("rounds") or [],
            "release": read("hf_release.json"),     # what went out to the Hub (scripts/hf_release.py writes it)
            # the filters' evidence: per-scene near road users, the rule's parameters, same-place stop groups
            "objects": {n: ((v.get("near") or {}).get("all") or {}).get("mean") for n, v in (counts.get("scenes") or {}).items()},
            "detector": {k: counts.get(k) for k in ("model", "conf", "near_px")},
            "params": result.get("params") or {},
            "stopped": sum((v or {}).get("stop_frac") is not None and v["stop_frac"] >= 0.8 for v in rule.values()),
            "stop_groups": [g for r in result.get("routes") or [] for g in r.get("stage1_groups") or []],
            "path_cover": [v["path_cover"] for v in rule.values() if (v or {}).get("path_cover") is not None],
            "filters": read("filters.json"),        # the app's settings: skipped filters, the overlap share
            "candidates": {n: FILTER_KEY.get((v or {}).get("kind"), "other") for n, v in rule.items()
                           if (v or {}).get("status") == "remove"},
            "human": {n: d["decision"] for n, d in decisions.items() if (d or {}).get("decision") in ("keep", "drop")},
            "backups": backups, "deleted": deleted, "deleted_logs": deleted_logs}


FILTER_KEY = {"정지 중복": "stops", "객체 부족": "objects", "경로 겹침": "overlap"}   # result.json `kind` -> state
CANDIDATES = ("stops", "objects", "overlap", "other")     # the filters' fixed order (curate.py); other kinds: "other"


def log_stage(data: dict, log: str) -> tuple:
    """(round number or None, release name or None) of a recording: the curation round whose 최종 확정 locked it,
    and the Hugging Face release it first went out in."""
    rnd = next((k for k, r in enumerate(data["rounds"], 1) if log in (r.get("logs") or [])), None)
    return rnd, ((data.get("release") or {}).get("logs") or {}).get(log)


def stage_text(data: dict, log: str, names: list) -> tuple:
    """Short label and colour: 새 데이터 (not finalized yet), 1라운드 · 미배포, 1라운드 · 배포 1002."""
    rnd, rel = log_stage(data, log)
    if any(scene_state(data, n) != "locked" for n in names) or rnd is None:
        return "새 데이터 · 확정 전", C["accent"]
    return (f"{rnd}라운드 · 배포 {rel}", C["ok"]) if rel else (f"{rnd}라운드 · 미배포", C["warn"])


def scene_state(data: dict, name: str) -> str:
    """keep / drop (confirmed by a person), objects / stops / other (a filter's candidate, not confirmed yet),
    pass (no filter caught it; also a scene the rule has not seen)."""
    if name in data["locked"]:
        return "locked"
    return data["human"].get(name) or data["candidates"].get(name) or "pass"


def curation_counts(data: dict, names: list[str]) -> dict:
    """Counts over some scenes (a course, a route): detected, judged by the rule, per state, and `cand` (all
    unconfirmed candidates)."""
    c = {"n": len(names), "detected": sum(n in data["detected"] for n in names),
         "ruled": sum(n in data["ruled"] for n in names), "keep": 0, "drop": 0, "pass": 0, "locked": 0,
         **dict.fromkeys(CANDIDATES, 0)}
    for n in names:
        c[scene_state(data, n)] += 1
    c["cand"] = sum(c[k] for k in CANDIDATES)
    return c


def curation_action(c: dict) -> str:
    """What to suggest next for a course; every action stays available regardless. The final deletion is
    for the whole dataset, on the page above the courses."""
    if not c["n"]:
        return "empty"
    if c["detected"] < c["n"]:
        return "detect"
    if c["ruled"] < c["n"]:
        return "rule"
    if c["cand"]:
        return "review"
    return "done"


def detector_candidates() -> list[str]:
    """Pythons that might have torch + ultralytics (the detector needs a GPU environment;
    the GUI's own Python, or the AppImage's, does not carry torch)."""
    home = Path.home()
    out = [str(SETTINGS.value("detect_python", "")), sys.executable]
    for base in (home / "anaconda3", home / "miniconda3", home / "miniforge3", home / "mambaforge",
                 Path("/opt/conda")):
        out += [str(p) for p in sorted((base / "envs").glob("*/bin/python"))] + [str(base / "bin" / "python")]
    out.append("/usr/bin/python3")
    seen, found = set(), []
    for p in out:
        if p and p not in seen and os.access(p, os.X_OK):
            seen.add(p)
            found.append(p)
    return found


def has_detector(python: str) -> bool:
    try:
        return subprocess.run([python, "-c", "import torch, ultralytics"], capture_output=True,
                              timeout=180).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def scan_source(root: Path, max_depth: int = 4) -> list[Path]:
    if is_recording(root):
        return [root]
    out = []

    def walk(path, depth):
        try:
            children = sorted(path.iterdir())
        except OSError:
            return
        for child in children:
            if child.name.startswith((".", "$")) or child.is_symlink():
                continue
            if is_recording(child):
                out.append(child)
            elif (child.is_dir() and depth < max_depth
                  and child.name not in ("lost+found", "System Volume Information")):
                walk(child, depth + 1)
    walk(root, 0)
    return sorted(out, key=lambda p: route_key(p.name))


def dataset_stamp(dataroot: Path) -> tuple:
    """Validation evidence: every table's name, mtime and size. Any table change (a parse,
    an exclusion) invalidates a successful check; a copy that keeps mtimes does not."""
    try:
        return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted((dataroot / VERSION).glob("*.json")))
    except OSError:
        return ()


def load_validation(data_root: Path) -> tuple:
    """The stamp of the last passed validation, kept in logs/validation.json across runs."""
    try:
        data = json.loads((data_root / "logs" / "validation.json").read_text())
        if data.get("dataset") == f"parsed/{DATASET}":
            return tuple(tuple(x) for x in data["stamp"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return ()


def save_validation(data_root: Path, stamp: tuple, log: Path | None) -> None:
    try:
        (data_root / "logs").mkdir(parents=True, exist_ok=True)
        (data_root / "logs" / "validation.json").write_text(json.dumps({
            "dataset": f"parsed/{DATASET}", "validated_at": datetime.now().isoformat(timespec="seconds"),
            "log": str(log) if log else "", "stamp": [list(x) for x in stamp]}, indent=1))
    except OSError:
        pass                                    # the check still holds for this session


class Job(QtCore.QObject):
    """One queue of subprocess, copy and move steps; no conversion code in the UI."""
    progress = Signal(float, str)
    output = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, title: str, steps: list, log_dir: Path):
        super().__init__()
        self.title, self.steps, self.i = title, steps, -1
        self.proc = None
        self.cancelled = False
        self.completed = False
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"gui_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
        self.log = self.log_path.open("w", encoding="utf-8")
        self._buf = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._bag = None                        # (k, n) of a converter's "[k/n] <bag>" line

    def start(self):
        self._next()

    def cancel(self):
        self.cancelled = True
        if self.proc is not None:
            self.proc.kill()

    def _emit(self, text):
        self.log.write(text + "\n")
        self.log.flush()
        self.output.emit(text)

    def _status(self, frac, extra=""):
        count = len(self.steps)
        step = self.steps[self.i][1] if 0 <= self.i < count else ""
        overall = (self.i + max(frac, 0)) / count if frac >= 0 and count else -1
        self.progress.emit(overall, f"{self.i + 1}/{count} · {step}" + (f" · {extra}" if extra else ""))

    def _next(self):
        if self.completed:
            return
        self.i += 1
        if self.cancelled:
            return self._done(False, "취소됨 · 완료된 단계는 유지됩니다")
        if self.i >= len(self.steps):
            return self._done(True, "완료")
        step = self.steps[self.i]
        self._emit(f"\n▶ {step[1]}")
        self._status(0)
        if step[0] == "proc":
            self._run_proc(step[2])
        else:
            QtCore.QTimer.singleShot(0, lambda: self._file_step(step))

    def _run_proc(self, argv):
        self._buf = ""
        self._bag = None
        self._decoder.reset()
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._proc_done)
        self.proc.errorOccurred.connect(self._proc_error)
        self._emit("$ " + " ".join(map(str, argv)))
        self._status(-1)
        self.proc.start(str(argv[0]), list(map(str, argv[1:])))

    def _read(self):
        if self.proc is None or self.completed:
            return
        self._buf += self._decoder.decode(bytes(self.proc.readAllStandardOutput()))
        parts = re.split(r"[\r\n]", self._buf)
        self._buf = parts.pop()
        for line in parts:
            if line.strip():
                self._emit(line)
            bag = re.match(r"\[(\d+)/(\d+)\] \S+\.bag$", line)   # not the "[1/5] Reading" stages
            if bag:
                self._bag = int(bag.group(1)), int(bag.group(2))
        matches = re.findall(r"(\d{1,3})%\|", "\n".join(parts) + self._buf)
        counted = re.findall(r"\((\d+)/(\d+)\)", "\n".join(parts))      # scene_detect: "scene-0001 (3/186)"
        if matches:
            frac = min(100, int(matches[-1])) / 100
            if self._bag:                       # several bags in one run: bar per bag
                k, n = self._bag
                self._status((k - 1 + frac) / n, f"bag {k}/{n}")
            else:
                self._status(frac)
        elif counted:
            k, n = map(int, counted[-1])
            self._status(k / max(n, 1), f"{k}/{n}")

    def _proc_error(self, error):
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            message = self.proc.errorString()
            self.proc = None
            self._done(False, f"실행할 수 없습니다: {message}")

    def _proc_done(self, code, status):
        if self.completed:
            return
        self._read()
        tail = self._buf + self._decoder.decode(b"", final=True)
        if tail.strip():
            self._emit(tail)
        self._buf = ""
        self.proc.deleteLater()
        self.proc = None
        if self.cancelled:
            return self._done(False, "취소됨 · 완료된 단계는 유지됩니다")
        if code != 0 or status == QtCore.QProcess.ExitStatus.CrashExit:
            return self._done(False, f"실패 (종료 코드 {code}) · 작업 로그를 확인하세요")
        QtCore.QTimer.singleShot(0, self._next)

    def _file_step(self, step):
        kind, _, src, dst = step
        src, dst = Path(src), Path(dst)
        part = None
        try:
            if self.cancelled:
                raise InterruptedError
            if dst.exists():
                raise FileExistsError(f"대상에 같은 이름이 있습니다: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if kind == "move":
                shutil.move(str(src), str(dst))
            else:
                if src.resolve() == dst.resolve():
                    raise ValueError("원본 위치로 복사할 수 없습니다")
                # Hidden partial file: never offered as a recording after cancellation.
                part = dst.parent / f".{dst.name}.part"
                total = src.stat().st_size or 1
                done, start = 0, time.monotonic()
                with src.open("rb") as reader, part.open("wb") as writer:
                    while chunk := reader.read(8 << 20):
                        if self.cancelled:
                            raise InterruptedError
                        writer.write(chunk)
                        done += len(chunk)
                        rate = done / max(time.monotonic() - start, .001)
                        self._status(done / total, f"{human_size(done)} / {human_size(total)} · {human_size(rate)}/s")
                        QtWidgets.QApplication.processEvents()
                shutil.copystat(src, part)
                if self.cancelled:
                    raise InterruptedError
                if dst.exists():
                    raise FileExistsError(str(dst))
                part.rename(dst)
                part = None
            self._emit(f"  {src} → {dst}")
        except InterruptedError:
            self._done(False, "취소됨 · 완료된 단계는 유지됩니다")
        except Exception as error:
            self._done(False, f"실패: {error}")
        else:
            QtCore.QTimer.singleShot(0, self._next)
        finally:
            if part is not None:
                part.unlink(missing_ok=True)

    def _done(self, ok, message):
        if self.completed:
            return
        self.completed = True
        self._emit(f"\n■ {self.title}: {message}")
        self.log.close()
        self.finished.emit(ok, message)


def label(text="", name="", wrap=False):
    widget = QtWidgets.QLabel(text)
    widget.setObjectName(name)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(wrap)
    return widget


def button(text, callback, name=""):
    widget = QtWidgets.QPushButton(text)
    widget.setObjectName(name)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.clicked.connect(callback)
    return widget


ACTION_COLOR = {"done": C["ok"], "import": C["faint"]}
# jobs that change the tables or selection/, after which the review viewer must reload
DATASET_WRITERS = ("parse", "exclude", "restore", "detect", "curate", "apply", "undo")
SUB_ROLE = Qt.ItemDataRole.UserRole + 1         # second, dimmer line of a cell
DOT_ROLE = Qt.ItemDataRole.UserRole + 2         # colour of a status dot before the text


def _weight(name):
    return getattr(QtGui.QFont.Weight, name) if hasattr(QtGui.QFont, "Weight") else getattr(QtGui.QFont, name)


class CellDelegate(QtWidgets.QStyledItemDelegate):
    """Draws a cell as `text` or `text / sub` on two lines, with an optional status dot."""
    def __init__(self, parent=None, pad=16):
        super().__init__(parent)
        self.pad = pad

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text, opt.text = opt.text, ""
        opt.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus
        widget = option.widget
        (widget.style() if widget else QtWidgets.QApplication.style()).drawControl(
            QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        rect = option.rect.adjusted(self.pad, 0, -10, 0)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        dot = index.data(DOT_ROLE)
        if dot:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(dot))
            painter.drawEllipse(QtCore.QRectF(rect.left(), rect.center().y() - 3, 7, 7))
            rect.setLeft(rect.left() + 16)
        fg = index.data(Qt.ItemDataRole.ForegroundRole)
        color = fg.color() if isinstance(fg, QtGui.QBrush) else QtGui.QColor(fg) if fg else QtGui.QColor(C["text"])
        sub = index.data(SUB_ROLE)
        font = QtGui.QFont(opt.font)
        painter.setPen(color)
        if sub:
            font.setWeight(_weight("Medium"))
            small = QtGui.QFont(opt.font)
            small.setPixelSize(12)
            half = rect.height() // 2
            painter.setFont(font)
            painter.drawText(rect.adjusted(0, 0, 0, -half + 1), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                             QtGui.QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, rect.width()))
            painter.setFont(small)
            painter.setPen(QtGui.QColor(C["sub"]))
            painter.drawText(rect.adjusted(0, half + 3, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                             QtGui.QFontMetrics(small).elidedText(sub, Qt.TextElideMode.ElideRight, rect.width()))
        else:
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             QtGui.QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, rect.width()))
        painter.restore()


class Stepper(QtWidgets.QWidget):
    """가져오기 → 파싱 → 검증 as numbered dots on a line; done steps get a check."""
    NAMES = ("가져오기", "파싱", "검증")

    def __init__(self):
        super().__init__()
        self.active, self.done = 0, 0
        self.setFixedHeight(30)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_state(self, active: int, done: int):
        """active: the current step (-1 = none); done: how many steps are finished."""
        self.active, self.done = active, done
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        font = QtGui.QFont(self.font())
        font.setPixelSize(13)
        bold = QtGui.QFont(font)
        bold.setWeight(_weight("DemiBold"))
        num_font = QtGui.QFont(bold)
        num_font.setPixelSize(11)
        r, cy, x, gap = 10, self.height() / 2, 0.5, 14
        links = len(self.NAMES) - 1
        widths = [QtGui.QFontMetrics(bold).horizontalAdvance(n) for n in self.NAMES]
        line_len = max(24.0, (self.width() - 1 - sum(2 * r + 8 + w for w in widths) - 2 * gap * links) / links)
        for i, name in enumerate(self.NAMES):
            done, active = i < self.done, i == self.active
            circle = QtCore.QRectF(x, cy - r, 2 * r, 2 * r)
            if done:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor(C["ok"]))
                p.drawEllipse(circle)
                pen = QtGui.QPen(QtGui.QColor(C["bg"]), 1.8)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x + 6, cy + 0.3), QtCore.QPointF(x + 8.8, cy + 3),
                                                QtCore.QPointF(x + 14, cy - 3)]))
            else:
                p.setPen(QtGui.QPen(QtGui.QColor(C["text"] if active else C["line2"]), 1.4))
                p.setBrush(QtGui.QColor(C["text"]) if active else QtGui.QBrush(Qt.BrushStyle.NoBrush))
                p.drawEllipse(circle.adjusted(0.7, 0.7, -0.7, -0.7))
                p.setFont(num_font)
                p.setPen(QtGui.QColor(C["bg"] if active else C["faint"]))
                p.drawText(circle, Qt.AlignmentFlag.AlignCenter, str(i + 1))
            x += 2 * r + 8
            p.setFont(bold if active else font)
            p.setPen(QtGui.QColor(C["text"] if active else C["sub"] if done else C["faint"]))
            p.drawText(QtCore.QRectF(x, 0, widths[i] + 4, self.height()), Qt.AlignmentFlag.AlignVCenter, name)
            x += widths[i]
            if i < links:
                x += gap
                p.setPen(QtGui.QPen(QtGui.QColor(C["ok"] if i < self.done else C["line2"]), 1.2))
                p.drawLine(QtCore.QPointF(x, cy), QtCore.QPointF(x + line_len, cy))
                x += line_len + gap


class RatioBar(QtWidgets.QWidget):
    """One long bar: (seconds, colour) segments over the whole recorded time."""
    def __init__(self, height=8):
        super().__init__()
        self.parts, self.total = [], 0.0
        self.setFixedHeight(height)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def set_parts(self, parts, total):
        self.parts, self.total = parts, total
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        r = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setClipPath(clip)
        p.fillRect(r, QtGui.QColor(C["line2"]))
        x = r.left()
        for value, colour in self.parts:
            if self.total > 0 and value > 0:
                w = r.width() * value / self.total
                p.fillRect(QtCore.QRectF(x, r.top(), w + 0.5, r.height()), QtGui.QColor(colour))
                x += w


BAR_ROLE = Qt.ItemDataRole.UserRole + 4


class BarDelegate(CellDelegate):
    """A recording's time as one long bar of segments; every row on the same time scale, so a longer
    recording has a longer bar."""
    def paint(self, painter, option, index):
        parts, total, longest = index.data(BAR_ROLE) or ([], 0.0, 0.0)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus
        style = option.widget.style() if option.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)
        if total <= 0 or longest <= 0:
            return
        r = QtCore.QRectF(option.rect.adjusted(12, 0, -12, 0))
        r.setTop(r.center().y() - 4)
        r.setHeight(8)
        r.setWidth(max(r.width() * total / longest, 4))
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(r, 4, 4)
        painter.setClipPath(clip)
        painter.fillRect(r, QtGui.QColor(C["line2"]))
        x = r.left()
        for value, colour in parts:
            if value > 0:
                w = r.width() * value / total
                painter.fillRect(QtCore.QRectF(x, r.top(), w + 0.5, r.height()), QtGui.QColor(colour))
                x += w
        painter.restore()


# the dataset overview's segments (parser page)
OV_PARSER = (("used", "데이터셋", C["accent"]), ("curated", "큐레이션으로 뺌", "#8c4a46"),
             ("lost", "파싱 때 버림", "#46464f"), ("waiting", "파싱 대기", "#6b6b75"))
# curation: a scene's state over the whole dataset (confirmed, candidates, deleted)
CUR_STATES = (("locked", "잠김 (이전 라운드 확정)", "#4f6f8f"),
              ("keep", "남기기 확정 (사람)", C["ok"]), ("pass", "남기기 확정 (필터 통과)", "#3f7a5c"),
              ("drop", "버리기 확정", C["bad"]),
              ("stops", "후보 · 정지 중복", "#c792ea"), ("objects", "후보 · 객체 부족", C["warn"]),
              ("overlap", "후보 · 경로 겹침", "#5ec8e5"), ("other", "후보 · 기타", "#d4a373"),
              ("deleted", "삭제 완료", "#8c4a46"))
ROUTE_COLORS = ("#8b95ff", "#4cc38a", "#e8a94f", "#f07068", "#5ec8e5", "#c792ea", "#f2d06b", "#7fd1b9",
                "#ff9e64", "#a3be8c", "#e06c9f", "#6cb6ff", "#d4a373", "#9ccfd8", "#eb6f92", "#b4f9f8",
                "#ffd580", "#9aa5ce", "#73daca", "#f7768e", "#bb9af7", "#e0af68")


def mercator(lat, lon):
    """Web Mercator in [0, 1] x [0, 1]: the OSM tile grid at zoom 0."""
    s = math.sin(math.radians(max(min(lat, 85.0), -85.0)))
    return (lon + 180.0) / 360.0, 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


class MiniHist(QtWidgets.QWidget):
    """A small bar chart: bars of (value, colour, tip), an optional marker line before bar `marker` with its text,
    and a caption under each end. Heights on a square-root scale, so a small bar next to a tall one still shows
    (the tips give the counts)."""
    def __init__(self, width=280, height=74):
        super().__init__()
        self.setFixedSize(width, height)
        self.bars, self.marker, self.marker_text, self.left, self.right = [], None, "", "", ""

    def set_bars(self, bars, marker=None, marker_text="", left="", right=""):
        self.bars, self.marker, self.marker_text, self.left, self.right = bars, marker, marker_text, left, right
        self.setToolTip("\n".join(tip for _, _, tip in bars if tip))
        self.update()

    def paintEvent(self, _):
        if not self.bars:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        font = p.font()
        font.setPixelSize(10)
        p.setFont(font)
        w, h, top, bottom = self.width(), self.height(), 14, 16
        top_v = math.sqrt(max(v for v, _, _ in self.bars) or 1)
        step = w / len(self.bars)
        for i, (v, colour, _) in enumerate(self.bars):
            bh = (h - top - bottom) * math.sqrt(v) / top_v
            p.fillRect(QtCore.QRectF(i * step + 1, h - bottom - bh, step - 2, bh), QtGui.QColor(colour))
        p.fillRect(QtCore.QRectF(0, h - bottom, w, 1), QtGui.QColor(C["line2"]))
        p.setPen(QtGui.QColor(C["faint"]))
        p.drawText(QtCore.QRectF(0, h - bottom + 2, w, bottom), int(Qt.AlignmentFlag.AlignLeft), self.left)
        p.drawText(QtCore.QRectF(0, h - bottom + 2, w, bottom), int(Qt.AlignmentFlag.AlignRight), self.right)
        if self.marker is not None:
            x = self.marker * step
            pen = QtGui.QPen(QtGui.QColor(C["text"]), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x, top - 2), QtCore.QPointF(x, h - bottom))
            p.setPen(QtGui.QColor(C["text"]))
            p.drawText(QtCore.QRectF(x + 4, 0, w - x, top), int(Qt.AlignmentFlag.AlignLeft), self.marker_text)


FREQ_COLOUR, FREQ_ALPHA = (94, 200, 229), 0.28    # the frequency view: every kept path this colour, see-through


def freq_swatch(k, bg=(16, 16, 19)):
    """The colour k overlapping paths add up to on the map's background."""
    a = 1 - (1 - FREQ_ALPHA) ** k
    return "#%02x%02x%02x" % tuple(round(b_ * (1 - a) + c * a) for b_, c in zip(bg, FREQ_COLOUR))


class CurationMap(QtWidgets.QWidget):
    """Every scene's GNSS track on OpenStreetMap, coloured by route or by curation state. Tiles come through the
    review server's /tile/ proxy (cached on disk), darkened like the review's own map. Wheel = zoom,
    drag = pan, click a track = pick its route, double-click = review that scene."""
    routePicked = Signal(str)
    sceneOpened = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(400)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.net = QtNetwork.QNetworkAccessManager(self)
        self.base = None                        # the review server, for /tile/
        self.tiles, self.pending, self.failed = {}, {}, {}   # pending: key -> reply (held, or PyQt drops the slot)
        self.tracks = []                        # (scene, log, [(x, y), ...] Web Mercator, bounding box)
        self.colors, self.dim, self.tips, self.legend, self.rank, self.points, self.hidden = {}, set(), {}, [], {}, {}, set()
        self.z, self.cx, self.cy, self.moved = 3.0, 0.5, 0.5, False
        self.roi, self.zmin = None, 2.0             # our region: zoom out and pan stop at it
        self._press, self.hover = None, None
        self.message = "지도를 준비하는 중… 검토 서버가 데이터셋의 GNSS 경로를 읽습니다"

    def set_base(self, url):
        if url != self.base:
            self.base, self.failed = url, {}
            self.update()

    def set_tracks(self, scenes):
        self.tracks = []
        for s in scenes:
            pts = [mercator(a, o) for a, o in s.get("pts") or []]
            if pts:
                xs, ys = [q[0] for q in pts], [q[1] for q in pts]
                self.tracks.append((s["name"], s["log"], pts, (min(xs), min(ys), max(xs), max(ys))))
        self.message = "" if self.tracks else "GNSS 경로가 있는 씬이 없습니다"
        self.roi = None
        if self.tracks:                             # our region: every track, 8 % around (always, moved or not)
            x0, y0 = min(t[3][0] for t in self.tracks), min(t[3][1] for t in self.tracks)
            x1, y1 = max(t[3][2] for t in self.tracks), max(t[3][3] for t in self.tracks)
            mx, my = (x1 - x0) * 0.08 + 1e-7, (y1 - y0) * 0.08 + 1e-7
            self.roi = (x0 - mx, y0 - my, x1 + mx, y1 + my)
        if not self.moved:
            self.fit()
        else:
            self._clamp()
        self.update()

    def set_style(self, colors, dim, tips, legend, rank=None, points=None, hidden=None):
        """colors: scene -> colour; dim: scenes drawn faint (outside the picked course / routes); tips: scene ->
        hover text; legend: rows of (text, colour) runs; rank: scene -> drawing order (higher on top);
        points: scene -> a colour per point (the segment after it), instead of one colour."""
        self.colors, self.dim, self.tips, self.legend = colors, dim, tips, legend
        self.rank, self.points, self.hidden = rank or {}, points or {}, hidden or set()
        self.update()

    def _fit_zoom(self):
        """The zoom at which our region just fills the map at its current size."""
        x0, y0, x1, y1 = self.roi
        w, h = max(self.width(), 100), max(self.height(), 100)
        return max(2.0, min(17.0, math.log2(min(w / max(x1 - x0, 1e-9), h / max(y1 - y0, 1e-9)) / 256)))

    def fit(self):
        """The whole region in view; zooming out stops here."""
        self.moved = False
        if self.roi:
            x0, y0, x1, y1 = self.roi
            self.z = self.zmin = self._fit_zoom()
            self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.update()

    def _clamp(self):
        """Keep the view on our region: no wider than it, the center only as far as its edge reaches the view's."""
        if not self.roi:
            return
        self.zmin = self._fit_zoom()                # the map may have been resized
        self.z = max(self.z, self.zmin)
        s, (x0, y0, x1, y1) = self._scale(), self.roi
        hw, hh = self.width() / 2 / s, self.height() / 2 / s
        self.cx = (x0 + x1) / 2 if x1 - x0 <= 2 * hw else min(max(self.cx, x0 + hw), x1 - hw)
        self.cy = (y0 + y1) / 2 if y1 - y0 <= 2 * hh else min(max(self.cy, y0 + hh), y1 - hh)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.moved:
            self.fit()
        else:
            self._clamp()

    def _scale(self):
        return 256.0 * 2 ** self.z

    def _screen(self, x, y):
        s = self._scale()
        return (x - self.cx) * s + self.width() / 2, (y - self.cy) * s + self.height() / 2

    def _request(self, key):
        if key in self.pending or self.failed.get(key, 0) > time.monotonic() or len(self.pending) >= 12:
            return
        reply = self.net.get(QtNetwork.QNetworkRequest(QtCore.QUrl(f"{self.base}/tile/{key[0]}/{key[1]}/{key[2]}.png")))
        self.pending[key] = reply
        reply.finished.connect(lambda k=key: self._tile(k))

    def _tile(self, key):
        reply = self.pending.pop(key, None)
        if reply is None:
            return
        image = QtGui.QImage.fromData(bytes(reply.readAll()))
        reply.deleteLater()
        if image.isNull():                      # offline, or the server went away: try again later
            self.failed[key] = time.monotonic() + 30
            return
        image = image.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
        image.invertPixels()                    # OSM is light; the app is dark
        self.tiles[key] = QtGui.QPixmap.fromImage(image)
        while len(self.tiles) > 800:
            self.tiles.pop(next(iter(self.tiles)))
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h, s = self.width(), self.height(), self._scale()
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(QtCore.QRectF(self.rect()), 8, 8)
        p.setClipPath(clip)
        p.fillRect(self.rect(), QtGui.QColor("#101013"))
        zi = int(max(0, min(19, round(self.z))))
        n, left, top = 2 ** zi, self.cx - w / 2 / s, self.cy - h / 2 / s
        p.save()
        p.setOpacity(0.5)
        for tx in range(math.floor(left * n), math.floor((left + w / s) * n) + 1):
            for ty in range(max(0, math.floor(top * n)), min(n - 1, math.floor((top + h / s) * n)) + 1):
                key = (zi, tx % n, ty)
                pm = self.tiles.get(key)
                if pm is None:
                    if self.base:
                        self._request(key)
                    continue
                x, y = (tx / n - self.cx) * s + w / 2, (ty / n - self.cy) * s + h / 2
                p.drawPixmap(QtCore.QRectF(x, y, s / n + 0.6, s / n + 0.6), pm, QtCore.QRectF(pm.rect()))
        p.restore()
        # faint tracks under, the picked course over them, the hovered one on top
        for name, _log, pts, _box in sorted(self.tracks, key=lambda t: (t[0] not in self.dim, self.rank.get(t[0], 0),
                                                                         t[0] == self.hover)):
            if name in self.hidden:
                continue
            colour = QtGui.QColor(self.colors.get(name, C["sub"]))
            faint = name in self.dim
            if faint:
                colour.setAlphaF(0.3)
            pen = QtGui.QPen(colour, 5.0 if name == self.hover else 1.5 if faint else 2.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            per = self.points.get(name)
            if per:                             # overlap: every segment its own colour
                sp = [QtCore.QPointF(*self._screen(*q)) for q in pts]
                for i in range(max(len(sp) - 1, 1)):
                    c = QtGui.QColor(per[min(i, len(per) - 1)])
                    if faint:
                        c.setAlphaF(0.3)
                    pen.setColor(c)
                    p.setPen(pen)
                    p.drawLine(sp[i], sp[min(i + 1, len(sp) - 1)])
                continue
            path = QtGui.QPainterPath()
            path.moveTo(*self._screen(*pts[0]))
            for q in pts[1:]:
                path.lineTo(*self._screen(*q))
            p.drawPath(path)
        font = p.font()
        if self.legend:
            font.setPixelSize(12)
            p.setFont(font)
            fm = QtGui.QFontMetrics(font)
            line = fm.height() + 5
            width = max(sum(fm.horizontalAdvance(t) for t, _ in row) for row in self.legend) + 22
            box = QtCore.QRectF(10, 10, width, line * len(self.legend) + 12)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(12, 12, 14, 215))
            p.drawRoundedRect(box, 7, 7)
            p.setBrush(Qt.BrushStyle.NoBrush)
            y = box.top() + 6
            for row in self.legend:
                x = box.left() + 11
                for text, colour in row:
                    p.setPen(QtGui.QColor(colour or C["text"]))
                    p.drawText(QtCore.QPointF(x, y + fm.ascent() + 2), text)
                    x += fm.horizontalAdvance(text)
                y += line
        font.setPixelSize(10)
        p.setFont(font)
        p.setPen(QtGui.QColor(C["faint"]))
        p.drawText(QtCore.QRectF(0, 0, w - 8, h - 5), int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom),
                   "© OpenStreetMap")
        if self.message:
            font.setPixelSize(13)
            p.setFont(font)
            p.setPen(QtGui.QColor(C["sub"]))
            p.drawText(QtCore.QRectF(self.rect()), int(Qt.AlignmentFlag.AlignCenter), self.message)

    def _nearest(self, pos):
        """(scene, log) of the track under the cursor (within 8 px), the picked course's first."""
        s, px, py = self._scale(), pos.x(), pos.y()
        mx, my = self.cx + (px - self.width() / 2) / s, self.cy + (py - self.height() / 2) / s
        pad, best, dist = 10 / s, None, 8.0
        for name, log, pts, (x0, y0, x1, y1) in self.tracks:
            if name in self.hidden or not (x0 - pad <= mx <= x1 + pad and y0 - pad <= my <= y1 + pad):
                continue
            sp = [self._screen(*q) for q in pts] or []
            extra = 3.0 if name in self.dim else 0.0
            for a, b in zip(sp, sp[1:] or sp):
                d = _seg_dist(px, py, *a, *b) + extra
                if d < dist:
                    best, dist = (name, log), d
        return best

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = (e.pos(), self.cx, self.cy, False)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._press:
            p0, cx, cy, _ = self._press
            d = e.pos() - p0
            if abs(d.x()) + abs(d.y()) > 3:
                s = self._scale()
                self.cx, self.cy, self.moved = cx - d.x() / s, cy - d.y() / s, True
                self._press = (p0, cx, cy, True)
                self._clamp()
                self.update()
            return
        hit = self._nearest(e.pos())
        name = hit[0] if hit else None
        if name != self.hover:
            self.hover = name
            self.update()
        where = e.globalPosition().toPoint() if hasattr(e, "globalPosition") else e.globalPos()
        if hit:
            QtWidgets.QToolTip.showText(where, self.tips.get(name, name), self)
        else:
            QtWidgets.QToolTip.hideText()
        self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, e):
        press, self._press = self._press, None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if press and not press[3]:
            hit = self._nearest(e.pos())
            if hit:
                self.routePicked.emit(hit[1])

    def mouseDoubleClickEvent(self, e):
        hit = self._nearest(e.pos())
        if hit:
            self.sceneOpened.emit(*hit)

    def wheelEvent(self, e):
        e.accept()                                  # the map's, not the page's: the page must not scroll too
        steps = e.angleDelta().y() / 120
        if not steps:
            return
        pos = e.position() if hasattr(e, "position") else e.posF()
        s = self._scale()
        mx, my = self.cx + (pos.x() - self.width() / 2) / s, self.cy + (pos.y() - self.height() / 2) / s
        self.z = max(self.zmin, min(18.0, self.z + 0.5 * steps))   # not wider than our region
        s = self._scale()
        self.cx, self.cy = mx - (pos.x() - self.width() / 2) / s, my - (pos.y() - self.height() / 2) / s
        self.moved = True
        self._clamp()
        self.update()

    def leaveEvent(self, _):
        if self.hover:
            self.hover = None
            self.update()


def setup_table(table, headers):
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setItemDelegate(CellDelegate(table))
    table.setMouseTracking(True)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(56)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    table.horizontalHeader().setStretchLastSection(False)


class ImportDialog(QtWidgets.QDialog):
    """Choose a source, review destinations, then copy; scanning never writes data."""
    def __init__(self, parent, data_root: Path, routes: list[str], parsed_logs: set[str],
                 source: Path | None = None):
        super().__init__(parent)
        self.data_root, self.routes, self.rows = data_root, routes, []
        self.parsed_logs = parsed_logs
        self.setWindowTitle("녹화 가져오기 · TCAR Parser")
        self.resize(1000, 740)
        self.setMinimumSize(860, 660)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 24)
        layout.setSpacing(12)
        layout.addWidget(label("녹화 가져오기", "title"))
        layout.addWidget(label("원본은 그대로 두고 데이터 폴더로 복사합니다. bag 옆의 integrity·recorder 기록도 함께 복사합니다.", "sub", True))
        layout.addSpacing(8)
        source_row = QtWidgets.QHBoxLayout()
        self.src = QtWidgets.QLineEdit(str(source) if source else str(SETTINGS.value("last_source", "")))
        self.src.setPlaceholderText("외장 SSD 또는 bag이 들어 있는 폴더를 선택하세요")
        self.src.setAccessibleName("가져올 원본 폴더")
        self.src.returnPressed.connect(self._scan)
        self.src.textChanged.connect(self._source_changed)
        source_row.addWidget(self.src, 1)
        source_row.addWidget(button("폴더 선택…", self._browse))
        self.scan_btn = button("녹화 찾기", self._scan)
        source_row.addWidget(self.scan_btn)
        layout.addLayout(source_row)
        review_row = QtWidgets.QHBoxLayout()
        review_row.addWidget(label("찾은 녹화", "section"))
        review_row.addStretch()
        self.select_all = QtWidgets.QCheckBox("전체 선택")
        self.select_all.setEnabled(False)
        self.select_all.clicked.connect(self._select_all)
        review_row.addWidget(self.select_all)
        layout.addLayout(review_row)
        self.stack = QtWidgets.QStackedWidget()
        empty = QtWidgets.QFrame()
        empty.setObjectName("emptyCard")
        empty_layout = QtWidgets.QVBoxLayout(empty)
        empty_layout.setContentsMargins(32, 36, 32, 36)
        empty_layout.addStretch()
        self.empty_title = label("원본 폴더를 선택하면 녹화를 찾습니다", "actionTitle", True)
        self.empty_desc = label(".bag 파일을 찾습니다.\n하위 4단계까지 검색하며, bag 파일이 든 폴더를 선택하면 됩니다.", "sub", True)
        empty_layout.addWidget(self.empty_title)
        empty_layout.addWidget(self.empty_desc)
        empty_layout.addStretch()
        self.stack.addWidget(empty)
        self.table = QtWidgets.QTableWidget(0, 4)
        setup_table(self.table, ["", "녹화", "저장할 코스", "상태"])
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        for column, width in ((0, 48), (2, 150), (3, 250)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.stack.addWidget(self.table)
        layout.addWidget(self.stack, 1)
        self.destination = label(f"{data_root}/raw/<코스>/ 로 복사합니다. 코스는 bag 이름의 알파벳(A-1 → A)으로 채웁니다.", "faint", True)
        layout.addWidget(self.destination)
        self.summary = label("폴더를 선택하고 가져올 녹화를 확인하세요.", "sub", True)
        layout.addWidget(self.summary)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button("취소", self.reject))
        self.go = button("녹화 선택 후 가져오기", self.accept, "primary")
        self.go.setEnabled(False)
        footer.addWidget(self.go)
        layout.addLayout(footer)
        if source:
            QtCore.QTimer.singleShot(0, self._scan)

    def _source_changed(self):
        self.rows = []
        self.table.setRowCount(0)
        self.stack.setCurrentIndex(0)
        self.empty_title.setText("원본 폴더에서 녹화를 찾아주세요")
        self.empty_desc.setText("폴더 선택 또는 ‘녹화 찾기’를 누르면 가져올 목록이 나타납니다.")
        self.select_all.setEnabled(False)
        self.select_all.setChecked(False)
        self.go.setEnabled(False)
        self.summary.setText("원본 폴더의 내용은 변경하지 않습니다.")

    def _browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "가져올 원본 폴더 선택", self.src.text())
        if path:
            self.src.setText(path)
            self._scan()

    def _scan(self):
        source = self.src.text().strip()
        self._source_changed()
        if not source or not Path(source).expanduser().exists():
            self.empty_title.setText("폴더를 찾을 수 없습니다")
            self.empty_desc.setText("원본 장치의 연결과 폴더 경로를 확인한 뒤 다시 선택하세요.")
            return
        root = Path(source).expanduser().resolve()
        SETTINGS.setValue("last_source", str(root))
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            recordings = [dict(recording_info(p), folder=str(p.parent.relative_to(root)) if p != root else "")
                          for p in scan_source(root)]
        except (OSError, ValueError) as error:
            self.empty_title.setText("폴더를 읽을 수 없습니다")
            self.empty_desc.setText(f"접근 권한과 연결을 확인하세요.\n{error}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not recordings:
            self.empty_title.setText("이 폴더에서 녹화를 찾지 못했습니다")
            self.empty_desc.setText(".bag 파일이 든 폴더 또는 그 상위 폴더를 선택하세요.\n폴더가 깊이 있으면 더 안쪽 폴더를 선택해 주세요.")
            self.summary.setText("찾은 녹화 0개 · 다른 폴더를 선택할 수 있습니다.")
            return
        self.table.setRowCount(len(recordings))
        for row, rec in enumerate(recordings):
            check = QtWidgets.QCheckBox()
            check.setAccessibleName(f"{rec['name']} 가져오기")
            holder = QtWidgets.QWidget()
            check_layout = QtWidgets.QHBoxLayout(holder)
            check_layout.setContentsMargins(16, 0, 0, 0)
            check_layout.addWidget(check)
            self.table.setCellWidget(row, 0, holder)
            folder = rec["folder"] if rec["folder"] not in ("", ".") else ""
            duration = f"{rec['duration'] / 60:.1f}분" if rec["duration"] else ""
            detail = " · ".join(filter(None, [folder, duration, human_size(rec["size"])]))
            item = QtWidgets.QTableWidgetItem(rec["name"])
            item.setData(SUB_ROLE, detail)
            item.setToolTip("\n".join(rec["files"]))
            self.table.setItem(row, 1, item)
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            combo.addItems(sorted(set(self.routes) | ({rec["route"]} if rec["route"] else set())))
            combo.setCurrentText(rec["route"] or "")
            combo.lineEdit().setPlaceholderText("예: A")
            combo.setAccessibleName(f"{rec['name']} 저장할 코스")
            combo.setFixedHeight(34)
            holder_combo = QtWidgets.QWidget()
            combo_layout = QtWidgets.QHBoxLayout(holder_combo)
            combo_layout.setContentsMargins(8, 0, 12, 0)
            combo_layout.addWidget(combo)
            self.table.setCellWidget(row, 2, holder_combo)
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem())
            failed = any(part.lower() == "fail" for part in Path(folder).parts)
            duplicate = rec["route"] and self._exists(rec["route"], rec["name"])
            check.setChecked(not duplicate and not failed and rec["name"] not in self.parsed_logs)
            self.rows.append((check, Path(rec["path"]), combo, rec["size"], failed))
            check.stateChanged.connect(self._update)
            combo.currentTextChanged.connect(self._update)
        self.stack.setCurrentIndex(1)
        self.select_all.setEnabled(True)
        self._update()

    def _exists(self, route, name):
        return any((self.data_root / "raw" / prefix / route / f"{name}.bag").exists() for prefix in ("", "_excluded"))

    def _select_all(self, checked):
        for check, *_ in self.rows:
            check.blockSignals(True)
            check.setChecked(checked)
            check.blockSignals(False)
        self._update()

    def _update(self):
        selected, total, errors, destinations = 0, 0, [], set()
        for row, (check, source, combo, size, failed) in enumerate(self.rows):
            route = combo.currentText().strip()
            target = self.data_root / "raw" / route / source.name
            issue, note = "", ""
            if not COURSE_RE.fullmatch(route):
                issue = "코스 이름 입력 필요 (예: A)"
            elif self._exists(route, source.stem):
                issue = "이미 있음 · 선택 해제"
            elif source.resolve() == target.resolve():
                issue = "이미 데이터 폴더에 있음"
            elif check.isChecked() and source.name in destinations:
                issue = "같은 이름 중복 선택"
            elif source.stem in self.parsed_logs:
                note = "데이터셋에 이미 있음 · 원본 보관만"
            elif failed:
                note = "fail 폴더 · 필요하면 선택"
            if check.isChecked():
                selected += 1
                total += size
                destinations.add(source.name)
                if issue:
                    errors.append(issue)
            state = issue or note or "가져오기 가능"
            item = self.table.item(row, 3)
            item.setText(state)
            item.setData(DOT_ROLE, C["warn"] if issue or failed else C["accent"] if note else C["ok"])
            item.setForeground(QtGui.QColor(C["text"] if issue else C["sub"]))
            item.setToolTip(state)
        self.select_all.blockSignals(True)
        self.select_all.setChecked(bool(self.rows) and selected == len(self.rows))
        self.select_all.blockSignals(False)
        try:
            free = shutil.disk_usage(self.data_root).free
            space = f" · 여유 {human_size(free)}"
            if total >= free:
                errors.append("저장 공간이 부족합니다")
        except OSError:
            space = ""
            errors.append("데이터 폴더에 접근할 수 없습니다")
        summary = f"{len(self.rows)}개 발견 · {selected}개 선택 ({human_size(total)}){space}"
        if errors:
            summary += "\n" + " / ".join(dict.fromkeys(errors))
        self.summary.setText(summary)
        self.go.setText(f"{selected}개 녹화 가져오기" if selected else "녹화 선택 후 가져오기")
        self.go.setEnabled(bool(selected) and not errors)

    def selection(self) -> list[tuple[Path, str]]:
        return [(p, combo.currentText().strip()) for check, p, combo, *_ in self.rows if check.isChecked()]

    def accept(self):
        self._update()
        if self.go.isEnabled():
            super().accept()


YIELD_CATS = {   # yield_check.py category -> (label, colour)
    "scene": ("씬", C["ok"]),
    "short": ("20초 미만 자투리", "#6b6b75"),
    "camera": ("카메라 누락·동기 벗어남", C["warn"]),
    "window": ("센서 시작·종료 구간", "#3a3a42"),
    "lidar_gap": ("LiDAR 프레임 끊김", C["bad"]),
}
TIMELINE_ROLE = Qt.ItemDataRole.UserRole + 3


def yield_reasons(rec: dict, top: int = 2) -> str:
    lost = sorted(((v, k) for k, v in rec["seconds"].items() if k != "scene" and v >= 0.1), reverse=True)
    return ", ".join(f"{YIELD_CATS[k][0]} {v:.0f}초" for v, k in lost[:top] if k in YIELD_CATS) or "손실 없음"


class TimelineDelegate(CellDelegate):
    """A recording's time line: each run of LiDAR frames coloured by what became of it."""
    def paint(self, painter, option, index):
        runs, span = index.data(TIMELINE_ROLE) or ([], 0)
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.state &= ~QtWidgets.QStyle.StateFlag.State_HasFocus
        style = option.widget.style() if option.widget else QtWidgets.QApplication.style()
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, option.widget)
        bar = QtCore.QRectF(option.rect.adjusted(12, 0, -12, 0))
        bar.setTop(bar.center().y() - 5)
        bar.setHeight(10)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(bar, 3, 3)
        painter.setClipPath(clip)
        painter.fillRect(bar, QtGui.QColor(C["hover"]))
        if span > 0:
            for t0, t1, cat in runs:
                x0 = bar.left() + bar.width() * t0 / span
                x1 = bar.left() + bar.width() * t1 / span
                painter.fillRect(QtCore.QRectF(x0, bar.top(), max(x1 - x0, 0.8), bar.height()),
                                 QtGui.QColor(YIELD_CATS.get(cat, ("", C["faint"]))[1]))
        painter.restore()


class YieldDialog(QtWidgets.QDialog):
    """yield_check.py's result: scenes per recording, and why the rest was lost."""
    def __init__(self, parent, result: dict, title: str):
        super().__init__(parent)
        self.recs = result.get("recordings", [])
        prm = result.get("params", {})
        self.setWindowTitle(f"수율 검사 · {title}")
        self.resize(1120, 720)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 22)
        layout.setSpacing(6)
        layout.addWidget(label(f"수율 검사 · {title}", "title"))
        layout.addWidget(label(
            f"기준: 카메라 {prm.get('rgb_cameras', 6)}대가 모두 LiDAR 기준 {prm.get('sync_ms', 25):g} ms 안에 있는 "
            f"프레임이 끊김 없이 {prm.get('scene_dur_s', 20):g}초 이어지면 씬 1개. 이미지는 읽지 않고 시각만 봅니다.",
            "sub", True))
        layout.addSpacing(14)
        scenes = sum(r["scenes"] for r in self.recs)
        span = sum(r.get("span_s", 0) for r in self.recs)
        used = sum(r["seconds"].get("scene", 0) for r in self.recs)
        totals = {k: sum(r["seconds"].get(k, 0) for r in self.recs) for k in YIELD_CATS}
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(28)
        for value, text in ((f"{scenes}", "씬"), (f"{len(self.recs)}", "녹화"),
                            (f"{100 * used / span:.0f}%" if span else "—", "씬으로 쓰인 시간"),
                            (f"{span / 60:.1f}분", "LiDAR 기록 시간")):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(0)
            v = label(value, "title")
            box.addWidget(v)
            box.addWidget(label(text, "sub"))
            head.addLayout(box)
        head.addStretch()
        layout.addLayout(head)
        layout.addSpacing(14)
        legend = QtWidgets.QHBoxLayout()
        legend.setSpacing(16)
        for k, (name, color) in YIELD_CATS.items():
            item = QtWidgets.QLabel(f"<span style='color:{color}'>■</span>&nbsp; {name} "
                                    f"<span style='color:{C['faint']}'>{totals[k] / 60:.1f}분</span>")
            item.setTextFormat(Qt.TextFormat.RichText)
            legend.addWidget(item)
        legend.addStretch()
        layout.addLayout(legend)
        layout.addSpacing(8)
        self.table = QtWidgets.QTableWidget(0, 5)
        setup_table(self.table, ["녹화", "씬", "사용", "시간 막대 (어디가 왜 잘렸나)", "주요 손실"])
        self.table.setItemDelegateForColumn(3, TimelineDelegate(self.table))
        h = self.table.horizontalHeader()
        for column, width in ((0, 250), (1, 56), (2, 64), (4, 310)):
            h.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        h.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setRowCount(len(self.recs))
        longest = max((r.get("span_s", 0) for r in self.recs), default=0)
        for row, r in enumerate(self.recs):
            name = QtWidgets.QTableWidgetItem(r["recording"])
            name.setData(SUB_ROLE, f"{r.get('span_s', 0) / 60:.1f}분" + (" · 확인 필요" if r.get("notes") else ""))
            name.setToolTip(r.get("path", ""))
            self.table.setItem(row, 0, name)
            n = QtWidgets.QTableWidgetItem(str(r["scenes"]))
            n.setForeground(QtGui.QColor(C["text"] if r["scenes"] else C["bad"]))
            self.table.setItem(row, 1, n)
            pct = 100 * r["seconds"].get("scene", 0) / r["span_s"] if r.get("span_s") else 0
            u = QtWidgets.QTableWidgetItem(f"{pct:.0f}%")
            u.setForeground(QtGui.QColor(C["sub"]))
            self.table.setItem(row, 2, u)
            bar = QtWidgets.QTableWidgetItem()
            # every bar on one time scale, so lengths compare across recordings
            bar.setData(TIMELINE_ROLE, (r.get("timeline", []), longest))
            self.table.setItem(row, 3, bar)
            why = QtWidgets.QTableWidgetItem(yield_reasons(r))
            why.setForeground(QtGui.QColor(C["sub"]))
            self.table.setItem(row, 4, why)
        self.table.itemSelectionChanged.connect(self._detail)
        layout.addWidget(self.table, 1)
        layout.addSpacing(8)
        self.detail = QtWidgets.QLabel()
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setMinimumHeight(96)
        self.detail.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.detail)
        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(label(f"검사 시각 {result.get('created_at', '')[:19].replace('T', ' ')} UTC", "faint"))
        foot.addStretch()
        foot.addWidget(button("닫기", self.accept, "primary"))
        layout.addLayout(foot)
        if self.recs:
            self.table.selectRow(0)

    def _detail(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = self.recs[rows[0].row()]
        span = r.get("span_s") or 1
        lines = [f"<b>{r['recording']}</b> &nbsp;<span style='color:{C['sub']}'>씬 {r['scenes']}개 · "
                 f"LiDAR {span:.0f}초 기록</span>"]
        parts = []
        for k, (name, color) in YIELD_CATS.items():
            v = r["seconds"].get(k, 0)
            if v < 0.1:
                continue
            extra = ""
            if k == "camera" and r.get("camera_blame_s"):
                worst = sorted(r["camera_blame_s"].items(), key=lambda kv: -kv[1])[:4]
                extra = " (" + ", ".join(f"{ch} {s:.0f}초" for ch, s in worst) + ")"
            parts.append(f"<span style='color:{color}'>■</span> {name} {v:.1f}초 "
                         f"<span style='color:{C['faint']}'>{100 * v / span:.0f}%{extra}</span>")
        lines.append(" &nbsp;&nbsp; ".join(parts))
        for n in r.get("notes", []):
            lines.append(f"<span style='color:{C['warn']}'>! {n}</span>")
        if r.get("sync_dev_ms"):
            d = r["sync_dev_ms"]
            lines.append(f"<span style='color:{C['faint']}'>통과한 프레임의 카메라–LiDAR 최대 차이: 중앙 {d['p50']} ms, "
                         f"99% {d['p99']} ms, 최대 {d['max']} ms</span>")
        self.detail.setText("<br>".join(lines))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_root: Path | None = None):
        super().__init__()
        self.setWindowTitle("TCAR Parser · 녹화 데이터 준비")
        self.resize(1240, 860)
        self.setMinimumSize(1000, 760)
        self.setAcceptDrops(True)
        self.data_root = data_root.resolve() if data_root and data_root.is_dir() else None
        if decisions_dir():
            os.environ["TCAR_DECISIONS_DIR"] = decisions_dir()     # inherited by the review server and the jobs
        self.job = None
        self.recs, self.active_recs = [], []
        self.status = {"logs": {}, "scenes": 0, "samples": 0, "error": ""}
        self.status_all = parsed_status(Path("/nonexistent"))
        self.validated = {}
        self.next_action = "root"
        self._job_kind, self._job_route = "", None
        self._close_when_done = False
        self._last_log = None
        self.mode = "parser"                    # parser | curation: two pages over the same courses
        self.reviewing = False                  # the review viewer fills the window
        self.cdata = curation_data(Path("/nonexistent"))
        self.cur_counts, self.cur_action, self.cur_logs = curation_counts(self.cdata, []), "empty", []
        self.viewer, self.viewer_url, self._viewer_buf, self._viewer_ready = None, None, "", False
        self._review_logs, self._review_scene = [], None
        self._cur_sig = None                    # the curation files' state, polled while the curation page shows
        self._map_color = "state" if str(SETTINGS.value("map_color", "freq")) == "state" else "freq"
        self._map_reply = None
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self._sidebar())
        self.pages = QtWidgets.QStackedWidget()
        for k, page in enumerate((self._main(), self._curation_page())):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            if k == 1:                                   # curation: the header (전체 검토) stays while the page scrolls
                holder = QtWidgets.QWidget()
                holder.setObjectName("content")
                col = QtWidgets.QVBoxLayout(holder)
                col.setContentsMargins(0, 0, 0, 0)
                col.setSpacing(0)
                col.addWidget(self._curation_header())
                col.addWidget(scroll, 1)
                scroll = holder
            self.pages.addWidget(scroll)
        self.pages.addWidget(self._review_page())
        body.addWidget(self.pages, 1)
        outer.addLayout(body, 1)
        outer.addWidget(self._jobbar())
        self.refresh()
        self._live = QtCore.QTimer(self)        # decisions made in a review show up here as they are made
        self._live.setInterval(1500)
        self._live.timeout.connect(self._poll_curation)
        self._live.start()

    @property
    def dataset(self) -> Path:
        return self.data_root / "parsed" / DATASET

    def _sidebar(self):
        widget = self.sidebar = QtWidgets.QWidget()
        widget.setObjectName("sidebar")
        widget.setFixedWidth(248)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(16, 20, 16, 16)
        layout.setSpacing(6)
        brand = QtWidgets.QHBoxLayout()
        brand.setSpacing(10)
        logo = QtWidgets.QLabel()
        logo.setPixmap(QtGui.QIcon(str(ASSETS / "tcar-parser.svg")).pixmap(24, 24))
        brand.addWidget(logo)
        brand.addWidget(label("TCAR Parser", "brand"))
        brand.addStretch()
        layout.addLayout(brand)
        layout.addSpacing(16)
        segment = QtWidgets.QFrame()
        segment.setObjectName("segment")
        seg_layout = QtWidgets.QHBoxLayout(segment)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(2)
        self.mode_btns = {}
        for mode, text in (("parser", "파서"), ("curation", "큐레이션")):
            b = button(text, lambda _=False, m=mode: self._set_mode(m), "seg")
            b.setCheckable(True)
            b.setChecked(mode == self.mode)
            seg_layout.addWidget(b, 1)
            self.mode_btns[mode] = b
        layout.addWidget(segment)
        layout.addSpacing(18)
        layout.addWidget(label("데이터 폴더", "section"))
        self.root_label = label("", "path", True)
        self.root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.root_label)
        self.change_btn = button("폴더 선택…", self._change_root, "quiet")
        layout.addWidget(self.change_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addSpacing(14)
        row = QtWidgets.QHBoxLayout()
        course_label = label("코스", "section")
        row.addWidget(course_label)
        row.addStretch()
        self.refresh_btn = button("새로고침", self.refresh, "quiet")
        row.addWidget(self.refresh_btn)
        layout.addLayout(row)
        self.route_list = QtWidgets.QListWidget()
        self.route_list.setAccessibleName("코스와 다음 작업")
        self.route_list.setItemDelegate(CellDelegate(self.route_list, pad=12))
        self.route_list.setMouseTracking(True)
        self.route_list.currentItemChanged.connect(self._select_route)
        self.route_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.route_list)
        layout.addSpacing(18)
        self.ov = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(self.ov)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(7)
        ov.addWidget(label("데이터셋 시간", "section"))
        self.ov_text = label("", "sub", True)
        ov.addWidget(self.ov_text)
        self.ov_bar = RatioBar(10)
        ov.addWidget(self.ov_bar)
        self.ov_legend = QtWidgets.QLabel()
        self.ov_legend.setTextFormat(Qt.TextFormat.RichText)
        self.ov_legend.setWordWrap(True)
        self.ov_legend.setObjectName("faint")
        ov.addWidget(self.ov_legend)
        ov.addSpacing(6)
        self.ov_rows = QtWidgets.QGridLayout()
        self.ov_rows.setHorizontalSpacing(8)
        self.ov_rows.setVerticalSpacing(8)
        self.ov_rows.setColumnStretch(1, 1)
        ov.addLayout(self.ov_rows)
        layout.addWidget(self.ov)
        layout.addStretch(1)
        self.sidebar_hint = label("가져온 녹화가 코스별로 여기에 표시됩니다.", "faint", True)
        layout.addWidget(self.sidebar_hint)
        self.import_btn = button("＋  녹화 가져오기", self._import)
        layout.addWidget(self.import_btn)
        self.yield_folder_btn = button("폴더 수율 검사…", self._yield_folder, "quiet")
        self.yield_folder_btn.setToolTip("가져오기 전에 SSD 같은 폴더의 bag을 바로 검사합니다")
        layout.addWidget(self.yield_folder_btn)
        hint = label("원본 폴더를 창에 끌어다 놓아도 됩니다.", "faint", True)
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(hint)
        self.parser_only = (self.import_btn, self.yield_folder_btn, hint, course_label, self.refresh_btn, self.route_list)
        self.cur_detector = label("", "faint", True)
        self.cur_detector_btn = button("검출 환경…", self._pick_detector, "quiet")
        self.cur_detector_btn.setToolTip("torch·ultralytics가 설치된 Python (예: conda 환경의 bin/python)")
        self.cur_undo_btn = button("마지막 최종 확정 되돌리기…", self._undo_apply, "quiet")
        self.cur_undo_btn.setToolTip("_removed/의 가장 최근 백업으로 데이터셋을 되돌립니다")
        layout.addWidget(self.cur_undo_btn)
        layout.addWidget(self.cur_detector)
        layout.addWidget(self.cur_detector_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self.curation_only = (self.cur_undo_btn, self.cur_detector, self.cur_detector_btn)
        for w in self.curation_only:
            w.hide()
        return widget

    def _main(self):
        widget = QtWidgets.QWidget()
        widget.setObjectName("content")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(36, 28, 36, 16)
        layout.setSpacing(0)
        heading = QtWidgets.QHBoxLayout()
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(4)
        self.title = label("", "title")
        self.title.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        titles.addWidget(self.title)
        self.subtitle = label("", "sub", True)
        titles.addWidget(self.subtitle)
        heading.addLayout(titles, 1)
        self.yield_last_btn = button("지난 수율 결과", self._yield_last, "quiet")
        heading.addWidget(self.yield_last_btn, alignment=Qt.AlignmentFlag.AlignTop)
        self.yield_btn = button("수율 검사", self._yield_route, "quiet")
        self.yield_btn.setToolTip("변환 없이 시각만 읽어, 녹화마다 씬이 몇 개 나오고 나머지는 왜 잘리는지 봅니다")
        heading.addWidget(self.yield_btn, alignment=Qt.AlignmentFlag.AlignTop)
        self.val_btn = button("다시 검증", self._validate, "quiet")
        heading.addWidget(self.val_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(heading)
        # original recording time vs what made it into the dataset, for the parsed bags
        self.stats = QtWidgets.QWidget()
        stats_row = QtWidgets.QHBoxLayout(self.stats)
        stats_row.setContentsMargins(0, 18, 0, 0)
        stats_row.setSpacing(32)
        self.stat_values = {}
        for key, text in (("orig", "원본 녹화 시간"), ("used", "데이터셋 시간"), ("rate", "사용률"), ("scenes", "씬")):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(1)
            value = label("—", "actionTitle")
            box.addWidget(value)
            box.addWidget(label(text, "faint"))
            stats_row.addLayout(box)
            self.stat_values[key] = value
        self.stats_note = label("", "faint", True)
        stats_row.addWidget(self.stats_note, 1, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.stats)
        layout.addSpacing(24)
        self.stepper = Stepper()
        layout.addWidget(self.stepper)
        layout.addSpacing(16)
        card = QtWidgets.QFrame()
        card.setObjectName("actionCard")
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 16, 16)
        card_layout.setSpacing(16)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(3)
        self.action_kicker = label("다음 할 일", "eyebrow")
        self.action_title = label("", "actionTitle", True)
        self.action_desc = label("", "sub", True)
        text.addWidget(self.action_kicker)
        text.addWidget(self.action_title)
        text.addWidget(self.action_desc)
        card_layout.addLayout(text, 1)
        self.primary_btn = button("", self._primary, "primary")
        card_layout.addWidget(self.primary_btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(card)
        layout.addSpacing(28)
        table_bar = QtWidgets.QHBoxLayout()
        self.recording_heading = label("녹화", "section")
        table_bar.addWidget(self.recording_heading)
        table_bar.addStretch()
        self.show_excluded = QtWidgets.QCheckBox("제외된 녹화 보기")
        self.show_excluded.stateChanged.connect(self._show_route)
        table_bar.addWidget(self.show_excluded)
        layout.addLayout(table_bar)
        layout.addSpacing(6)
        self.recording_stack = QtWidgets.QStackedWidget()
        self.table = QtWidgets.QTableWidget(0, 5)
        setup_table(self.table, ["녹화", "시간  (데이터셋 · 큐레이션으로 뺌 · 파싱 때 버림 · 파싱 대기)", "원본", "데이터셋", "사용률", "상태"])
        self.table.setItemDelegateForColumn(1, BarDelegate(self.table))
        self.table.setMinimumHeight(180)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 290), (2, 72), (3, 80), (4, 72), (5, 200)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.table.setColumnWidth(column, width)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.recording_stack.addWidget(self.table)
        empty = QtWidgets.QFrame()
        empty.setObjectName("emptyCard")
        empty_layout = QtWidgets.QVBoxLayout(empty)
        empty_layout.setContentsMargins(24, 22, 24, 22)
        empty_layout.addStretch()
        self.empty_title = label("", "actionTitle", True)
        self.empty_desc = label("", "sub", True)
        for w in (self.empty_title, self.empty_desc):
            w.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            empty_layout.addWidget(w)
        empty_layout.addStretch()
        self.recording_stack.addWidget(empty)
        layout.addWidget(self.recording_stack, 1)
        layout.addSpacing(10)
        footer = QtWidgets.QHBoxLayout()
        self.selection_hint = label("", "faint", True)
        footer.addWidget(self.selection_hint, 1)
        self.open_btn = button("폴더 열기", self._open_folder, "quiet")
        self.excl_btn = button("녹화 제외…", self._exclude, "danger")
        footer.addWidget(self.open_btn)
        footer.addWidget(self.excl_btn)
        layout.addLayout(footer)
        return widget

    def _curation_header(self):
        """Pinned above the scrolling curation page: the title and 전체 검토, always the whole dataset."""
        widget = QtWidgets.QWidget()
        widget.setObjectName("content")
        heading = QtWidgets.QHBoxLayout(widget)
        heading.setContentsMargins(36, 22, 36, 14)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(label("큐레이션", "title"))
        self.cur_subtitle = label("", "sub", True)
        titles.addWidget(self.cur_subtitle)
        heading.addLayout(titles, 1)
        self.cur_review_btn = button("전체 검토", self._review_all, "primary")
        self.cur_review_btn.setToolTip("데이터셋의 모든 씬을 한 목록으로 검토합니다 (카메라 6대·LiDAR·지도, 남기기 / 버리기 확정). "
                                       "루트만 보려면 표에서 루트를 고르고 ‘고른 루트 검토’")
        heading.addWidget(self.cur_review_btn, alignment=Qt.AlignmentFlag.AlignTop)
        heading.addSpacing(6)
        self.cur_rule_btn = button("규칙 다시 실행", self._curate, "quiet")
        self.cur_rule_btn.setToolTip("데이터셋 전체에 필터(① 정지 중복 → ② 객체 부족 → ③ 경로 겹침)를 다시 돌립니다 (사람이 확정한 결정은 그대로)")
        heading.addWidget(self.cur_rule_btn, alignment=Qt.AlignmentFlag.AlignTop)
        return widget

    def _review_all(self):
        logs = sorted(self.status_all["names_by_log"], key=route_key)
        self._open_review(logs, None, f"데이터셋 전체 · 씬 {self.status_all['scenes']:,}개")

    def _curation_page(self):
        widget = QtWidgets.QWidget()
        widget.setObjectName("content")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(36, 6, 36, 16)
        layout.setSpacing(0)
        # the whole dataset: confirmed, candidates, deleted (only here; not per course or route)
        whole = QtWidgets.QFrame()
        whole.setObjectName("actionCard")
        whole_layout = QtWidgets.QVBoxLayout(whole)
        whole_layout.setContentsMargins(20, 16, 16, 16)
        whole_layout.setSpacing(10)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(30)
        self.cur_values = {}
        for key, text in (("total", "전체"), ("drop", "삭제 예정 (버리기 확정)"), ("after", "삭제 후 예상"),
                          ("candidates", "후보 (필터에 걸림)"), ("deleted", "삭제 완료")):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(2)
            value, sub = label("—", "actionTitle"), label("", "faint")
            box.addWidget(label(text, "eyebrow"))
            box.addWidget(value)
            box.addWidget(sub)
            top.addLayout(box)
            self.cur_values[key] = (value, sub)
        top.addStretch()
        final = QtWidgets.QVBoxLayout()
        final.setSpacing(6)
        self.cur_final_btn = button("", self._final_delete, "danger")
        self.cur_final_btn.setToolTip("이번 라운드를 끝냅니다: 버리기 확정 씬을 빼고(_removed/로 옮김), 남는 씬을 모두 잠급니다. "
                                      "잠긴 씬은 이후 새 데이터가 들어와도 절대 삭제되지 않습니다")
        final.addWidget(self.cur_final_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.cur_final_note = label("", "faint")
        final.addWidget(self.cur_final_note, alignment=Qt.AlignmentFlag.AlignRight)
        top.addLayout(final)
        whole_layout.addLayout(top)
        self.cur_bar = RatioBar(12)
        whole_layout.addWidget(self.cur_bar)
        self.cur_legend = QtWidgets.QLabel()
        self.cur_legend.setTextFormat(Qt.TextFormat.RichText)
        self.cur_legend.setWordWrap(True)
        self.cur_legend.setObjectName("faint")
        whole_layout.addWidget(self.cur_legend)
        self.cur_kept = label("", "faint", True)
        whole_layout.addWidget(self.cur_kept)
        layout.addWidget(whole)
        layout.addSpacing(16)
        # curation rounds and Hugging Face releases: what is finalized / published, and what is new
        stage = QtWidgets.QFrame()
        stage.setObjectName("actionCard")
        stage_layout = QtWidgets.QVBoxLayout(stage)
        stage_layout.setContentsMargins(20, 14, 16, 14)
        stage_layout.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label("라운드 · 배포", "section"))
        head.addSpacing(10)
        self.stage_note = label("", "faint", True)
        head.addWidget(self.stage_note, 1)
        stage_layout.addLayout(head)
        self.stage_table = QtWidgets.QTableWidget(0, 6)
        setup_table(self.stage_table, ["구분", "최종 확정", "녹화", "남긴 씬", "지운 씬", "허깅페이스 배포"])
        self.stage_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.stage_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.stage_table.verticalHeader().setDefaultSectionSize(52)
        header = self.stage_table.horizontalHeader()
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 150), (1, 130), (3, 90), (4, 90), (5, 200)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.stage_table.setColumnWidth(column, width)
        self.stage_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        stage_layout.addWidget(self.stage_table)
        layout.addWidget(stage)
        layout.addSpacing(16)
        # every scene's GNSS track on OSM
        map_card = QtWidgets.QFrame()
        map_card.setObjectName("actionCard")
        map_layout = QtWidgets.QVBoxLayout(map_card)
        map_layout.setContentsMargins(16, 12, 16, 16)
        map_layout.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label("지도", "section"))
        head.addSpacing(10)
        head.addWidget(label("휠 = 확대 · 드래그 = 이동 · 경로를 누르면 그 루트를 고르고, 두 번 누르면 그 씬을 검토합니다",
                             "faint", True), 1)
        segment = QtWidgets.QFrame()
        segment.setObjectName("segment")
        seg_layout = QtWidgets.QHBoxLayout(segment)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(2)
        self.map_color_btns = {}
        for key, text in (("freq", "빈도"), ("state", "상태별")):
            b = button(text, lambda _=False, k=key: self._set_map_color(k), "seg")
            b.setCheckable(True)
            b.setChecked(key == self._map_color)
            seg_layout.addWidget(b)
            self.map_color_btns[key] = b
        head.addWidget(segment)
        head.addWidget(button("전체 보기", lambda: self.cur_map.fit(), "quiet"))
        map_layout.addLayout(head)
        self.cur_map = CurationMap()
        self.cur_map.routePicked.connect(self._map_pick_route)
        self.cur_map.sceneOpened.connect(self._map_open_scene)
        map_layout.addWidget(self.cur_map)
        layout.addWidget(map_card)
        layout.addSpacing(26)
        # the picked course: its next step and its routes
        self.cur_heading = label("", "section")
        layout.addWidget(self.cur_heading)
        layout.addSpacing(8)
        card = QtWidgets.QFrame()
        card.setObjectName("actionCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 16, 16)
        card_layout.setSpacing(12)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(16)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(3)
        self.cur_kicker = label("다음 할 일", "eyebrow")
        self.cur_title = label("", "actionTitle", True)
        self.cur_desc = label("", "sub", True)
        for w in (self.cur_kicker, self.cur_title, self.cur_desc):
            text.addWidget(w)
        top.addLayout(text, 1)
        self.cur_primary = button("", self._curation_primary, "primary")
        top.addWidget(self.cur_primary, alignment=Qt.AlignmentFlag.AlignVCenter)
        card_layout.addLayout(top)
        acts = QtWidgets.QHBoxLayout()
        acts.setSpacing(8)
        self.cur_pick_btn = button("", lambda: self._open_review())
        self.cur_pick_btn.setToolTip("표에서 고른 루트의 씬만 검토합니다 (두 번 눌러도 됩니다)")
        acts.addWidget(self.cur_pick_btn)
        self.cur_detect_btn = button("", self._detect)
        acts.addWidget(self.cur_detect_btn)
        acts.addStretch()
        self.cur_scope = label("", "faint", True)
        acts.addWidget(self.cur_scope)
        card_layout.addLayout(acts)
        layout.addWidget(card)
        layout.addSpacing(20)
        bar = QtWidgets.QHBoxLayout()
        self.cur_table_heading = label("루트", "section")
        bar.addWidget(self.cur_table_heading)
        bar.addStretch()
        self.cur_clear_btn = button("선택 해제 (코스 전체)", self._clear_scope, "quiet")
        bar.addWidget(self.cur_clear_btn)
        layout.addLayout(bar)
        layout.addSpacing(6)
        self.cur_table = QtWidgets.QTableWidget(0, 9)
        setup_table(self.cur_table, ["루트", "확정 · 배포", "씬", "검출", "① 정지 중복", "② 객체 부족", "③ 경로 겹침",
                                     "남기기 확정", "버리기 확정"])
        self.cur_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.cur_table.setMinimumHeight(240)
        header = self.cur_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 150), (2, 56), (3, 80), (4, 96), (5, 96), (6, 96), (7, 92), (8, 92)):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.cur_table.setColumnWidth(column, width)
        self.cur_table.itemSelectionChanged.connect(self._scope_changed)
        self.cur_table.itemDoubleClicked.connect(lambda *_: self._open_review())
        layout.addWidget(self.cur_table, 1)
        layout.addSpacing(10)
        self.cur_paths = label("", "faint", True)
        self.cur_paths.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.cur_paths)
        layout.addSpacing(24)
        # last: the filters, what each one looks at, its numbers, why, and the data behind it
        filters = QtWidgets.QFrame()
        filters.setObjectName("actionCard")
        filters_layout = QtWidgets.QVBoxLayout(filters)
        filters_layout.setContentsMargins(20, 14, 16, 16)
        filters_layout.setSpacing(12)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(label("필터", "section"))
        head.addSpacing(10)
        head.addWidget(label("① → ② → ③ 순서로 걸러, 앞에서 걸린 씬은 뒤에서 다시 보지 않습니다. 걸린 씬이 후보가 되고, "
                             "사람이 확정하기 전까지 데이터셋에 그대로 있습니다. 건너뛴 필터는 후보를 만들지 않습니다.", "faint", True), 1)
        filters_layout.addLayout(head)
        self.filter_rows = {}
        self.filter_skip = {}
        for i, (key, title, colour) in enumerate((("stops", "① 정지 중복", "#c792ea"), ("objects", "② 객체 부족", "#e8a94f"),
                                                  ("overlap", "③ 경로 겹침", "#5ec8e5"))):
            if i:
                sep = QtWidgets.QFrame()
                sep.setFixedHeight(1)
                sep.setStyleSheet(f"background: {C['line']};")
                filters_layout.addWidget(sep)
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(24)
            text = QtWidgets.QVBoxLayout()
            text.setSpacing(4)
            top = QtWidgets.QHBoxLayout()
            top.setSpacing(8)
            dot = QtWidgets.QLabel(f"<span style='color:{colour}'>●</span>")
            dot.setTextFormat(Qt.TextFormat.RichText)
            top.addWidget(dot)
            top.addWidget(label(title, "actionTitle"))
            count = label("", "sub")
            top.addWidget(count)
            top.addStretch()
            if key == "overlap":
                top.addWidget(button("지도에서 보기", lambda: self._set_map_color("freq"), "quiet"))
            skip = button("건너뛰기", lambda on=False, k=key: self._skip_filter(k, on), "quiet")
            skip.setCheckable(True)
            skip.setToolTip("이 필터를 건너뛰고(후보를 만들지 않고) 규칙을 다시 돌립니다. 다시 누르면 되돌립니다")
            top.addWidget(skip)
            self.filter_skip[key] = skip
            text.addLayout(top)
            rule, basis, why = label("", "", True), label("", "sub", True), label("", "faint", True)
            for w in (rule, basis, why):
                w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
                text.addWidget(w)
            row.addLayout(text, 1)
            chart = MiniHist()
            row.addWidget(chart, alignment=Qt.AlignmentFlag.AlignTop)
            filters_layout.addLayout(row)
            self.filter_rows[key] = (count, rule, basis, why, chart)
        layout.addWidget(filters)
        return widget

    def _review_page(self):
        widget = QtWidgets.QWidget()
        widget.setObjectName("content")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(8)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(button("←  큐레이션", lambda: self._close_review(), "quiet"))   # (clicked's bool must not land in `refresh`)
        bar.addSpacing(8)
        self.review_title = label("검토", "actionTitle")
        bar.addWidget(self.review_title)
        bar.addSpacing(12)
        self.review_status = label("", "sub", True)
        bar.addWidget(self.review_status, 1)
        self.review_final_btn = button("", self._final_from_review, "danger")
        self.review_final_btn.setToolTip("검토를 닫고 버리기 확정 씬을 데이터셋에서 뺍니다 (계획을 보여 주고 한 번 더 묻습니다)")
        bar.addWidget(self.review_final_btn)
        bar.addSpacing(8)
        bar.addWidget(button("새로고침", self._reload_review, "quiet"))
        bar.addWidget(button("브라우저로 열기", self._review_in_browser, "quiet"))
        layout.addLayout(bar)
        if QWebEngineView is not None:
            self.web = QWebEngineView()
            layout.addWidget(self.web, 1)
        else:
            self.web = None
            layout.addWidget(label("이 환경에는 Qt WebEngine이 없어 검토 화면을 브라우저로 엽니다.", "sub", True), 1)
        return widget

    def _jobbar(self):
        frame = self.jobbar = QtWidgets.QFrame()
        frame.setObjectName("jobBar")
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(2)
        outer.addWidget(self.progress)
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(24, 8, 16, 8)
        layout.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        self.job_status = label("대기 중", "sub", True)
        row.addWidget(self.job_status, 1)
        self.log_btn = button("로그 보기", self._toggle_log, "quiet")
        row.addWidget(self.log_btn)
        self.cancel_btn = button("취소", self._cancel, "quiet")
        self.cancel_btn.setEnabled(False)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)
        self.log_panel = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(0, 0, 8, 6)
        log_layout.setSpacing(6)
        self.log_path = label("작업 로그는 데이터 폴더의 logs/에 저장됩니다.", "faint", True)
        self.log_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        log_layout.addWidget(self.log_path)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setFixedHeight(160)
        log_layout.addWidget(self.log_view)
        layout.addWidget(self.log_panel)
        outer.addWidget(inner)
        opened = str(SETTINGS.value("log_open", "false")).lower() == "true"
        self.log_panel.setVisible(opened)
        self.log_btn.setText("로그 닫기" if opened else "로그 보기")
        return frame

    def route_dirs(self) -> list[str]:
        if not self.data_root:
            return []
        names = {detect_route(Path(n)) for n in self.status_all["imported"]} - {None}
        for base in (self.data_root / "raw", self.data_root / "raw" / "_excluded"):
            if base.is_dir():
                names.update(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
        return sorted(names)

    def current_route(self) -> str | None:
        item = self.route_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _read_route(self, route):
        recs = []
        for excluded, base in ((False, self.data_root / "raw" / route),
                               (True, self.data_root / "raw" / "_excluded" / route)):
            if base.is_dir():
                recs.extend(dict(recording_info(p), excluded=excluded) for p in base.iterdir() if is_recording(p))
        status = course_status(self.status_all, route)
        have = {r["name"] for r in recs}
        # parsed before this data folder held its bag (or the bag was deleted): still listed
        recs.extend(dict(dataset_only_info(self.dataset, name), excluded=False)
                    for name in status["imported"] if name not in have)
        recs.sort(key=lambda r: route_key(r["name"]))
        for r in recs:
            r["stats"] = (log_time(self.dataset, r["name"], status["logs"].get(r["name"], 0))
                          if r["name"] in status["imported"] else None)
        return recs, status

    def _route_action(self, route, recs, status):
        active = [r for r in recs if not r["excluded"] and r["raw"]]
        if status["error"]:
            return "validate", "데이터셋 확인 필요"
        pending = sum(r["name"] not in status["imported"] for r in active)
        if pending:
            return "parse", f"{pending}개 파싱 대기"
        if status["logs"]:
            stamp = dataset_stamp(self.dataset)
            if stamp and self.validated.get(DATASET) == stamp:
                return "done", "모두 완료"
            return "validate", "검증 대기"
        return "import", "녹화 가져오기"

    def refresh(self):
        if self.job:
            return
        self.root_label.setText(str(self.data_root) if self.data_root else "아직 선택하지 않았습니다")
        self.root_label.setToolTip(str(self.data_root or ""))
        self.change_btn.setText("폴더 변경…" if self.data_root else "폴더 선택…")
        keep = self.current_route() or getattr(self, "_last_route", None)
        self.route_list.blockSignals(True)
        self.route_list.clear()
        self._route_cache = {}
        try:
            self.status_all = parsed_status(self.dataset) if self.data_root else parsed_status(Path("/nonexistent"))
            self.cdata = curation_data(self.dataset) if self.data_root else curation_data(Path("/nonexistent"))
            if self.data_root and DATASET not in self.validated:
                saved = load_validation(self.data_root)     # a check from an earlier run
                if saved:
                    self.validated[DATASET] = saved
            for route in self.route_dirs():
                recs, status = self._read_route(route)
                self._route_cache[route] = recs, status
                if self.mode == "parser":
                    action, next_text = self._route_action(route, recs, status)
                    dot = ACTION_COLOR.get(action, C["accent"])
                else:
                    c = curation_counts(self.cdata, self._course_scenes(route))
                    action = curation_action(c)
                    next_text = self._curation_copy(c)[action][3]
                    dot = self._cur_dot(action)
                item = QtWidgets.QListWidgetItem(f"코스 {route}")
                item.setData(Qt.ItemDataRole.UserRole, route)
                item.setData(SUB_ROLE, next_text)
                item.setData(DOT_ROLE, dot)
                item.setSizeHint(QtCore.QSize(0, 52))
                item.setToolTip(f"코스 {route}\n{next_text}")
                self.route_list.addItem(item)
                if route == keep:
                    self.route_list.setCurrentItem(item)
        except OSError as error:
            self.job_status.setText(f"데이터 폴더를 읽을 수 없습니다: {error}")
        finally:
            self.route_list.blockSignals(False)
        if self.route_list.currentItem() is None and self.route_list.count():
            self.route_list.blockSignals(True)
            self.route_list.setCurrentRow(0)
            self.route_list.blockSignals(False)
        self.sidebar_hint.setVisible(self.route_list.count() == 0)
        self.route_list.setFixedHeight(min(max(self.route_list.count(), 1), 7) * 54 + 6)
        self._show_overview()
        saved = str(SETTINGS.value("detect_python", ""))
        self.cur_detector.setText(f"검출 환경: {Path(saved).parent.parent.name or saved}" if saved else
                                  "검출 환경: 처음 검출할 때 찾아 기억합니다")
        self.cur_detector.setToolTip(saved)
        if self.mode == "parser":
            self._show_route()
        else:
            self._show_curation()
        self._cur_sig = self._curation_signature() if self.data_root else None

    def _overview(self) -> dict:
        """Seconds per course: recorded, in the dataset, taken out by curation, lost at parse time, waiting to
        be parsed."""
        out = {}
        for route, (recs, status) in self._route_cache.items():
            c = dict.fromkeys(("orig", "used", "curated", "lost", "waiting"), 0.0)
            for r in recs:
                if r["excluded"]:
                    continue
                st = r.get("stats")
                if st:
                    c["orig"] += st["orig_s"]
                    c["used"] += st["used_s"]
                    c["curated"] += st["curated_s"]
                    c["lost"] += max(st["orig_s"] - st["parsed_s"], 0.0)
                elif r["raw"] and r["duration"]:
                    c["waiting"] += r["duration"]
            c["total"] = c["orig"] + c["waiting"]
            out[route] = c
        return out

    def _show_overview(self):
        """The sidebar's recorded-time bars (parser page; the curation page has its own, for the whole dataset)."""
        per = self._overview()
        keys = OV_PARSER
        tot = {k: sum(c[k] for c in per.values()) for k in ("orig", "used", "curated", "lost", "waiting", "total")}
        self.ov.setVisible(tot["total"] > 0 and self.mode == "parser")
        if not tot["total"] or self.mode != "parser":
            return
        mins = lambda v: f"{v / 60:.1f}분"
        self.ov_text.setText(f"원본 {mins(tot['total'])} 중 데이터셋 {mins(tot['used'])} ({100 * tot['used'] / tot['total']:.1f}%)")
        self.ov_bar.set_parts([(tot[k], col) for k, _, col in keys], tot["total"])
        self.ov_legend.setText("<br>".join(
            f"<span style='color:{col}'>●</span>&nbsp;{name} <span style='color:{C['sub']}'>{mins(tot[k])}</span>"
            for k, name, col in keys if tot[k] >= 1))
        while self.ov_rows.count():
            w = self.ov_rows.takeAt(0).widget()
            if w:
                w.deleteLater()
        for row, (route, c) in enumerate(sorted(per.items())):
            if not c["total"]:
                continue
            bar = RatioBar(6)
            bar.set_parts([(c[k], col) for k, _, col in keys], c["total"])
            share = c["used"]
            bar.setToolTip(f"코스 {route}\n" + "\n".join(f"{name} {mins(c[k])}" for k, name, _ in keys if c[k] >= 1))
            self.ov_rows.addWidget(label(route, "section"), row, 0)
            self.ov_rows.addWidget(bar, row, 1)
            pct = label(f"{100 * share / c['total']:.0f}%", "faint")
            pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct.setMinimumWidth(34)
            self.ov_rows.addWidget(pct, row, 2)

    def _set_mode(self, mode):
        if self.job or mode == self.mode:
            self.mode_btns[self.mode].setChecked(True)
            self.mode_btns[mode].setChecked(mode == self.mode)
            return
        if self.reviewing:
            self._close_review(refresh=False)
        self.mode = mode
        for m, b in self.mode_btns.items():
            b.setChecked(m == mode)
        for w in self.parser_only:
            w.setVisible(mode == "parser")
        for w in self.curation_only:
            w.setVisible(mode == "curation")
        self.pages.setCurrentIndex(0 if mode == "parser" else 1)
        self.refresh()
        self._warm_viewer()

    def _select_route(self, item, _previous=None):
        if item is None:
            return
        self._last_route = item.data(Qt.ItemDataRole.UserRole)
        if self.mode == "parser":
            self._show_route()
        else:
            self.cur_table.clearSelection()
            self._show_curation()

    # ---- curation: per course, and per route inside it ----
    def _course_logs(self, route) -> list[str]:
        return sorted((n for n in self.status_all["names_by_log"] if detect_route(Path(n)) == route),
                      key=route_key)

    def _course_scenes(self, route) -> list[str]:
        return [x for n in self._course_logs(route) for x in self.status_all["names_by_log"][n]]

    def _scope(self) -> tuple[list[str], str]:
        """The logs the curation actions work on: the selected routes, else the whole dataset."""
        rows = sorted({i.row() for i in self.cur_table.selectionModel().selectedRows()})
        logs = [self.cur_table.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]
        if logs:
            names = ", ".join(BAG_RE.match(n)["course"] + "-" + BAG_RE.match(n)["num"] if BAG_RE.match(n) else n
                              for n in logs)
            return logs, names
        every = sorted(self.status_all["names_by_log"], key=route_key)
        return (every, "데이터셋 전체") if every else ([], "")

    def _curation_copy(self, c: dict) -> dict:
        """action -> (card title, card text, primary button, sidebar line)."""
        missing, dc, conf = c["n"] - c["detected"], c["cand"], c["keep"] + c["drop"]
        per = " · ".join(f"{t.split(' · ')[1]} {c[k]}" for k, t, _ in CUR_STATES if k in CANDIDATES and c[k])
        minutes = max(1, round(missing * 6 / 60))
        return {
            "empty": ("파싱된 씬이 없습니다", "파서에서 녹화를 파싱하면 여기서 큐레이션합니다.", "", "데이터셋 없음"),
            "detect": (f"씬 {missing:,}개가 아직 객체 검출 전입니다",
                       f"검출된 씬은 지금 바로 검토할 수 있습니다. 나머지는 GPU로 약 {minutes}분, 끝나면 규칙을 다시 돌립니다.",
                       f"{missing:,}개 검출하기", f"검출 {c['detected']}/{c['n']}" + (f" · 후보 {dc}" if dc else "")),
            "rule": (f"씬 {c['n'] - c['ruled']:,}개에 아직 규칙을 돌리지 않았습니다",
                     "필터(① 정지 중복 → ② 객체 부족 → ③ 경로 겹침)에 걸린 씬을 후보로 만듭니다. 데이터셋은 바뀌지 않습니다.",
                     "규칙 실행", "규칙 실행 대기"),
            "review": (f"필터에 걸린 후보 {dc}개를 확인하세요 ({per})",
                       "맨 위 ‘전체 검토’(또는 루트를 고르고 ‘고른 루트 검토’)에서 카메라·LiDAR·지도를 보고 남기기나 버리기를 누르면 확정되고 바로 저장됩니다 "
                       "(keep.txt · drop.txt). 누르지 않은 씬은 후보로 남습니다.",
                       "", f"후보 {dc}" + (f" · 확정 {conf}" if conf else "")),
            "done": ("필터에 걸린 후보를 모두 확정했습니다",
                     (f"버리기 확정 {c['drop']}개는 위의 ‘최종 확정’으로 데이터셋에서 빼고, 남는 씬은 잠급니다." if c["drop"] else
                      "이 범위에는 버리기로 확정한 씬이 없습니다.") + " 다른 씬도 검토에서 언제든 확정할 수 있습니다.",
                     "", f"확정 완료 · 버리기 {c['drop']}" if c["drop"] else "확정 완료"),
        }

    def _cur_dot(self, action):
        return C["ok"] if action == "done" else C["faint"] if action == "empty" else C["accent"]

    def _show_curation(self):
        route = None                                     # curation: the whole dataset, not a course
        routes = sorted(self.status_all["names_by_log"], key=route_key)
        # the route table (kept while only the selection changes)
        if getattr(self, "_cur_table_route", None) != (route, tuple(routes), id(self.cdata)):
            same_course = (getattr(self, "_cur_table_route", None) or (None,))[0] == route
            keep = {self.cur_table.item(i.row(), 0).data(Qt.ItemDataRole.UserRole)
                    for i in self.cur_table.selectionModel().selectedRows()} if same_course else set()
            self._cur_table_route = (route, tuple(routes), id(self.cdata))
            self.cur_table.blockSignals(True)
            self.cur_table.selectionModel().blockSignals(True)
            self.cur_table.clearSelection()
            self.cur_table.setRowCount(len(routes))
            self.cur_table.setMinimumHeight(min(max(len(routes), 3) * 56 + 46, 760))   # every route without scrolling
            for row, name in enumerate(routes):
                rc = curation_counts(self.cdata, self.status_all["names_by_log"][name])
                m, when = BAG_RE.match(name), _bag_time(name)
                st_text, st_col = stage_text(self.cdata, name, self.status_all["names_by_log"][name])
                cells = (name, st_text, f"{rc['n']}", f"{rc['detected']} / {rc['n']}", f"{rc['stops']}", f"{rc['objects']}",
                         f"{rc['overlap']}", f"{rc['keep']}", f"{rc['drop']}")
                for column, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(text)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, name)
                        item.setData(SUB_ROLE, " · ".join(filter(None, [f"{m['course']}-{m['num']}" if m else "",
                                                                        f"{when:%Y-%m-%d %H:%M}" if when else ""])))
                    elif column == 1:
                        item.setForeground(QtGui.QColor(st_col))
                    elif column == 3 and rc["detected"] < rc["n"]:
                        item.setForeground(QtGui.QColor(C["warn"]))
                    elif column >= 4 and rc[("stops", "objects", "overlap", "keep", "drop")[column - 4]]:
                        item.setForeground(QtGui.QColor(("#c792ea", "#e8a94f", "#5ec8e5", C["ok"], C["bad"])[column - 4]))
                    else:
                        item.setForeground(QtGui.QColor(C["sub"] if column else C["text"]))
                    self.cur_table.setItem(row, column, item)
            flags = QtCore.QItemSelectionModel.SelectionFlag
            for row, name in enumerate(routes):         # a refresh keeps the routes picked before it
                if name in keep:
                    self.cur_table.selectionModel().select(self.cur_table.model().index(row, 0),
                                                           flags.Select | flags.Rows)
            self.cur_table.selectionModel().blockSignals(False)
            self.cur_table.blockSignals(False)
        # the whole dataset: what goes (버리기 확정), what is open (후보), what went (삭제 완료); the rest stays
        w = curation_counts(self.cdata, [x for names in self.status_all["names_by_log"].values() for x in names])
        deleted = self.cdata["deleted"]
        total = w["n"] + deleted
        pc = lambda v: f"{100 * v / total:.1f}%" if total else "—"
        cand, kept = w["cand"], w["keep"] + w["pass"] + w["locked"]   # passing every filter is a confirmed keep
        # time: each route's scene length (its import record), the scenes already taken out by their route too
        by_log = self.status_all["names_by_log"]
        sec = {lg: (import_stats(self.dataset, lg) or {}).get("scene_s") or 20.0 for lg in set(by_log) | set(self.cdata["deleted_logs"])}
        now_s = sum(sec[lg] * len(ns) for lg, ns in by_log.items())
        drop_s = sum(sec[lg] for lg, ns in by_log.items() for n in ns if scene_state(self.cdata, n) == "drop")
        del_s = sum(sec.get(lg, 20.0) for lg in self.cdata["deleted_logs"]) + 20.0 * (deleted - len(self.cdata["deleted_logs"]))
        total_s, after, after_s = now_s + del_s, w["n"] - w["drop"], now_s - drop_s
        mins = lambda v: f"{v / 60:.1f}분" if v < 3600 else f"{int(v // 3600)}시간 {round(v % 3600 / 60)}분"
        for key, value, sub in (
                ("total", f"{total:,}개  ·  {mins(total_s)}",
                 (f"잠김 {w['locked']:,} · 새 데이터 {w['n'] - w['locked']:,}" if w["locked"] else f"지금 데이터셋 {w['n']:,}개")
                 + (f" · 삭제 완료 {deleted:,}" if deleted else "")),
                ("drop", f"{w['drop']:,}개  ·  {pc(w['drop'])}", f"{mins(drop_s)} · 최종 삭제 때 빠짐"),
                ("after", f"{after:,}개  ·  {pc(after)}",
                 f"{mins(after_s)}" + (f" · 후보 {cand}개 미확정" if cand else " · 후보 모두 확정")),
                ("candidates", f"{cand:,}개  ·  {pc(cand)}",
                 " · ".join(f"{t.split(' · ')[1]} {w[k]:,}" for k, t, _ in CUR_STATES if k in CANDIDATES and (w[k] or k != "other"))),
                ("deleted", f"{deleted:,}개  ·  {pc(deleted)}",
                 f"{mins(del_s)} · 최종 삭제 {len(self.cdata['backups'])}번 (되돌리기 가능)" if self.cdata["backups"] else "아직 없음")):
            self.cur_values[key][0].setText(value)
            self.cur_values[key][1].setText(sub)
        seg = dict(w, deleted=deleted)
        shown = [(k, t, col) for k, t, col in CUR_STATES if k not in ("keep", "pass", "locked")]   # 버리기, 후보, 삭제 완료
        self.cur_bar.set_parts([(seg[k], col) for k, _, col in shown], total)
        self.cur_bar.setToolTip("\n".join(f"{t} {seg[k]:,} ({pc(seg[k])})" for k, t, _ in shown)
                                + f"\n남음 (회색) {kept:,} ({pc(kept)})")
        self.cur_legend.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(
            f"<span style='color:{col}'>●</span>&nbsp;{t}&nbsp;<span style='color:{C['sub']}'>{seg[k]:,} ({pc(seg[k])})</span>"
            for k, t, col in shown if seg[k] or k != "other"))
        self.cur_kept.setText(f"회색 = 남는 씬 {kept:,}개 ({pc(kept)}): " + (f"잠김(이전 라운드) {w['locked']:,} · " if w["locked"] else "")
                              + f"필터 통과 {w['pass']:,} · 사람이 남기기로 확정 {w['keep']:,}"
                              + ("  ·  잠긴 씬은 절대 삭제되지 않고, 새로 들어온 데이터만 후보·삭제 대상입니다" if w["locked"] else ""))
        new = w["n"] - w["locked"]
        self.cur_final_btn.setText(f"최종 확정 · 버리기 {w['drop']}개 삭제" if w["drop"] else "최종 확정 · 삭제 없음")
        self._final_ready = bool(new) and not cand
        self.cur_final_note.setText(
            "새 데이터가 없습니다 · 모두 잠김" if not new else
            f"후보 {cand}개를 먼저 확정하세요" if cand else
            f"버리기 {w['drop']}개를 빼고 남는 새 씬 {new - w['drop']}개를 잠급니다 (이후 삭제 불가)")
        self.cur_subtitle.setText(
            (f"데이터셋 전체 씬 {total:,}개" + (f" (지금 {w['n']:,}개 + 삭제 완료 {deleted:,}개)" if deleted else "")
             + "  ·  필터를 통과한 씬은 바로 남기고, 걸린 씬은 후보가 되어 사람이 검토에서 확정합니다") if total else
            "아직 파싱된 씬이 없습니다.")
        # the scope: the routes picked in the table, else the whole dataset
        logs, scope_name = self._scope()
        self.cur_logs = logs
        self.cur_counts = c = curation_counts(self.cdata, [x for n in logs for x in self.status_all["names_by_log"].get(n, [])])
        self.cur_action = curation_action(c)
        self.cur_heading.setText(f"데이터셋 전체  ·  루트 {len(routes)}개  ·  씬 {w['n']:,}개" if routes else "데이터셋")
        title, desc, primary, _ = self._curation_copy(c)[self.cur_action]
        self.cur_kicker.setText("완료" if self.cur_action == "done" else "다음 할 일")
        self.cur_kicker.setStyleSheet(f"color: {C['ok']};" if self.cur_action == "done" else "")
        self.cur_title.setText(title)
        self.cur_desc.setText(desc)
        self.cur_primary.setText(primary)
        self.cur_primary.setVisible(bool(primary))
        missing = c["n"] - c["detected"]
        self.cur_detect_btn.setText(f"미검출 {missing}개 검출" if missing else "검출 완료")
        self.cur_scope.setText(f"범위: {scope_name}" if scope_name else "")
        self.cur_review_btn.setText(f"전체 검토 · 씬 {w['n']:,}개")
        picked = bool(self.cur_table.selectionModel().selectedRows())
        self.cur_pick_btn.setVisible(picked)
        self.cur_pick_btn.setText("고른 루트 검토 · " + (scope_name if len(logs) <= 3 else f"루트 {len(logs)}개"))
        self.cur_table_heading.setText(f"루트  {len(routes)}" if routes else "루트")
        if self.data_root:
            sel = self.dataset / "selection"
            copy = os.environ.get("TCAR_DECISIONS_DIR")
            self.cur_paths.setText(f"확정: {sel / 'keep.txt'} · drop.txt   ·   삭제 완료: deleted.txt   ·   "
                                   f"최종 삭제 백업: {self.dataset / '_removed'}\n"
                                   + (f"같은 목록과 human_decisions.json을 {copy}에도 복사합니다.\n" if copy else "")
                                   + "루트를 고르면(여러 개: Ctrl/Shift) 검토·검출이 그 루트에만, 고르지 않으면 데이터셋 전체에 "
                                   "적용됩니다. 두 번 누르면 그 루트를 바로 검토합니다.")
        self._show_stages()
        self._show_filters()
        self._style_map()
        self._update_buttons()

    def _show_stages(self):
        """The 라운드 · 배포 table: each curation round (what its 최종 확정 locked and deleted, and the release it went
        out in), then the recordings not finalized yet."""
        d, by_log = self.cdata, self.status_all["names_by_log"]
        rel = d.get("release") or {}
        rel_date = {r["name"]: (r.get("created") or "")[5:10].replace("-", "/") for r in rel.get("releases") or []}

        def courses(logs):
            c = {}
            for lg in logs:
                m = BAG_RE.match(lg)
                c[m["course"] if m else "?"] = c.get(m["course"] if m else "?", 0) + 1
            return " · ".join(f"{k} {v}" for k, v in sorted(c.items())) + f"  ({len(logs)}개)"
        rows = []
        for k, r in enumerate(d["rounds"], 1):
            logs = r.get("logs") or []
            rels = sorted({(rel.get("logs") or {}).get(lg) for lg in logs} - {None})
            st = r.get("stamp", "")
            when = f"{st[4:6]}/{st[6:8]} {st[9:11]}:{st[11:13]}" if len(st) >= 13 else st
            out = ", ".join(f"{x} ({rel_date.get(x, '')})" for x in rels) if rels else "아직 안 올림"
            rows.append((f"{k}라운드", when, courses(logs), f"{r.get('n', 0):,}", f"{r.get('deleted', 0):,}", out,
                         C["ok"] if rels else C["warn"]))
        new_logs = [lg for lg, names in by_log.items() if stage_text(d, lg, names)[0].startswith("새 데이터")]
        if new_logs:
            c = curation_counts(d, [n for lg in new_logs for n in by_log[lg]])
            rows.append(("새 데이터 · 확정 전", "—", courses(new_logs), f"{c['n']:,} (후보 {c['cand']})", "—",
                         "최종 확정 뒤 배포", C["accent"]))
        if not rows:
            rows.append(("아직 최종 확정한 라운드가 없습니다", "", "", "", "", "", C["sub"]))
        self.stage_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, text in enumerate(row[:6]):
                item = QtWidgets.QTableWidgetItem(text)
                item.setForeground(QtGui.QColor(row[6] if j in (0, 5) else C["sub"] if j else C["text"]))
                self.stage_table.setItem(i, j, item)
        self.stage_table.setFixedHeight(len(rows) * 52 + 40)
        latest = rel.get("latest")
        self.stage_note.setText((f"허깅페이스 {rel.get('repo')} · main = 최신 배포 {latest}" if latest else "아직 허깅페이스에 올린 배포가 없습니다")
                                + " · 새 녹화는 파싱 → 검토 → 최종 확정 → scripts/hf_release.py 순서로 새 배포가 됩니다")

    def _scope_changed(self):
        self._show_curation()

    def _clear_scope(self):
        self.cur_table.clearSelection()

    def _curation_primary(self):
        actions = {"detect": self._detect, "rule": self._curate, "review": self._open_review}
        if not self.job and self.cur_action in actions:
            actions[self.cur_action]()

    def _detector_python(self) -> str | None:
        saved = str(SETTINGS.value("detect_python", ""))
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.job_status.setText("검출용 GPU 환경(torch · ultralytics)을 확인하는 중…")
        QtWidgets.QApplication.processEvents()
        try:
            if saved and has_detector(saved):
                return saved
            for python in detector_candidates():
                QtWidgets.QApplication.processEvents()
                if has_detector(python):
                    SETTINGS.setValue("detect_python", python)
                    return python
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.job_status.setText("대기 중")
        self._message("검출 환경을 찾지 못했습니다",
                      "torch와 ultralytics가 설치된 Python이 없습니다.\n‘검출 환경…’에서 conda 환경의 bin/python 같은 "
                      "파일을 직접 고르세요.")
        return None

    def _pick_detector(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "검출용 Python (torch · ultralytics)",
                                                        str(Path.home()))
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = has_detector(path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not ok:
            return self._message("검출 환경이 아닙니다", f"{path}\n이 Python에서 torch와 ultralytics를 불러올 수 없습니다.")
        SETTINGS.setValue("detect_python", path)
        self.refresh()

    def _detect(self):
        logs, scope_name = self._scope()
        missing = self.cur_counts["n"] - self.cur_counts["detected"]
        if self.job or not logs or not missing:
            return
        python = self._detector_python()
        if python:
            self.run(f"객체 검출 · {scope_name}", [
                ("proc", f"key frame 객체 검출 (YOLO · GPU) · 씬 {missing}개",
                 [python, "-u", CURATION / "scene_detect.py", "--dataroot", self.dataset, "--logs", ",".join(logs)]),
                ("proc", "새 검출로 규칙 다시 실행", self._py("curation/curate.py", "--dataroot", self.dataset))],
                kind="detect")

    def _curate(self):
        if self.job or not self.status_all["scenes"]:
            return
        self.run("규칙 제안", [("proc", "삭제 후보 규칙 실행 (데이터셋 전체)",
                                self._py("curation/curate.py", "--dataroot", self.dataset))], kind="curate")

    def _final_delete(self):
        """최종 확정: the round ends. The confirmed 버리기 scenes go, every scene left is locked for good (a later round
        with new recordings can only delete new scenes)."""
        if self.job or not self.data_root:
            return
        sys.path.insert(0, str(CURATION))
        try:
            import apply_selection
            plan = apply_selection.plan(str(self.dataset))
        except (ValueError, OSError) as error:
            return self._message("계획을 만들 수 없습니다", str(error))
        finally:
            sys.path.remove(str(CURATION))
        if plan["candidates"]:
            return self._message("후보를 먼저 확정하세요", f"아직 확정하지 않은 후보 {plan['candidates']}개가 있습니다. "
                                 "잠그면 다시 볼 수 없으니, 검토에서 모두 확정한 뒤 최종 확정하세요.")
        k, new = len(plan["deleted"]), plan["n_before"] - plan["n_locked"] - len(plan["deleted"])
        text = (f"이번 라운드를 최종 확정합니다.\n\n· 버리기 확정 {k}개를 데이터셋에서 뺍니다"
                + (f" (씬 {plan['n_before']}개 → {plan['n_after']}개, 이름 당김 {len(plan['rename'])}개, "
                   f"옮길 파일 {plan['bytes_moved'] / 1e9:.1f} GB · _removed/로 옮기니 되돌릴 수 있음)" if k else "")
                + f"\n· 남는 새 씬 {new}개를 잠급니다" + (f" (이미 잠긴 {plan['n_locked']}개와 함께)" if plan["n_locked"] else "")
                + ".\n\n잠긴 씬은 앞으로 새 데이터가 들어와도 후보가 되지 않고 절대 삭제되지 않습니다. "
                  "새로 파싱한 녹화의 씬만 필터·검토·삭제 대상이 됩니다."
                + ("\n\n지우기 전에 표에만 먼저 적용해 모든 참조(scene · sample · sample_data · ego_pose · prev/next · log/map)와 "
                   "devkit 불러오기를 확인하고, 문제가 있으면 아무것도 지우지 않습니다. 지운 뒤에도 devkit · CAN bus로 다시 확인합니다."
                   if k else ""))
        if not self._confirm("최종 확정할까요?", text, f"최종 확정 · {k}개 삭제" if k else "최종 확정", True):
            return
        tool = lambda *a: self._py("curation/apply_selection.py", "--dataroot", self.dataset, *a)
        steps = ([("proc", "미리 검사: 표에만 삭제를 적용해 모든 참조와 devkit 불러오기 확인 (문제가 있으면 여기서 멈춤)",
                   tool("--simulate"))] if k else [])
        steps += [("proc", "버리기 확정 씬 옮기고 남은 씬 잠그기", tool("--finalize", "--no-precheck"))]
        steps += ([("proc", "삭제 후 점검: 표 참조 · key frame 파일 · devkit · CAN bus", tool("--check"))] if k else [])
        steps += [("proc", "필터 다시 실행 (잠긴 씬 제외)", self._py("curation/curate.py", "--dataroot", self.dataset))]
        self.run(f"최종 확정 · 버리기 {k}개 삭제, 새 씬 {new}개 잠금", steps, kind="apply")

    def _undo_apply(self):
        """Take back the last 최종 확정 (its deletion and its lock), or an older plain deletion."""
        if self.job:
            return
        rounds = self.cdata["rounds"]
        last = rounds[-1] if rounds else None
        backup = Path(last["backup"]) if last and last.get("backup") else (self.cdata["backups"][-1] if not last and self.cdata["backups"] else None)
        if last is None and backup is None:
            return
        what = (f"{last['stamp']}의 최종 확정을 되돌립니다: " if last else f"{backup.name}의 삭제를 되돌립니다: ") + (
            "옮긴 파일, 표, 이름, 확정 기록이 그때로 돌아가고 " if backup else "") + ("그때 잠근 씬이 다시 풀립니다." if last else "")
        if not self._confirm("최종 확정 되돌리기", what, "되돌리기", True):
            return
        step = (["--undo", backup] if backup else ["--unlock-last"])
        self.run("최종 확정 되돌리기", [
            ("proc", f"되돌리기 · {last['stamp'] if last else backup.name}",
             self._py("curation/apply_selection.py", "--dataroot", self.dataset, *step))]
            + ([("proc", "되돌린 뒤 점검: 표 참조 · key frame 파일 · devkit · CAN bus",
                 self._py("curation/apply_selection.py", "--dataroot", self.dataset, "--check"))] if backup else [])
            + [("proc", "필터 다시 실행", self._py("curation/curate.py", "--dataroot", self.dataset))], kind="undo")

    # ---- the dataset map, and the curation state as it changes ----
    def _load_map(self):
        if not self._viewer_ready:
            return
        self._map_reply = self.cur_map.net.get(QtNetwork.QNetworkRequest(QtCore.QUrl(f"{self.viewer_url}/api/map")))
        self._map_reply.finished.connect(self._map_loaded)

    def _map_loaded(self):
        reply, self._map_reply = self._map_reply, None
        if reply is None:
            return
        data = bytes(reply.readAll())
        reply.deleteLater()
        try:
            scenes = json.loads(data)["scenes"]
        except (ValueError, KeyError, TypeError):
            return
        self.cur_map.set_tracks(scenes)
        self._style_map()
        self._show_filters()

    def _style_map(self):
        """빈도: the scenes that stay (locked, passed, kept by a person), one see-through colour each, so a road driven
        once is faint and one driven many times is strong; 상태별: every scene by its state. Picked routes bright."""
        by_log = self.status_all["names_by_log"]
        state_col, state_ko = {k: col for k, _, col in CUR_STATES}, {k: t for k, t, _ in CUR_STATES}
        freq = self._map_color != "state"
        short_of = lambda lg: f"{BAG_RE.match(lg)['course']}-{BAG_RE.match(lg)['num']}" if BAG_RE.match(lg) else lg
        colors, tips, dim, hidden, count, rank = {}, {}, set(), set(), dict.fromkeys(state_col, 0), {}
        scope = set(self.cur_logs)
        blend = QtGui.QColor(*FREQ_COLOUR)
        blend.setAlphaF(FREQ_ALPHA)
        for lg, names in by_log.items():
            for n in names:
                st = scene_state(self.cdata, n)
                count[st] += 1
                tips[n] = f"{short_of(lg)} · {n} · {state_ko[st]}"
                if freq:
                    if st in ("keep", "pass", "locked"):
                        colors[n] = blend.name(QtGui.QColor.NameFormat.HexArgb)
                    else:
                        hidden.add(n)                   # going or undecided: not part of what the dataset keeps
                else:
                    colors[n] = state_col[st]
                    rank[n] = 0 if st in ("pass", "locked") else 2 if st in ("keep", "drop") else 1   # decisions on top
                if scope and lg not in scope:
                    dim.add(n)
        if freq:
            kept = sum(count[k] for k in ("keep", "pass", "locked"))
            legend = [[(f"남는 씬 {kept:,}개 · 겹칠수록 진하게", None)],
                      [(" ● ", freq_swatch(1)), ("1번  ", C["sub"]), ("● ", freq_swatch(2)), ("2번  ", C["sub"]),
                       ("● ", freq_swatch(4)), ("4번  ", C["sub"]), ("● ", freq_swatch(8)), ("8번+", C["sub"])]]
        else:
            legend = [[("●  ", state_col[k]), (f"{state_ko[k]}  ", None), (f"{count[k]:,}", C["sub"])]
                      for k, _, _ in CUR_STATES if k != "deleted" and (count[k] or k not in ("other", "locked"))]
        self.cur_map.set_style(colors, dim, tips, legend, rank, None, hidden)

    def _show_filters(self):
        """The filter card, in the filters' order: each one's rule, how it counts, why, the data behind it (the rule's
        own parameters and results), and whether it is skipped."""
        d, P, det = self.cdata, self.cdata["params"], self.cdata["detector"]
        skip = set((d["filters"] or {}).get("skip") or [])
        caught = {k: sum(v == k for v in d["candidates"].values()) for k in CANDIDATES}
        open_ = {k: sum(d["candidates"].get(n) == k and not d["human"].get(n) for n in d["candidates"]) for k in CANDIDATES}
        place, head = P.get("place_m", 10), P.get("heading_deg", 30)
        for key, b in self.filter_skip.items():
            b.blockSignals(True)
            b.setChecked(key in skip)
            b.setText("건너뜀 · 되돌리기" if key in skip else "건너뛰기")
            b.blockSignals(False)
            b.setEnabled(self.job is None and bool(self.status_all["scenes"]))
        counted = lambda k: ("건너뜀 · 이 필터는 후보를 만들지 않습니다" if k in skip else
                             f"걸린 씬 {caught[k]} · 아직 후보 {open_[k]}")

        count, rule, basis, why, chart = self.filter_rows["stops"]
        groups = d["stop_groups"]
        sizes = [len(g) for g in groups]
        count.setText(counted("stops"))
        rule.setText("같은 곳에 멈춘 씬끼리 묶어, 주변 움직임이 가장 큰 하나만 남기고 나머지를 후보로")
        basis.setText(f"멈춤: 시간의 80% 이상 1 km/h 미만 · 같은 곳: 같은 루트에서 서로 경로의 80% 이상이 {place:g} m · {head:g}° 안, "
                      f"속도 흐름 차 ≤ {P.get('ego_tau', 1):g} · 주변 움직임: LiDAR 0.5초 간격 프레임에서 바뀐 칸 수"
                      + (f"\n근거: 멈춘 씬 {d['stopped']}개 → 같은 곳 그룹 {len(groups)}개 (그룹당 {min(sizes)}–{max(sizes)}개) → "
                         f"대표 {len(groups)}개를 남기고 후보 {sum(sizes) - len(groups)}개" if groups else
                         "\n근거: 같은 곳에 멈춘 씬 그룹이 없습니다"))
        why.setText("왜: 신호 대기처럼 거의 같은 장면이 반복되면 대표 하나로 충분합니다. 가장 먼저 거르고, 그룹에 든 씬은 남긴 대표까지 "
                    "②③에서 다시 보지 않습니다(대표가 뒤에서 걸려 그룹이 통째로 빠지지 않게). 그래프: 그룹 크기별 그룹 수")
        top = max(sizes, default=2)
        chart.set_bars([(sizes.count(k), "#c792ea", f"{k}개짜리 그룹 {sizes.count(k)}개") for k in range(2, top + 1)],
                       left="2개", right=f"{top}개 (그룹 크기)")

        count, rule, basis, why, chart = self.filter_rows["objects"]
        mo = P.get("min_objects", 3.0)
        vals = [v for v in d["objects"].values() if v is not None]
        settled = {m for g in groups for m in g}          # ①'s groups, the kept scenes too, never reach ②
        left = [v for n, v in d["objects"].items() if v is not None and n not in settled]
        count.setText(counted("objects"))
        rule.setText(f"①의 정지 그룹(대표 포함)을 뺀 씬 중, 가까운 도로 이용자가 샘플당 평균 {mo:g}개 미만인 씬")
        basis.setText(f"셈: 차량 · 버스 · 트럭 · 보행자 · 자전거 · 오토바이, 박스 높이 {det.get('near_px') or 50}px 이상(멀리 있는 것 제외), "
                      f"카메라 6대 합, 씬의 key frame 전부의 평균 · 검출 {det.get('model') or 'YOLO'} (신뢰도 ≥ {det.get('conf') or 0.4}), "
                      "자차 차체와 이륜차 탑승자는 빼고 셈"
                      + (f"\n근거: 검출된 씬 {len(vals)}개에서 ①의 정지 그룹 {len(settled)}개를 뺀 {len(left)}개 중 "
                         f"{sum(v < mo for v in left)}개가 기준 미만"
                         if vals else "\n근거: 아직 검출한 씬이 없습니다"))
        why.setText("왜: 주변에 배울 대상이 거의 없는 장면은 학습에 주는 정보가 적습니다. 그래프: 씬별 샘플당 평균(전체), 점선 왼쪽이 기준 미만")
        bins = [0] * 16
        for v in vals:
            bins[min(int(v), 15)] += 1
        chart.set_bars([(c, "#e8a94f" if i < mo else "#46505e", f"샘플당 {i}{'개 이상' if i == 15 else f'–{i + 1}개'}: 씬 {c}개")
                        for i, c in enumerate(bins)], marker=mo, marker_text=f"기준 {mo:g}", left="0", right="15+ 개/샘플")

        count, rule, basis, why, chart = self.filter_rows["overlap"]
        mv = P.get("min_overlap", (d["filters"] or {}).get("min_overlap", 0.6))
        cov = d["path_cover"]
        count.setText(counted("overlap"))
        rule.setText(f"①②가 정하지 않은 달리는 씬 중, 먼저 남긴 씬이 이 씬 경로의 {mv:.0%} 이상을 같은 방향으로 지나간 씬")
        basis.setText(f"같은 길: {place:g} m · {head:g}° 안 (정지 중복과 같은 기준), 0.5초 간격 경로, 모든 루트끼리 · 남길 씬은 가까운 "
                      "도로 이용자가 많은 순으로 먼저 고름"
                      + (f"\n근거: 달리는 씬 {len(cov)}개 중 다른 씬이 {mv:.0%} 이상 덮는 씬 {sum(c >= mv for c in cov)}개 "
                         f"(80% 이상 {sum(c >= 0.8 for c in cov)}개)" if cov else "\n근거: 규칙을 다시 돌리면 계산합니다"))
        why.setText("왜: 같은 길을 같은 방향으로 다시 달린 씬은 새로 보여 주는 것이 적습니다. 씬이 20초라 시작 지점이 어긋나 통째로 "
                    "겹치는 일은 드뭅니다. 그래프: 씬마다 다른 씬이 덮는 비율(가장 많이), 점선 오른쪽이 기준 이상 · 지도의 ‘빈도’ 보기")
        bins = [0] * 10
        for c in cov:
            bins[min(int(c * 10), 9)] += 1
        chart.set_bars([(c, "#5ec8e5" if (i + 1) / 10 > mv else "#46505e", f"{i * 10}–{i * 10 + 10}% 덮임: 씬 {c}개")
                        for i, c in enumerate(bins)], marker=mv * 10, marker_text=f"기준 {mv:.0%}", left="0%", right="100% 덮임")

    def _skip_filter(self, key, on):
        """Skip (or bring back) one filter: selection/filters.json, which curate.py reads, then the rule again."""
        if self.job or not self.data_root:
            return
        conf = dict(self.cdata["filters"] or {})
        skip = set(conf.get("skip") or [])
        (skip.add if on else skip.discard)(key)
        conf.update(skip=[k for k in ("stops", "objects", "overlap") if k in skip], min_overlap=conf.get("min_overlap", 0.6))
        path = self.dataset / "selection" / "filters.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(conf, ensure_ascii=False, indent=1))
        except OSError as error:
            return self._message("필터 설정을 저장할 수 없습니다", str(error))
        self.cdata["filters"] = conf
        self._show_filters()
        self._curate()

    def _set_map_color(self, key):
        self._map_color = key
        SETTINGS.setValue("map_color", key)
        for k, b in self.map_color_btns.items():
            b.setChecked(k == key)
        self._style_map()

    def _map_pick_route(self, log):
        if self.job:
            return
        for row in range(self.cur_table.rowCount()):
            if self.cur_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == log:
                self.cur_table.selectRow(row)
                self.cur_table.scrollToItem(self.cur_table.item(row, 0))

    def _map_open_scene(self, name, log):
        m = BAG_RE.match(log)
        self._open_review([log], name, f"{m['course']}-{m['num']} · {name}" if m else f"{log} · {name}")

    def _curation_signature(self):
        ds = self.dataset
        sig = [_mtime(ds / "selection" / f) for f in ("human_decisions.json", "result.json", "object_counts.json")]
        sig.append(_mtime(ds / VERSION / "scene.json"))
        sig += [(m.parent.name, _mtime(m)) for m in sorted((ds / "_removed").glob("*/manifest.json"))]
        sig += sorted(u.parent.name for u in (ds / "_removed").glob("*/UNDONE"))
        return tuple(sig)

    def _final_buttons(self):
        """최종 삭제 on the curation page and in the review bar: the confirmed 버리기 scenes of the dataset."""
        names = [n for ns in self.status_all["names_by_log"].values() for n in ns]
        drop = sum(scene_state(self.cdata, n) == "drop" for n in names)
        cand = sum(scene_state(self.cdata, n) in CANDIDATES for n in names)
        new = sum(scene_state(self.cdata, n) != "locked" for n in names)
        for b in (self.cur_final_btn, self.review_final_btn):
            b.setText(f"최종 확정 · 버리기 {drop}개 삭제" if drop else "최종 확정 · 삭제 없음")
            b.setEnabled(self.job is None and bool(new) and not cand)
            b.setToolTip("새 데이터가 없습니다 (모두 잠김)" if not new else f"후보 {cand}개를 먼저 확정하세요" if cand else
                         f"버리기 {drop}개를 빼고 남는 새 씬을 잠급니다")

    def _final_from_review(self):
        self._close_review()
        self._final_delete()

    def _poll_curation(self):
        """Confirmations made in a review (in this window or a browser) show on the curation page as they happen;
        while the review fills the window only the 최종 삭제 count follows (a light read, playback stays smooth)."""
        if self.mode != "curation" or self.job or not self.data_root:
            return
        sig = self._curation_signature()
        old, self._cur_sig = self._cur_sig, sig
        if old is None or old == sig:
            return
        if self.reviewing:
            self.cdata = curation_data(self.dataset)
            self._final_buttons()               # closing the review refreshes everything else
            return
        if old[3] != sig[3]:                    # the tables changed (e.g. a final deletion from the command line)
            return self.refresh()
        self.cdata = curation_data(self.dataset)
        for i in range(self.route_list.count()):
            item = self.route_list.item(i)
            route = item.data(Qt.ItemDataRole.UserRole)
            c = curation_counts(self.cdata, self._course_scenes(route))
            action = curation_action(c)
            text = self._curation_copy(c)[action][3]
            item.setData(SUB_ROLE, text)
            item.setData(DOT_ROLE, self._cur_dot(action))
            item.setToolTip(f"코스 {route}\n{text}")
        self._show_curation()

    # ---- review viewer: curation/sample_viewer.py, served by this app, shown in the window ----
    def _start_viewer(self):
        if self.viewer is not None:
            return
        port = free_port()
        self.viewer_url, self._viewer_buf, self._viewer_ready = f"http://127.0.0.1:{port}", "", False
        proc = QtCore.QProcess(self)
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._viewer_output)
        proc.finished.connect(self._viewer_finished)
        self.viewer = proc
        wait = max(10, round(self.status_all["scenes"] * 0.23 / 10) * 10)     # 334 scenes: 76 s from the /data HDD
        self.review_status.setText(f"검토 화면을 준비하는 중… 데이터셋 표를 읽습니다 (씬 {self.status_all['scenes']}개, "
                                   f"약 {wait}초). 한 번 띄우면 데이터셋이 바뀔 때까지 그대로 씁니다.")
        proc.start(sys.executable, ["-u", str(CURATION / "sample_viewer.py"), "--dataroot", str(self.dataset),
                                    "--port", str(port), "--no-browser", "--fill"])    # the cache fills while idle

    def _warm_viewer(self):
        """In curation mode the review server starts in the background, so opening a review is instant."""
        if self.mode == "curation" and not self.job and self.data_root and self.status_all["scenes"]:
            self._start_viewer()

    def _review_url(self) -> str:
        # one screen for every scope: all its scenes in the list (candidates and confirmed marked); app=1 hides
        # the links that lead out of the review
        query = f"?app=1&logs={','.join(self._review_logs)}" if self._review_logs else "?app=1"
        where = f"#scene={self._review_scene}&view=all" if self._review_scene else "#view=all"
        return f"{self.viewer_url}/review{query}{where}"

    def _viewer_output(self):
        if self.viewer is None:
            return
        self._viewer_buf = (self._viewer_buf + bytes(self.viewer.readAllStandardOutput()).decode(errors="replace"))[-20000:]
        if not self._viewer_ready and "[viewer] serving" in self._viewer_buf:
            self._viewer_ready = True
            self.review_status.setText("")
            self.cur_map.set_base(self.viewer_url)
            self._load_map()
            self.review_title.setToolTip(f"{self.dataset}\n{self.viewer_url}")
            if self.reviewing and self.web is not None:
                self.web.load(QtCore.QUrl(self._review_url()))
            elif self.web is None:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._review_url()))

    def _viewer_finished(self, code, _status):
        tail = "\n".join(self._viewer_buf.strip().splitlines()[-6:])
        self.viewer, self._viewer_ready = None, False
        self.cur_map.set_base(None)
        if self.reviewing and code not in (0, 9, -9):
            self.review_status.setText(f"검토 서버가 멈췄습니다 (종료 코드 {code})")
            self.log_view.appendPlainText("[viewer]\n" + tail)
            self.log_panel.show()

    def _stop_viewer(self):
        if self.viewer is not None:
            proc, self.viewer = self.viewer, None
            proc.finished.disconnect()
            proc.kill()
            proc.waitForFinished(3000)
        self._viewer_ready = False
        self.cur_map.set_base(None)
        if self.web is not None:
            self.web.setUrl(QtCore.QUrl("about:blank"))

    def _open_review(self, logs=None, scene=None, scope_name=""):
        if logs is None:
            logs, scope_name = self._scope()
        if self.job or not logs:
            return
        self._review_logs, self._review_scene = logs, scene
        self.review_title.setText(f"검토 · {scope_name}")
        self._start_viewer()
        if self.web is None:
            if self._viewer_ready:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._review_url()))
            return
        self.reviewing = True
        self.sidebar.hide()                 # the viewer wants the whole window
        self.jobbar.hide()
        self.pages.setCurrentIndex(2)
        if self._viewer_ready:
            self.web.load(QtCore.QUrl(self._review_url()))

    def _close_review(self, refresh=True):
        self.reviewing = False
        self.sidebar.show()
        self.jobbar.show()
        self.pages.setCurrentIndex(0 if self.mode == "parser" else 1)
        if refresh:
            self.refresh()                  # decisions made in the viewer change the rule's final list

    def _reload_review(self):
        if self.web is not None and self._viewer_ready:
            self.web.reload()

    def _review_in_browser(self):
        if self._viewer_ready:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._review_url()))

    def _show_route(self, *_):
        route = self.current_route()
        self.table.setRowCount(0)
        self.recs, self.active_recs = [], []
        self.status = {"logs": {}, "scenes": 0, "samples": 0, "error": ""}
        if route:
            all_recs, self.status = self._route_cache[route]
            self.active_recs = [r for r in all_recs if not r["excluded"] and r["raw"]]
            self.recs = [r for r in all_recs if not r["excluded"] or self.show_excluded.isChecked()]
            self.next_action, _ = self._route_action(route, all_recs, self.status)
            self.title.setText(f"코스 {route}")
            self.title.setToolTip(route)
            dates = ", ".join(sorted({r["date"] for r in all_recs if r["date"]}))
            meta = [dates, f"녹화 {sum(not r['excluded'] for r in all_recs)}개"]
            if self.status["samples"]:
                meta.append(f"샘플 {self.status['samples']:,}개")
            self.subtitle.setText("  ·  ".join(filter(None, meta)))
            self._show_stats(all_recs)
            excluded_count = sum(r["excluded"] for r in all_recs)
            self.show_excluded.setText(f"제외된 녹화 보기 ({excluded_count})")
            self.show_excluded.setVisible(bool(excluded_count) or self.show_excluded.isChecked())
            self.table.setRowCount(len(self.recs))
            longest = max(((r["stats"] or {}).get("orig_s") or r["duration"] or 0.0 for r in self.recs), default=0.0)
            for row, rec in enumerate(self.recs):
                st = rec["stats"]
                size = human_size(rec["size"]) if rec["size"] is not None else ""
                detail = " · ".join(filter(None, [rec["label"], rec["when"], size]))
                if rec["excluded"]:
                    state, color = "제외됨", C["muted"]
                elif self.status["error"]:
                    state, color = "데이터셋 확인 필요", C["warn"]
                elif rec["name"] in self.status["logs"]:
                    st_text, _ = stage_text(self.cdata, rec["name"], self.status_all["names_by_log"].get(rec["name"], []))
                    state, color = f"데이터셋에 포함 ({self.status['logs'][rec['name']]}씬) · {st_text}", C["ok"]
                elif rec["name"] in self.status["imported"]:
                    state, color = "큐레이션으로 모두 제외", C["muted"]
                else:
                    state, color = "파싱 대기", C["accent"]
                if not rec["raw"]:
                    detail += " · 원본은 데이터 폴더에 없음"
                orig = st["orig_s"] if st else rec["duration"]
                rate = 100 * st["used_s"] / st["orig_s"] if st and st["orig_s"] else None
                cells = (rec["name"], "", f"{orig / 60:.1f}분" if orig is not None else "—",
                         f"{st['used_s'] / 60:.1f}분" if st else "—",
                         f"{rate:.1f}%" if rate is not None else "—", state)
                if st:
                    seg = {"used": st["used_s"], "curated": st["curated_s"], "lost": max(st["orig_s"] - st["parsed_s"], 0.0),
                           "waiting": 0.0}
                else:
                    seg = {"used": 0.0, "curated": 0.0, "lost": 0.0, "waiting": rec["duration"] or 0.0}
                parts = [(seg[k], C["line2"] if rec["excluded"] else col) for k, _, col in OV_PARSER]
                for column, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setToolTip((rec["path"] or rec["name"]) if column == 0 else
                                    loss_text(st) if st and column in (1, 2, 3, 4) else text)
                    if column == 0:
                        item.setData(SUB_ROLE, detail)
                    if column == 1:
                        item.setData(BAR_ROLE, (parts, sum(seg.values()), longest))
                    if column == 5:
                        item.setData(DOT_ROLE, color)
                    if rec["excluded"]:
                        item.setForeground(QtGui.QColor(C["faint"]))
                    elif column == 4 and rate is not None and rate < 90:
                        item.setForeground(QtGui.QColor(C["warn"]))
                    elif column in (2, 3, 4):
                        item.setForeground(QtGui.QColor(C["sub"]))
                    self.table.setItem(row, column, item)
            if excluded_count and not self.recs:
                self.empty_title.setText("사용할 녹화가 모두 제외되어 있습니다")
                self.empty_desc.setText("‘제외된 녹화 보기’를 켜고 녹화를 복원하거나, 새 녹화를 가져오세요.")
            else:
                self.empty_title.setText("아직 이 코스에 녹화가 없습니다")
                self.empty_desc.setText("녹화를 가져올 때 저장할 코스를 이 코스로 지정하세요.")
        else:
            self.next_action = "import" if self.data_root else "root"
            self.title.setText("녹화에서 데이터셋까지")
            self.subtitle.setText("단계를 따라가면 됩니다. 각 단계가 끝나면 다음 할 일을 알려드립니다.")
            self.empty_title.setText("가져온 녹화가 여기에 표시됩니다")
            self.empty_desc.setText("녹화별로 파싱 대기, 데이터셋 포함 여부를 한눈에 확인할 수 있습니다.")
        if not route:
            self.show_excluded.setVisible(False)
            self.stats.setVisible(False)
        self.recording_stack.setCurrentIndex(0 if self.recs else 1)
        self.recording_heading.setText(f"녹화  {len(self.recs)}" if route else "녹화")
        self._show_next_action()
        self._update_buttons()

    def _show_stats(self, all_recs):
        """Course totals over its parsed recordings, and the whole dataset's for scale."""
        parsed = [r["stats"] for r in all_recs if not r["excluded"] and r["stats"]]
        self.stats.setVisible(bool(parsed))
        if not parsed:
            return
        orig, used = sum(s["orig_s"] for s in parsed), sum(s["used_s"] for s in parsed)
        self.stat_values["orig"].setText(f"{orig / 60:.1f}분")
        self.stat_values["used"].setText(f"{used / 60:.1f}분")
        self.stat_values["rate"].setText(f"{100 * used / orig:.1f}%" if orig else "—")
        self.stat_values["scenes"].setText(f"{self.status['scenes']:,}")
        tip = {k: sum(s[k] for s in parsed) for k in ("short_s", "camera_s", "other_s", "curated_n", "curated_s")}
        tip.update(orig_s=orig, used_s=used, lidar_gaps=sum(s["lidar_gaps"] for s in parsed))
        for value in self.stat_values.values():
            value.setToolTip(loss_text(tip))
        notes = []
        pending = sum(not r["excluded"] and r["raw"] and not r["stats"] for r in all_recs)
        if pending:
            notes.append(f"파싱 대기 {pending}개는 빼고 셌습니다")
        every = [s for s in (log_time(self.dataset, n, self.status_all["logs"].get(n, 0))
                             for n in self.status_all["imported"]) if s]
        if len(every) > len(parsed):
            o, u = sum(s["orig_s"] for s in every), sum(s["used_s"] for s in every)
            notes.append(f"데이터셋 전체 {len(every)}개 녹화: 원본 {o / 60:.1f}분 중 {u / 60:.1f}분 "
                         f"({100 * u / o:.1f}%), 씬 {self.status_all['scenes']:,}개")
        self.stats_note.setText("\n".join(notes))

    def _calib_dir(self) -> Path | None:
        calib = self.data_root / "calib" if self.data_root else None
        return calib if calib and calib.is_dir() else None

    def _show_next_action(self):
        action = self.next_action
        pending = sum(r["name"] not in self.status["logs"] for r in self.active_recs)
        calib = self._calib_dir()
        calib_note = (f" 보정값은 {calib}을 씁니다." if calib else
                      " 데이터 폴더에 calib/이 없어 기본 보정값(카메라 비보정)으로 변환합니다.")
        copy = {
            "root": ("먼저 데이터 폴더를 선택하세요", "녹화 사본, 파싱 결과, 작업 로그를 함께 보관할 폴더입니다. 선택한 위치는 다음 실행에도 기억합니다.", "데이터 폴더 선택…"),
            "import": ("녹화를 가져오세요", "외장 SSD 같은 원본 폴더를 고르면 bag을 코스별로 데이터 폴더에 복사합니다.", "녹화 가져오기…"),
            "parse": (f"녹화 {pending}개를 데이터셋에 추가하세요", f"새 녹화만 파싱해 통합 데이터셋 parsed/{DATASET}에 이어 붙입니다." + calib_note, f"{pending}개 파싱하기"),
            "validate": ("데이터셋을 검증하세요", self.status["error"] or "nuScenes devkit으로 통합 데이터셋 전체를 열어 구조를 확인합니다. 통과하면 이 코스는 끝입니다.", "검증하기"),
            "done": ("데이터셋 준비가 끝났습니다", f"이 코스 씬 {self.status['scenes']:,}개 · 샘플 {self.status['samples']:,}개, 검증 통과. 새 녹화는 언제든 추가할 수 있습니다.", "모두 완료"),
        }
        title, description, primary = copy[action]
        self.action_kicker.setText("완료" if action == "done" else "다음 할 일")
        self.action_kicker.setStyleSheet(f"color: {C['ok']};" if action == "done" else "")
        self.action_title.setText(title)
        self.action_desc.setText(description)
        self.primary_btn.setText(primary)
        active_step = {"root": 0, "import": 0, "parse": 1, "validate": 2, "done": -1}[action]
        self.stepper.set_state(active_step, len(Stepper.NAMES) if action == "done" else max(active_step, 0))
        self.primary_btn.setVisible(action != "done")

    def selected_rec(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows()
        return self.recs[rows[0].row()] if rows and rows[0].row() < len(self.recs) else None

    def _update_buttons(self):
        busy = self.job is not None
        rec = self.selected_rec()
        raw = bool(rec and rec["raw"])
        self.change_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy and bool(self.data_root))
        self.route_list.setEnabled(not busy)
        self.import_btn.setEnabled(not busy and bool(self.data_root))
        self.primary_btn.setEnabled(not busy and self.next_action != "done")
        self.val_btn.setVisible(bool(self.status["logs"]) and self.next_action != "validate")
        self.val_btn.setEnabled(not busy)
        route = self.current_route()
        self.yield_btn.setVisible(bool(self.active_recs))
        self.yield_btn.setEnabled(not busy)
        self.yield_last_btn.setVisible(bool(route and self._yield_file(route).exists()))
        self.yield_folder_btn.setEnabled(not busy and bool(self.data_root))
        self.open_btn.setEnabled(raw)
        self.excl_btn.setEnabled(not busy and raw and not self.status["error"])
        restore = bool(rec and rec["excluded"])
        self.excl_btn.setText("녹화 복원" if restore else "녹화 제외…")
        self.excl_btn.setObjectName("" if restore else "danger")
        self.excl_btn.style().unpolish(self.excl_btn)
        self.excl_btn.style().polish(self.excl_btn)
        self.selection_hint.setText("복원한 뒤 파싱하면 데이터셋에 다시 들어갑니다." if restore else
                                    "제외하면 데이터셋에서도 빠집니다. 실행 전에 한 번 더 묻습니다." if raw else
                                    "원본 bag이 데이터 폴더에 없는 녹화입니다. 원본을 가져오면 제외·복원할 수 있습니다." if rec else
                                    "녹화를 선택하면 폴더 열기와 제외를 할 수 있습니다.")
        self.open_btn.setVisible(raw)
        self.excl_btn.setVisible(raw)
        for b in self.mode_btns.values():
            b.setEnabled(not busy)
        c = self.cur_counts
        self.cur_primary.setEnabled(not busy and self.cur_action not in ("done", "empty"))
        self.cur_review_btn.setEnabled(not busy and bool(self.status_all["scenes"]))
        self.cur_pick_btn.setEnabled(not busy and bool(c["n"]))
        self.cur_detect_btn.setEnabled(not busy and c["detected"] < c["n"])
        self._final_buttons()
        self.cur_rule_btn.setEnabled(not busy and bool(self.status_all["scenes"]))
        self.cur_clear_btn.setVisible(bool(self.cur_table.selectionModel().selectedRows()))
        self.cur_table.setEnabled(not busy)
        self.cur_undo_btn.setVisible(self.mode == "curation" and bool(self.cdata["backups"] or self.cdata["rounds"]))
        self.cur_undo_btn.setEnabled(not busy)
        self.cur_detector_btn.setEnabled(not busy)

    def _primary(self):
        actions = {"root": self._change_root, "import": self._import,
                   "parse": self._parse, "validate": self._validate}
        if not self.job and self.next_action in actions:
            actions[self.next_action]()

    def _py(self, script, *args):
        return [sys.executable, "-u", str(HERE / script), *map(str, args)]

    def run(self, title, steps, kind="", route=None):
        if self.job or not self.data_root or not steps:
            return
        try:
            job = Job(title, steps, self.data_root / "logs")
        except OSError as error:
            self._message("작업을 시작할 수 없습니다", f"로그 저장 위치를 확인하세요.\n{error}")
            return
        self.job = job
        self._job_kind, self._job_route = kind, route
        if kind in ("parse", "exclude", "restore", "apply", "undo"):
            self.validated.pop(DATASET, None)
        if kind in DATASET_WRITERS:
            self._stop_viewer()             # it serves the tables it read at start; restarted on the next review
        self._validation_stamp = dataset_stamp(self.dataset) if kind == "validate" else ()
        self._last_log = job.log_path
        job.progress.connect(self._on_progress)
        job.output.connect(self.log_view.appendPlainText)
        job.finished.connect(self._on_finished)
        self.log_view.clear()
        self.log_path.setText(f"로그 저장: {job.log_path}")
        self.job_status.setStyleSheet("")
        self.job_status.setText(f"{title} · 작업을 시작합니다")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.cancel_btn.setEnabled(True)
        on_course = self.mode == "parser"
        kicker, head, desc, primary = ((self.action_kicker, self.action_title, self.action_desc, self.primary_btn)
                                       if on_course else
                                       (self.cur_kicker, self.cur_title, self.cur_desc, self.cur_primary))
        kicker.setText("작업 진행 중")
        kicker.setStyleSheet("")
        head.setText(title)
        desc.setText("아래에서 진행 상황을 확인하세요. 작업이 끝나면 다음 할 일을 안내합니다.")
        primary.setText("진행 중…")
        primary.setVisible(True)
        self._update_buttons()
        job.start()

    def _on_progress(self, fraction, text):
        self.progress.setRange(0, 0 if fraction < 0 else 1000)
        if fraction >= 0:
            self.progress.setValue(int(fraction * 1000))
            text += f" · {int(fraction * 100)}%"
        self.job_status.setText(text)

    def _on_finished(self, ok, message):
        job = self.job
        if ok and self._job_kind == "validate":
            stamp = dataset_stamp(self.dataset)
            if stamp and stamp == self._validation_stamp:
                self.validated[DATASET] = stamp
                save_validation(self.data_root, stamp, job.log_path)
        self.job = None
        self.cancel_btn.setEnabled(False)
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 if ok else 0)
        self.job_status.setStyleSheet(f"color: {C['ok'] if ok else C['warn']};")
        self.job_status.setText(f"{job.title} · {message}")
        self.refresh()
        self._warm_viewer()
        if not ok:
            self.log_panel.show()
            self.log_btn.setText("로그 닫기")
        elif self._job_kind == "yield" and not self._close_when_done:
            QtCore.QTimer.singleShot(0, lambda: self._show_yield(self._yield_out, self._yield_title))
        if self._close_when_done:
            QtCore.QTimer.singleShot(0, self.close)

    def _confirm(self, title, text, action, danger=False):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(title)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning if danger else QtWidgets.QMessageBox.Icon.Question)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(text)
        cancel = box.addButton("돌아가기", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        proceed = box.addButton(action, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        if danger:
            proceed.setObjectName("danger")
        box.setDefaultButton(cancel)
        box.exec() if hasattr(box, "exec") else box.exec_()
        return box.clickedButton() == proceed

    def _message(self, title, text):
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(title)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(text)
        box.addButton("확인", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.exec() if hasattr(box, "exec") else box.exec_()

    def _cancel(self):
        if self.job and self._confirm("작업 취소", "현재 작업을 중단합니다. 완료된 복사·파싱은 되돌리지 않습니다.\n미완료 복사본은 지웁니다. 다음 실행 전 목록과 로그를 확인하세요.", "작업 중단", True):
            self.cancel_btn.setEnabled(False)
            self.job_status.setText("작업을 중단하고 있습니다…")
            self.job.cancel()

    def _toggle_log(self):
        visible = self.log_panel.isHidden()
        self.log_panel.setVisible(visible)
        self.log_btn.setText("로그 닫기" if visible else "로그 보기")
        SETTINGS.setValue("log_open", "true" if visible else "false")

    def _change_root(self):
        if self.job:
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "데이터 폴더 선택 · raw / parsed / logs를 보관할 위치", str(self.data_root or DEFAULT_ROOT))
        if path:
            self.data_root = Path(path).resolve()
            SETTINGS.setValue("data_root", str(self.data_root))
            self.validated.clear()
            self.refresh()

    def _import(self, _checked=False, source=None):
        if self.job or not self.data_root:
            return
        dialog = ImportDialog(self, self.data_root, self.route_dirs(), set(self.status_all["imported"]), source=source)
        if (dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()) != QtWidgets.QDialog.DialogCode.Accepted:
            return
        steps, copies = [], 0
        for src, route in dialog.selection():
            if not COURSE_RE.fullmatch(route):
                return self._message("코스를 확인하세요", "코스 이름은 A, B처럼 대문자로 시작해야 합니다.")
            for f in [src, *companions(src)]:
                dst = self.data_root / "raw" / route / f.name
                excluded = self.data_root / "raw" / "_excluded" / route / f.name
                if dst.exists() or excluded.exists():
                    return self._message("이미 있는 녹화입니다", f"{f.name}\n목록을 새로 확인하세요. 제외된 녹화는 복원할 수 있습니다.")
                steps.append(("copy", f"복사 · {f.name}", str(f), str(dst)))
            copies += 1
        self.run(f"녹화 {copies}개 가져오기", steps, kind="import")

    def _parse(self):
        route = self.current_route()
        if self.job or not route or self.next_action != "parse":
            return
        todo = sorted((r for r in self.active_recs if r["name"] not in self.status["logs"]),
                      key=lambda r: route_key(r["name"]))
        args = [r["path"] for r in todo] + ["--out", self.dataset, "--split", "train", "--no-validate"]
        calib = self._calib_dir()
        if calib:
            args += ["--calib", calib]
        self.run(f"코스 {route} 파싱", [("proc", f"새 녹화 {len(todo)}개 파싱", self._py("bag2nuscenes.py", *args))],
                 kind="parse", route=route)

    def _yield_file(self, name: str) -> Path:
        return self.data_root / "logs" / "yield" / f"{name}.json"

    def _run_yield(self, paths: list[Path], name: str, title: str, route=None):
        self._yield_out, self._yield_title = self._yield_file(name), title
        self.run(f"수율 검사 · {title}", [("proc", "bag 시각 읽고 프레임 기준 적용",
                                         self._py("yield_check.py", *paths, "--json", self._yield_out))],
                 kind="yield", route=None)

    def _yield_route(self):
        route = self.current_route()
        if self.job or not route or not self.active_recs:
            return
        self._run_yield([Path(r["path"]) for r in self.active_recs], route, f"코스 {route}")

    def _yield_folder(self):
        if self.job or not self.data_root:
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "수율을 검사할 폴더 (bag이 든 폴더)",
                                                          str(SETTINGS.value("last_source", "")))
        if not path:
            return
        folder = Path(path)
        SETTINGS.setValue("last_source", str(folder))
        if not scan_source(folder):
            return self._message("녹화를 찾지 못했습니다", ".bag 파일이 든 폴더 또는 그 상위 폴더를 고르세요.")
        self._run_yield([folder], f"folder_{folder.name}", folder.name)

    def _yield_last(self):
        route = self.current_route()
        if route:
            self._show_yield(self._yield_file(route), f"코스 {route}")

    def _show_yield(self, path: Path, title: str):
        try:
            result = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            return self._message("수율 결과를 읽을 수 없습니다", f"{path}\n{error}")
        dialog = YieldDialog(self, result, title)
        dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()

    def _validate(self):
        if self.job or not self.data_root or not self.status_all["logs"] and not self.status_all["error"]:
            return
        # The existing helper skips missing devkit; require it here so a skip is
        # never presented to an operator as a successful validation.
        code = ("import sys; from pathlib import Path; "
                "from nuscenes.nuscenes import NuScenes; "
                f"sys.path.insert(0, {str(HERE)!r}); "
                "from nuscenes_writer import validate_with_devkit; "
                f"validate_with_devkit(Path(sys.argv[1]), {VERSION!r})")
        self.run("데이터셋 검증", [("proc", "nuScenes 데이터셋 로드 확인", [sys.executable, "-u", "-c", code, str(self.dataset)])],
                 kind="validate", route=self.current_route())

    def _exclude(self):
        rec, route = self.selected_rec(), self.current_route()
        if self.job or not rec or not rec["raw"] or self.status["error"]:
            return
        base = self.data_root / "raw" / route
        gone = self.data_root / "raw" / "_excluded" / route
        files = [Path(f) for f in rec["files"]]
        moves = [(f, (base if rec["excluded"] else gone) / f.name) for f in files]
        clash = next((dst for _, dst in moves if dst.exists()), None)
        if clash:
            return self._message("같은 이름의 녹화가 있습니다", f"덮어쓰지 않았습니다. 다음 위치를 확인하세요.\n{clash}")
        # in the tables, or only its import record left after curation took every scene out
        included = rec["name"] in self.status["imported"]
        n_scenes = self.status["logs"].get(rec["name"], 0)
        steps = []
        if rec["excluded"]:
            title = f"{rec['name']} 복원"
            # A previously interrupted exclusion may leave a parsed log behind.
            if included:
                if not self._confirm("녹화 복원", "남아 있는 데이터셋 항목을 제거한 뒤 녹화를 복원합니다.\n복원 후 다시 파싱하면 데이터셋에 포함됩니다.", "녹화 복원"):
                    return
                steps.append(("proc", "남아 있는 데이터셋 항목 제거", self._py("remove_log.py", self.dataset, rec["name"])))
            steps += [("move", f"사용할 녹화로 복원 · {src.name}", str(src), str(dst)) for src, dst in moves]
        else:
            message = f"{rec['name']}\n\nbag 사본과 기록 파일을 raw/_excluded/{route}/로 옮깁니다. 외부 원본은 변경하지 않습니다."
            if included and n_scenes:
                message += f"\n\n통합 데이터셋에 포함된 {n_scenes}개 씬과 관련 파일도 삭제합니다. 복원 후 다시 파싱해야 데이터셋에 돌아옵니다."
            elif included:
                message += "\n\n큐레이션으로 씬이 모두 빠진 녹화입니다. 남은 파싱 기록을 지워, 복원하면 다시 파싱할 수 있게 합니다."
            else:
                message += "\n나중에 ‘제외된 녹화 보기’에서 복원할 수 있습니다."
            if not self._confirm("이 녹화를 제외할까요?", message, "데이터셋에서 제거 후 제외" if included else "녹화 제외", True):
                return
            title = f"{rec['name']} 제외"
            if included:
                steps.append(("proc", "데이터셋에서 녹화 제거", self._py("remove_log.py", self.dataset, rec["name"])))
            steps += [("move", f"제외 폴더로 이동 · {src.name}", str(src), str(dst)) for src, dst in moves]
        self.run(title, steps, kind="restore" if rec["excluded"] else "exclude", route=route)

    def _open_folder(self):
        rec = self.selected_rec()
        if rec and rec["raw"]:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(rec["path"]).parent)))

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if not self.job and len(urls) == 1 and urls[0].isLocalFile() and Path(urls[0].toLocalFile()).exists():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if self.job or len(urls) != 1 or not urls[0].isLocalFile():
            return
        source = Path(urls[0].toLocalFile())
        if not source.exists():
            return
        event.acceptProposedAction()
        if not self.data_root:
            self._change_root()
        if self.data_root:
            self._import(source=source)

    def closeEvent(self, event):
        if self.job:
            event.ignore()
            if not self._close_when_done and self._confirm("작업 중 창 닫기", "작업을 취소한 뒤 창을 닫습니다. 완료된 단계는 유지됩니다.\n취소 후 데이터 폴더의 logs/에서 작업 기록을 확인할 수 있습니다.", "취소하고 닫기", True):
                self._close_when_done = True
                self.job.cancel()
            return
        self._stop_viewer()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help=f"data root (raw/, parsed/, logs/); default {DEFAULT_ROOT}")
    args, rest = parser.parse_known_args()
    if QWebEngineView is not None:          # Qt WebEngine wants this set before the application exists
        QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(["tcar-parser", *rest])      # WM_CLASS matches tcar-parser.desktop
    app.setApplicationName("TCAR Parser")
    for font in sorted((ASSETS / "fonts").glob("*.otf")):
        QtGui.QFontDatabase.addApplicationFont(str(font))
    app.setDesktopFileName("tcar-parser")
    app.setWindowIcon(QtGui.QIcon(str(ASSETS / "tcar-parser.svg")))
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    saved = str(SETTINGS.value("data_root", ""))
    candidate = args.data or (Path(saved).expanduser() if saved else DEFAULT_ROOT)
    root = candidate.expanduser().resolve() if candidate and candidate.expanduser().is_dir() else None
    if root:
        SETTINGS.setValue("data_root", str(root))
    window = MainWindow(root)
    window.show()
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":
    main()
