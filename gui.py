#!/usr/bin/env python3
"""Korean desktop workflow for T-Car ROS 1 bags → nuScenes.

Run: python gui.py [--data <data root>]   (default /data)
Data layout: raw/<course>/<bag>, raw/_excluded/<course>/<bag>, parsed/tcar_nuscenes, logs/.
A course is the letter of a bag's name (A-1_2026-09-28-14-40-58.bag → A); every course
feeds the one dataset. Conversion remains in the existing subprocess CLIs.
"""
from __future__ import annotations

import argparse
import codecs
import json
import re
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    Signal = QtCore.Signal
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    Signal = QtCore.pyqtSignal

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
    empty = {"logs": {}, "samples_by_log": {}, "scenes": 0, "samples": 0, "error": ""}
    if not version.exists():
        return empty
    try:
        logs = json.loads((version / "log.json").read_text())
        scenes = json.loads((version / "scene.json").read_text())
        by_token = {row["token"]: row["logfile"] for row in logs}
        per = {row["logfile"]: 0 for row in logs}
        per_samples = {row["logfile"]: 0 for row in logs}
        samples = 0
        for scene in scenes:
            per[by_token[scene["log_token"]]] += 1
            per_samples[by_token[scene["log_token"]]] += scene["nbr_samples"]
            samples += scene["nbr_samples"]
        return {"logs": per, "samples_by_log": per_samples, "scenes": len(scenes),
                "samples": samples, "error": ""}
    except (OSError, ValueError, KeyError, TypeError):
        return dict(empty, error="데이터셋 목록을 읽을 수 없습니다. 검증 로그를 확인하세요.")


def course_status(status: dict, course: str) -> dict:
    """The dataset-wide status narrowed to one course's logs."""
    logs = {n: k for n, k in status["logs"].items() if detect_route(Path(n)) == course}
    return {"logs": logs, "scenes": sum(logs.values()),
            "samples": sum(status["samples_by_log"].get(n, 0) for n in logs),
            "error": status["error"]}


def dataset_only_info(dataroot: Path, name: str) -> dict:
    """A log in the dataset whose bag is not under raw/: what its import record says."""
    rec = {"name": name, "path": "", "files": [], "label": "", "date": "", "when": "",
           "duration": None, "size": None, "route": detect_route(Path(name)), "raw": False}
    m, when = BAG_RE.match(name), _bag_time(name)
    if m:
        rec.update(label=f"{m['course']}-{m['num']}", date=f"{when:%Y-%m-%d}", when=f"{when:%Y-%m-%d %H:%M}")
    try:
        data = json.loads((dataroot / f"{name}.import.json").read_text())
        cw = data.get("coverage_window", {})
        if cw.get("latest_ns") and cw.get("earliest_ns"):
            rec["duration"] = (cw["latest_ns"] - cw["earliest_ns"]) / 1e9
        rec["path"] = data.get("source_bag", "") or ""
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return rec


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
    """Session-only validation evidence; table changes invalidate a successful check."""
    try:
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                     for p in sorted((dataroot / VERSION).glob("*.json")))
    except OSError:
        return ()


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
        if matches:
            frac = min(100, int(matches[-1])) / 100
            if self._bag:                       # several bags in one run: bar per bag
                k, n = self._bag
                self._status((k - 1 + frac) / n, f"bag {k}/{n}")
            else:
                self._status(frac)

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
        self.job = None
        self.recs, self.active_recs = [], []
        self.status = {"logs": {}, "scenes": 0, "samples": 0, "error": ""}
        self.status_all = parsed_status(Path("/nonexistent"))
        self.validated = {}
        self.next_action = "root"
        self._job_kind, self._job_route = "", None
        self._close_when_done = False
        self._last_log = None
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self._sidebar())
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._main())
        body.addWidget(scroll, 1)
        outer.addLayout(body, 1)
        outer.addWidget(self._jobbar())
        self.refresh()

    @property
    def dataset(self) -> Path:
        return self.data_root / "parsed" / DATASET

    def _sidebar(self):
        widget = QtWidgets.QWidget()
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
        layout.addSpacing(20)
        layout.addWidget(label("데이터 폴더", "section"))
        self.root_label = label("", "path", True)
        self.root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.root_label)
        self.change_btn = button("폴더 선택…", self._change_root, "quiet")
        layout.addWidget(self.change_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addSpacing(14)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label("코스", "section"))
        row.addStretch()
        self.refresh_btn = button("새로고침", self.refresh, "quiet")
        row.addWidget(self.refresh_btn)
        layout.addLayout(row)
        self.route_list = QtWidgets.QListWidget()
        self.route_list.setAccessibleName("코스와 다음 작업")
        self.route_list.setItemDelegate(CellDelegate(self.route_list, pad=12))
        self.route_list.setMouseTracking(True)
        self.route_list.currentItemChanged.connect(self._show_route)
        layout.addWidget(self.route_list, 1)
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
        self.table = QtWidgets.QTableWidget(0, 4)
        setup_table(self.table, ["녹화", "길이", "크기", "상태"])
        self.table.setMinimumHeight(180)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 84), (2, 96), (3, 210)):
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

    def _jobbar(self):
        frame = QtWidgets.QFrame()
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
        names = {detect_route(Path(n)) for n in self.status_all["logs"]} - {None}
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
                    for name in status["logs"] if name not in have)
        recs.sort(key=lambda r: route_key(r["name"]))
        return recs, status

    def _route_action(self, route, recs, status):
        active = [r for r in recs if not r["excluded"] and r["raw"]]
        if status["error"]:
            return "validate", "데이터셋 확인 필요"
        pending = sum(r["name"] not in status["logs"] for r in active)
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
        keep = self.current_route()
        self.route_list.blockSignals(True)
        self.route_list.clear()
        self._route_cache = {}
        try:
            self.status_all = parsed_status(self.dataset) if self.data_root else parsed_status(Path("/nonexistent"))
            for route in self.route_dirs():
                recs, status = self._read_route(route)
                self._route_cache[route] = recs, status
                action, next_text = self._route_action(route, recs, status)
                item = QtWidgets.QListWidgetItem(f"코스 {route}")
                item.setData(Qt.ItemDataRole.UserRole, route)
                item.setData(SUB_ROLE, next_text)
                item.setData(DOT_ROLE, ACTION_COLOR.get(action, C["accent"]))
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
            self.route_list.setCurrentRow(0)
        self.sidebar_hint.setVisible(self.route_list.count() == 0)
        self._show_route()

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
            if self.status["scenes"]:
                meta += [f"씬 {self.status['scenes']:,}개", f"샘플 {self.status['samples']:,}개"]
            if self.status_all["scenes"] > self.status["scenes"]:
                meta.append(f"데이터셋 전체 씬 {self.status_all['scenes']:,}개")
            self.subtitle.setText("  ·  ".join(filter(None, meta)))
            excluded_count = sum(r["excluded"] for r in all_recs)
            self.show_excluded.setText(f"제외된 녹화 보기 ({excluded_count})")
            self.show_excluded.setVisible(bool(excluded_count) or self.show_excluded.isChecked())
            self.table.setRowCount(len(self.recs))
            for row, rec in enumerate(self.recs):
                detail = " · ".join(filter(None, [rec["label"], rec["when"]]))
                if rec["excluded"]:
                    state, color = "제외됨", C["muted"]
                elif self.status["error"]:
                    state, color = "데이터셋 확인 필요", C["warn"]
                elif rec["name"] in self.status["logs"]:
                    state, color = f"데이터셋에 포함 ({self.status['logs'][rec['name']]}씬)", C["ok"]
                else:
                    state, color = "파싱 대기", C["accent"]
                if not rec["raw"]:
                    detail += " · 원본은 데이터 폴더에 없음"
                duration = f"{rec['duration'] / 60:.1f}분" if rec["duration"] is not None else "—"
                size = human_size(rec["size"]) if rec["size"] is not None else "—"
                for column, text in enumerate((rec["name"], duration, size, state)):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setToolTip((rec["path"] or rec["name"]) if column == 0 else text)
                    if column == 0:
                        item.setData(SUB_ROLE, detail)
                    if column == 3:
                        item.setData(DOT_ROLE, color)
                    if rec["excluded"]:
                        item.setForeground(QtGui.QColor(C["faint"]))
                    elif column in (1, 2):
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
        self.recording_stack.setCurrentIndex(0 if self.recs else 1)
        self.recording_heading.setText(f"녹화  {len(self.recs)}" if route else "녹화")
        self._show_next_action()
        self._update_buttons()

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
        if kind in ("parse", "exclude", "restore"):
            self.validated.pop(DATASET, None)
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
        self.action_kicker.setText("작업 진행 중")
        self.action_title.setText(title)
        self.action_desc.setText("아래에서 진행 상황을 확인하세요. 작업이 끝나면 다음 할 일을 안내합니다.")
        self.primary_btn.setText("진행 중…")
        self.primary_btn.setVisible(True)
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
        self.job = None
        self.cancel_btn.setEnabled(False)
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000 if ok else 0)
        self.job_status.setStyleSheet(f"color: {C['ok'] if ok else C['warn']};")
        self.job_status.setText(f"{job.title} · {message}")
        self.refresh()
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
        dialog = ImportDialog(self, self.data_root, self.route_dirs(), set(self.status_all["logs"]), source=source)
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
        included = rec["name"] in self.status["logs"]
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
            if included:
                message += f"\n\n통합 데이터셋에 포함된 {self.status['logs'][rec['name']]}개 씬과 관련 파일도 삭제합니다. 복원 후 다시 파싱해야 데이터셋에 돌아옵니다."
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
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, help=f"data root (raw/, parsed/, logs/); default {DEFAULT_ROOT}")
    args, rest = parser.parse_known_args()
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
