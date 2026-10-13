#!/usr/bin/env python3
"""Timeline of a ROS 1 bag: when every topic has messages, on the header-stamp axis — rqt_bag's timeline without ROS.

    bagview                      # pick a .bag in a dialog
    bagview some.bag             # or: python scripts/bag_timeline.py some.bag
    python scripts/bag_timeline.py some.bag --text     # the per-topic table only, no window

One row per topic: a mark where a message is, red where messages are missing (a gap of more than 2.5 periods of a
regular topic). The axis is the messages' header.stamp (topics without a std_msgs/Header, or whose stamps are all
zero, are drawn at the time the bag recorded them and say so); the toolbar switches every row to the recording time.
Wheel: zoom at the cursor. Drag: pan. Double click or F: everything. Hover: the time, and the nearest message.

Made to be quick on bags of 30-150 GB:
  1. the index only (the records after every chunk, read with a thread pool): every message's recording time and
     place in the file — a few seconds, shown at once;
  2. the header stamps without deserialising anything: 12 bytes of each message. A chunk of many small messages is
     read whole, of a chunk of camera frames only the bytes needed;
  3. both are kept in ~/.cache/tcar_bagview/, so the same bag opens instantly the next time.
"""
from __future__ import annotations

import argparse
import bz2
import datetime as dt
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "tcar_bagview"
IDX = np.dtype([("s", "<u4"), ("ns", "<u4"), ("off", "<u4")])
WORKERS = 16
HEAD = 96                                   # bytes of a message record that hold its header and the stamp
KST = dt.timezone(dt.timedelta(hours=9))


# ------------------------------------------------------------------ bag reading
def _fields(h: bytes) -> dict:
    out, i = {}, 0
    while i < len(h):
        n = struct.unpack_from("<I", h, i)[0]
        k, _, v = h[i + 4:i + 4 + n].partition(b"=")
        out[k] = v
        i += 4 + n
    return out


def _has_header(msgdef: str) -> bool:
    """The message starts with a std_msgs/Header (its stamp then sits at bytes 4..12 of the serialised message)."""
    for line in msgdef.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("="):
            return False
        parts = line.split()
        return len(parts) >= 2 and parts[0] in ("Header", "std_msgs/Header") and "=" not in line
    return False


def read_index(path: str, progress=lambda text, frac: None) -> dict:
    """Connections and, per message, its recording time and where it lies: {conns, chunks, per-chunk arrays}."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        head = os.pread(fd, 4096 + 17, 0)
        if not head.startswith(b"#ROSBAG V2.0\n"):
            raise ValueError("not a ROS 1 bag (V2.0)")
        hl = struct.unpack_from("<I", head, 13)[0]
        fh = _fields(head[17:17 + hl])
        index_pos = struct.unpack("<Q", fh[b"index_pos"])[0]
        n_conn, n_chunk = struct.unpack("<I", fh[b"conn_count"])[0], struct.unpack("<I", fh[b"chunk_count"])[0]
        if not index_pos:
            raise ValueError("the bag has no index (recording was not closed): run `rosbag reindex` first")
        tail = os.pread(fd, size - index_pos, index_pos)
        conns, chunk_pos, i = {}, [], 0
        for _ in range(n_conn + n_chunk):
            hl = struct.unpack_from("<I", tail, i)[0]
            h = _fields(tail[i + 4:i + 4 + hl])
            dl = struct.unpack_from("<I", tail, i + 4 + hl)[0]
            data = tail[i + 8 + hl:i + 8 + hl + dl]
            i += 8 + hl + dl
            if h[b"op"] == b"\x07":
                c = _fields(data)
                conns[struct.unpack("<I", h[b"conn"])[0]] = {
                    "topic": h[b"topic"].decode(), "type": c.get(b"type", b"").decode(),
                    "header": _has_header(c.get(b"message_definition", b"").decode(errors="replace"))}
            elif h[b"op"] == b"\x06":
                chunk_pos.append(struct.unpack("<Q", h[b"chunk_pos"])[0])
        chunk_pos.sort()
        ends = chunk_pos[1:] + [index_pos]
        layout = {}                                         # header length -> (offset of conn, offset of count)

        def one(k):
            pos = chunk_pos[k]
            first = os.pread(fd, 256, pos)
            hl = struct.unpack_from("<I", first, 0)[0]
            ch = _fields(first[4:4 + hl])
            dlen = struct.unpack_from("<I", first, 4 + hl)[0]
            start = pos + 8 + hl
            blk = os.pread(fd, ends[k] - (start + dlen), start + dlen)
            cs, arrs, j = [], [], 0
            while j + 4 <= len(blk):
                hl2 = struct.unpack_from("<I", blk, j)[0]
                lay = layout.get(hl2)
                if lay is None:                             # find the fields once per header layout
                    h, o, lay_c, lay_n = blk[j + 4:j + 4 + hl2], 0, None, None
                    while o < hl2:
                        n = struct.unpack_from("<I", h, o)[0]
                        key = h[o + 4:o + 4 + n].split(b"=", 1)[0]
                        if key == b"conn":
                            lay_c = o + 4 + 5
                        elif key == b"count":
                            lay_n = o + 4 + 6
                        elif key == b"op" and h[o + 4 + 3:o + 4 + 4] != b"\x04":
                            lay_c = lay_n = -1              # not an index record
                        o += 4 + n
                    lay = layout[hl2] = (lay_c, lay_n)
                dl = struct.unpack_from("<I", blk, j + 4 + hl2)[0]
                if lay[0] is not None and lay[0] >= 0:
                    conn = struct.unpack_from("<I", blk, j + 4 + lay[0])[0]
                    arrs.append(np.frombuffer(blk, IDX, dl // 12, j + 8 + hl2))
                    cs.append(np.full(dl // 12, conn, np.uint16))
                j += 8 + hl2 + dl
            idx = np.concatenate(arrs) if arrs else np.zeros(0, IDX)
            return start, dlen, ch.get(b"compression", b"none"), (np.concatenate(cs) if cs else np.zeros(0, np.uint16)), idx

        chunks, done = [None] * len(chunk_pos), 0
        with ThreadPoolExecutor(WORKERS) as pool:
            for k, res in enumerate(pool.map(one, range(len(chunk_pos)))):
                chunks[k] = res
                done += 1
                if done % 500 == 0:
                    progress("인덱스 읽는 중", done / len(chunk_pos))
        return {"conns": conns, "chunks": chunks, "size": size}
    finally:
        os.close(fd)


def read_stamps(path: str, index: dict, progress=lambda text, frac: None) -> list:
    """header.stamp (ns) of every message of a header-carrying connection, chunk by chunk; -1 where there is none."""
    has = np.zeros(65536, bool)
    for c, v in index["conns"].items():
        has[c] = v["header"]
    fd = os.open(path, os.O_RDONLY)

    def stamps_in(buf: np.ndarray, offs: np.ndarray) -> np.ndarray:
        """buf: bytes of a chunk (uint8); offs: where each message record starts in it."""
        hl = (buf[offs].astype(np.int64) | buf[offs + 1].astype(np.int64) << 8 | buf[offs + 2].astype(np.int64) << 16
              | buf[offs + 3].astype(np.int64) << 24)
        p = offs + hl + 12                                  # record header, data length, seq
        ok = p + 8 <= len(buf)
        p = np.where(ok, p, 0)
        b = buf[p[:, None] + np.arange(8)].astype(np.int64)
        sec = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16 | b[:, 3] << 24
        nsec = b[:, 4] | b[:, 5] << 8 | b[:, 6] << 16 | b[:, 7] << 24
        return np.where(ok, sec * 1_000_000_000 + nsec, -1)

    def one(k):
        start, dlen, comp, cs, idx = index["chunks"][k]
        out = np.full(len(cs), -1, np.int64)
        sel = np.flatnonzero(has[cs])
        if not len(sel):
            return out
        offs = idx["off"][sel].astype(np.int64)
        if comp != b"none":
            raw = os.pread(fd, dlen, start)
            if comp == b"bz2":
                raw = bz2.decompress(raw)
            elif comp == b"lz4":
                import lz4.frame
                raw = lz4.frame.decompress(raw)
            else:
                return out
            out[sel] = stamps_in(np.frombuffer(raw, np.uint8), offs)
        elif len(sel) * 16384 >= dlen:                      # many small messages: the whole chunk in one read
            out[sel] = stamps_in(np.frombuffer(os.pread(fd, dlen, start), np.uint8), offs)
        else:                                               # a few big ones (images, point clouds): their heads only
            for i, o in zip(sel, offs):
                b = os.pread(fd, HEAD, start + int(o))
                hl = struct.unpack_from("<I", b, 0)[0]
                if hl + 20 <= len(b):
                    s, ns = struct.unpack_from("<II", b, hl + 12)
                    out[i] = s * 1_000_000_000 + ns
        return out

    try:
        n = len(index["chunks"])
        res, done = [None] * n, 0
        with ThreadPoolExecutor(WORKERS) as pool:
            for k, r in enumerate(pool.map(one, range(n))):
                res[k] = r
                done += 1
                if done % 200 == 0:
                    progress("header stamp 읽는 중", done / n)
        return res
    finally:
        os.close(fd)


def by_topic(index: dict, stamps: list | None) -> dict:
    """{topic: {type, header, bag (ns, sorted), hdr (ns, in the same order) or None}}"""
    cs = np.concatenate([c[3] for c in index["chunks"]]) if index["chunks"] else np.zeros(0, np.uint16)
    idx = np.concatenate([c[4] for c in index["chunks"]]) if index["chunks"] else np.zeros(0, IDX)
    bag = idx["s"].astype(np.int64) * 1_000_000_000 + idx["ns"]
    hdr = np.concatenate(stamps) if stamps is not None and len(stamps) else None
    topics = {}
    for conn, v in sorted(index["conns"].items(), key=lambda kv: kv[1]["topic"]):
        m = cs == conn
        t = topics.setdefault(v["topic"], {"type": v["type"], "header": v["header"], "bag": [], "hdr": []})
        t["bag"].append(bag[m])
        if hdr is not None and v["header"]:
            t["hdr"].append(hdr[m])
    for t in topics.values():
        b = np.concatenate(t["bag"])
        order = np.argsort(b, kind="stable")
        t["bag"] = b[order]
        t["hdr"] = np.concatenate(t["hdr"])[order] if t["hdr"] and sum(map(len, t["hdr"])) == len(b) else None
    return topics


def cache_path(path: str) -> Path:
    st = os.stat(path)
    return CACHE / f"{Path(path).name}.{st.st_size}.{st.st_mtime_ns}.npz"


def save_cache(path: str, topics: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, []
    for k, (name, t) in enumerate(topics.items()):
        meta.append({"topic": name, "type": t["type"], "header": t["header"], "hdr": t["hdr"] is not None})
        arrays[f"b{k}"] = t["bag"]
        if t["hdr"] is not None:
            arrays[f"h{k}"] = t["hdr"]
    tmp = str(cache_path(path)) + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, cache_path(path))


def load_cache(path: str) -> dict | None:
    f = cache_path(path)
    if not f.exists():
        return None
    try:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        return {m["topic"]: {"type": m["type"], "header": m["header"], "bag": z[f"b{k}"],
                             "hdr": z[f"h{k}"] if m["hdr"] else None} for k, m in enumerate(meta)}
    except Exception:                                       # noqa: BLE001 - a broken cache is read again from the bag
        return None


# ------------------------------------------------------------------ per-topic rows
GROUPS = ("카메라 영상", "LiDAR 포인트", "레이더", "GNSS · INS", "CAN", "카메라 정보 · LiDAR 상태", "그 밖")


def sensor_group(topic: str, msgtype: str) -> int:
    """Which sensor a topic belongs to (index into GROUPS): the cameras' images and the LiDARs' points first."""
    t, ty = topic.lower(), msgtype.lower()
    if "camera_info" in t or ty.endswith("camerainfo") or ("lidar" in t and "status" in t):
        return 5
    if ty.endswith(("compressedimage", "/image")) or ("camera" in t and "image" in ty):
        return 0
    if ty.endswith("pointcloud2") and "radar" not in t or "lidar" in t or "velodyne" in t or "ouster" in t:
        return 1
    if "radar" in t:
        return 2
    if any(k in t for k in ("novatel", "gnss", "gps", "/ins", "imu")):
        return 3
    if "can" in t.replace("scan", ""):
        return 4
    return 6


def natural(topic: str) -> tuple:
    """camera_2 before camera_10; the top LiDAR (middle / top) before the others."""
    import re
    lead = 0 if any(k in topic for k in ("middle", "top")) else 1
    return (lead, [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", topic)])


def make_row(name: str, t: dict, use_header: bool) -> dict:
    """What a row draws and says: its times on the chosen axis, rate, gaps, and how the stamps relate to the bag."""
    bag, hdr = t["bag"], t["hdr"]
    note, lat = "", None
    times, axis = bag, "bag"
    if hdr is not None and len(hdr):
        good = hdr > 0
        if not good.any():
            note = "stamp=0"
        else:
            d = (bag[good] - hdr[good]) / 1e6               # ms the recorder got it after its stamp
            lat = (float(np.median(d)), float(np.percentile(d, 99)))
            if use_header:
                times, axis = np.sort(hdr[good]), "hdr"
                if (~good).any():
                    note = f"stamp=0 {int((~good).sum())}개"
                if abs(lat[0]) > 5000:
                    note = (note + " · " if note else "") + f"stamp가 기록보다 {-lat[0] / 1e3:+.1f} s"
    elif not t["header"]:
        note = "header 없음"
    n = len(times)
    row = {"topic": name, "type": t["type"], "times": times, "axis": axis, "n": n, "note": note, "lat": lat,
           "group": sensor_group(name, t["type"]),
           "hz": 0.0, "period": 0.0, "max_gap": 0.0, "gap_a": np.zeros(0, np.int64), "gap_b": np.zeros(0, np.int64),
           "regular": False}
    if n >= 2:
        d = np.diff(times)
        med = float(np.median(d))
        row["period"], row["max_gap"] = med, float(d.max())
        span = float(times[-1] - times[0])
        row["hz"] = (n - 1) * 1e9 / span if span > 0 else 0.0     # the average rate (bursty topics: not 1 / median)
        # regular: most intervals near the median -> a longer one is missing data
        row["regular"] = n >= 20 and med > 0 and float(np.percentile(d, 90)) < 1.6 * med
        if row["regular"]:
            g = np.flatnonzero(d > 2.5 * med)
            row["gap_a"], row["gap_b"] = times[g], times[g + 1]
    return row


def table(rows: list) -> str:
    lines = [f"{'topic':46} {'n':>9} {'Hz':>8} {'max gap':>9} {'gaps':>5}  {'axis':4} {'stamp→기록 ms (중앙/p99)':>22}  note"]
    for r in rows:
        lat = f"{r['lat'][0]:9.1f} /{r['lat'][1]:9.1f}" if r["lat"] else ""
        lines.append(f"{r['topic'][:46]:46} {r['n']:>9,} {r['hz']:>8.1f} {r['max_gap'] / 1e9:>8.3f}s "
                     f"{len(r['gap_a']):>5}  {r['axis']:4} {lat:>22}  {r['note']}")
    return "\n".join(lines)


def load(path: str, progress=lambda text, frac: None, on_index=lambda topics: None) -> dict:
    topics = load_cache(path)
    if topics is not None:
        return topics
    index = read_index(path, progress)
    on_index(by_topic(index, None))                         # the recording-time view, at once
    topics = by_topic(index, read_stamps(path, index, progress))
    save_cache(path, topics)
    return topics


# ------------------------------------------------------------------ window
def gui(path: str | None) -> None:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        Signal = QtCore.Signal
    except ImportError:
        from PyQt5 import QtCore, QtGui, QtWidgets
        Signal = QtCore.pyqtSignal
    Qt = QtCore.Qt

    class Loader(QtCore.QThread):
        progress = Signal(str, float)
        indexed = Signal(object)
        loaded = Signal(object, float)
        failed = Signal(str)

        def __init__(self, bag):
            super().__init__()
            self.bag = bag

        def run(self):
            t = time.time()
            try:
                topics = load(self.bag, lambda s, f: self.progress.emit(s, f), lambda tp: self.indexed.emit(tp))
                self.loaded.emit(topics, time.time() - t)
            except Exception as e:                          # noqa: BLE001 - shown in the window
                self.failed.emit(f"{type(e).__name__}: {e}")

    ROW, TOP, LABEL = 22, 34, 700
    COLS, COLW = (304, 388, 470, 524), (80, 78, 50, 168)        # count, rate, gaps, axis / note
    C = {"bg": "#14161a", "row": "#191c21", "text": "#d8dce3", "sub": "#8a93a3", "axis": "#5a6272",
         "msg": (88, 166, 255), "bagmsg": (150, 158, 172), "gap": (235, 87, 87), "cursor": "#e6b450", "warn": "#e6b450"}

    class Timeline(QtWidgets.QWidget):
        hover = Signal(str)

        def __init__(self):
            super().__init__()
            self.rows, self.shown, self.t0, self.t1, self.lo, self.hi = [], [], 0, 1, 0, 1
            self.scroll, self.cursor, self.drag = 0, None, None
            self.setMouseTracking(True)
            self.setMinimumSize(900, 300)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if hasattr(Qt, "FocusPolicy") else Qt.StrongFocus)

        def set_rows(self, rows, keep_view=False, grouped=False):
            self.rows, self.grouped = rows, grouped
            ts = [r["times"] for r in rows if r["n"]]
            if ts:
                # the common span: rows whose stamps are far off the rest (another clock) do not stretch it
                a = np.array([t[0] for t in ts], np.float64)
                b = np.array([t[-1] for t in ts], np.float64)
                ma, mb = np.median(a), np.median(b)
                near = (np.abs(a - ma) < 3600e9) & (np.abs(b - mb) < 3600e9)
                self.lo, self.hi = int(a[near].min()), int(b[near].max())
            if not keep_view:
                self.fit()
            self.filter(getattr(self, "_filter", ""))

        def filter(self, text):
            self._filter = text
            words = text.lower().split()
            self.shown = [r for r in self.rows if all(w in (r["topic"] + " " + r["type"]).lower() for w in words)]
            if getattr(self, "grouped", False):             # a title line before every sensor's topics
                out, last = [], None
                for r in self.shown:
                    if r["group"] != last:
                        last = r["group"]
                        k = sum(x["group"] == last for x in self.shown)
                        out.append({"title": f"{GROUPS[last]}  ·  토픽 {k}개", "topic": GROUPS[last], "type": "", "n": 0,
                                    "times": np.zeros(0, np.int64), "gap_a": np.zeros(0, np.int64), "lat": None,
                                    "max_gap": 0.0, "group": last})
                    out.append(r)
                self.shown = out
            self.scroll = min(self.scroll, max(0, len(self.shown) * ROW - (self.height() - TOP)))
            self.update()

        def fit(self):
            pad = max((self.hi - self.lo) * 0.01, 1e6)
            self.t0, self.t1 = self.lo - pad, self.hi + pad
            self.update()

        def x_of(self, t):
            return LABEL + (t - self.t0) / (self.t1 - self.t0) * (self.width() - LABEL)

        def t_of(self, x):
            return self.t0 + (x - LABEL) / max(self.width() - LABEL, 1) * (self.t1 - self.t0)

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            W, H = self.width(), self.height()
            p.fillRect(0, 0, W, H, QtGui.QColor(C["bg"]))
            w = W - LABEL
            if w < 10 or not self.shown:
                p.setPen(QtGui.QColor(C["sub"]))
                p.drawText(20, TOP + 30, "bag을 여세요 (열기… 또는 창에 끌어다 놓기)" if not self.rows else "맞는 토픽이 없습니다")
                return
            first = self.scroll // ROW
            n_vis = min(len(self.shown) - first, (H - TOP) // ROW + 2)
            img = np.zeros((max(n_vis, 1) * ROW, w, 3), np.uint8)
            img[:] = (0x14, 0x16, 0x1a)
            edges = self.t0 + (self.t1 - self.t0) * np.arange(w + 1) / w
            px = (self.t1 - self.t0) / w                    # ns per pixel
            span = (self.t1 - self.t0) / 1e9
            step = 10 ** np.floor(np.log10(max(span / 8, 1e-6)))
            step = next(s for s in (step, 2 * step, 5 * step, 10 * step) if span / s <= 10)
            ticks, tk = [], np.ceil((self.t0 - self.lo) / 1e9 / step) * step
            while self.lo + tk * 1e9 < self.t1:
                ticks.append((int(self.x_of(self.lo + tk * 1e9)), tk))
                tk += step
            # The grid and the cursor are painted first and the marks over them: a line drawn over the rows hid
            # the message under it, which read as a missing stamp.
            lines = [(x - LABEL, (0x3a, 0x40, 0x4c)) for x, _ in ticks]
            if self.cursor is not None and self.cursor >= LABEL:
                lines.append((self.cursor - LABEL, (0xe6, 0xb4, 0x50)))
            for k in range(n_vis):
                r = self.shown[first + k]
                y = k * ROW
                if "title" in r:
                    img[y:y + ROW] = (0x23, 0x28, 0x31)
                elif (first + k) % 2:
                    img[y:y + ROW] = (0x19, 0x1c, 0x21)
            for x, colour in lines:
                if 0 <= x < w:
                    img[:, x] = colour
            for k in range(n_vis):
                r = self.shown[first + k]
                y = k * ROW
                if "title" in r or not r["n"]:
                    continue
                t = r["times"]
                if len(r["gap_a"]):                          # missing data: red from the last message to the next
                    ga = np.clip(((r["gap_a"] - self.t0) / px).astype(np.int64), -1, w)
                    gb = np.clip(((r["gap_b"] - self.t0) / px).astype(np.int64), -1, w)
                    for a, b in zip(ga[(gb >= 0) & (ga < w)], gb[(gb >= 0) & (ga < w)]):
                        img[y + 8:y + ROW - 8, max(a, 0):max(b, a + 1)] = C["gap"]
                cnt = np.diff(np.searchsorted(t, edges))
                on = cnt > 0
                col = np.array(C["msg"] if r["axis"] == "hdr" else C["bagmsg"], np.float32)
                if on.any() and np.median(cnt[on]) >= 3:    # many per pixel: dimmer where the row is thinner than usual
                    full = np.clip(cnt[on] / max(float(np.percentile(cnt[on], 90)), 1.0), 0.3, 1.0)[:, None]
                    img[y + 4:y + ROW - 4, on] = (col * full).astype(np.uint8)
                else:
                    img[y + 4:y + ROW - 4, on] = col.astype(np.uint8)
            qi = QtGui.QImage(img.tobytes(), w, img.shape[0], 3 * w, QtGui.QImage.Format.Format_RGB888
                              if hasattr(QtGui.QImage, "Format") else QtGui.QImage.Format_RGB888)
            p.drawImage(LABEL, TOP - self.scroll % ROW, qi)
            # labels
            f = p.font(); f.setPointSizeF(9.0); p.setFont(f)
            fm = p.fontMetrics()
            el = Qt.TextElideMode.ElideMiddle if hasattr(Qt, "TextElideMode") else Qt.ElideMiddle
            ar = (Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) if hasattr(Qt, "AlignmentFlag") else (Qt.AlignRight | Qt.AlignVCenter)
            p.fillRect(0, TOP, LABEL, H - TOP, QtGui.QColor(C["bg"]))
            for k in range(n_vis):
                r = self.shown[first + k]
                y = TOP + k * ROW - self.scroll % ROW
                if "title" in r:
                    p.fillRect(0, y, LABEL, ROW, QtGui.QColor("#232831"))
                    p.setPen(QtGui.QColor("#e6b450"))
                    p.drawText(10, y + 15, r["title"])
                    continue
                if (first + k) % 2:
                    p.fillRect(0, y, LABEL, ROW, QtGui.QColor(C["row"]))
                p.setPen(QtGui.QColor(C["text"]))
                p.drawText(10 + (12 if self.grouped else 0), y + 15, fm.elidedText(r["topic"], el, 290 - (12 if self.grouped else 0)))
                p.setPen(QtGui.QColor(C["sub"]))
                p.drawText(QtCore.QRect(COLS[0], y, COLW[0], ROW), ar, f"{r['n']:,}")
                p.drawText(QtCore.QRect(COLS[1], y, COLW[1], ROW), ar, f"{r['hz']:.1f} Hz" if r["hz"] else "")
                gaps = len(r["gap_a"])
                p.setPen(QtGui.QColor("#eb5757" if gaps else C["sub"]))
                p.drawText(QtCore.QRect(COLS[2], y, COLW[2], ROW), ar, f"{gaps}" if gaps else ("0" if r["regular"] else "비주기"))
                p.setPen(QtGui.QColor(C["warn"] if r["note"] else C["sub"]))
                p.drawText(QtCore.QRect(COLS[3], y, COLW[3], ROW), ar,
                           fm.elidedText(r["note"] or ("기록 시각" if r["axis"] == "bag" else "header"), el, COLW[3]))
            # axis
            p.fillRect(0, 0, W, TOP, QtGui.QColor(C["bg"]))
            p.setPen(QtGui.QColor(C["sub"]))
            p.drawText(10, 22, "토픽")
            for cx, cw, name in zip(COLS, COLW, ("메시지", "평균 빈도", "갭", "시간축 · 비고")):
                p.drawText(QtCore.QRect(cx, 4, cw, 24), ar, name)
            below = TOP + n_vis * ROW - self.scroll % ROW       # under the last row the lines go on (nothing to hide)
            for x, tk in ticks:
                p.setPen(QtGui.QColor(C["axis"]))
                p.drawLine(x, TOP - 6, x, TOP - 1)
                if below < H:
                    p.drawLine(x, below, x, H)
                p.setPen(QtGui.QColor(C["sub"]))
                p.drawText(x + 4, 22, f"{tk:.{max(0, -int(np.floor(np.log10(step))))}f} s")
            if self.cursor is not None and self.cursor >= LABEL:
                p.setPen(QtGui.QColor(C["cursor"]))
                p.drawLine(self.cursor, TOP - 10, self.cursor, TOP - 1)
                if below < H:
                    p.drawLine(self.cursor, below, self.cursor, H)

        def _pos(self, e):
            q = e.position() if hasattr(e, "position") else e.pos()
            return int(q.x()), int(q.y())

        def mouseMoveEvent(self, e):
            x, y = self._pos(e)
            if self.drag is not None:
                dx = (x - self.drag[0]) / max(self.width() - LABEL, 1) * (self.drag[2] - self.drag[1])
                self.t0, self.t1 = self.drag[1] - dx, self.drag[2] - dx
            self.cursor = x
            k = (y - TOP + self.scroll) // ROW
            text = ""
            if x >= LABEL:
                t = self.t_of(x)
                text = f"{dt.datetime.fromtimestamp(t / 1e9, KST):%H:%M:%S.%f}"[:-3] + f"  (+{(t - self.lo) / 1e9:.3f} s)"
            if y >= TOP and 0 <= k < len(self.shown) and "title" not in self.shown[k]:
                r = self.shown[k]
                text += f"   {r['topic']}  [{r['type']}]"
                if x >= LABEL and r["n"]:
                    i = int(np.clip(np.searchsorted(r["times"], t), 1, r["n"] - 1)) if r["n"] > 1 else 0
                    i = i - 1 if r["n"] > 1 and t - r["times"][i - 1] < r["times"][i] - t else i
                    m = r["times"][i]
                    text += f"   가까운 메시지 #{i} +{(m - self.lo) / 1e9:.4f} s"
                    if 0 < i:
                        text += f" · 앞과 {(m - r['times'][i - 1]) / 1e6:.1f} ms"
                    if i < r["n"] - 1:
                        text += f" · 뒤와 {(r['times'][i + 1] - m) / 1e6:.1f} ms"
                if r["lat"]:
                    text += f"   stamp→기록 {r['lat'][0]:.1f} ms (p99 {r['lat'][1]:.1f})"
                if r["max_gap"]:
                    text += f"   최대 간격 {r['max_gap'] / 1e9:.3f} s"
            self.hover.emit(text)
            self.update()

        def mousePressEvent(self, e):
            x, _ = self._pos(e)
            if x >= LABEL:
                self.drag = (x, self.t0, self.t1)

        def mouseReleaseEvent(self, _):
            self.drag = None

        def mouseDoubleClickEvent(self, _):
            self.fit()

        def wheelEvent(self, e):
            x, _ = self._pos(e)
            d = e.angleDelta().y()
            mods = e.modifiers()
            shift = Qt.KeyboardModifier.ShiftModifier if hasattr(Qt, "KeyboardModifier") else Qt.ShiftModifier
            if x < LABEL or mods & shift:                    # over the names: scroll the rows
                self.scroll = int(np.clip(self.scroll - d / 120 * ROW * 3, 0, max(0, len(self.shown) * ROW - (self.height() - TOP))))
            else:
                t = self.t_of(x)
                k = 0.8 ** (d / 120)
                span = float(np.clip((self.t1 - self.t0) * k, 1e5, (self.hi - self.lo) * 20 + 1e9))
                a = (t - self.t0) / (self.t1 - self.t0)
                self.t0, self.t1 = t - a * span, t + (1 - a) * span
            self.update()

        def keyPressEvent(self, e):
            if e.key() == (Qt.Key.Key_F if hasattr(Qt, "Key") else Qt.Key_F):
                self.fit()

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("bag 타임라인")
            self.resize(1700, 950)
            self.setAcceptDrops(True)
            self.topics, self.loader, self.path = None, None, None
            bar = self.addToolBar("bar")
            bar.setMovable(False)
            act = bar.addAction("열기…")
            act.triggered.connect(self.pick)
            bar.addSeparator()
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["header stamp 축", "bag 기록 시각 축"])
            self.mode.currentIndexChanged.connect(lambda _: self.show_rows(keep=False))
            bar.addWidget(self.mode)
            self.find = QtWidgets.QLineEdit()
            self.find.setPlaceholderText("토픽 · 타입 거르기 (예: cam, lidar, can)")
            self.find.setFixedWidth(320)
            self.find.textChanged.connect(lambda s: self.tl.filter(s))
            bar.addWidget(self.find)
            self.order = QtWidgets.QComboBox()
            self.order.addItems(["센서별", "이름순", "갭 많은 순", "빈도 높은 순"])
            self.order.currentIndexChanged.connect(lambda _: self.show_rows(keep=True))
            bar.addWidget(self.order)
            fit = bar.addAction("전체 보기 (F)")
            fit.triggered.connect(lambda: self.tl.fit())
            self.tl = Timeline()
            self.setCentralWidget(self.tl)
            self.note = QtWidgets.QLabel("휠: 확대 · 드래그: 이동 · 더블클릭: 전체 · 이름 위에서 휠: 토픽 스크롤")
            self.bar_p = QtWidgets.QProgressBar()
            self.bar_p.setFixedWidth(220)
            self.bar_p.setVisible(False)
            self.statusBar().addWidget(self.note, 1)
            self.statusBar().addPermanentWidget(self.bar_p)
            self.tl.hover.connect(lambda s: self.note.setText(s or self.summary))
            self.summary = self.note.text()

        def dragEnterEvent(self, e):
            if e.mimeData().hasUrls():
                e.acceptProposedAction()

        def dropEvent(self, e):
            f = e.mimeData().urls()[0].toLocalFile()
            if f:
                self.open(f)

        def pick(self):
            s = QtCore.QSettings("IRCV", "bagview")
            start = str(s.value("dir", "/media")) if os.path.isdir(str(s.value("dir", "/media"))) else str(Path.home())
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "bag 파일 선택", start, "ROS bag (*.bag);;모든 파일 (*)")
            if f:
                s.setValue("dir", os.path.dirname(f))
                self.open(f)

        def open(self, f):
            if self.loader is not None and self.loader.isRunning():
                return
            self.path, self.topics = f, None
            self.setWindowTitle(f"bag 타임라인 — {os.path.basename(f)}  ({os.path.getsize(f) / 1e9:.1f} GB)")
            self.bar_p.setVisible(True)
            self.bar_p.setRange(0, 0)
            self.summary = "여는 중…"
            self.note.setText(self.summary)
            self.loader = Loader(f)
            self.loader.progress.connect(self.on_progress)
            self.loader.indexed.connect(lambda tp: self.set_topics(tp, "기록 시각만 먼저 (header stamp 읽는 중…)"))
            self.loader.loaded.connect(lambda tp, sec: self.set_topics(
                tp, "저장해 둔 것에서 읽음" if sec < 0.5 else f"{sec:.1f}초에 읽음", done=True))
            self.loader.failed.connect(self.on_failed)
            self.loader.start()

        def on_progress(self, text, frac):
            self.bar_p.setRange(0, 1000)
            self.bar_p.setValue(int(frac * 1000))
            self.summary = f"{text} {frac * 100:.0f}%"
            self.note.setText(self.summary)

        def on_failed(self, msg):
            self.bar_p.setVisible(False)
            self.summary = "읽지 못했습니다: " + msg
            self.note.setText(self.summary)

        def set_topics(self, topics, how, done=False):
            first = self.topics is None
            self.topics = topics
            if done:
                self.bar_p.setVisible(False)
            self.show_rows(keep=not first, how=how)
            shot = os.environ.get("BAGVIEW_SHOT")           # tests: save the window as an image and leave
            if done and shot:
                view = os.environ.get("BAGVIEW_SHOT_VIEW")  # "a,b": seconds from the start
                if view:
                    a, b = map(float, view.split(","))
                    self.tl.t0, self.tl.t1 = self.tl.lo + a * 1e9, self.tl.lo + b * 1e9
                QtCore.QTimer.singleShot(400, lambda: (self.grab().save(shot), QtWidgets.QApplication.quit()))

        def show_rows(self, keep=True, how=None):
            if not self.topics:
                return
            use_header = self.mode.currentIndex() == 0
            rows = [make_row(n, t, use_header) for n, t in self.topics.items()]
            mode = self.order.currentIndex()
            key = [lambda r: (r["group"], natural(r["topic"])), lambda r: natural(r["topic"]),
                   lambda r: (-len(r["gap_a"]), r["topic"]), lambda r: (-r["hz"], r["topic"])][mode]
            rows.sort(key=key)
            self.tl.set_rows(rows, keep_view=keep, grouped=mode == 0)
            n = sum(r["n"] for r in rows)
            gaps = sum(len(r["gap_a"]) for r in rows)
            span = (self.tl.hi - self.tl.lo) / 1e9
            start = dt.datetime.fromtimestamp(self.tl.lo / 1e9, KST)
            if how:
                self._how = how
            self.summary = (f"토픽 {len(rows)}개 · 메시지 {n:,}개 · {span:.1f} s · 시작 {start:%Y-%m-%d %H:%M:%S} KST · "
                            f"갭 {gaps}곳 · {getattr(self, '_how', '')}   |   휠: 확대 · 드래그: 이동 · 더블클릭: 전체 · 이름 위 휠: 스크롤")
            self.note.setText(self.summary)

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, col in (("Window", "#1b1e24"), ("WindowText", "#d8dce3"), ("Base", "#14161a"), ("Text", "#d8dce3"),
                      ("Button", "#252a32"), ("ButtonText", "#d8dce3"), ("Highlight", "#3a6ea5"), ("PlaceholderText", "#7a8394")):
        cr = getattr(QtGui.QPalette.ColorRole, role, None) if hasattr(QtGui.QPalette, "ColorRole") else getattr(QtGui.QPalette, role, None)
        if cr is not None:
            pal.setColor(cr, QtGui.QColor(col))
    app.setPalette(pal)
    win = Window()
    win.show()
    if path:
        win.open(path)
    else:
        QtCore.QTimer.singleShot(0, win.pick)
    sys.exit(app.exec() if hasattr(app, "exec") else app.exec_())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", help="a ROS 1 .bag (a dialog asks when left out)")
    ap.add_argument("--text", action="store_true", help="print the per-topic table and exit (no window)")
    ap.add_argument("--bag-time", action="store_true", help="--text: the recording time axis instead of header stamps")
    ap.add_argument("--no-cache", action="store_true", help="read the bag again even when it is cached")
    a = ap.parse_args()
    if a.no_cache and a.bag and cache_path(a.bag).exists():
        cache_path(a.bag).unlink()
    if a.text:
        if not a.bag:
            ap.error("--text needs a bag")
        t = time.time()
        topics = load(a.bag, lambda s, f: print(f"\r{s} {f * 100:3.0f}%", end="", file=sys.stderr, flush=True))
        print(f"\r{' ' * 40}\r", end="", file=sys.stderr)
        rows = [make_row(n, v, not a.bag_time) for n, v in topics.items()]
        print(table(rows))
        print(f"\n{len(rows)} topics, {sum(r['n'] for r in rows):,} messages, read in {time.time() - t:.1f} s")
        return
    gui(a.bag)


if __name__ == "__main__":
    main()
