#!/usr/bin/env python3
"""A recording's messages back from the full set (tcar_nuscenes_full) into a ROS bag, for time its bag lost.

    python scripts/nuscenes_to_bag.py --log A-4_2026-09-28-14-51-38 --template <bag> --out <bag>
        [--dataroot <full set>] [--files <dir with samples/ sweeps/ ext/>] [--until-bag <bag>] [--append <bag>]

What comes back, for the time the log's scenes cover (or up to the first message of --until-bag):
  cameras (7)     sensor_msgs/CompressedImage with the JPEG bytes as recorded (the full set keeps them as they were);
                  header stamp to the microsecond, seq 0
  LiDARs (5)      sensor_msgs/PointCloud2 from .pcd.bin + .time.bin, the same returns (x, y, z, intensity, ring and
                  each point's time, to ~1 us). The roof Ruby 128 goes back into its 128x1800 grid (a laser's row is
                  fixed, the column follows the firing time; cells without a return get NaN, their row's ring and
                  column's time, as the driver writes them); the bottom M1s' 15750x5 grids cannot be placed reliably
                  (their rows are 126 time blocks with jitter as large as a row's step), so their returns come in one row
  other topics    every message (ext/<scene>/<topic>.json keeps all fields), serialized again with the template's
                  definitions; recording time to the microsecond. Byte for byte the original, except NaN payload
                  bits (JSON keeps NaN, not its bits: /radar/ObjectList has such NaNs)
  not kept        /radar/PointCloudDetection (the driver's cloud of /radar/DetectionList, which comes back; its own
                  stamp is the driver's clock) and /middle/rslidar_packets
Checked on G-4 (scene-0953 against the bag): 59 of 63 topics byte for byte, every camera JPEG the same, every LiDAR
sweep the same points (point times within ~1 us); every roof sweep in its original grid, 37 of 196 byte for byte.
A camera's or LiDAR's recording time is its stamp plus that topic's median delay in the template bag.

Connection records (types, definitions, md5, callerid, latching) and per-topic constants (frame_id, image format,
point fields) come from --template, a bag of the same vehicle setup (for a bag that is partly there: that bag).
--append copies a bag's own messages after the rebuilt ones, so the result is one bag: rebuilt time + recovered
time. --files is where the scenes' sensor files are (e.g. a release's tars unpacked) when the dataroot's are gone.
"""
import argparse
import heapq
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.interfaces.typing import Nodetype
from rosbags.rosbag1 import Writer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (CAM_CHANNEL_TO_TOPIC, EXTRA_LIDAR_TOPIC_TO_CHANNEL,  # noqa: E402
                    LIDAR_POINTS_TOPIC)

VERSION = "v1.0-trainval"
LIDAR_CHANNEL_TO_TOPIC = {"LIDAR_TOP": LIDAR_POINTS_TOPIC,
                          **{c: t for t, c in EXTRA_LIDAR_TOPIC_TO_CHANNEL.items()}}
NP = {"bool": np.bool_, "int8": np.int8, "uint8": np.uint8, "byte": np.uint8, "char": np.uint8,
      "int16": np.int16, "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "int64": np.int64,
      "uint64": np.uint64, "float32": np.float32, "float64": np.float64}
PF = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}   # PointField datatype


def iter_json_array(path, block=1 << 26):
    """The objects of a big JSON array file, one at a time (the whole file is never in memory)."""
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
                    break                                     # the object goes on in the next block
                yield obj
                i = j
            if not more:
                return


def build(ts, msgtype, v):
    """msg_to_json's output -> the rosbags message of msgtype."""
    if msgtype in ("builtin_interfaces/msg/Time", "builtin_interfaces/msg/Duration"):
        s, ns = divmod(int(v), 10**9)
        return ts.types[msgtype](sec=s, nanosec=ns)
    _, fields = ts.fielddefs[msgtype]
    return ts.types[msgtype](**{name: value(ts, desc, v[name]) for name, desc in fields})


def value(ts, desc, x):
    kind, d = desc
    if kind == Nodetype.BASE:
        return x
    if kind == Nodetype.NAME:
        return build(ts, d, x)
    sub, _ = d                                                # ARRAY / SEQUENCE: (element, size)
    if sub[0] == Nodetype.BASE and sub[1][0] != "string":
        return np.array(x, dtype=NP[sub[1][0]])
    return [value(ts, sub, e) for e in x]


def tables(dataroot, log, only=None):
    """The log's scenes (or only those named) and their sample_data [(channel, timestamp us, filename)]."""
    v = Path(dataroot) / VERSION
    rd = lambda n: json.loads((v / f"{n}.json").read_text())
    tok = {l["token"] for l in rd("log") if l["logfile"] == log}
    if not tok:
        sys.exit(f"{log}: not in {dataroot}")
    scenes = {s["token"]: s["name"] for s in rd("scene") if s["log_token"] in tok and (not only or s["name"] in only)}
    samples = {s["token"] for s in rd("sample") if s["scene_token"] in scenes}
    sensor = {s["token"]: s["channel"] for s in rd("sensor")}
    chan = {c["token"]: sensor[c["sensor_token"]] for c in rd("calibrated_sensor")}
    sd = [(chan[x["calibrated_sensor_token"]], x["timestamp"], x["filename"])
          for x in iter_json_array(v / "sample_data.json") if x["sample_token"] in samples]
    return sorted(scenes.values()), sorted(sd, key=lambda r: r[1])


def point_dtype(m):
    return np.dtype({"names": [f.name for f in m.fields], "formats": [PF[f.datatype] for f in m.fields],
                     "offsets": [f.offset for f in m.fields], "itemsize": m.point_step})


def lidar_grid(pts, rel, stamp_s, grid, dt, sweep_s=0.1):
    """The returns put back into the organized cloud they came from: a laser's fixed row (its ring), the column
    of their firing time (each row fires at a fixed offset within a column); the cells without a return get NaN,
    their row's ring and their column's time, as the driver writes them. None when the returns do not fit."""
    ring_of_row, row_of_ring, offset, H, W = grid
    step = sweep_s / W
    ring = pts[:, 4].astype(np.int64)
    if ring.min() < 0 or ring.max() >= len(row_of_ring) or (row_of_ring[ring] < 0).any():
        return None
    row = row_of_ring[ring]
    t = stamp_s + rel.astype(np.float64)
    base = t - offset[row]
    order = np.argsort(base, kind="stable")
    b = base[order]
    newcol = np.concatenate([[0], np.cumsum(np.diff(b) > step / 2)])
    k = int(newcol[-1]) + 1
    if k > W:
        return None
    med = np.array([np.median(x) for x in np.split(b, np.searchsorted(newcol, np.arange(1, k)))])
    if k == W:
        colidx = np.arange(W)
    else:                                                     # columns without a single return
        extra = np.maximum(np.round(np.diff(med) / step).astype(np.int64) - 1, 0)
        lead = max(0, int(round((med[0] - stamp_s) / step)))
        while lead + k + extra.sum() > W:
            if extra.sum():
                extra[np.argmax(extra)] -= 1
            else:
                lead -= 1
        colidx = lead + np.concatenate([[0], np.cumsum(1 + extra)])
    col = np.empty(len(t), np.int64)
    col[order] = colidx[newcol]
    if len(np.unique(row * W + col)) != len(t):
        return None
    g = np.zeros((H, W), dt)
    for f in ("x", "y", "z"):
        g[f] = np.float32("nan")
    g["ring"] = ring_of_row[:, None]
    g["timestamp"] = np.interp(np.arange(W), colidx, med)[None, :] + offset[:, None]
    g["x"][row, col], g["y"][row, col], g["z"][row, col], g["intensity"][row, col] = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    g["timestamp"][row, col] = t
    return g


def template_info(path):
    """Connections by topic, a first message per topic, and each payload topic's median recording delay (ns)."""
    conns, first, delay = {}, {}, {}
    with AnyReader([Path(path)]) as r:
        ts = r.typestore
        for c in r.connections:
            conns[c.topic] = c
        want = set(CAM_CHANNEL_TO_TOPIC.values()) | set(LIDAR_CHANNEL_TO_TOPIC.values())
        lat = {}
        cs = [c for c in r.connections if c.topic in want]
        for c, t, raw in r.messages(connections=cs):
            if len(lat.setdefault(c.topic, [])) >= 50:
                if all(len(lat.get(x.topic, [])) >= 50 for x in cs):
                    break
                continue
            m = r.deserialize(raw, c.msgtype)
            first.setdefault(c.topic, m)
            lat[c.topic].append(t - (m.header.stamp.sec * 10**9 + m.header.stamp.nanosec))
        delay = {k: int(statistics.median(v)) for k, v in lat.items() if v}
    grids = {}
    m = first.get(LIDAR_POINTS_TOPIC)
    if m is not None and m.height > 1 and m.width > 1:            # organized: rows = lasers, columns = time
        g = np.frombuffer(bytes(m.data), point_dtype(m)).reshape(m.height, m.width)
        ring_of_row = g["ring"][:, 0].astype(np.int64)
        if len(set(ring_of_row)) == m.height and (g["ring"] == g["ring"][:, :1]).all():
            row_of_ring = np.full(ring_of_row.max() + 1, -1)
            row_of_ring[ring_of_row] = np.arange(m.height)
            offset = np.median(g["timestamp"] - g["timestamp"][:1, :], axis=1)
            grids[LIDAR_POINTS_TOPIC] = (ring_of_row, row_of_ring, offset, m.height, m.width)
    return ts, conns, first, delay, grids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataroot", default=os.path.join(os.environ.get("TCAR_DATA_ROOT", "/data"), "parsed",
                                                       "tcar_nuscenes_full"))
    ap.add_argument("--files", help="where samples/ sweeps/ ext/ are (default: the dataroot)")
    ap.add_argument("--until-bag", help="rebuild only what lies before this bag's first message")
    ap.add_argument("--append", help="then copy this bag's messages in")
    ap.add_argument("--scenes", help="only these scenes (names, comma separated)")
    a = ap.parse_args()
    files = Path(a.files or a.dataroot)
    scenes, sd = tables(a.dataroot, a.log, set(a.scenes.split(",")) if a.scenes else None)
    until = None
    if a.until_bag:
        with AnyReader([Path(a.until_bag)]) as r:
            until = r.start_time
    ts, conns, first, delay, grids = template_info(a.template)
    print(f"{a.log}: {len(scenes)} scenes in the full set ({scenes[0]} .. {scenes[-1]}), {len(sd):,} sensor frames"
          + (f"; rebuilding up to {until / 1e9:.3f}" if until else ""), flush=True)

    gridded = {"grid": 0, "list": 0}

    def sensor_msgs():
        for ch, t_us, fn in sd:
            stamp = t_us * 1000
            if until is not None and stamp >= until:
                continue
            if ch in CAM_CHANNEL_TO_TOPIC:
                topic = CAM_CHANNEL_TO_TOPIC[ch]
                m = first[topic]
                jpeg = (files / fn).read_bytes()
                msg = ts.types["sensor_msgs/msg/CompressedImage"](
                    header=ts.types["std_msgs/msg/Header"](seq=0, stamp=build(ts, "builtin_interfaces/msg/Time",
                                                                               stamp),
                                                           frame_id=m.header.frame_id),
                    format=m.format, data=np.frombuffer(jpeg, np.uint8))
            elif ch in LIDAR_CHANNEL_TO_TOPIC:
                topic = LIDAR_CHANNEL_TO_TOPIC[ch]
                m = first[topic]
                pts = np.fromfile(files / fn, np.float32).reshape(-1, 5)
                tf = files / (fn[:-len(".pcd.bin")] + ".time.bin")
                rel = np.fromfile(tf, np.float32) if tf.exists() else np.zeros(len(pts), np.float32)
                dt = point_dtype(m)
                arr = lidar_grid(pts, rel, stamp / 1e9, grids[topic], dt) if topic in grids and tf.exists() else None
                if arr is None:                                   # the returns in one row
                    arr = np.zeros((1, len(pts)), dt)
                    for k, name in enumerate(("x", "y", "z", "intensity", "ring")):
                        if name in dt.names:
                            arr[name][0] = pts[:, k]
                    if "timestamp" in dt.names:
                        arr["timestamp"][0] = stamp / 1e9 + rel.astype(np.float64)
                    gridded["list"] += 1
                else:
                    gridded["grid"] += 1
                h, wd = arr.shape
                msg = ts.types["sensor_msgs/msg/PointCloud2"](
                    header=ts.types["std_msgs/msg/Header"](seq=0, stamp=build(ts, "builtin_interfaces/msg/Time",
                                                                               stamp),
                                                           frame_id=m.header.frame_id),
                    height=h, width=wd, fields=m.fields, is_bigendian=False, point_step=m.point_step,
                    row_step=m.point_step * wd, data=np.frombuffer(arr.tobytes(), np.uint8), is_dense=h == 1)
            else:
                continue                                      # RADAR_FRONT: comes back as /radar/DetectionList
            if topic not in conns:
                continue
            yield stamp + delay.get(topic, 0), topic, ts.serialize_ros1(msg, conns[topic].msgtype)

    def ext_msgs():
        """The scenes' sidecars in recording-time order, a scene at a time; their margins overlap, so repeats go."""
        seen, horizon = {}, 0
        for name in scenes:
            d = files / "ext" / name
            idx = d / "_index.json"
            if not idx.exists():
                continue
            streams = []
            for slug, info in json.loads(idx.read_text()).items():
                p = d / f"{slug}.json"
                if p.exists() and info["topic"] in conns:
                    rows = json.loads(p.read_text())
                    streams.append([(r["bag_utime"] * 1000, info["topic"], r["msg"]) for r in rows])
            for t, topic, msg in heapq.merge(*streams, key=lambda x: (x[0], x[1])):
                if until is not None and t >= until:
                    continue
                raw = ts.serialize_ros1(build(ts, conns[topic].msgtype, msg), conns[topic].msgtype)
                key = (topic, t, hash(bytes(raw)))
                if key in seen:
                    continue
                seen[key] = t
                if t > horizon + 30 * 10**9:                      # forget what lies well before
                    horizon = t
                    seen = {k: v for k, v in seen.items() if v > t - 30 * 10**9}
                yield t, topic, raw

    out = Path(a.out)
    if out.exists():
        out.unlink()
    n, per = 0, {}
    with Writer(out) as w:
        wc = {}
        for topic, c in sorted(conns.items(), key=lambda kv: kv[1].id):
            wc[topic] = w.add_connection(topic, c.msgtype, msgdef=c.msgdef.data, md5sum=c.digest,
                                         callerid=c.ext.callerid, latching=c.ext.latching)
        for t, topic, raw in heapq.merge(sensor_msgs(), ext_msgs(), key=lambda x: (x[0], x[1])):
            w.write(wc[topic], t, raw)
            n += 1
            per[topic] = per.get(topic, 0) + 1
            if n % 20000 == 0:
                print(f"  {n:,} messages rebuilt", flush=True)
        print(f"rebuilt {n:,} messages on {len(per)} topics; roof LiDAR sweeps in their original grid "
              f"{gridded['grid']:,}, as a list {gridded['list']:,}", flush=True)
        if a.append:
            m = 0
            with AnyReader([Path(a.append)]) as r:
                for c, t, raw in r.messages():
                    w.write(wc[c.topic], t, raw)
                    m += 1
            print(f"then {m:,} messages of {a.append}", flush=True)
    print(f"{out}: {out.stat().st_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
