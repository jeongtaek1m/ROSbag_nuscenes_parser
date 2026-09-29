#!/usr/bin/env python3
"""How many scenes would each bag give, and where is the rest lost? No conversion.

    python yield_check.py <.bag or a folder of them> ... [--json out.json]

Reads only the header stamp of every LIDAR_TOP, standard-camera and INSPVA message
(straight from the bag through its chunk index: a few hundred bytes per message, no
image or point cloud is read) and runs the converter's frame rule on them: coverage
window, a complete camera set within --sync-ms of every LiDAR frame, unbroken runs cut
into --scene-dur scenes.

Every LiDAR frame ends up in one category, in seconds:
  scene      in a scene
  short      usable, but in a run shorter than a scene (or a run's remainder)
  camera     no complete camera set within --sync-ms (per camera: missing or out of sync)
  window     before every sensor started or after the first one stopped
  lidar_gap  time with LiDAR frames missing
The INS stream is taken at its header stamps; the converter uses the receiver's GPS
measurement time, which the header trails by ~1-8 ms — nothing a coverage window notices.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAM_CHANNEL_TO_TOPIC, INSPVA_TOPIC, LIDAR_POINTS_TOPIC,  # noqa: E402
                    NUSCENES_CAMS)
from nuscenes_writer import (coverage_window, partition_scenes, plan_frames,  # noqa: E402
                             required_streams)

CATEGORIES = ("scene", "short", "camera", "window", "lidar_gap")
_U32 = struct.Struct("<I")


def find_bags(paths: list[Path]) -> list[Path]:
    """Files as given, directories as every finished *.bag under them, in route order."""
    out: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"  [!] not found: {p}", flush=True)
            continue
        found = sorted(p.rglob("*.bag")) if p.is_dir() else [p]
        out.extend(f for f in found if f.is_file() and f not in out
                   and not any(part.startswith(".") for part in f.parts))
    return sorted(out, key=route_key)


def route_key(bag: Path) -> tuple:
    """A-2 before A-10: the letter, then the route number, then the name."""
    m = re.match(r"([A-Za-z]+)-(\d+)", bag.name)
    return (m.group(1), int(m.group(2)), bag.name) if m else ("~", 0, bag.name)


def _fields(header: bytes) -> dict[str, bytes]:
    out, i = {}, 0
    while i + 4 <= len(header):
        n = _U32.unpack_from(header, i)[0]
        name, _, value = header[i + 4:i + 4 + n].partition(b"=")
        out[name.decode()] = value
        i += 4 + n
    return out


def _stamp_at(buf: bytes, pos: int) -> tuple[int, int] | None:
    """header.stamp (sec, nsec) of the message record at buf[pos:], skipping connection
    records in front of it; None when buf ends first."""
    while True:
        if pos + 4 > len(buf):
            return None
        hlen = _U32.unpack_from(buf, pos)[0]
        if pos + 8 + hlen > len(buf):
            return None
        op = _fields(buf[pos + 4:pos + 4 + hlen]).get("op", b"\0")[0]
        dlen = _U32.unpack_from(buf, pos + 4 + hlen)[0]
        data = pos + 8 + hlen
        if op == 7:                                     # connection record: skip
            pos = data + dlen
            continue
        if op != 2:
            raise ValueError(f"expected message data, found record op {op}")
        if data + 12 > len(buf):
            return None
        # std_msgs/Header: uint32 seq, then stamp (uint32 sec, uint32 nsec)
        return _U32.unpack_from(buf, data + 4)[0], _U32.unpack_from(buf, data + 8)[0]


def header_stamps(bag: Path, topics: list[str]) -> dict[str, np.ndarray]:
    """topic -> sorted unique header stamps (ns), reading each message's first bytes."""
    stamps: dict[str, list[int]] = {t: [] for t in topics}
    with Reader(bag) as reader, open(bag, "rb") as f:
        by_id = {c.id: c.topic for c in reader.connections if c.topic in stamps}
        entries = sorted((e.chunk_pos, e.offset, cid) for cid in by_id
                         for e in reader.indexes.get(cid, []))
        chunk_pos, chunk_buf, plain = None, b"", {}
        for pos, offset, cid in entries:
            chunk = reader.chunks[pos]
            if pos not in plain:
                plain[pos] = _is_plain(f, pos)
            if plain[pos]:
                size, stamp = 512, None
                while stamp is None:                    # grow past long connection records
                    f.seek(chunk.datapos + offset)
                    buf = f.read(min(size, chunk.datasize - offset))
                    stamp = _stamp_at(buf, 0)
                    if stamp is None and size >= chunk.datasize - offset:
                        raise ValueError(f"{bag.name}: truncated message at {pos}+{offset}")
                    size *= 8
            else:                                       # compressed: inflate the chunk once
                if chunk_pos != pos:
                    f.seek(chunk.datapos)
                    chunk_buf, chunk_pos = chunk.decompressor(f.read(chunk.datasize)), pos
                stamp = _stamp_at(chunk_buf, offset)
                if stamp is None:
                    raise ValueError(f"{bag.name}: truncated message at {pos}+{offset}")
            stamps[by_id[cid]].append(stamp[0] * 10**9 + stamp[1])
    return {t: np.unique(np.array(v, dtype=np.int64)) for t, v in stamps.items()}


def _is_plain(f, pos: int) -> bool:
    """Whether the chunk record at pos is stored uncompressed (messages readable in place)."""
    f.seek(pos)
    hlen = _U32.unpack(f.read(4))[0]
    return _fields(f.read(hlen)).get("compression", b"none") in (b"none", b"")


def check(bag: Path, sync_ms: float, scene_s: float, group_ms: float) -> dict:
    gating = list(NUSCENES_CAMS)
    cam_topics = {ch: CAM_CHANNEL_TO_TOPIC[ch] for ch in gating}
    stamps = header_stamps(bag, [LIDAR_POINTS_TOPIC, INSPVA_TOPIC, *cam_topics.values()])
    frames = {ch: stamps[t] for ch, t in cam_topics.items()}
    L = stamps[LIDAR_POINTS_TOPIC]
    res = {"recording": bag.stem, "path": str(bag), "notes": [],
           "frames": {"LIDAR_TOP": int(len(L)), **{ch: int(len(v)) for ch, v in frames.items()}},
           "scenes": 0, "seconds": {c: 0.0 for c in CATEGORIES}, "camera_blame_s": {},
           "timeline": [], "span_s": 0.0}
    dead = [ch for ch in gating if not len(frames[ch])]
    if dead:
        res["notes"].append("프레임 없는 카메라: " + ", ".join(dead) + " - 이 녹화는 씬이 나올 수 없음")
    if len(L) < 2:
        res["notes"].append(f"LiDAR 프레임 없음 ({LIDAR_POINTS_TOPIC})")
        return res
    period = float(np.median(np.diff(L)))
    t0 = int(L[0])
    res["span_s"] = round((L[-1] - L[0]) / 1e9 + period / 1e9, 2)

    sync_ns = int(sync_ms * 1e6)
    cat = np.full(len(L), "camera", dtype=object)
    pose = stamps[INSPVA_TOPIC]
    if not dead and len(pose):
        window = coverage_window(required_streams(L, frames, gating, pose), sync_ns)
        plan = plan_frames(L, frames, gating, sync_ns, window, int(group_ms * 1e6),
                           int(1.5 * period))
        cut, _ = partition_scenes(plan["segments"], int(round(scene_s * 1e9 / period)))
        in_window = (L >= window["start_ns"]) & (L <= window["end_ns"])
        # Name the stream that starts late or stops early when that costs real time.
        late = (window["start_ns"] - L[0]) / 1e9
        early = (L[-1] - window["end_ns"]) / 1e9
        if late > 2:
            res["notes"].append(f"{window['last_start']}가 {late:.0f}초 늦게 시작 - 그 앞은 쓸 수 없음")
        if early > 2:
            res["notes"].append(f"{window['first_end']}가 끝나기 {early:.0f}초 전에 끊김 - 그 뒤는 쓸 수 없음")
        in_scene = np.zeros(len(L), dtype=bool)
        for _, idx in cut:
            in_scene[idx] = True
        cat[~in_window] = "window"
        cat[plan["valid"] & ~in_scene] = "short"
        cat[in_scene] = "scene"
        res["scenes"] = len(cut)
        sd = plan["stats"]["sync_dev_ms"]
        res["sync_dev_ms"] = {k: round(v, 2) for k, v in sd.items()} if sd else None
    elif not len(pose):
        res["notes"].append(f"{INSPVA_TOPIC} 없음 - ego pose를 만들 수 없음")
        cat[:] = "window"

    # Which cameras were missing or out of sync where a frame was lost to "camera".
    lost = L[cat == "camera"]
    for ch in gating:
        T = frames[ch]
        if not len(lost):
            break
        if not len(T):
            bad = np.ones(len(lost), dtype=bool)
        else:
            j = np.clip(np.searchsorted(T, lost), 1, max(len(T) - 1, 1))
            bad = np.minimum(np.abs(T[j] - lost), np.abs(T[j - 1] - lost)) > sync_ns
        if bad.any():
            res["camera_blame_s"][ch] = round(float(bad.sum()) * period / 1e9, 1)

    step = period / 1e9
    for c in CATEGORIES[:-1]:
        res["seconds"][c] = round(float((cat == c).sum()) * step, 1)
    gaps = np.flatnonzero(np.diff(L) > 1.5 * period)
    res["seconds"]["lidar_gap"] = round(float(sum((L[g + 1] - L[g]) / 1e9 - step for g in gaps)), 1)

    # Run-length timeline in seconds from the first LiDAR frame, gaps as their own runs.
    timeline, start = [], 0
    rel = (L - t0) / 1e9
    gap_after = set(gaps.tolist())
    for k in range(1, len(L) + 1):
        if k == len(L) or cat[k] != cat[start] or k - 1 in gap_after:
            timeline.append([round(rel[start], 2), round(rel[k - 1] + step, 2), cat[start]])
            if k < len(L) and k - 1 in gap_after:
                timeline.append([round(rel[k - 1] + step, 2), round(rel[k], 2), "lidar_gap"])
            start = k
    res["timeline"] = timeline
    return res


def summary_line(r: dict) -> str:
    s = r["seconds"]
    lost = sorted(((v, k) for k, v in s.items() if k != "scene" and v >= 0.1), reverse=True)
    return (f"{r['recording']}: {r['scenes']} scenes, {s['scene']:.0f}/{r['span_s']:.0f} s used"
            + ("; lost " + ", ".join(f"{k} {v:.1f} s" for v, k in lost) if lost else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--json", type=Path, help="write the full result here")
    p.add_argument("--sync-ms", type=float, default=25.0)
    p.add_argument("--scene-dur", type=float, default=20.0)
    p.add_argument("--camera-group-ms", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    bags = find_bags(a.paths)
    if not bags:
        raise SystemExit("no .bag files found")
    print(f"{len(bags)} bag(s): {len(NUSCENES_CAMS)} cameras within {a.sync_ms:g} ms of "
          f"LiDAR, {a.scene_dur:g} s scenes", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(check, b, a.sync_ms, a.scene_dur, a.camera_group_ms): b for b in bags}
        for k, f in enumerate(as_completed(futs), 1):
            b = futs[f]
            try:
                res = f.result()
            except Exception as e:                      # noqa: BLE001 - report and go on
                res = {"recording": b.stem, "path": str(b), "error": repr(e), "scenes": 0,
                       "notes": [f"검사 실패: {e!r}"], "seconds": {c: 0.0 for c in CATEGORIES},
                       "camera_blame_s": {}, "timeline": [], "span_s": 0.0, "frames": {}}
            results[b] = res
            print(summary_line(res), flush=True)
            for n in res["notes"]:
                print(f"  [!] {n}", flush=True)
            print(f"{100 * k // len(bags):3d}%| {k}/{len(bags)}", flush=True)
    ordered = [results[b] for b in bags]
    print(f"total: {sum(r['scenes'] for r in ordered)} scenes from {len(ordered)} bag(s)")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "paths": [str(x) for x in a.paths],
            "params": {"sync_ms": a.sync_ms, "scene_dur_s": a.scene_dur,
                       "camera_group_ms": a.camera_group_ms, "rgb_cameras": len(NUSCENES_CAMS)},
            "recordings": ordered}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
