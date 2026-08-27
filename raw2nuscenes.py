#!/usr/bin/env python3
"""Stage 2: intermediate -> NuScenes v1.0-trainval layout.

Reads <intermediate>/<bag_name>/ produced by decode_lidar.py + bag2raw.py and
emits a NuScenes-compatible dataset under <out_root>/.

Pipeline:
  1. Read calib + odom + per-channel cam ts + lidar ts (filesystem glob).
  2. Pick keyframes at fixed rate (every Kth lidar at 10Hz -> 2Hz default).
  3. For each keyframe lidar ts, match nearest cam ts per channel within
     <sync_ms> tolerance. If any channel misses, drop the keyframe.
  4. Partition surviving keyframes into scenes by fixed time window
     (default 20s). Drop the trailing incomplete scene.
  5. Build 13 NuScenes JSON tables.
  6. Materialize files: hard-link JPEGs into samples/sweeps; decompress lidar
     .bin.zst -> .pcd.bin (5xfloat32) into samples/sweeps.
  7. Optional: validate by loading NuScenes(version, dataroot).

Convention notes:
  - Intermediate calib stores OpenCV-style extrinsics (P_sensor = R @ P_ego + t).
    NuScenes stores sensor-in-ego pose. We invert when writing
    calibrated_sensor.json.
  - Annotations are empty (sample_annotation/instance = []) on purpose: labels
    are produced externally. category/attribute/visibility hold one placeholder
    record each so foreign keys stay valid; replace them with the real taxonomy
    before handing the dataset to a labeling vendor.
  - Images are NOT undistorted. NuScenes has no distortion field, so consumers
    treat camera_intrinsic as a pinhole K, but five of the six cameras are
    fisheye. See "Known limitations" in README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import zstandard as zstd
from scipy.spatial.transform import Rotation, Slerp

from common import (
    NUSCENES_CAMS,
    opencv_ext_to_nuscenes_pose,
    quat_wxyz_to_R,
)


def new_token() -> str:
    return uuid.uuid4().hex


@dataclass
class IntermediateData:
    root: Path
    calib: dict
    cam_ts: dict[str, np.ndarray]   # channel -> sorted ts_ns (int64)
    lidar_ts: np.ndarray            # sorted ts_ns
    odom_ts: np.ndarray             # sorted ts_ns
    odom_t: np.ndarray              # (N, 3) translation
    odom_R: Rotation                # rotation samples for SLERP
    cam_size: dict[str, tuple[int, int]]  # channel -> (height, width)
    bag_start_ns: int
    bag_end_ns: int


def load_intermediate(path: Path) -> IntermediateData:
    calib = json.loads((path / "calib.json").read_text())
    meta = json.loads((path / "meta.json").read_text())

    # camera ts per channel
    cam_root = path / "cameras"
    cam_ts: dict[str, np.ndarray] = {}
    cam_size: dict[str, tuple[int, int]] = {}
    for ch_dir in sorted(cam_root.iterdir()):
        if not ch_dir.is_dir():
            continue
        ts = np.array(sorted(int(p.stem) for p in ch_dir.glob("*.jpg")), dtype=np.int64)
        cam_ts[ch_dir.name] = ts
        if len(ts) > 0:
            sample = cv2.imread(str(ch_dir / f"{ts[0]}.jpg"))
            if sample is None:
                raise SystemExit(f"unreadable JPEG: {ch_dir / f'{ts[0]}.jpg'}")
            cam_size[ch_dir.name] = (sample.shape[0], sample.shape[1])

    # lidar ts
    lidar_ts = np.array(
        sorted(int(p.name.split(".", 1)[0]) for p in (path / "lidar").glob("*.bin.zst")),
        dtype=np.int64,
    )

    # odom
    odom_tbl = pq.read_table(path / "odom.parquet")
    odom_ts = np.asarray(odom_tbl.column("ts_ns")).astype(np.int64)
    sort_idx = np.argsort(odom_ts)
    odom_ts = odom_ts[sort_idx]
    odom_t = np.stack([
        np.asarray(odom_tbl.column("tx"))[sort_idx],
        np.asarray(odom_tbl.column("ty"))[sort_idx],
        np.asarray(odom_tbl.column("tz"))[sort_idx],
    ], axis=-1)
    qw = np.asarray(odom_tbl.column("qw"))[sort_idx]
    qx = np.asarray(odom_tbl.column("qx"))[sort_idx]
    qy = np.asarray(odom_tbl.column("qy"))[sort_idx]
    qz = np.asarray(odom_tbl.column("qz"))[sort_idx]
    odom_R = Rotation.from_quat(np.stack([qx, qy, qz, qw], axis=-1))

    return IntermediateData(
        root=path,
        calib=calib,
        cam_ts=cam_ts,
        lidar_ts=lidar_ts,
        odom_ts=odom_ts,
        odom_t=odom_t,
        odom_R=odom_R,
        cam_size=cam_size,
        bag_start_ns=meta["bag_start_ns"],
        bag_end_ns=meta["bag_end_ns"],
    )


def nearest_ts(query: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each q in query, return (matched_target_ts, abs_diff_ns)."""
    if len(target) == 0:
        return np.full_like(query, -1), np.full_like(query, np.iinfo(np.int64).max)
    idx = np.searchsorted(target, query)
    idx_l = np.clip(idx - 1, 0, len(target) - 1)
    idx_r = np.clip(idx, 0, len(target) - 1)
    d_l = np.abs(target[idx_l] - query)
    d_r = np.abs(target[idx_r] - query)
    pick_l = d_l <= d_r
    matched = np.where(pick_l, target[idx_l], target[idx_r])
    diff = np.where(pick_l, d_l, d_r)
    return matched, diff


def sync_keyframes(data: IntermediateData, channels: list[str], sync_ns: int,
                   keyframe_stride: int) -> list[dict]:
    """Pick every Kth lidar ts as keyframe anchor. Match nearest cam ts per
    channel; drop keyframes that miss any channel within tolerance.

    Returns list of dicts: {"lidar_ts": int, "cam_ts": {channel: int}}.
    """
    anchors = data.lidar_ts[::keyframe_stride]
    print(f"  candidate keyframes: {len(anchors)} (every {keyframe_stride}th of {len(data.lidar_ts)} lidar)")

    cam_match: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ch in channels:
        cam_match[ch] = nearest_ts(anchors, data.cam_ts[ch])

    valid = np.ones(len(anchors), dtype=bool)
    for ch in channels:
        _, diff = cam_match[ch]
        valid &= diff <= sync_ns
    n_drop = int((~valid).sum())
    print(f"  dropped {n_drop} ({100*n_drop/len(anchors):.2f}%) for sync miss; kept {valid.sum()}")

    out: list[dict] = []
    for i in np.where(valid)[0]:
        out.append({
            "lidar_ts": int(anchors[i]),
            "cam_ts": {ch: int(cam_match[ch][0][i]) for ch in channels},
        })
    return out


def partition_scenes(keyframes: list[dict], scene_dur_s: float) -> list[list[dict]]:
    """Group keyframes into scenes by fixed time window. By construction all
    scenes except possibly the last have full duration; the last scene is
    dropped if it didn't reach >=90% of the target duration."""
    if not keyframes:
        return []
    scene_dur_ns = int(scene_dur_s * 1e9)
    scenes: list[list[dict]] = []
    current: list[dict] = []
    scene_start_ns = keyframes[0]["lidar_ts"]
    for kf in keyframes:
        if kf["lidar_ts"] - scene_start_ns >= scene_dur_ns and current:
            scenes.append(current)
            current = []
            scene_start_ns = kf["lidar_ts"]
        current.append(kf)
    if current:
        scenes.append(current)

    # Drop trailing incomplete scene (time-based, not count-based, since sync
    # drops can leave a fully-spanning scene with fewer than expected samples).
    if scenes:
        last = scenes[-1]
        last_span_ns = last[-1]["lidar_ts"] - last[0]["lidar_ts"]
        threshold = int(scene_dur_s * 0.9 * 1e9)
        if last_span_ns < threshold:
            print(f"  dropping last scene (span {last_span_ns/1e9:.1f}s < {threshold/1e9:.1f}s)")
            scenes = scenes[:-1]

    print(f"  {len(scenes)} scenes  ({sum(len(s) for s in scenes)} total samples)")
    return scenes


def interp_pose(query_ns: np.ndarray, data: IntermediateData) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ego pose at each query ns. Returns (translations, quaternions[wxyz])."""
    odom_ts = data.odom_ts
    odom_t = data.odom_t
    # Clip query to odom range to avoid SLERP errors at the edges
    q = np.clip(query_ns, odom_ts[0], odom_ts[-1])

    # Linear interp translation
    tx = np.interp(q, odom_ts, odom_t[:, 0])
    ty = np.interp(q, odom_ts, odom_t[:, 1])
    tz = np.interp(q, odom_ts, odom_t[:, 2])
    trans = np.stack([tx, ty, tz], axis=-1)

    # SLERP rotation
    slerp = Slerp(odom_ts.astype(np.float64), data.odom_R)
    rots = slerp(q.astype(np.float64))
    quats_xyzw = rots.as_quat()
    quats_wxyz = np.stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]],
        axis=-1,
    )
    return trans, quats_wxyz


def assign_to_nearest_sample(target_ts: np.ndarray, sample_ts: np.ndarray) -> np.ndarray:
    """For each target ts, return index into sample_ts (nearest)."""
    if len(sample_ts) == 0:
        return np.full_like(target_ts, -1, dtype=np.int64)
    idx = np.searchsorted(sample_ts, target_ts)
    idx_l = np.clip(idx - 1, 0, len(sample_ts) - 1)
    idx_r = np.clip(idx, 0, len(sample_ts) - 1)
    d_l = np.abs(sample_ts[idx_l] - target_ts)
    d_r = np.abs(sample_ts[idx_r] - target_ts)
    return np.where(d_l <= d_r, idx_l, idx_r)


def build_tables(data: IntermediateData, scenes: list[list[dict]],
                 channels: list[str], log_token: str, log_name: str,
                 existing: dict | None = None
                 ) -> tuple[dict, list[tuple[Path, Path, str]]]:
    """Build all 13 NuScenes JSON tables.

    If `existing` (loaded JSONs) is provided, reuse stable tokens (sensor,
    category, attribute, visibility, map) and continue the scene index.
    The returned tables contain ONLY the new records — caller merges with
    existing ones via merge_tables().

    Returns (tables, plan) where plan is a list of (src_path, target_relpath,
    format) used by materialize_files. Format is 'jpg' (hard link) or 'lidar'
    (zstd decompress).
    """
    plan: list[tuple[Path, Path, str]] = []
    scene_idx_offset = 0
    existing_sensor_tokens: dict[str, str] = {}
    if existing:
        scene_idx_offset = len(existing.get("scene.json", []))
        for s in existing.get("sensor.json", []):
            existing_sensor_tokens[s["channel"]] = s["token"]
    # ---------- sensor.json (reuse existing tokens if any) ----------
    sensor_tokens: dict[str, str] = {}
    sensors = []
    for ch, modality in [("LIDAR_TOP", "lidar")] + [(c, "camera") for c in channels]:
        if ch in existing_sensor_tokens:
            sensor_tokens[ch] = existing_sensor_tokens[ch]
        else:
            tk = new_token()
            sensor_tokens[ch] = tk
            sensors.append({"token": tk, "channel": ch, "modality": modality})

    # ---------- calibrated_sensor.json (8 records, one per (sensor, log)) ----------
    cs_tokens: dict[str, str] = {}
    cs_records = []
    # LIDAR_TOP
    L = data.calib["LIDAR_TOP"]
    R_lidar = quat_wxyz_to_R(L["rotation"])
    t_lidar = np.asarray(L["translation"], dtype=np.float64)
    rot_q, trans = opencv_ext_to_nuscenes_pose(R_lidar, t_lidar)
    cs_tokens["LIDAR_TOP"] = new_token()
    cs_records.append({
        "token": cs_tokens["LIDAR_TOP"],
        "sensor_token": sensor_tokens["LIDAR_TOP"],
        "translation": trans,
        "rotation": rot_q,
        "camera_intrinsic": [],
    })
    for ch in channels:
        cs_tokens[ch] = new_token()
        if ch in data.calib:
            params = data.calib[ch]
            R_c = quat_wxyz_to_R(params["rotation"])
            t_c = np.asarray(params["translation"], dtype=np.float64)
            rot_q, trans = opencv_ext_to_nuscenes_pose(R_c, t_c)
            K = params["intrinsic"]
        else:
            # placeholder for CAM_TRAFFIC (no calib) — identity transform
            rot_q = [1.0, 0.0, 0.0, 0.0]
            trans = [0.0, 0.0, 0.0]
            h, w = data.cam_size.get(ch, (1080, 1920))
            K = [[w / 2, 0.0, w / 2], [0.0, h / 2, h / 2], [0.0, 0.0, 1.0]]
        cs_records.append({
            "token": cs_tokens[ch],
            "sensor_token": sensor_tokens[ch],
            "translation": trans,
            "rotation": rot_q,
            "camera_intrinsic": K,
        })

    # ---------- log.json ----------
    log_record = {
        "token": log_token,
        "logfile": log_name,
        "vehicle": "tcar",
        "date_captured": datetime.fromtimestamp(
            data.bag_start_ns / 1e9, tz=timezone.utc).date().isoformat(),
        "location": "korea-test",
    }

    # ---------- placeholder category/attribute/visibility/map (reuse if existing) ----------
    if existing and existing.get("category.json"):
        categories: list = []  # already there, don't duplicate
    else:
        categories = [{
            "token": new_token(),
            "name": "vehicle.unknown",
            "description": "placeholder; annotations to be filled later",
            "index": 0,
        }]
    if existing and existing.get("attribute.json"):
        attributes: list = []
    else:
        attributes = [{
            "token": new_token(),
            "name": "vehicle.moving",
            "description": "placeholder",
        }]
    if existing and existing.get("visibility.json"):
        visibilities: list = []
    else:
        visibilities = [{
            "token": new_token(),
            "level": "v80-100",
            "description": "placeholder",
        }]
    # Map: append a record per log so log_tokens reference is set per bag.
    # devkit's render_sample expects a real PNG at filename; we point all maps
    # at a single shared placeholder created in materialize_files().
    maps = [{
        "token": new_token(),
        "category": "semantic_prior",
        "filename": "maps/placeholder.png",
        "log_tokens": [log_token],
    }]

    # ---------- scenes / samples / sample_data / ego_pose ----------
    scene_records = []
    sample_records = []
    sample_data_records = []
    ego_pose_records = []

    # All sample_data we will emit (samples + sweeps), per channel, indexed by ts_ns
    # so we can chain prev/next within scene.
    # Strategy:
    #   - For samples: lidar @ keyframe ts + 7 cams @ matched ts (is_key_frame=True)
    #   - For sweeps:  every other lidar/cam frame in [scene_start, scene_end] window
    #                  whose nearest sample is within scene
    #
    # We'll build sample_data records per scene and link prev/next per sensor channel.

    # Pre-compute keyframe sample tokens & timestamps so sweeps can attach.
    for scene_idx, scene_kfs in enumerate(scenes):
        scene_token = new_token()
        scene_kf_ts = np.array([kf["lidar_ts"] for kf in scene_kfs], dtype=np.int64)
        scene_kf_tokens = [new_token() for _ in scene_kfs]

        scene_start_ns = scene_kf_ts[0]
        scene_end_ns = scene_kf_ts[-1]

        # ----- sample records (keyframes) -----
        for i, (kf, sample_token) in enumerate(zip(scene_kfs, scene_kf_tokens)):
            sample_records.append({
                "token": sample_token,
                "timestamp": kf["lidar_ts"] // 1000,  # ns -> us
                "scene_token": scene_token,
                "next": scene_kf_tokens[i + 1] if i + 1 < len(scene_kfs) else "",
                "prev": scene_kf_tokens[i - 1] if i > 0 else "",
            })

        # ----- collect all sample_data per (channel) for this scene window -----
        channel_frames: dict[str, list[tuple[int, bool, str]]] = {}
        # (frame_ts_ns, is_key_frame, sample_data_token)

        # LIDAR_TOP: keyframe lidar ts + every other lidar in window
        lidar_in_scene = data.lidar_ts[
            (data.lidar_ts >= scene_start_ns) & (data.lidar_ts <= scene_end_ns)
        ]
        kf_lidar_set = set(scene_kf_ts.tolist())
        ld_frames = []
        for t in lidar_in_scene:
            ld_frames.append((int(t), int(t) in kf_lidar_set, new_token()))
        channel_frames["LIDAR_TOP"] = ld_frames

        # Cameras: keyframe cam ts (from scene_kfs) + sweeps in window
        for ch in channels:
            kf_cam_ts_set = {kf["cam_ts"][ch] for kf in scene_kfs}
            cam_in_scene_mask = (
                (data.cam_ts[ch] >= scene_start_ns) & (data.cam_ts[ch] <= scene_end_ns)
            )
            cam_in_scene = data.cam_ts[ch][cam_in_scene_mask]
            # ensure all keyframe cam ts are included
            full_set = set(int(t) for t in cam_in_scene) | kf_cam_ts_set
            cam_sorted = sorted(full_set)
            ch_frames = []
            for t in cam_sorted:
                is_kf = t in kf_cam_ts_set
                ch_frames.append((int(t), is_kf, new_token()))
            channel_frames[ch] = ch_frames

        # Map keyframe lidar_ts -> sample_token for sample_data linkage
        kf_lidar_to_sample = {int(t): tok for t, tok in zip(scene_kf_ts, scene_kf_tokens)}

        # Build channel-keyed nearest-sample mapping for sweeps using the sorted
        # scene_kf_ts as anchors (so all sample_data in this scene attach to a
        # sample within this scene).
        for ch_name, frames in channel_frames.items():
            frame_ts_arr = np.array([f[0] for f in frames], dtype=np.int64)
            nearest_kf = assign_to_nearest_sample(frame_ts_arr, scene_kf_ts)

            # interp ego pose for every frame ts in this channel (one ego_pose per sample_data)
            trans, quats = interp_pose(frame_ts_arr, data)

            prev_token = ""
            for i, (ts_ns, is_kf, sd_token) in enumerate(frames):
                ego_token = new_token()
                ego_pose_records.append({
                    "token": ego_token,
                    "translation": trans[i].tolist(),
                    "rotation": [float(x) for x in quats[i]],
                    "timestamp": int(ts_ns) // 1000,
                })

                # Determine sample_token: keyframes use their own; sweeps use nearest keyframe lidar_ts -> sample_token
                if is_kf and ch_name == "LIDAR_TOP":
                    sample_token = kf_lidar_to_sample[ts_ns]
                else:
                    nearest_kf_ts = int(scene_kf_ts[nearest_kf[i]])
                    sample_token = kf_lidar_to_sample[nearest_kf_ts]

                # filename: samples/<channel>/<token>.<ext> or sweeps/<channel>/<token>.<ext>
                bucket = "samples" if is_kf else "sweeps"
                if ch_name == "LIDAR_TOP":
                    fname = f"{bucket}/LIDAR_TOP/{sd_token}.pcd.bin"
                    fileformat = "pcd"
                    height, width = 0, 0
                    src = data.root / "lidar" / f"{ts_ns}.bin.zst"
                    plan.append((src, Path(fname), "lidar"))
                else:
                    fname = f"{bucket}/{ch_name}/{sd_token}.jpg"
                    fileformat = "jpg"
                    height, width = data.cam_size.get(ch_name, (0, 0))
                    src = data.root / "cameras" / ch_name / f"{ts_ns}.jpg"
                    plan.append((src, Path(fname), "jpg"))

                sd = {
                    "token": sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_token,
                    "calibrated_sensor_token": cs_tokens[ch_name],
                    "filename": fname,
                    "fileformat": fileformat,
                    "is_key_frame": bool(is_kf),
                    "height": int(height),
                    "width": int(width),
                    "timestamp": int(ts_ns) // 1000,
                    "next": "",  # filled in second pass below
                    "prev": prev_token,
                }
                sample_data_records.append(sd)
                if prev_token:
                    # Set prev->next link
                    for r in reversed(sample_data_records):
                        if r["token"] == prev_token:
                            r["next"] = sd_token
                            break
                prev_token = sd_token

        # ----- scene record -----
        scene_records.append({
            "token": scene_token,
            "name": f"scene-{scene_idx_offset + scene_idx + 1:04d}",
            "description": f"auto-generated from {log_name}",
            "log_token": log_token,
            "nbr_samples": len(scene_kfs),
            "first_sample_token": scene_kf_tokens[0],
            "last_sample_token": scene_kf_tokens[-1],
        })

    # ---------- empty/placeholder annotation tables ----------
    sample_annotations: list = []
    instances: list = []

    tables = {
        "sensor.json": sensors,
        "calibrated_sensor.json": cs_records,
        "log.json": [log_record],
        "scene.json": scene_records,
        "sample.json": sample_records,
        "sample_data.json": sample_data_records,
        "ego_pose.json": ego_pose_records,
        "sample_annotation.json": sample_annotations,
        "instance.json": instances,
        "category.json": categories,
        "attribute.json": attributes,
        "visibility.json": visibilities,
        "map.json": maps,
    }
    return tables, plan


def _ensure_placeholder_map(out_root: Path) -> None:
    """nuscenes-devkit's render_sample loads map raster; write a tiny PNG so
    devkit doesn't crash on render. Real maps can replace this later."""
    map_path = out_root / "maps" / "placeholder.png"
    if map_path.exists():
        return
    map_path.parent.mkdir(parents=True, exist_ok=True)
    blank = np.full((100, 100), 255, dtype=np.uint8)
    cv2.imwrite(str(map_path), blank)


def materialize_files(plan: list[tuple[Path, Path, str]], data: IntermediateData,
                      out_root: Path) -> None:
    """Hard-link JPEGs + decompress lidar .bin.zst -> .pcd.bin into samples/sweeps.

    Hard links, not symlinks: the dataset must survive deleting the intermediate
    dump. Both share the same inode so this costs no extra disk, but the image
    data stays reachable once the intermediate directory entry is gone. Falls
    back to a copy if the two trees are on different filesystems.
    """
    _ensure_placeholder_map(out_root)
    for sub in ["samples", "sweeps"]:
        for ch in ["LIDAR_TOP"] + sorted(data.cam_ts.keys()):
            (out_root / sub / ch).mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    n_cam = n_lidar = 0
    for src, rel_target, fmt in plan:
        target = out_root / rel_target
        if target.exists():
            continue
        if fmt == "jpg":
            try:
                os.link(src, target)
            except OSError:  # cross-filesystem — fall back to a real copy
                target.write_bytes(src.read_bytes())
            n_cam += 1
        else:  # lidar
            target.write_bytes(dctx.decompress(src.read_bytes()))
            n_lidar += 1
    print(f"  cam hardlinks: {n_cam}   lidar decompressed: {n_lidar}")


def load_existing_tables(json_dir: Path) -> dict | None:
    """If <json_dir>/scene.json exists with records, load all 13 tables."""
    if not (json_dir / "scene.json").exists():
        return None
    out: dict = {}
    for name in [
        "sensor.json", "calibrated_sensor.json", "log.json", "scene.json",
        "sample.json", "sample_data.json", "ego_pose.json",
        "sample_annotation.json", "instance.json",
        "category.json", "attribute.json", "visibility.json", "map.json",
    ]:
        f = json_dir / name
        out[name] = json.loads(f.read_text()) if f.exists() else []
    return out if out["scene.json"] else None


def merge_tables(existing: dict, new: dict) -> dict:
    """Concatenate new records onto existing, preserving order. Tables that
    'new' is empty for (sensor/category/attribute/visibility when reusing
    tokens) keep the existing rows untouched."""
    out: dict = {}
    for name, new_recs in new.items():
        old_recs = existing.get(name, [])
        out[name] = old_recs + new_recs
    return out


def write_tables(tables: dict, out_root: Path, version: str) -> Path:
    json_dir = out_root / version
    json_dir.mkdir(parents=True, exist_ok=True)
    for name, records in tables.items():
        (json_dir / name).write_text(json.dumps(records, indent=2))
    return json_dir


def validate_with_devkit(out_root: Path, version: str) -> None:
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("  [skip] nuscenes-devkit not installed")
        return
    print(f"  loading NuScenes(version='{version}', dataroot='{out_root}')...")
    nusc = NuScenes(version=version, dataroot=str(out_root), verbose=False)
    print(f"  ✓ loaded: {len(nusc.scene)} scene, {len(nusc.sample)} sample, "
          f"{len(nusc.sample_data)} sample_data, {len(nusc.ego_pose)} ego_pose")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("intermediate", type=Path,
                   help="Stage-1 intermediate dir, e.g. /data/intermediate/<bag>")
    p.add_argument("--out", type=Path, default=Path("/data/tcar_nuscenes"),
                   help="NuScenes dataroot (default: /data/tcar_nuscenes).")
    p.add_argument("--version", default="v1.0-trainval",
                   help="NuScenes version subdir name (default: v1.0-trainval).")
    p.add_argument("--keyframe-stride", type=int, default=5,
                   help="Pick every Kth lidar frame as keyframe (default 5 = 2Hz from 10Hz lidar).")
    p.add_argument("--sync-ms", type=float, default=25.0,
                   help="Per-channel sync tolerance in ms (default 25).")
    p.add_argument("--scene-dur", type=float, default=20.0,
                   help="Scene length in seconds (default 20). Trailing scene shorter than 90%% is dropped.")
    p.add_argument("--include-traffic-cam", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include CAM_TRAFFIC as a 7th channel "
                        "(--no-include-traffic-cam to omit it).")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip NuScenes(...) load check at the end.")
    args = p.parse_args()

    if not args.intermediate.exists():
        raise SystemExit(f"intermediate not found: {args.intermediate}")

    print("[1/6] Loading intermediate...")
    data = load_intermediate(args.intermediate)
    print(f"  channels: {sorted(data.cam_ts.keys())}")
    print(f"  lidar: {len(data.lidar_ts)} frames")
    print(f"  odom: {len(data.odom_ts)} samples")

    channels = list(NUSCENES_CAMS)
    if args.include_traffic_cam and "CAM_TRAFFIC" in data.cam_ts:
        channels.append("CAM_TRAFFIC")

    print(f"\n[2/6] Sync matching (tolerance {args.sync_ms} ms, all {len(channels)} cams must pass)...")
    keyframes = sync_keyframes(
        data, channels,
        sync_ns=int(args.sync_ms * 1e6),
        keyframe_stride=args.keyframe_stride,
    )

    print(f"\n[3/6] Scene partitioning ({args.scene_dur}s windows, trailing-scene drop if <90% span)...")
    scenes = partition_scenes(keyframes, args.scene_dur)
    if not scenes:
        raise SystemExit("no scenes survived partitioning")

    log_name = args.intermediate.name
    log_token = new_token()

    json_dir = args.out / args.version
    existing = load_existing_tables(json_dir)
    if existing is not None:
        # Sanity: prevent re-importing the same bag (matched by logfile name)
        if any(r.get("logfile") == log_name for r in existing.get("log.json", [])):
            raise SystemExit(
                f"log '{log_name}' already present in {json_dir}/log.json — "
                "skip or remove existing log entry first."
            )
        print(f"\n[4/6] Append mode: existing dataset found "
              f"({len(existing['scene.json'])} scenes, "
              f"{len(existing['sample.json'])} samples). "
              f"New scenes will continue from idx {len(existing['scene.json']) + 1}.")
    else:
        print(f"\n[4/6] Fresh dataset — no existing tables in {json_dir}")

    print(f"  Building tables ({len(scenes)} scenes)...")
    tables, plan = build_tables(data, scenes, channels, log_token, log_name,
                                existing=existing)
    counts = {k: len(v) for k, v in tables.items()}
    for k, n in counts.items():
        print(f"  + {k:30} {n:>8}")
    print(f"  materialization plan: {len(plan)} files")

    print(f"\n[5/6] Materializing files under {args.out}...")
    args.out.mkdir(parents=True, exist_ok=True)
    materialize_files(plan, data, args.out)
    if existing is not None:
        tables = merge_tables(existing, tables)
        print(f"  merged: total {len(tables['scene.json'])} scenes, "
              f"{len(tables['sample.json'])} samples")
    json_dir = write_tables(tables, args.out, args.version)
    print(f"  JSON tables -> {json_dir}")

    if args.no_validate:
        print("\n[6/6] (skipped validation)")
    else:
        print(f"\n[6/6] Validating with nuscenes-devkit...")
        validate_with_devkit(args.out, args.version)

    print("\nDone.")


if __name__ == "__main__":
    main()
