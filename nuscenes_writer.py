"""Turn synchronized sensor timestamps into the 13 NuScenes JSON tables.

Knows nothing about rosbags or about where the sensor files live: the caller
hands it timestamps plus a calibration dict and gets back the tables and a
materialization plan of (timestamp_ns, channel, destination_relpath) that it is
free to satisfy however it likes.

Conventions:
  - Calibration arrives in the OpenCV extrinsic convention
    (P_sensor = R @ P_ego + t); NuScenes wants the sensor pose in the ego frame,
    so it is inverted when writing calibrated_sensor.json.
  - sample_annotation/instance are emitted empty on purpose: labels are produced
    externally. category/attribute carry the label taxonomy (from the perception
    stack's enums, see common.py) and visibility the four nuScenes bins with
    nuScenes' literal tokens "1".."4", so a vendor fills the two empty tables
    against a complete, standard-looking schema.
  - Images are not undistorted anywhere in this pipeline; NuScenes has no
    distortion field. See "Known limitations" in README.md.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common import (
    MOTION_TYPE_TO_ATTRIBUTE,
    OBJECT_TYPE_TO_CATEGORY,
    VISIBILITY_LEVELS,
    opencv_ext_to_nuscenes_pose,
    quat_wxyz_to_R,
)


def new_token() -> str:
    return uuid.uuid4().hex


@dataclass
class SensorData:
    """Everything the table builder needs, however the caller obtained it."""
    calib: dict
    cam_ts: dict[str, np.ndarray]   # channel -> sorted ts_ns (int64)
    lidar_ts: np.ndarray            # sorted ts_ns
    odom_ts: np.ndarray             # sorted ts_ns
    odom_t: np.ndarray              # (N, 3) translation
    odom_R: Rotation                # rotation samples for SLERP
    cam_size: dict[str, tuple[int, int]]  # channel -> (height, width)
    bag_start_ns: int
    bag_end_ns: int
    # Built lazily by interp_pose and reused. Slerp's constructor preprocesses
    # every odom sample (O(N)); build_tables interpolates once per
    # (scene, channel), so rebuilding it per call cost scenes x channels x N.
    _slerp: Slerp | None = field(default=None, repr=False, compare=False)



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


def required_streams(data: SensorData, required: list[str]) -> dict[str, np.ndarray]:
    """The streams a keyframe cannot do without: LiDAR, the gating cameras, odom."""
    out: dict[str, np.ndarray] = {"LIDAR_TOP": data.lidar_ts}
    out.update({ch: data.cam_ts[ch] for ch in required})
    out["ODOM"] = data.odom_ts
    return out


def coverage_window(streams: dict[str, np.ndarray], sync_ns: int) -> dict:
    """The interval in which every stream in `streams` has data, shrunk by sync_ns.

    Sensors start and stop at different times (up to ~1.4 s apart on the
    2026-08-19 bags). A keyframe outside this interval would have no image or
    no pose to attach; the margin exists because a camera frame matched to a
    keyframe may sit up to sync_ns away from it and still needs a pose.
    `streams` is name -> sorted timestamps (ns). The result is JSON-friendly.
    """
    present = {k: v for k, v in streams.items() if len(v)}
    if not present:
        raise ValueError("coverage_window: no timestamps in any stream")
    firsts = {k: int(v[0]) for k, v in present.items()}
    lasts = {k: int(v[-1]) for k, v in present.items()}
    earliest, latest = min(firsts.values()), max(lasts.values())
    last_start = max(firsts, key=firsts.__getitem__)
    first_end = min(lasts, key=lasts.__getitem__)
    start = firsts[last_start] + int(sync_ns)
    end = lasts[first_end] - int(sync_ns)
    return {
        "start_ns": start, "end_ns": end, "sync_ns": int(sync_ns),
        "earliest_ns": earliest, "latest_ns": latest,
        "last_start": last_start, "first_end": first_end,
        "head_cut_s": (start - earliest) / 1e9,
        "tail_cut_s": (latest - end) / 1e9,
        "streams": {k: {"first_ns": firsts[k], "last_ns": lasts[k],
                        "start_offset_s": (firsts[k] - earliest) / 1e9,
                        "end_offset_s": (lasts[k] - latest) / 1e9}
                    for k in present},
    }


def format_coverage(window: dict) -> list[str]:
    """Human-readable lines for a coverage_window() result."""
    lines = [f"{'stream':16} {'starts':>9} {'ends':>9}"]
    for k, w in window["streams"].items():
        lines.append(f"{k:16} {w['start_offset_s']:+8.3f}s {w['end_offset_s']:+8.3f}s")
    lines.append(
        f"coverage window: head cut {window['head_cut_s']:.3f}s "
        f"({window['last_start']} starts last), tail cut {window['tail_cut_s']:.3f}s "
        f"({window['first_end']} ends first), margin {window['sync_ns'] / 1e6:.0f} ms")
    return lines


def sync_keyframes(data: SensorData, channels: list[str], sync_ns: int,
                   keyframe_stride: int,
                   required: list[str] | None = None,
                   window: dict | None = None) -> list[dict]:
    """Pick every Kth lidar ts as a keyframe anchor and match cameras to it.

    Candidates are restricted to the coverage window (see coverage_window): the
    span in which LiDAR, every required camera and odom are all present. Since
    scenes and sweeps only ever lie between keyframes, that is what guarantees
    every frame in the dataset has both an image and a pose. `window` may be
    passed in when the caller has already computed it.

    `required` channels must all land within `sync_ns` or the keyframe is
    dropped. Channels outside `required` are best-effort: they are attached when
    they happen to be in tolerance and simply omitted when they are not, so they
    can never veto a keyframe. That is how CAM_TRAFFIC is carried — it is a
    seventh, non-standard channel with placeholder calibration, and letting it
    gate the six real cameras would throw away good samples for nothing.

    Returns [{"lidar_ts": int, "cam_ts": {channel: ts}}], where cam_ts is
    guaranteed to hold every required channel and may hold the optional ones.
    """
    required = list(channels) if required is None else [c for c in required if c in channels]
    optional = [c for c in channels if c not in required]

    if window is None:
        window = coverage_window(required_streams(data, required), sync_ns)
    all_anchors = data.lidar_ts[::keyframe_stride]
    in_window = (all_anchors >= window["start_ns"]) & (all_anchors <= window["end_ns"])
    anchors = all_anchors[in_window]
    print(f"  candidate keyframes: {len(anchors)} "
          f"(every {keyframe_stride}th of {len(data.lidar_ts)} lidar; "
          f"{int((~in_window).sum())} outside the coverage window)")
    print(f"  gating on {len(required)} channel(s); "
          f"best-effort: {optional or 'none'}")

    cam_match = {ch: nearest_ts(anchors, data.cam_ts[ch]) for ch in channels}

    valid = np.ones(len(anchors), dtype=bool)
    for ch in required:
        valid &= cam_match[ch][1] <= sync_ns
    n_drop = int((~valid).sum())
    print(f"  dropped {n_drop} ({100 * n_drop / max(len(anchors), 1):.2f}%) "
          f"for sync miss; kept {int(valid.sum())}")

    out: list[dict] = []
    for i in np.where(valid)[0]:
        cam_ts = {ch: int(cam_match[ch][0][i]) for ch in required}
        for ch in optional:
            matched, diff = cam_match[ch]
            if diff[i] <= sync_ns:
                cam_ts[ch] = int(matched[i])
        out.append({"lidar_ts": int(anchors[i]), "cam_ts": cam_ts})

    for ch in optional:
        n = sum(1 for kf in out if ch in kf["cam_ts"])
        print(f"    {ch}: attached to {n}/{len(out)} keyframes "
              f"({100 * n / max(len(out), 1):.1f}%)")
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


def interp_pose(query_ns: np.ndarray, data: SensorData) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ego pose at each query ns. Returns (translations, quaternions[wxyz])."""
    odom_ts = data.odom_ts
    odom_t = data.odom_t
    # Every frame must lie inside odom coverage; coverage_window guarantees it
    # for anything sync_keyframes lets through. Clipping here instead would
    # silently freeze the pose at the first/last odom sample.
    outside = (query_ns < odom_ts[0]) | (query_ns > odom_ts[-1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} frame timestamp(s) outside odom coverage "
            f"[{int(odom_ts[0])}, {int(odom_ts[-1])}] ns — keyframes were not "
            "restricted to the coverage window")
    q = query_ns

    # Linear interp translation
    tx = np.interp(q, odom_ts, odom_t[:, 0])
    ty = np.interp(q, odom_ts, odom_t[:, 1])
    tz = np.interp(q, odom_ts, odom_t[:, 2])
    trans = np.stack([tx, ty, tz], axis=-1)

    # SLERP rotation. The interpolator is cached on `data`: constructing it
    # walks every odom sample, which dwarfs the interpolation itself.
    if data._slerp is None:
        data._slerp = Slerp(odom_ts.astype(np.float64), data.odom_R)
    rots = data._slerp(q.astype(np.float64))
    quats_xyzw = rots.as_quat()
    quats_wxyz = np.stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]],
        axis=-1,
    )
    return trans, quats_wxyz


def assign_to_following_sample(target_ts: np.ndarray, sample_ts: np.ndarray) -> np.ndarray:
    """For each target ts, the index of the first sample at or after it.

    nuScenes' rule for sweeps: a sample owns the sweeps recorded since the
    previous sample, up to its own instant. In the official data every sweep
    (camera, lidar, radar) points at the keyframe that follows it, even when
    the one before is closer. Targets past the last sample map to the last one.
    """
    if len(sample_ts) == 0:
        return np.full_like(target_ts, -1, dtype=np.int64)
    idx = np.searchsorted(sample_ts, target_ts, side="left")
    return np.clip(idx, 0, len(sample_ts) - 1)


def build_tables(data: SensorData, scenes: list[list[dict]],
                 channels: list[str], log_token: str, log_name: str,
                 existing: dict | None = None
                 ) -> tuple[dict, list[tuple[Path, Path, str]]]:
    """Build all 13 NuScenes JSON tables.

    If `existing` (loaded JSONs) is provided, reuse stable tokens (sensor,
    category, attribute, visibility, map) and continue the scene index.
    The returned tables contain ONLY the new records — caller merges with
    existing ones via merge_tables().

    Returns (tables, plan) where plan is a list of
    (source_timestamp_ns, channel, destination_relpath) that the caller turns
    into real files.
    """
    plan: list[tuple[int, str, str]] = []
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

    # ---------- taxonomy (reuse across logs; written once) ----------
    # These three tables define what a labelling vendor is being asked to
    # produce. category/attribute come from msg/ObjectType.msg and
    # msg/MotionType.msg so the perception enum and the label set stay in step;
    # visibility is NuScenes' standard four bins. sample_annotation and instance
    # are emitted empty — the vendor fills those.
    if existing and existing.get("category.json"):
        categories: list = []  # already written, don't duplicate
    else:
        categories = [
            {"token": new_token(), "name": name, "description": desc, "index": i}
            for i, (_type_id, (name, desc))
            in enumerate(sorted(OBJECT_TYPE_TO_CATEGORY.items()))
        ]
    if existing and existing.get("attribute.json"):
        attributes: list = []
    else:
        attributes = [
            {"token": new_token(), "name": name, "description": desc}
            for _mid, (name, desc) in sorted(MOTION_TYPE_TO_ATTRIBUTE.items())
        ]
    if existing and existing.get("visibility.json"):
        visibilities: list = []
    else:
        # nuScenes gives these four rows the literal tokens "1".."4" (the one
        # table that does not use uuids), and downstream tools filter on those
        # strings — e.g. mmdetection3d keeps a box only if its visibility_token
        # is in {"1","2","3","4"}. Random tokens here would silently drop every
        # vendor-produced annotation from such tools.
        visibilities = [
            {"token": str(i + 1), "level": level, "description": desc}
            for i, (level, desc) in enumerate(VISIBILITY_LEVELS)
        ]
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
            kf_cam_ts_set = {kf["cam_ts"][ch] for kf in scene_kfs
                             if ch in kf["cam_ts"]}
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

        # A keyframe frame belongs to the sample it was synchronized into: the
        # lidar frame by its own timestamp, a camera frame by the match
        # sync_keyframes made — it may sit a few ms after the lidar instant, so
        # it must not be re-assigned by time. Sweeps attach to the following
        # keyframe, as in nuScenes (see assign_to_following_sample). Anchors are
        # this scene's keyframes only, so every sample_data stays in its scene.
        kf_frame_to_sample: dict[tuple[str, int], str] = {}
        for kf, tok in zip(scene_kfs, scene_kf_tokens):
            kf_frame_to_sample[("LIDAR_TOP", int(kf["lidar_ts"]))] = tok
            for ch, t in kf["cam_ts"].items():
                kf_frame_to_sample[(ch, int(t))] = tok

        for ch_name, frames in channel_frames.items():
            frame_ts_arr = np.array([f[0] for f in frames], dtype=np.int64)
            following_kf = assign_to_following_sample(frame_ts_arr, scene_kf_ts)

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

                if is_kf:
                    sample_token = kf_frame_to_sample[(ch_name, int(ts_ns))]
                else:
                    sample_token = scene_kf_tokens[following_kf[i]]

                # filename: samples/<channel>/<token>.<ext> or sweeps/<channel>/<token>.<ext>
                bucket = "samples" if is_kf else "sweeps"
                if ch_name == "LIDAR_TOP":
                    fname = f"{bucket}/LIDAR_TOP/{sd_token}.pcd.bin"
                    fileformat = "pcd"
                    height, width = 0, 0
                else:
                    fname = f"{bucket}/{ch_name}/{sd_token}.jpg"
                    fileformat = "jpg"
                    height, width = data.cam_size.get(ch_name, (0, 0))
                plan.append((int(ts_ns), ch_name, fname))

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

