"""Turn synchronized sensor timestamps into the 13 NuScenes JSON tables.

Knows nothing about rosbags or about where the sensor files live: the caller
hands it timestamps plus a calibration dict and gets back the tables and a
materialization plan of (timestamp_ns, channel, destination_relpath) that it is
free to satisfy however it likes.

Frame selection (plan_frames / partition_scenes):
  - LIDAR_TOP is the anchor. Every LiDAR frame must have a *complete camera
    set*: one capture instant at which every gating camera delivered a frame,
    all within --sync-ms of the LiDAR stamp. The cameras share a trigger, so a
    set is picked as a whole, never camera by camera.
  - A LiDAR frame without one, a missing LiDAR frame, or leaving the coverage
    window breaks the sequence. The unbroken runs are cut into scenes of exactly
    --scene-dur seconds; a remainder shorter than that is not used.
  - Samples (keyframes) are every --keyframe-stride-th LiDAR frame of a scene.
    Sweeps are every frame of every channel between a scene's first and last
    sample — all 30 fps camera frames, all 10 Hz LiDAR frames.

Conventions:
  - Calibration arrives in the OpenCV extrinsic convention
    (P_sensor = R @ P_ego + t); NuScenes wants the sensor pose in the ego frame,
    so it is inverted when writing calibrated_sensor.json.
  - Camera intrinsics are whatever the caller passes in `SensorData.intrinsic`:
    the pinhole K of the rectified images (see rectify.py).
  - Scene names are taken from the official nuScenes train/val lists, so every
    tool that splits by `nuscenes.utils.splits` works unchanged.
  - sample_annotation/instance are emitted empty on purpose: labels are produced
    externally. category/attribute carry the label taxonomy (from the perception
    stack's enums, see common.py) and visibility the four nuScenes bins with
    nuScenes' literal tokens "1".."4", so a vendor fills the two empty tables
    against a complete, standard-looking schema.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
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

# nuscenes-devkit's NuScenesCanBus refuses these scene numbers outright
# (can_bus_api.py, `can_blacklist`), so they are never handed out.
CAN_BUS_BLACKLIST = frozenset({161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172,
                               173, 174, 175, 176, 309, 310, 311, 312, 313, 314})

FILE_EXT = {"lidar": ".pcd.bin", "radar": ".pcd", "camera": ".jpg"}


def new_token() -> str:
    return uuid.uuid4().hex


@dataclass
class SensorData:
    """Everything the table builder needs, however the caller obtained it."""
    calib: dict                                # channel -> calibration (common.load_calib)
    frames: dict[str, np.ndarray]              # channel -> sorted ts_ns of the staged frames
    modality: dict[str, str]                   # channel -> "lidar" | "camera" | "radar"
    intrinsic: dict[str, list]                 # camera channel -> K for calibrated_sensor
    cam_size: dict[str, tuple[int, int]]       # camera channel -> (height, width)
    odom_ts: np.ndarray                        # sorted ts_ns
    odom_t: np.ndarray                         # (N, 3) translation
    odom_R: Rotation                           # rotation samples for SLERP
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


def required_streams(lidar_ts: np.ndarray, cam_ts: dict[str, np.ndarray],
                     gating: list[str], odom_ts: np.ndarray) -> dict[str, np.ndarray]:
    """The streams a frame cannot do without: LiDAR, the gating cameras, odom."""
    out: dict[str, np.ndarray] = {"LIDAR_TOP": lidar_ts}
    out.update({ch: cam_ts[ch] for ch in gating})
    out["ODOM"] = odom_ts
    return out


def coverage_window(streams: dict[str, np.ndarray], sync_ns: int) -> dict:
    """The interval in which every stream in `streams` has data, shrunk by sync_ns.

    Sensors start and stop at different times (up to ~1.4 s apart on the
    2026-08-19 bags). A frame outside this interval would have no image or
    no pose to attach; the margin exists because a camera frame matched to a
    LiDAR frame may sit up to sync_ns away from it and still needs a pose.
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


# ------------------------------------------------------------ frame selection
def camera_instants(cam_ts: dict[str, np.ndarray], channels: list[str], group_ns: int
                    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Group the cameras' frames into capture instants.

    The cameras share a trigger: frames taken together carry stamps within
    ~0.04 ms of each other, but never bit-identical ones. Stamps from all
    `channels` are merged and split wherever consecutive stamps are more than
    `group_ns` apart. Returns (instant_ns, members): instant_ns[i] is the
    earliest stamp of instant i and members[ch][i] is ch's stamp there, or -1
    where that camera dropped the frame.
    """
    parts = [np.asarray(cam_ts[ch], dtype=np.int64) for ch in channels]
    ts = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    who = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)]) if parts \
        else np.zeros(0, dtype=np.int64)
    order = np.argsort(ts, kind="stable")
    ts, who = ts[order], who[order]
    if not len(ts):
        return ts, {ch: ts.copy() for ch in channels}
    new_group = np.concatenate([[True], np.diff(ts) > group_ns])
    gid = np.cumsum(new_group) - 1
    instants = ts[new_group]
    members = {ch: np.full(len(instants), -1, dtype=np.int64) for ch in channels}
    for k, ch in enumerate(channels):
        sel = who == k
        members[ch][gid[sel]] = ts[sel]
    return instants, members


def plan_frames(lidar_ts: np.ndarray, cam_ts: dict[str, np.ndarray], gating: list[str],
                sync_ns: int, window: dict, group_ns: int, max_gap_ns: int) -> dict:
    """Decide which LiDAR frames the dataset can use, and with which camera set.

    For each LiDAR frame, of the camera instants whose every gating frame lies
    within sync_ns of the LiDAR stamp, the nearest *complete* one is taken — one
    where no gating camera dropped its frame. A frame without such an instant
    cannot be used; neither can one outside the coverage window. Unusable frames
    and LiDAR gaps longer than max_gap_ns cut the sequence into segments.

    Returns a dict:
      valid     bool[n_lidar]
      cam       {channel: int64[n_lidar]}   chosen camera stamps, -1 where invalid
      dev_ns    int64[n_lidar]              worst |camera - lidar| of the chosen set
      segments  [int array of lidar indices] unbroken runs of valid frames
      stats     JSON-friendly counters
    """
    L = np.asarray(lidar_ts, dtype=np.int64)
    n = len(L)
    instants, members = camera_instants(cam_ts, gating, group_ns)
    complete = (np.all(np.stack([members[ch] >= 0 for ch in gating]), axis=0)
                if len(instants) else np.zeros(0, dtype=bool))
    ci = np.flatnonzero(complete)
    C = instants[ci]

    chosen = np.full(n, -1, dtype=np.int64)
    best = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    if len(C):
        pos = np.searchsorted(C, L)
        for cand in (pos - 1, pos):   # the nearest complete instant on either side
            inside = (cand >= 0) & (cand < len(C))
            c = ci[np.clip(cand, 0, len(C) - 1)]
            dev = np.max(np.stack([np.abs(members[ch][c] - L) for ch in gating]), axis=0)
            better = inside & (dev <= sync_ns) & (dev < best)
            chosen[better] = c[better]
            best[better] = dev[better]

    in_window = (L >= window["start_ns"]) & (L <= window["end_ns"])
    has_set = chosen >= 0
    valid = in_window & has_set
    gap_before = np.concatenate([[False], np.diff(L) > max_gap_ns])

    idx = np.flatnonzero(valid)
    if len(idx):
        cuts = np.flatnonzero((np.diff(idx) > 1) | gap_before[idx[1:]]) + 1
        segments = np.split(idx, cuts)
    else:
        segments = []

    cam = {ch: np.where(valid, members[ch][np.clip(chosen, 0, None)], -1)
           if len(instants) else np.full(n, -1, dtype=np.int64) for ch in gating}

    # Why frames inside the window were lost: which cameras had dropped their
    # frame at the nearest instant (complete or not).
    blamed: Counter = Counter()
    lost = np.flatnonzero(in_window & ~has_set)
    if len(lost) and len(instants):
        near = np.clip(np.searchsorted(instants, L[lost]), 0, len(instants) - 1)
        left = np.clip(near - 1, 0, len(instants) - 1)
        pick = np.where(np.abs(instants[left] - L[lost]) <= np.abs(instants[near] - L[lost]),
                        left, near)
        for i in pick:
            for ch in gating:
                if members[ch][i] < 0:
                    blamed[ch] += 1
    dev_ms = best[valid] / 1e6
    stats = {
        "n_lidar": int(n),
        "n_in_window": int(in_window.sum()),
        "n_valid": int(valid.sum()),
        "n_no_camera_set": int((in_window & ~has_set).sum()),
        "n_lidar_gaps": int((gap_before & in_window).sum()),
        "n_segments": len(segments),
        "n_camera_instants": int(len(instants)),
        "n_complete_instants": int(len(ci)),
        "frames_lost_by_camera": dict(blamed),
        "sync_dev_ms": ({"p50": float(np.percentile(dev_ms, 50)),
                         "p99": float(np.percentile(dev_ms, 99)),
                         "max": float(dev_ms.max())} if len(dev_ms) else None),
        "sync_ns": int(sync_ns), "group_ns": int(group_ns), "max_gap_ns": int(max_gap_ns),
    }
    return {"valid": valid, "cam": cam, "dev_ns": best, "segments": segments, "stats": stats}


def format_plan(plan: dict) -> list[str]:
    s = plan["stats"]
    lines = [
        f"lidar frames: {s['n_lidar']}, in coverage window {s['n_in_window']}, "
        f"usable {s['n_valid']}",
        f"camera instants: {s['n_camera_instants']}, complete {s['n_complete_instants']} "
        f"({100 * s['n_complete_instants'] / max(s['n_camera_instants'], 1):.2f}%)",
        f"breaks: {s['n_no_camera_set']} frame(s) without a complete camera set within "
        f"{s['sync_ns'] / 1e6:g} ms, {s['n_lidar_gaps']} lidar gap(s) "
        f"-> {s['n_segments']} segment(s)",
    ]
    if s["frames_lost_by_camera"]:
        lines.append("  camera missing at the lost frames: " + ", ".join(
            f"{ch} {n}" for ch, n in sorted(s["frames_lost_by_camera"].items(),
                                              key=lambda kv: -kv[1])))
    if s["sync_dev_ms"]:
        d = s["sync_dev_ms"]
        lines.append(f"camera set vs lidar: p50 {d['p50']:.1f} ms, p99 {d['p99']:.1f} ms, "
                     f"max {d['max']:.1f} ms")
    return lines


def partition_scenes(segments: list[np.ndarray], frames_per_scene: int
                     ) -> tuple[list[tuple[int, np.ndarray]], dict]:
    """Cut every segment into consecutive scenes of exactly frames_per_scene frames.

    Returns ([(segment_index, lidar_indices)], stats). A segment's remainder
    shorter than a full scene is not used; stats says how much that was.
    """
    scenes: list[tuple[int, np.ndarray]] = []
    unused = 0
    for k, seg in enumerate(segments):
        n_full = len(seg) // frames_per_scene
        for j in range(n_full):
            scenes.append((k, seg[j * frames_per_scene:(j + 1) * frames_per_scene]))
        unused += len(seg) - n_full * frames_per_scene
    stats = {"n_scenes": len(scenes), "frames_per_scene": int(frames_per_scene),
             "frames_in_scenes": len(scenes) * frames_per_scene,
             "frames_unused": int(unused)}
    return scenes, stats


def official_scene_names(split: str, used: set[str], n: int) -> list[str]:
    """The next n unused names from the official nuScenes list for `split`.

    Our scenes are named after official ones so that anything splitting a
    nuScenes dataset by `nuscenes.utils.splits` — the devkit's detection and
    tracking evaluation, mmdetection3d / BEVFusion info generation — puts them
    in the intended split without modification. Numbers the CAN bus API
    refuses are skipped. The lists hold 700 train and 150 val names.
    """
    from nuscenes.utils.splits import create_splits_scenes  # local: optional dep elsewhere
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', not {split!r}")
    names = sorted(create_splits_scenes()[split])
    free = [s for s in names if s not in used and int(s[-4:]) not in CAN_BUS_BLACKLIST]
    if len(free) < n:
        raise SystemExit(f"the official '{split}' list has {len(free)} unused scene "
                         f"names left, {n} needed — the dataset is full for this split")
    return free[:n]


# ---------------------------------------------------------------- ego pose
def interp_pose(query_ns: np.ndarray, data: SensorData) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate ego pose at each query ns. Returns (translations, quaternions[wxyz])."""
    odom_ts = data.odom_ts
    odom_t = data.odom_t
    # Every frame must lie inside odom coverage; the coverage window guarantees
    # it for anything plan_frames lets through. Clipping here instead would
    # silently freeze the pose at the first/last odom sample.
    outside = (query_ns < odom_ts[0]) | (query_ns > odom_ts[-1])
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} frame timestamp(s) outside odom coverage "
            f"[{int(odom_ts[0])}, {int(odom_ts[-1])}] ns — frames were not "
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


# ------------------------------------------------------------------ tables
def build_tables(data: SensorData, scenes: list[dict], channels: list[str],
                 gating: list[str], kf_tol_ns: dict[str, int],
                 log_token: str, log_name: str, existing: dict | None = None
                 ) -> tuple[dict, list[tuple[int, str, str]], dict]:
    """Build all 13 NuScenes JSON tables.

    `scenes` are {"name", "description", "keyframes": [{"lidar_ts", "cam_ts"}]}
    where cam_ts holds the chosen frame of every gating camera. `channels` is
    every sample_data channel to emit, LIDAR_TOP first. Channels that are
    neither LIDAR_TOP nor gating are best-effort: a sample gets the frame
    nearest its LiDAR instant when that is within kf_tol_ns[channel], and no
    frame of that channel otherwise.

    If `existing` (loaded JSONs) is provided, reuse stable tokens (sensor,
    category, attribute, visibility) and return ONLY the new records — the
    caller merges them via merge_tables().

    Returns (tables, plan, attach) where plan is a list of
    (source_timestamp_ns, channel, destination_relpath) that the caller turns
    into real files, and attach counts best-effort keyframes per channel.
    """
    plan: list[tuple[int, str, str]] = []
    existing_sensor_tokens: dict[str, str] = {}
    if existing:
        for s in existing.get("sensor.json", []):
            existing_sensor_tokens[s["channel"]] = s["token"]
    # ---------- sensor.json (reuse existing tokens if any) ----------
    sensor_tokens: dict[str, str] = {}
    sensors = []
    for ch in channels:
        if ch in existing_sensor_tokens:
            sensor_tokens[ch] = existing_sensor_tokens[ch]
        else:
            tk = new_token()
            sensor_tokens[ch] = tk
            sensors.append({"token": tk, "channel": ch, "modality": data.modality[ch]})

    # ---------- calibrated_sensor.json (one per (sensor, log)) ----------
    cs_tokens: dict[str, str] = {}
    cs_records = []
    for ch in channels:
        params = data.calib[ch]
        rot_q, trans = opencv_ext_to_nuscenes_pose(
            quat_wxyz_to_R(params["rotation"]),
            np.asarray(params["translation"], dtype=np.float64))
        cs_tokens[ch] = new_token()
        cs_records.append({
            "token": cs_tokens[ch],
            "sensor_token": sensor_tokens[ch],
            "translation": trans,
            "rotation": rot_q,
            "camera_intrinsic": ([list(map(float, row)) for row in data.intrinsic[ch]]
                                 if data.modality[ch] == "camera" else []),
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
    # at a single shared placeholder created by the caller.
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
    attach: Counter = Counter()
    odom_lo, odom_hi = int(data.odom_ts[0]), int(data.odom_ts[-1])

    for scene in scenes:
        scene_kfs = scene["keyframes"]
        scene_token = new_token()
        scene_kf_ts = np.array([kf["lidar_ts"] for kf in scene_kfs], dtype=np.int64)
        scene_kf_tokens = [new_token() for _ in scene_kfs]
        scene_start_ns, scene_end_ns = scene_kf_ts[0], scene_kf_ts[-1]

        # ----- sample records (keyframes) -----
        for i, (kf, sample_token) in enumerate(zip(scene_kfs, scene_kf_tokens)):
            sample_records.append({
                "token": sample_token,
                "timestamp": kf["lidar_ts"] // 1000,  # ns -> us
                "scene_token": scene_token,
                "next": scene_kf_tokens[i + 1] if i + 1 < len(scene_kfs) else "",
                "prev": scene_kf_tokens[i - 1] if i > 0 else "",
            })

        for ch_name in channels:
            ts_all = data.frames[ch_name]
            # A keyframe frame belongs to the sample it was synchronized into:
            # LIDAR_TOP by its own timestamp, a gating camera by the set
            # plan_frames chose — it may sit a few ms after the lidar instant, so
            # it must not be re-assigned by time.
            if ch_name == "LIDAR_TOP":
                kf_map = {int(t): tok for t, tok in zip(scene_kf_ts, scene_kf_tokens)}
            elif ch_name in gating:
                kf_map = {int(kf["cam_ts"][ch_name]): tok
                          for kf, tok in zip(scene_kfs, scene_kf_tokens)}
            else:
                cand = ts_all[(ts_all >= odom_lo) & (ts_all <= odom_hi)]
                matched, diff = nearest_ts(scene_kf_ts, cand)
                kf_map = {int(m): tok for m, d, tok in zip(matched, diff, scene_kf_tokens)
                          if d <= kf_tol_ns[ch_name]}
                attach[ch_name] += len(kf_map)
            in_scene = ts_all[(ts_all >= scene_start_ns) & (ts_all <= scene_end_ns)]
            frame_ts_arr = np.array(sorted(set(in_scene.tolist()) | set(kf_map)),
                                    dtype=np.int64)
            # Sweeps attach to the following keyframe, as in nuScenes (see
            # assign_to_following_sample). Anchors are this scene's keyframes
            # only, so every sample_data stays in its scene.
            following_kf = assign_to_following_sample(frame_ts_arr, scene_kf_ts)

            # interp ego pose for every frame ts in this channel (one ego_pose per sample_data)
            trans, quats = interp_pose(frame_ts_arr, data)

            modality = data.modality[ch_name]
            ext = FILE_EXT[modality]
            if modality == "camera":
                height, width = data.cam_size[ch_name]
                fileformat = "jpg"
            else:
                height, width = 0, 0
                fileformat = "pcd"

            prev_sd: dict | None = None
            for i, ts_ns in enumerate(frame_ts_arr.tolist()):
                ego_token = new_token()
                ego_pose_records.append({
                    "token": ego_token,
                    "translation": trans[i].tolist(),
                    "rotation": [float(x) for x in quats[i]],
                    "timestamp": ts_ns // 1000,
                })
                is_kf = ts_ns in kf_map
                sample_token = kf_map[ts_ns] if is_kf else scene_kf_tokens[following_kf[i]]
                sd_token = new_token()
                bucket = "samples" if is_kf else "sweeps"
                fname = f"{bucket}/{ch_name}/{sd_token}{ext}"
                plan.append((ts_ns, ch_name, fname))
                sd = {
                    "token": sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_token,
                    "calibrated_sensor_token": cs_tokens[ch_name],
                    "filename": fname,
                    "fileformat": fileformat,
                    "is_key_frame": is_kf,
                    "height": int(height),
                    "width": int(width),
                    "timestamp": ts_ns // 1000,
                    "next": "",
                    "prev": prev_sd["token"] if prev_sd else "",
                }
                if prev_sd:
                    prev_sd["next"] = sd_token
                sample_data_records.append(sd)
                prev_sd = sd

        # ----- scene record -----
        scene_records.append({
            "token": scene_token,
            "name": scene["name"],
            "description": scene["description"],
            "log_token": log_token,
            "nbr_samples": len(scene_kfs),
            "first_sample_token": scene_kf_tokens[0],
            "last_sample_token": scene_kf_tokens[-1],
        })

    tables = {
        "sensor.json": sensors,
        "calibrated_sensor.json": cs_records,
        "log.json": [log_record],
        "scene.json": scene_records,
        "sample.json": sample_records,
        "sample_data.json": sample_data_records,
        "ego_pose.json": ego_pose_records,
        "sample_annotation.json": [],
        "instance.json": [],
        "category.json": categories,
        "attribute.json": attributes,
        "visibility.json": visibilities,
        "map.json": maps,
    }
    return tables, plan, dict(attach)


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
    """Write every table to a temporary file first, then swap them all in.

    In append mode these files hold every earlier import too; a run stopped
    halfway through writing them must not leave tables that disagree.
    """
    json_dir = out_root / version
    json_dir.mkdir(parents=True, exist_ok=True)
    tmp = {name: json_dir / f".{name}.tmp" for name in tables}
    for name, records in tables.items():
        tmp[name].write_text(json.dumps(records, indent=2))
    for name in tables:
        tmp[name].replace(json_dir / name)
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

