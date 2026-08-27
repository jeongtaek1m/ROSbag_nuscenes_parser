"""Visualize one NuScenes scene in rerun.io.

Logs ego pose, 7 cameras (with intrinsics), and 3D lidar over a timeline.
Lidar is logged in TWO variants which can be toggled in the entity tree:

  world/ego/LIDAR_TOP/intensity    — all points colored by intensity (default)
  world/ego/LIDAR_TOP/cam_colored  — points colored by the camera that sees
                                     them (positive depth + within image
                                     bounds). Points with negative depth in
                                     ALL cameras are NOT logged here.

Usage:
    pip install rerun-sdk
    python scripts/rerun_viz.py /data/tcar_nuscenes
    python scripts/rerun_viz.py /data/tcar_nuscenes --scene scene-0010
    python scripts/rerun_viz.py /data/tcar_nuscenes --scene 0 --max-samples 5 \
        --save /tmp/preview.rrd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_calib  # noqa: E402


CAM_PRIORITY = [  # which camera "wins" if multiple see the same lidar point
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    "CAM_TRAFFIC",
]


def quat_wxyz_to_R(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    if intensity.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    norm = np.clip(intensity / max(intensity.max(), 1e-3), 0, 1)
    lut = cv2.applyColorMap(
        np.arange(256).reshape(1, -1).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    ).reshape(-1, 3)  # BGR
    return lut[(norm * 255).astype(np.int32)][:, ::-1].astype(np.uint8)  # -> RGB


def find_snapshot_calib(calib_dir: Path | None) -> dict | None:
    """Load the calibration snapshot, for the distortion NuScenes cannot store."""
    if calib_dir is None or not calib_dir.exists():
        return None
    return load_calib(calib_dir)


def colorize_lidar_from_cams(
    P_lidar: np.ndarray,            # (N, 3) in lidar frame
    cam_imgs: dict[str, np.ndarray],          # ch -> (H, W, 3) RGB
    cam_K: dict[str, np.ndarray],             # ch -> (3, 3)
    cam_dist: dict[str, np.ndarray | None],   # ch -> distortion or None
    cam_model: dict[str, str],                # ch -> 'pinhole'|'fisheye'|'plain'
    cam_R_ego: dict[str, np.ndarray],         # ch -> R_ego_cam (3, 3)
    cam_t_ego: dict[str, np.ndarray],         # ch -> t_ego_cam (3,)
    R_ego_lidar: np.ndarray,
    t_ego_lidar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (points, colors uint8) for points seen by SOME camera with depth>0."""
    N = P_lidar.shape[0]
    if N == 0:
        return P_lidar, np.zeros((0, 3), dtype=np.uint8)

    pts_lidar_3xN = P_lidar.T.astype(np.float64)
    pts_ego = R_ego_lidar @ pts_lidar_3xN + t_ego_lidar.reshape(3, 1)

    colors = np.zeros((N, 3), dtype=np.uint8)
    colored = np.zeros(N, dtype=bool)

    for ch in CAM_PRIORITY:
        if ch not in cam_imgs:
            continue
        H, W = cam_imgs[ch].shape[:2]
        # ego -> cam
        pts_cam = cam_R_ego[ch].T @ (pts_ego - cam_t_ego[ch].reshape(3, 1))
        depth = pts_cam[2]
        eligible = (depth > 0.1) & (~colored)
        if not eligible.any():
            continue
        idxs = np.where(eligible)[0]
        sub = pts_cam[:, idxs].T  # (M, 3)

        K = cam_K[ch]
        D = cam_dist[ch]
        model = cam_model[ch]
        if model == "fisheye" and D is not None:
            uv, _ = cv2.fisheye.projectPoints(
                sub.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                K, D.reshape(-1, 1),
            )
            uv = uv.reshape(-1, 2)
        elif model == "pinhole" and D is not None:
            uv, _ = cv2.projectPoints(
                sub, np.zeros(3), np.zeros(3), K, D.reshape(-1),
            )
            uv = uv.reshape(-1, 2)
        else:
            proj = K @ sub.T
            uv = (proj[:2] / proj[2:3]).T

        u, v = uv[:, 0], uv[:, 1]
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        good = idxs[in_img]
        u_g = u[in_img].astype(np.int32)
        v_g = v[in_img].astype(np.int32)
        colors[good] = cam_imgs[ch][v_g, u_g]
        colored[good] = True

    return P_lidar[colored], colors[colored]


def log_sample(nusc: NuScenes, dataroot: Path, sample: dict,
               snapshot_calib: dict | None) -> None:
    rr.set_time("ts", duration=sample["timestamp"] / 1e6)

    # Ego pose (world -> ego)
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    qw, qx, qy, qz = ego["rotation"]
    rr.log(
        "world/ego",
        rr.Transform3D(
            translation=ego["translation"],
            rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
        ),
    )

    # Sensor transforms + cameras + lidar
    cam_imgs: dict[str, np.ndarray] = {}
    cam_K: dict[str, np.ndarray] = {}
    cam_dist: dict[str, np.ndarray | None] = {}
    cam_model: dict[str, str] = {}
    cam_R_ego: dict[str, np.ndarray] = {}
    cam_t_ego: dict[str, np.ndarray] = {}
    R_ego_lidar = None
    t_ego_lidar = None
    pc = None

    for ch, sd_token in sample["data"].items():
        sd = nusc.get("sample_data", sd_token)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        qw, qx, qy, qz = cs["rotation"]
        ent = f"world/ego/{ch}"
        rr.log(
            ent,
            rr.Transform3D(
                translation=cs["translation"],
                rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw]),
            ),
        )

        if sd["sensor_modality"] == "camera":
            K = np.asarray(cs["camera_intrinsic"], dtype=np.float64).reshape(3, 3)
            rr.log(
                f"{ent}/image",
                rr.Pinhole(image_from_camera=K,
                           resolution=[int(sd["width"]), int(sd["height"])]),
            )
            img = cv2.imread(str(dataroot / sd["filename"]))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rr.log(f"{ent}/image", rr.Image(img_rgb))

            cam_imgs[ch] = img_rgb
            cam_K[ch] = K
            cam_R_ego[ch] = quat_wxyz_to_R(cs["rotation"])
            cam_t_ego[ch] = np.asarray(cs["translation"], dtype=np.float64)
            # NuScenes has no distortion field; take it from the snapshot
            if snapshot_calib and ch in snapshot_calib:
                p = snapshot_calib[ch]
                cam_dist[ch] = np.asarray(p["distortion"], dtype=np.float64) if p.get("distortion") else None
                cam_model[ch] = p.get("model", "pinhole")
            else:
                cam_dist[ch] = None
                cam_model[ch] = "plain"

        elif sd["sensor_modality"] == "lidar":
            pc = LidarPointCloud.from_file(str(dataroot / sd["filename"]))
            R_ego_lidar = quat_wxyz_to_R(cs["rotation"])
            t_ego_lidar = np.asarray(cs["translation"], dtype=np.float64)
            P_lidar = pc.points[:3].T  # (N, 3)
            rr.log(
                f"{ent}/intensity",
                rr.Points3D(P_lidar, colors=intensity_colors(pc.points[3]), radii=0.04),
            )

    # Camera-colored variant — overlay on the same parent lidar entity tree
    if pc is not None and cam_imgs:
        kept_pts, kept_colors = colorize_lidar_from_cams(
            pc.points[:3].T, cam_imgs, cam_K, cam_dist, cam_model,
            cam_R_ego, cam_t_ego, R_ego_lidar, t_ego_lidar,
        )
        rr.log(
            "world/ego/LIDAR_TOP/cam_colored",
            rr.Points3D(kept_pts, colors=kept_colors, radii=0.05),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataroot", type=Path, help="NuScenes dataroot")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--scene", default="0",
                   help="Scene index (int) or scene name (e.g. scene-0001).")
    p.add_argument("--calib", type=Path, default=None,
                   help="Calibration snapshot, for the distortion NuScenes cannot "
                        "store. Without it, projection ignores distortion.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit number of samples (for quick preview).")
    p.add_argument("--save", type=Path, default=None,
                   help="Save .rrd file instead of spawning viewer.")
    args = p.parse_args()

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    print(f"Loaded {len(nusc.scene)} scenes from {args.dataroot}")

    if args.scene.isdigit():
        scene = nusc.scene[int(args.scene)]
    else:
        matches = [s for s in nusc.scene if s["name"] == args.scene]
        if not matches:
            raise SystemExit(f"scene '{args.scene}' not found")
        scene = matches[0]
    print(f"Scene: {scene['name']}  ({scene['nbr_samples']} samples)")

    log_name = nusc.get("log", scene["log_token"])["logfile"]
    snapshot_calib = find_snapshot_calib(args.calib)
    if snapshot_calib is None:
        print("  [warn] no calibration snapshot — projecting without distortion "
              "(cam_colored may misalign at image edges, esp. fisheye)")

    if args.save:
        rr.init("tcar_nuscenes", spawn=False)
        rr.save(str(args.save))
    else:
        rr.init("tcar_nuscenes", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    tok = scene["first_sample_token"]
    n = 0
    while tok:
        s = nusc.get("sample", tok)
        log_sample(nusc, args.dataroot, s, snapshot_calib)
        n += 1
        if args.max_samples and n >= args.max_samples:
            break
        tok = s["next"]

    print(f"Logged {n} samples.")
    if args.save:
        print(f"Saved rrd: {args.save}")


if __name__ == "__main__":
    main()
