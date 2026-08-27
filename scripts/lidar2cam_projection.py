"""Project lidar points onto camera images for extrinsic sanity check.

Picks one lidar frame, finds nearest camera frame per channel, projects
lidar points to image, overlays colored by depth.

Calibration convention assumed (OpenCV-style extrinsic):
    P_cam = R_cam_ego @ P_ego + t_cam_ego
    P_ego = R_ego_lidar @ P_lidar + t_lidar_ego_origin_in_lidar
That is, calib stores rotation that rotates ego→sensor frame, and
translation = ego origin expressed in the sensor frame. If projections look
flipped/rotated, the convention may be inverse — try --invert-extrinsic.

Usage:
    python scripts/lidar2cam_projection.py /data/intermediate/<bag>
    python scripts/lidar2cam_projection.py <int> --frame mid --max-depth 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import quat_wxyz_to_R  # noqa: E402


def project_to_image(P_cam, K, dist, model):
    """P_cam: (N, 3) points in camera frame, z>0 already filtered."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    if model == "fisheye":
        D = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        pts = P_cam.reshape(-1, 1, 3).astype(np.float64)
        img_pts, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K, D)
    elif model == "pinhole":
        D = np.asarray(dist, dtype=np.float64).reshape(-1)
        pts = P_cam.astype(np.float64)
        img_pts, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    else:
        proj = (K @ P_cam.T).T
        img_pts = (proj[:, :2] / proj[:, 2:3]).reshape(-1, 1, 2)
    return img_pts.reshape(-1, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("intermediate", type=Path,
                   help="Stage-1 intermediate dir, e.g. /data/intermediate/<bag>")
    p.add_argument("--out", type=Path, default=Path("viz_projection"),
                   help="Output dir (default: viz_projection/).")
    p.add_argument("--frame", default="mid",
                   help="first | mid | last | <integer index>")
    p.add_argument("--max-depth", type=float, default=80.0,
                   help="Max depth (m) for color scale + filtering.")
    p.add_argument("--point-size", type=int, default=2,
                   help="Radius (px) of overlay dots.")
    p.add_argument("--invert-extrinsic", action="store_true",
                   help="Use opposite calib convention (sensor->ego instead of ego->sensor).")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    calib = json.loads((args.intermediate / "calib.json").read_text())
    lidar = calib["LIDAR_TOP"]
    R_lidar = quat_wxyz_to_R(lidar["rotation"])
    t_lidar = np.array(lidar["translation"], dtype=np.float64)

    files = sorted((args.intermediate / "lidar").glob("*.bin.zst"))
    if not files:
        raise SystemExit(f"no lidar in {args.intermediate}/lidar")

    if args.frame == "first":
        idx = 0
    elif args.frame == "last":
        idx = len(files) - 1
    elif args.frame == "mid":
        idx = len(files) // 2
    else:
        idx = int(args.frame)
    f = files[idx]
    lidar_ts = int(f.name.split(".", 1)[0])
    print(f"lidar frame {idx}/{len(files)}: ts={lidar_ts}")

    arr = np.frombuffer(zstd.ZstdDecompressor().decompress(f.read_bytes()),
                        dtype=np.float32).reshape(-1, 5)
    P_lidar = arr[:, :3].astype(np.float64)
    print(f"  n_points: {len(P_lidar)}")

    # Build LUT for jet colormap (BGR)
    cmap = cv2.applyColorMap(np.arange(256).reshape(1, -1).astype(np.uint8),
                             cv2.COLORMAP_JET).reshape(-1, 3)  # (256, 3) BGR

    cam_root = args.intermediate / "cameras"
    for cam_name, params in calib.items():
        if cam_name == "LIDAR_TOP":
            continue
        cam_dir = cam_root / cam_name
        if not cam_dir.exists():
            print(f"  [skip] {cam_name}: no cam dir")
            continue
        cam_ts = sorted(int(p.stem) for p in cam_dir.glob("*.jpg"))
        if not cam_ts:
            print(f"  [skip] {cam_name}: no images")
            continue
        cam_ts_arr = np.array(cam_ts)
        nearest_idx = int(np.argmin(np.abs(cam_ts_arr - lidar_ts)))
        nearest_ts = cam_ts[nearest_idx]
        diff_ms = abs(nearest_ts - lidar_ts) / 1e6
        cam_path = cam_dir / f"{nearest_ts}.jpg"

        img = cv2.imread(str(cam_path))
        if img is None:
            print(f"  [skip] {cam_name}: cv2.imread failed")
            continue
        h, w = img.shape[:2]

        R_cam = quat_wxyz_to_R(params["rotation"])
        t_cam = np.array(params["translation"], dtype=np.float64)

        if args.invert_extrinsic:
            # Treat calib as sensor->ego: P_ego = R @ P_sensor + t
            R_cam_ego = R_cam.T
            t_cam_ego = -R_cam.T @ t_cam
            R_lidar_ego = R_lidar.T
            t_lidar_ego = -R_lidar.T @ t_lidar
        else:
            R_cam_ego = R_cam
            t_cam_ego = t_cam
            R_lidar_ego = R_lidar
            t_lidar_ego = t_lidar

        # Compose T_cam_lidar = T_cam_ego @ inv(T_lidar_ego)
        # P_ego = R_lidar_ego.T @ (P_lidar - t_lidar_ego)  [for OpenCV ext convention]
        # P_cam = R_cam_ego @ P_ego + t_cam_ego
        P_ego = (R_lidar_ego.T @ (P_lidar - t_lidar_ego).T).T
        P_cam = (R_cam_ego @ P_ego.T).T + t_cam_ego

        in_front = P_cam[:, 2] > 0.1
        P_cam_v = P_cam[in_front]
        if len(P_cam_v) == 0:
            print(f"  {cam_name}: 0 points in front of camera — extrinsic likely wrong")
            continue

        img_pts = project_to_image(P_cam_v, params["intrinsic"], params["distortion"], params["model"])
        depths = P_cam_v[:, 2]

        u, v = img_pts[:, 0], img_pts[:, 1]
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths > 0) & (depths < args.max_depth)
        u_v = u[valid].astype(np.int32)
        v_v = v[valid].astype(np.int32)
        d_v = depths[valid]

        depth_norm = np.clip(d_v / args.max_depth, 0, 1)
        colors = cmap[(depth_norm * 255).astype(np.int32)]

        out_img = img.copy()
        for ui, vi, c in zip(u_v, v_v, colors):
            cv2.circle(out_img, (int(ui), int(vi)), args.point_size,
                       (int(c[0]), int(c[1]), int(c[2])), -1)

        info = f"{cam_name} | model={params['model']} | sync_diff={diff_ms:.1f}ms | proj={len(u_v)}/{len(P_lidar)}"
        cv2.putText(out_img, info, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out_img, info, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        out_path = args.out / f"{cam_name}_proj.jpg"
        cv2.imwrite(str(out_path), out_img)
        print(f"  {cam_name:18}  sync={diff_ms:5.1f}ms  proj={len(u_v):6d}/{len(P_lidar)}  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
