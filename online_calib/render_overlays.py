#!/usr/bin/env python3
"""LiDAR points projected into camera images, one or more calibrations side by side.

    python online_calib/render_overlays.py --calib new=final_CAM_FRONT.json old=old_CAM_FRONT.json \
        --clip CLIP --ins INS.npz --channel CAM_FRONT --frames 2 --out DIR [--mask MASK.png]

For each picked frame (the car moving, spread over the clip) and calibration:
  <stem>_<label>.jpg          the whole image, points coloured by range
  <stem>_c<k>_<label>.jpg     2x crops, chosen once (with the first calibration)
                              so every label shows the same pixels: near objects
                              against a far background at the image centre and at
                              its border (points by range), and road markings (only
                              the most reflective ground points, drawn in magenta,
                              which must land on the paint; picked once with the
                              first calibration, so each label projects the same
                              LiDAR points)
Points are one deskewed sweep placed in the world with the INS and projected at
each pixel row's exposure time. DIR/index.json lists the files and crop boxes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate import Calib, frame_points, zbuffer_visible
from traj import Trajectory

CROP_W, CROP_H, ZOOM = 360, 220, 2


def project(cal: Calib, clip: Path, traj: Trajectory, t_img: int, n_sweeps: int):
    P, E, I, _ = frame_points(clip, t_img, traj, cal.T_el, n_sweeps)
    uv, Xc = cal.project_world(P, t_img, traj)
    dist = np.linalg.norm(Xc, axis=1)
    vis = zbuffer_visible(uv, dist, cal.W, cal.H) & (dist < 70)
    # road surface: the ego frame's origin sits ~0.3 m above it
    ground = (P - traj.p(t_img)[0]) @ traj.R(t_img)[0]
    ground = ground[:, 2] < -0.15
    return uv, dist, vis, E, I, ground


def colours(rng: np.ndarray, max_range: float) -> np.ndarray:
    v = np.clip(255 * (1 - np.sqrt(rng / max_range)), 0, 255).astype(np.uint8)[:, None]
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[:, 0]


def draw(img: np.ndarray, uv: np.ndarray, rng: np.ndarray, radius: int, max_range: float) -> np.ndarray:
    out = img.copy()
    o = np.argsort(-rng)
    for (u, v), c in zip(np.round(uv[o]).astype(int), colours(rng[o], max_range)):
        cv2.circle(out, (int(u), int(v)), radius, tuple(int(x) for x in c), -1, lineType=cv2.LINE_AA)
    return out


def pick_crops(uv, dist, vis, E, W, H, mask) -> list[tuple[int, int]]:
    """Top-left corners of a centre and a border window where near objects (poles,
    trunks, cars) stand in front of a far background: their outlines show a
    misalignment best."""
    inside = vis & np.isfinite(uv).all(1)
    u_all, v_all, d_all = uv[inside, 0], uv[inside, 1], dist[inside]
    edge = (E[inside] & (d_all < 18)).astype(float)
    far = (d_all > 25).astype(float)
    half = np.hypot(W, H) / 2
    best = {"centre": (-1, None), "border": (-1, None)}
    for x0 in range(0, W - CROP_W + 1, 40):
        for y0 in range(0, H - CROP_H + 1, 40):
            if mask is not None and np.mean(mask[y0:y0 + CROP_H, x0:x0 + CROP_W] > 0) < 0.9:
                continue
            r = np.hypot(x0 + CROP_W / 2 - W / 2, y0 + CROP_H / 2 - H / 2) / half
            zone = "centre" if r < 0.35 else "border" if r > 0.55 else None
            if zone is None:
                continue
            w = (u_all >= x0) & (u_all < x0 + CROP_W) & (v_all >= y0) & (v_all < y0 + CROP_H)
            if w.sum() < 200:
                continue
            n = edge[w].sum() * min(far[w].mean(), 0.5)
            if n > best[zone][0]:
                best[zone] = (n, (x0, y0))
    return [b[1] for b in best.values() if b[1] is not None]


def marking_points(I, dist, vis, ground) -> np.ndarray:
    cand = vis & ground & (dist < 30)
    if cand.sum() < 100:
        return np.zeros(len(I), bool)
    return cand & (I > np.percentile(I[cand], 92))


def pick_marking_crop(uv, hot, W, H, mask) -> tuple[int, int] | None:
    u, v = uv[hot, 0], uv[hot, 1]
    best = (0, None)
    for x0 in range(0, W - CROP_W + 1, 40):
        for y0 in range(H // 3, H - CROP_H + 1, 40):
            if mask is not None and np.mean(mask[y0:y0 + CROP_H, x0:x0 + CROP_W] > 0) < 0.9:
                continue
            n = int(np.sum((u >= x0) & (u < x0 + CROP_W) & (v >= y0) & (v < y0 + CROP_H)))
            if n > best[0]:
                best = (n, (x0, y0))
    return best[1] if best[0] >= 30 else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calib", nargs="+", required=True, help="LABEL=CALIB.json ...")
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--channel", required=True)
    p.add_argument("--frames", type=int, default=2)
    p.add_argument("--sweeps", type=int, default=1)
    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--width", type=int, default=1280, help="width of the saved full images")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(a.ins)
    cals = [(s.split("=", 1)[0], Calib(Path(s.split("=", 1)[1]))) for s in a.calib]
    mask = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE) if a.mask else None

    files = sorted((a.clip / "cam" / a.channel).glob("*.jpg"))
    files = [f for f in files if traj.covers(int(f.stem))[0]]
    moving = [f for f in files if traj.speed(int(f.stem))[0] > 2.0] or files
    step = len(moving) / a.frames
    pick = [moving[int((k + 0.5) * step)] for k in range(a.frames)]
    idx_path = a.out / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    short = a.clip.name.split("_")[0]
    for f in pick:
        t_img = int(f.stem)
        img = cv2.imread(str(f))
        stem = f"{a.channel}_{short}_{t_img}"
        entry = {"channel": a.channel, "clip": a.clip.name, "t_ns": t_img,
                 "speed_mps": float(traj.speed(t_img)[0]), "full": {}, "crops": []}
        crops = None
        for label, cal in cals:
            uv, dist, vis, E, I, ground = project(cal, a.clip, traj, t_img, a.sweeps)
            if crops is None:                     # the same LiDAR points for every calibration
                hot0 = marking_points(I, dist, vis, ground)
            hot = hot0 & np.isfinite(uv).all(1)
            if crops is None:
                crops = [(*c, "range") for c in pick_crops(uv, dist, vis, E, cal.W, cal.H, mask)]
                m = pick_marking_crop(uv, hot, cal.W, cal.H, mask)
                crops += [(*m, "markings")] if m else []
                entry["crops"] = [{"box": [x0, y0, CROP_W, CROP_H], "kind": kind, "files": {}}
                                  for x0, y0, kind in crops]
            s = a.width / cal.W
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            full = draw(small, uv[vis] * s, dist[vis], 1, 50.0)
            name = f"{stem}_{label}.jpg"
            cv2.imwrite(str(a.out / name), full, [cv2.IMWRITE_JPEG_QUALITY, 85])
            entry["full"][label] = name
            for k, (x0, y0, kind) in enumerate(crops):
                roi = cv2.resize(img[y0:y0 + CROP_H, x0:x0 + CROP_W], None, fx=ZOOM, fy=ZOOM,
                                 interpolation=cv2.INTER_CUBIC)
                inb = vis & (uv[:, 0] >= x0 - 2) & (uv[:, 0] < x0 + CROP_W + 2) & \
                    (uv[:, 1] >= y0 - 2) & (uv[:, 1] < y0 + CROP_H + 2)
                if kind == "markings":
                    c = roi.copy()
                    for u, v in np.round((uv[inb & hot] - [x0, y0]) * ZOOM + (ZOOM - 1) / 2).astype(int):
                        cv2.circle(c, (int(u), int(v)), 2, (255, 0, 255), -1, lineType=cv2.LINE_AA)
                else:
                    mr = float(np.percentile(dist[inb], 90)) if inb.sum() > 20 else 50.0
                    c = draw(roi, (uv[inb] - [x0, y0]) * ZOOM + (ZOOM - 1) / 2, dist[inb], 2, mr)
                name = f"{stem}_c{k}_{label}.jpg"
                cv2.imwrite(str(a.out / name), c, [cv2.IMWRITE_JPEG_QUALITY, 88])
                entry["crops"][k]["files"][label] = name
        index = [e for e in index if not (e["channel"] == a.channel and e["t_ns"] == t_img)] + [entry]
        print(f"{stem}: {len(entry['crops'])} crops")
    idx_path.write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
