#!/usr/bin/env python3
"""Per-camera mask of the ego vehicle's own body (hood, trunk, mirrors).

Pixels that barely change across frames while the car drives belong to the
car; everything the calibration tracks or matches must stay out of them.
    python online_calib/masks.py CLIP --ins INS.npz --out DIR
writes DIR/<CHANNEL>.png (255 = usable, 0 = ego body / never usable).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj import Trajectory  # noqa: E402


def body_mask(files: list[Path], traj: Trajectory, n: int = 60) -> np.ndarray:
    moving = [f for f in files if traj.speed(int(f.stem))[0] > 3.0]
    pick = moving[:: max(1, len(moving) // n)][:n]
    g = np.stack([cv2.GaussianBlur(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), (0, 0), 3).astype(np.float32)
                  for f in pick])
    change = np.median(np.abs(np.diff(g, axis=0)), axis=0)
    # body ~< 4 grey levels (its paint mirrors the scene a little); textureless
    # road ~10-16. Pixels that are always dark lie outside the lens image circle.
    static = (change < 5.0).astype(np.uint8)
    dark = (np.median(g, axis=0) < 12).astype(np.uint8)
    # keep only large static regions (the body), not flat sky patches
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN, k)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(static)
    h, w = static.shape
    body = np.zeros_like(static)
    for i in range(1, nlab):
        x, y, ww, hh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + ww == w or y + hh == h
        if area > 0.01 * h * w and touches_border and y + hh > 0.5 * h:
            body[lab == i] = 1
    body |= dark
    body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return np.where(body > 0, 0, 255).astype(np.uint8)


def static_feature_mask(files: list[Path], traj: Trajectory, shape: tuple[int, int],
                        n_starts: int = 12, span: int = 15) -> np.ndarray:
    """Where corners stay put for `span` frames while the car drives: its body.

    Robust to glossy paint, which fools the intensity-change test: the body's
    outline and fittings are fixed in the image whatever they reflect.
    """
    h, w = shape
    votes = np.zeros((h, w), np.float32)
    moving = [i for i, f in enumerate(files) if traj.speed(int(f.stem))[0] > 3.0]
    starts = [i for i in moving[:: max(1, len(moving) // n_starts)] if i + span < len(files)][:n_starts]
    for i in starts:
        prev = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        pts = cv2.goodFeaturesToTrack(prev, 3000, 0.01, 7)
        if pts is None:
            continue
        p0 = pts.copy()
        alive = np.ones(len(pts), bool)
        for j in range(i + 1, i + span + 1):
            cur = cv2.imread(str(files[j]), cv2.IMREAD_GRAYSCALE)
            pts, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            alive &= st.ravel() == 1
            prev = cur
        disp = np.linalg.norm((pts - p0).reshape(-1, 2), axis=1)
        hit = np.zeros((h, w), np.float32)      # one vote per start, not per corner
        for x, y in p0.reshape(-1, 2)[alive & (disp < 2.0)]:
            cv2.circle(hit, (int(x), int(y)), 20, 1.0, -1)
        votes += hit
    body = (votes >= max(3, len(starts) // 3)).astype(np.uint8)
    return cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("clip", type=Path)
    p.add_argument("--ins", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    traj = Trajectory(a.ins)
    a.out.mkdir(parents=True, exist_ok=True)
    for cam in sorted((a.clip / "cam").iterdir()):
        files = sorted(cam.glob("*.jpg"))
        m = body_mask(files, traj)
        body = static_feature_mask(files, traj, m.shape)
        body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        # the body continues to the bottom of the image below any part of it
        body = np.maximum.accumulate(body, axis=0)
        m[body > 0] = 0
        cv2.imwrite(str(a.out / f"{cam.name}.png"), m)
        print(cam.name, f"masked {100 * np.mean(m == 0):.1f}%")


if __name__ == "__main__":
    main()
