#!/usr/bin/env python3
"""Draw a calibration's windshield field over one of the camera's images.

    python online_calib/plot_field.py CALIB.json --image FRAME.jpg --out FIELD.jpg [--range 10]

Arrows are the far-field pixel offsets S_inf (what the windshield adds beyond the
lens model for distant points) on a regular grid, magnified; their colour is the
magnitude in pixels. With --range, the near term S_near / range is added
for points at that distance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from camera_model import Grid


def field(c: dict, u: np.ndarray, v: np.ndarray, rng: float | None) -> np.ndarray:
    g = c["grid"]
    grid = Grid(c["width"], c["height"], g["gx"], g["gy"])
    G = np.array(g["G"])
    idx, w = grid.basis(torch.tensor(u, dtype=torch.float64), torch.tensor(v, dtype=torch.float64))
    idx, w = idx.numpy(), w.numpy()
    f = np.stack([(G[k][idx] * w).sum(1) for k in range(4)], 1)
    off = f[:, :2]
    if rng:
        off = off + f[:, 2:] / rng
    return off


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("calib", type=Path)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--range", type=float, default=None, help="add the near term for this distance (m)")
    p.add_argument("--gain", type=float, default=60.0, help="arrow magnification")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--step", type=int, default=90, help="arrow spacing in image pixels")
    a = p.parse_args()
    c = json.loads(a.calib.read_text())
    W, H = c["width"], c["height"]
    img = cv2.imread(str(a.image))
    s = a.width / W
    bg = cv2.cvtColor(cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    bg = (0.55 * bg + 0.45 * 255).astype(np.uint8)
    step = a.step
    u, v = np.meshgrid(np.arange(step / 2, W, step), np.arange(step / 2, H, step))
    u, v = u.ravel(), v.ravel()
    off = field(c, u, v, a.range)
    mag = np.hypot(off[:, 0], off[:, 1])
    vmax = max(3.0, float(np.percentile(mag, 99)))
    col = cv2.applyColorMap((40 + np.clip(215 * mag / vmax, 0, 215)).astype(np.uint8)[:, None], cv2.COLORMAP_PLASMA)[:, 0]
    for (x, y), (dx, dy), cc in zip(np.c_[u, v] * s, off * a.gain * s, col):
        cv2.circle(bg, (int(x), int(y)), 2, tuple(int(k) for k in cc), -1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(bg, (int(x), int(y)), (int(x + dx), int(y + dy)), tuple(int(k) for k in cc), 3,
                        line_type=cv2.LINE_AA, tipLength=0.3)
    cv2.imwrite(str(a.out), bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps({"median_px": float(np.median(mag)), "p95_px": float(np.percentile(mag, 95)),
                      "max_px": float(mag.max()), "colour_max_px": vmax}))


if __name__ == "__main__":
    main()
