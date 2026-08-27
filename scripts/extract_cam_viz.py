"""Extract one frame from each /camera_N/compressed topic at the bag midpoint.

Useful for figuring out which physical camera maps to which nuscenes channel.
Also saves a combined `grid.jpg` arranged as the standard 6 nuscenes channels
in a 2×3 grid + CAM_TRAFFIC as a 7th tile centered below.

Usage:
    python scripts/extract_cam_viz.py /path/to/bag.bag --out viz_cameras
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


# Topic → (grid row, col) for the 3×3 layout.
# Row 0 = front row, Row 1 = back row, Row 2 = traffic (col 1 only).
GRID_LAYOUT = {
    "/camera_6/compressed": ((0, 0), "CAM_FRONT_LEFT"),
    "/camera_4/compressed": ((0, 1), "CAM_FRONT"),
    "/camera_1/compressed": ((0, 2), "CAM_FRONT_RIGHT"),
    "/camera_5/compressed": ((1, 0), "CAM_BACK_LEFT"),
    "/camera_2/compressed": ((1, 1), "CAM_BACK"),
    "/camera_0/compressed": ((1, 2), "CAM_BACK_RIGHT"),
    "/camera_3/compressed": ((2, 1), "CAM_TRAFFIC"),
}
CELL_W, CELL_H = 640, 360  # 16:9 tiles


def _put_label(img, text):
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def build_grid(out_dir: Path) -> Path:
    grid = np.zeros((CELL_H * 3, CELL_W * 3, 3), dtype=np.uint8)
    for topic, ((row, col), ch_name) in GRID_LAYOUT.items():
        idx = topic.split("_")[1].split("/")[0]
        src = out_dir / f"cam_{idx}.jpg"
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        img = cv2.resize(img, (CELL_W, CELL_H))
        _put_label(img, ch_name)
        y0, y1 = row * CELL_H, (row + 1) * CELL_H
        x0, x1 = col * CELL_W, (col + 1) * CELL_W
        grid[y0:y1, x0:x1] = img
    out = out_dir / "grid.jpg"
    cv2.imwrite(str(out), grid)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bag", type=Path, help="Path to a .bag file.")
    p.add_argument("--out", type=Path, default=Path("viz_cameras"),
                   help="Output dir (default: viz_cameras/).")
    p.add_argument("--n-cams", type=int, default=7,
                   help="Number of /camera_0../camera_{N-1} topics to try.")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with AnyReader([args.bag]) as reader:
        mid = (reader.start_time + reader.end_time) // 2
        print(f"bag mid (ns): {mid}")

        cam_topics = [f"/camera_{i}/compressed" for i in range(args.n_cams)]
        conns = [c for c in reader.connections if c.topic in cam_topics]
        if not conns:
            raise SystemExit(f"No /camera_*/compressed topics found in {args.bag}")

        found = {c.topic: None for c in conns}
        # rosbags AnyReader.messages supports start= for fast skip past most of the bag
        for conn, ts, raw in reader.messages(connections=conns, start=mid):
            if found[conn.topic] is not None:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            idx = conn.topic.split("_")[1].split("/")[0]
            out_path = args.out / f"cam_{idx}.jpg"
            out_path.write_bytes(bytes(msg.data))
            found[conn.topic] = ts
            print(f"  cam_{idx}: ts={ts}  -> {out_path}  ({len(msg.data)} bytes)")
            if all(v is not None for v in found.values()):
                break

    grid_path = build_grid(args.out)
    print(f"  grid: {grid_path}")
    print("Done.")


if __name__ == "__main__":
    main()
