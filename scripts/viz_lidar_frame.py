"""Visualize one lidar sweep from a converted dataset (BEV + side view).

Usage:
    python scripts/viz_lidar_frame.py /data/tcar_nuscenes --out viz_lidar
    python scripts/viz_lidar_frame.py <dataroot> --frame mid|first|last|<index>
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataroot", type=Path,
                   help="NuScenes dataroot, e.g. /data/tcar_nuscenes")
    p.add_argument("--out", type=Path, default=Path("viz_lidar"),
                   help="Output dir (default: viz_lidar/).")
    p.add_argument("--frame", default="mid",
                   help="Which frame: 'first' | 'mid' | 'last' | integer index.")
    args = p.parse_args()

    files = sorted((args.dataroot / "samples" / "LIDAR_TOP").glob("*.pcd.bin"))
    if not files:
        raise SystemExit(f"No lidar files in {args.dataroot / 'samples' / 'LIDAR_TOP'}")

    if args.frame == "first":
        idx = 0
    elif args.frame == "last":
        idx = len(files) - 1
    elif args.frame == "mid":
        idx = len(files) // 2
    else:
        idx = int(args.frame)

    f = files[idx]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"frame {idx}/{len(files)}: {f.name}")

    arr = np.fromfile(f, dtype=np.float32).reshape(-1, 5)
    print(f"n_points={arr.shape[0]}")
    x, y, z, intensity, ring = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    n = arr.shape[0]
    pidx = np.arange(n) if n < 80000 else np.random.choice(n, 80000, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    ax = axes[0]
    sc = ax.scatter(x[pidx], y[pidx], c=ring[pidx], s=0.3, cmap="tab20", alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_title(f"BEV — colored by ring  (n={n})")
    ax.axhline(0, color="white", lw=0.3, alpha=0.4)
    ax.axvline(0, color="white", lw=0.3, alpha=0.4)
    ax.scatter([0], [0], c="red", s=80, marker="x", label="lidar origin")
    ax.legend(loc="upper right")
    ax.set_facecolor("#222")
    plt.colorbar(sc, ax=ax, label="ring (channel)")

    ax = axes[1]
    sc = ax.scatter(x[pidx], z[pidx], c=intensity[pidx], s=0.3, cmap="viridis",
                    alpha=0.6, vmin=0, vmax=np.percentile(intensity, 99))
    ax.set_xlim(-80, 80)
    ax.set_ylim(-5, 15)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("z (m, up)")
    ax.set_title("Side view (x-z) — colored by intensity")
    ax.axhline(0, color="white", lw=0.3, alpha=0.4)
    ax.scatter([0], [0], c="red", s=80, marker="x")
    ax.set_facecolor("#222")
    plt.colorbar(sc, ax=ax, label="intensity")

    plt.tight_layout()
    out_path = args.out / f"frame_{args.frame}.png"
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
