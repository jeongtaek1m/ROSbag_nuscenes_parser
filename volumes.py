"""Volumes of a dataset: tcar_nuscenes, tcar_nuscenes_2, tcar_nuscenes_3, ...

A nuScenes dataroot can name its scenes only from the official split lists (`nuscenes.utils.splits`: 685 usable
train names), so a set that has used them up continues in a sibling folder `<name>_<N>` (N = 2, 3, ...). Each volume
is a complete dataroot of its own; scene names start again in every one, recording (log) names are unique across
them. The standard and the full set have one volume family each, kept in step: volume N of the one is the twin of
volume N of the other.
"""
from __future__ import annotations

import re
from pathlib import Path

_VOL = re.compile(r"^(?P<base>.+?)_(?P<n>[2-9]|[1-9]\d+)$")


def base_of(root: Path) -> Path:
    """The first volume's folder: .../tcar_nuscenes_2 -> .../tcar_nuscenes (itself when it is the first)."""
    root = Path(root)
    m = _VOL.match(root.name)
    return root.with_name(m["base"]) if m and root.with_name(m["base"]).is_dir() else root


def index_of(root: Path) -> int:
    """1 for the first volume, N for <name>_<N>."""
    root = Path(root)
    return 1 if base_of(root) == root else int(root.name.rsplit("_", 1)[1])


def volumes(root: Path) -> list[Path]:
    """Every volume of root's family that exists, in order; the first one is listed even when it does not."""
    base = base_of(root)
    more = [p for p in base.parent.glob(f"{base.name}_*")
            if p.is_dir() and re.fullmatch(rf"{re.escape(base.name)}_([2-9]|[1-9]\d+)", p.name)]
    return [base, *sorted(more, key=lambda p: int(p.name.rsplit("_", 1)[1]))]


def newest(root: Path) -> Path:
    """Where new recordings go."""
    return volumes(root)[-1]


def next_volume(root: Path) -> Path:
    """The folder after the newest volume (highest number + 1: a volume set aside leaves no name to reuse)."""
    base = base_of(root)
    return base.with_name(f"{base.name}_{index_of(newest(root)) + 1}")


def volume(root: Path, n: int) -> Path:
    """Volume n of root's family (whether it exists or not)."""
    base = base_of(root)
    return base if n == 1 else base.with_name(f"{base.name}_{n}")
