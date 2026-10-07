#!/usr/bin/env python3
"""Publish the curated T-Car nuScenes dataset to a Hugging Face dataset repo, append-only.

    python scripts/hf_release.py --branch 1002 --from main                 # plan only (nothing uploaded)
    python scripts/hf_release.py --branch 1002 --from main --upload        # do it (resumes where it stopped)
    python scripts/hf_release.py --branch 1015 --from 1002 --upload        # next delivery: only the new recordings

How the data is kept as it grows:
  - Only finalized data goes up: every scene must be locked by curation's 최종 확정
    (selection/locked.json). Locked scenes are never deleted or changed again, so a sensor tar, once
    published, never changes either. New recordings only ever add tars.
  - Sensor files (samples/, sweeps/) are packed per recording (log), split at scene boundaries into
    tars of at most --part-gb: sensors/<log>/<log>.partNN.tar.gz. A release uploads only the parts that
    are not in the repo yet.
  - The tables, CAN bus, maps, import records and the curation lists are one tar, meta/TCar_meta.tar.gz,
    replaced by every release (the only file that changes; scene names follow the tables).
  - manifest.json lists every part (log, scene tokens, files, bytes, sha256, release) and the release
    history with the parser commit that produced the data; README.md is written from it; assemble.py
    (scripts/hf_assemble.py) builds or updates a dataset directory from the tars.
  - One branch per delivery (e.g. 1002, then 1015 made from 1002): each branch is a whole, self-contained
    release; earlier branches stay as they were. Files shared between branches are stored once.
Before anything is uploaded the dataset is checked (every table reference, devkit, CAN bus), and the
parts already published must still be in it (their scenes locked). The Hugging Face token: $HF_TOKEN, or
--token-file (any text file holding an hf_... token; the token is never printed).
"""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
sys.path.insert(0, str(REPO_DIR / 'curation'))
import apply_selection  # noqa: E402

VERSION = 'v1.0-trainval'
META_PATH = 'meta/TCar_meta.tar.gz'
GZ_LEVEL = 3                                      # LiDAR shrinks ~37 %, JPEG not at all; pigz keeps up with the HDD
CURATION_FILES = ('locked.json', 'keep.txt', 'drop.txt', 'deleted.txt', 'filters.json', 'result.json', 'mirror.json')
LAYOUT = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- the dataset
def load_dataset(root):
    tdir = os.path.join(root, VERSION)
    rd = lambda n: json.load(open(os.path.join(tdir, n + '.json'), encoding='utf-8'))
    scenes, logs, samples = rd('scene'), rd('log'), rd('sample')
    logfile = {l['token']: l['logfile'] for l in logs}
    scene_of = {s['token']: s['scene_token'] for s in samples}
    span = {}
    for smp in samples:
        a, b = span.get(smp['scene_token'], (smp['timestamp'], smp['timestamp']))
        span[smp['scene_token']] = (min(a, smp['timestamp']), max(b, smp['timestamp']))
    seconds = sum((b - a) / 1e6 + 0.5 for a, b in span.values())     # key frames every 0.5 s
    files, chans = {}, {}                         # scene token -> [sensor file names]; channel -> [frames, w, h, key]
    full = os.path.isdir(os.path.join(root, 'ext'))   # the full-data converter's dataset
    for sd in rd('sample_data'):
        fl = files.setdefault(scene_of[sd['sample_token']], [])
        fl.append(sd['filename'])
        if full:                                  # per-point LiDAR time next to the frame
            fl += [t for t in apply_selection.point_time_file(sd['filename']) if os.path.isfile(os.path.join(root, t))]
        c = chans.setdefault(sd['filename'].split('/')[1], [0, sd['width'], sd['height'], 0])
        c[0] += 1
        c[3] += sd['is_key_frame']
    if full:                                      # every other topic, per scene: ext/<scene name>/
        for s in scenes:
            d = os.path.join(root, 'ext', s['name'])
            if os.path.isdir(d):
                files.setdefault(s['token'], []).extend(f'ext/{s["name"]}/{n}' for n in sorted(os.listdir(d)))
    by_log = {}
    for s in sorted(scenes, key=lambda s: s['name']):
        by_log.setdefault(logfile[s['log_token']], []).append(s)
    hz = {ch: round(c[0] / seconds, 1) for ch, c in chans.items()}
    return {'profile': 'full' if full else 'standard',
            'scenes': scenes, 'by_log': by_log, 'files': files, 'samples': len(samples), 'seconds': seconds,
            'sample_data': sum(len(v) for v in files.values()),
            'channels': {ch: {'frames': c[0], 'hz': hz[ch], 'width': c[1], 'height': c[2], 'key_frames': c[3]}
                         for ch, c in sorted(chans.items())},
            'log_info': {l['logfile']: {'date': l.get('date_captured', ''), 'location': l.get('location', '')} for l in logs},
            'annotations': len(rd('sample_annotation'))}


def scene_bytes(root, ds):
    """Bytes per scene: selection/scene_bytes.json (curation keeps it), else stat."""
    sizes = ((apply_selection._read(os.path.join(root, 'selection', 'scene_bytes.json'), {}) or {}).get('scenes') or {})
    out = {}
    for s in ds['scenes']:
        b = (sizes.get(s['token']) or {}).get('bytes')
        out[s['token']] = b if b else sum(os.path.getsize(os.path.join(root, f)) for f in ds['files'].get(s['token'], []))
    return out


def _split(items, w, cap):
    """Contiguous groups, as few as cap allows and then as even as possible: the smallest largest-group for
    that count (binary search on the bound, greedy fill)."""
    def fill(bound):
        out, cur, acc = [], [], 0
        for it, b in zip(items, w):
            if cur and acc + b > bound:
                out.append(cur); cur, acc = [], 0
            cur.append(it); acc += b
        return out + [cur] if cur else out
    n = len(fill(max(cap, max(w))))               # fewest groups with none above cap (one scene alone may be)
    lo, hi = max(w), max(cap, max(w))
    while lo < hi:                                # smallest bound that still needs no more than n groups
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if len(fill(mid)) <= n else (mid + 1, hi)
    return fill(lo)


def plan_parts(ds, sizes, published, part_bytes):
    """New parts: per log, the scenes no published part has, split into ~equal contiguous groups."""
    have = {t for p in published for t in p['scene_tokens']}
    used = {p['path'] for p in published}
    parts = []
    for lg, scs in sorted(ds['by_log'].items()):
        todo = [s for s in scs if s['token'] not in have]
        if not todo:
            continue
        groups = _split(todo, [sizes[s['token']] for s in todo], part_bytes)
        k = 0
        for g in groups:
            while f'sensors/{lg}/{lg}.part{k:02d}.tar.gz' in used:
                k += 1
            path = f'sensors/{lg}/{lg}.part{k:02d}.tar.gz'
            used.add(path)
            parts.append({'path': path, 'log': lg, 'scenes': [s['name'] for s in g], 'scene_tokens': [s['token'] for s in g],
                          'bytes_est': sum(sizes[s['token']] for s in g)})
    return parts


# ---------------------------------------------------------------- tars
class _HashWriter:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)

    def tell(self):
        return self.n


def _inodes(root, names):
    """file name -> inode, from the directory listings (no stat per file): reading in inode order is close to
    the order the converter wrote them, so the HDD streams instead of seeking."""
    dirs = {os.path.dirname(n) for n in names}
    ino = {}
    for d in dirs:
        with os.scandir(os.path.join(root, d)) as it:
            for e in it:
                ino[f'{d}/{e.name}'] = e.inode()
    return ino


class _Gz:
    """A gzip file written through pigz (all cores) when there is one, else zlib: `with _Gz(dest) as out` gives a
    writable stream of the uncompressed tar; afterwards .raw (uncompressed bytes), .bytes and .sha256 of the file.
    pigz -n leaves out the name and time, so the same tar always gives the same file."""
    def __init__(self, dest):
        self.dest = str(dest)

    def __enter__(self):
        self.raw = 0
        if shutil.which('pigz'):
            self.fh = open(self.dest, 'wb')
            self.proc = subprocess.Popen(['pigz', f'-{GZ_LEVEL}', '-n', '-c'], stdin=subprocess.PIPE, stdout=self.fh)
            self.sink = self.proc.stdin
        else:
            import gzip
            self.proc, self.fh = None, None
            self.sink = gzip.GzipFile(self.dest, 'wb', compresslevel=GZ_LEVEL, mtime=0)
        return self

    def write(self, b):
        self.raw += len(b)
        return self.sink.write(b)

    def tell(self):
        return self.raw

    def __exit__(self, *exc):
        self.sink.close()
        if self.proc:
            if self.proc.wait() != 0:
                raise RuntimeError(f'pigz failed on {self.dest}')
            self.fh.close()
        self.bytes, self.sha256 = os.path.getsize(self.dest), apply_selection_sha(self.dest)


def write_tar(root, members, dest, workers=16, window=96):
    """members: relative paths under root (files), into a .tar.gz. Deterministic; returns (bytes, sha256, files, raw).
    Files are read `workers` at a time, up to `window` ahead, so the HDD can order the seeks (one reader
    gets ~35 MB/s from the scattered token-named files); they are written in order."""
    ino = _inodes(root, [m for m in members if '/' in m])
    members = sorted(members, key=lambda m: (ino.get(m, 0), m))

    def read(m):
        p = os.path.join(root, m)
        with open(p, 'rb') as f:
            return os.fstat(f.fileno()), f.read()
    with _Gz(dest) as gz, ThreadPoolExecutor(workers) as pool:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.GNU_FORMAT) as tf:
            futs = [pool.submit(read, m) for m in members[:window]]
            for i, m in enumerate(members):
                st, data = futs[i].result()
                futs[i] = None
                if i + window < len(members):
                    futs.append(pool.submit(read, members[i + window]))
                ti = tarfile.TarInfo(m)
                ti.size, ti.mtime, ti.mode = len(data), int(st.st_mtime), 0o644
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ''
                tf.addfile(ti, io.BytesIO(data))
    return gz.bytes, gz.sha256, len(members), gz.raw


def meta_members(root):
    """(arcname, path): the tables, CAN bus, maps, calibration, import records, and selection/'s curation lists as
    curation/."""
    out = []
    for d in (VERSION, 'can_bus', 'maps', 'calibration'):
        base = os.path.join(root, d)
        for dirpath, dirs, names in sorted(os.walk(base)):
            rel = os.path.relpath(dirpath, root)
            out += [(f'{rel}/{n}', os.path.join(dirpath, n)) for n in sorted(names)]
    out += [(n, os.path.join(root, n)) for n in sorted(os.listdir(root)) if n.endswith('.import.json')]
    for n in CURATION_FILES:
        p = os.path.join(root, 'selection', n)
        if os.path.isfile(p):
            out.append((f'curation/{n}', p))
    return out


def write_meta(root, dest):
    with _Gz(dest) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.GNU_FORMAT) as tf:
            tf.copybufsize = 4 << 20
            for arc, p in meta_members(root):
                st = os.stat(p)
                ti = tarfile.TarInfo(arc)
                ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = st.st_size, int(st.st_mtime), 0o644, 0, 0, '', ''
                with open(p, 'rb') as f:
                    tf.addfile(ti, f)
    return gz.bytes, gz.sha256


# ---------------------------------------------------------------- where it goes
class HfRemote:
    def __init__(self, repo, branch, token):
        from huggingface_hub import HfApi
        self.api, self.repo, self.branch, self.token = HfApi(token=token), repo, branch, token

    def has_branch(self, name):
        refs = self.api.list_repo_refs(self.repo, repo_type='dataset')
        return name in {b.name for b in refs.branches} | {t.name for t in refs.tags}

    def files(self, rev=None):
        """path -> sha256 (LFS files) or None (small git files)."""
        out = {}
        for f in self.api.list_repo_tree(self.repo, repo_type='dataset', revision=rev or self.branch, recursive=True):
            if hasattr(f, 'size'):
                out[f.path] = f.lfs.sha256 if getattr(f, 'lfs', None) else None
        return out

    def read_json(self, path, rev=None):
        from huggingface_hub import hf_hub_download
        try:
            return json.load(open(hf_hub_download(self.repo, path, repo_type='dataset', revision=rev or self.branch,
                                                  token=self.token), encoding='utf-8'))
        except Exception:
            return None

    def create_branch(self, base):
        self.api.create_branch(self.repo, branch=self.branch, revision=base, repo_type='dataset', exist_ok=True)

    def put(self, local, path, msg):
        self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=self.repo, repo_type='dataset',
                             revision=self.branch, commit_message=msg)

    def mirror(self, src, dst='main'):
        """Make branch dst hold exactly what release src holds: files copied on the server (nothing re-uploaded),
        the ones src does not have removed. One commit."""
        from huggingface_hub import CommitOperationCopy, CommitOperationDelete
        have, want = self.files(dst), self.files(src)
        ops = [CommitOperationDelete(path_in_repo=p) for p in have if p not in want]
        ops += [CommitOperationCopy(src_path_in_repo=p, path_in_repo=p, src_revision=src)
                for p, sha in want.items() if sha is None or have.get(p) != sha]   # small files: always (cheap)
        self.api.create_commit(self.repo, repo_type='dataset', revision=dst, operations=ops,
                               commit_message=f'{dst} = release {src}')
        return len(ops)

    def commit(self, adds, deletes, msg):
        """adds: path in repo -> Path (a local file) or str (text)."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        ops = [CommitOperationDelete(path_in_repo=p) for p in deletes]
        ops += [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(c) if isinstance(c, Path) else c.encode())
                for p, c in adds.items()]                 # Path: a file to upload; str: the file's text
        self.api.create_commit(self.repo, repo_type='dataset', revision=self.branch, operations=ops, commit_message=msg)


class DirRemote:
    """A directory standing in for the repo: <dir>/<branch>/... (tests, or a copy to hand over on a disk)."""
    def __init__(self, base, branch):
        self.base, self.branch = Path(base), branch

    def has_branch(self, name):
        return (self.base / name).is_dir()

    def files(self, rev=None):
        d = self.base / (rev or self.branch)
        out = {}
        for p in d.rglob('*') if d.is_dir() else []:
            if p.is_file():
                rel = p.relative_to(d).as_posix()
                out[rel] = apply_selection_sha(p) if rel.endswith(('.tar', '.tar.gz')) else None
        return out

    def read_json(self, path, rev=None):
        p = self.base / (rev or self.branch) / path
        return json.load(open(p, encoding='utf-8')) if p.is_file() else None

    def create_branch(self, base):
        src, dst = self.base / base, self.base / self.branch
        if not dst.exists():
            shutil.copytree(src, dst, copy_function=os.link) if src.is_dir() else dst.mkdir(parents=True)

    def mirror(self, src, dst='main'):
        shutil.rmtree(self.base / dst, ignore_errors=True)
        shutil.copytree(self.base / src, self.base / dst, copy_function=os.link)
        return 1

    def put(self, local, path, msg):
        dst = self.base / self.branch / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dst)

    def commit(self, adds, deletes, msg):
        for p in deletes:
            (self.base / self.branch / p).unlink(missing_ok=True)
        for p, c in adds.items():
            dst = self.base / self.branch / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()                      # a hard link from the base branch: never write through it
            if isinstance(c, Path):
                shutil.copyfile(c, dst)
            else:
                dst.write_text(c, encoding='utf-8')


def apply_selection_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- README
FRAMES = """## Coordinate frames

`ego_pose` is in the location's global frame, as in nuScenes: metres, x east, y north, z up, on the
tangent plane at 37.20° N, 126.83° E (height 0 on the WGS84 ellipsoid). `log.json` names it in the
extra key `global_frame` (`enu@37.200000,126.830000,0.000`). Ego poses and the CAN bus `pose` come from
the NovAtel INSPVA solution (100 Hz, RTK), placed at the receiver's GPS measurement time.
"""


STANDARD_NOTES = """Key frames (`samples/`) are 2 Hz, synchronised across the seven sensors as in nuScenes; every other frame
is a sweep (`sweeps/`). LiDAR files hold five float32 per point (x, y, z, intensity, ring), as in nuScenes.
`calibrated_sensor` holds the intrinsics and extrinsics, one set per recording.
"""

FULL_NOTES = """Key frames (`samples/`) are 2 Hz, chosen where `LIDAR_TOP` and the six standard cameras are synchronised
(the same frames as the standard set). `CAM_TRAFFIC`, the bottom LiDARs and the radar join a key frame when
their nearest frame is close enough (25 ms for the camera, half a frame period otherwise); every other frame is
a sweep (`sweeps/`). The images are the JPEGs as recorded: not rectified.

- **LiDAR** — `.pcd.bin`: five float32 per point (x, y, z, intensity, ring) in the sensor frame, as in nuScenes.
  Next to each one, `<same name>.time.bin` holds each point's time: float32 seconds relative to the frame's
  `sample_data` timestamp, one per point in the same order. `LIDAR_TOP` is the roof RoboSense Ruby 128; `LIDAR_BOTTOM_FRONT` /
  `_REAR` are the M1s, `LIDAR_BOTTOM_LEFT` / `_RIGHT` the Bpearls.
- **Radar** — `RADAR_FRONT` is the Continental ARS548's detection point cloud (`/radar/PointCloudDetection`)
  as nuScenes radar `.pcd`, in the sensor frame. The full detection list, object list and status are in
  `ext/`.
- **`ext/<scene>/<topic>.json`** (topic `/` → `__`) — every other topic of the recording, for the scene ± 0.5 s,
  as a JSON array of `{"utime", "bag_utime", "msg"}`: `utime` is the message header stamp (the bag time when
  there is none) in the same microseconds as `sample_data`, `msg` the message fields as recorded (their own
  units; constants omitted). `_index.json` lists each file's topic and ROS message type. About 240 MB per
  scene, 51 topics:
  - INS / GNSS (`/novatel/oem7/*`): `inspva`, `inspvax`, `corrimu`, `insstdev`, `bestpos`, `bestgnsspos`,
    `bestutm`, `bestvel`, `ppppos`, `heading2`, `rxstatus`, `time`, `odom`, and `oem7raw` (`message_data` is
    the receiver's binary log; concatenated in order it is a file NovAtel tools read, with logs the decoded
    topics lack, e.g. RAWIMUSX);
  - vehicle: decoded CAN `/bsw/vehicle_can`, `/bsw/vehicle_can_extended`, raw CAN frames `/can_rx0`,
    `/can_rx1` (CAN FD), `/can_tx0`, `/app/loc/vehicle_state`, `/bsw/ad_state`, the mode switches
    (`/bsw/*_mode_sw`), `/hmi/logging_status`;
  - radar: `/radar/DetectionList` (all 800 slots as sent; only the first `list_numofdetections` are this
    cycle's detections), `/radar/ObjectList`, `/radar/PointCloudObject`, `/radar/DirectionVelocity`,
    `/radar/Status`, and the corner / vehicle radar objects (`/bsw/corner_radar_objects*`,
    `/bsw/vehicle_radar_objects`);
  - the cameras' `camera_info` (the driver's placeholder, not the calibration), the LiDARs' `rslidar_status`,
    and diagnostics `/diag/ptp`, `/diag/disk`.
- **`can_bus/`** — the nuScenes CAN bus expansion (`pose`, `ms_imu`, `meta`) from the INS, as in the standard set.

## Calibration

`calibration/` is the calibration of 2026-09-23 (`tcar_calib_20260923`; its `README.md` documents the model
and the accuracy): the seven cameras and `LIDAR_TOP`. Its values are also in `calibrated_sensor`, the same
for every recording:

- `translation`, `rotation` — sensor → ego (`T_ego_cam`, `T_ego_lidar`). The ego frame is the NovAtel output
  point (x forward, y left, z up), the frame `ego_pose` places.
- `camera_intrinsic` — fx, fy, cx, cy of the **Kannala-Brandt** (OpenCV `cv2.fisheye`) model of the raw image.
  The distortion is in extra keys of the record: `camera_model` (`"kannala_brandt"`), `camera_distortion`
  (k1…k4), `image_size`, `time_offset_s` and `rolling_shutter_readout_s` (exposure time of each image row), and
  `windshield` — the file of the windshield refraction field, a smooth pixel offset added after the lens model
  (up to 3.6 px at 10 m without it).
- `calibration` — the calibration's name; `null` for the bottom LiDARs and the radar, which are not calibrated
  (identity placeholder).

The devkit's own projection (`view_points`) is a plain pinhole and ignores the distortion; to project LiDAR
into the images use `calibration/project.py` (`load_camera`, `project_lidar`), which applies the lens model and
the windshield field.
"""


def _course(log):
    return log.split('-', 1)[0] if '-' in log else log


def _short(log):
    return log.split('_', 1)[0]                   # A-1_2026-09-28-14-40-58 -> A-1


def _order(log):
    m = re.match(r'([A-Za-z]+)-(\d+)', log)
    return (m.group(1), int(m.group(2)), log) if m else (log, 0, log)


def readme(man, repo):
    """The dataset card (README.md), written from the manifest every release."""
    rel, tot = man['release'], man['totals']
    hours = tot['seconds'] / 3600
    shards = man['shards']
    by_log = {}
    for sh in shards:
        by_log.setdefault(sh['log'], []).append(sh)
    info = man.get('logs', {})
    courses = {}
    for lg, ss in by_log.items():
        c = courses.setdefault(_course(lg), {'logs': 0, 'scenes': 0, 'gb': 0.0, 'dates': set()})
        c['logs'] += 1
        c['scenes'] += sum(len(x['scene_tokens']) for x in ss)
        c['gb'] += sum(x['bytes'] for x in ss) / 1e9
        c['dates'].add((info.get(lg) or {}).get('date') or lg.split('_')[-1][:10])
    course_rows = '\n'.join(f"| {k} | {', '.join(sorted(v['dates']))} | {v['logs']} | {v['scenes']} | {v['scenes'] * 20 / 60:.0f} min | {v['gb']:.0f} |"
                            for k, v in sorted(courses.items()))
    full = man.get('profile') == 'full'
    ch = man.get('channels', {})

    def ch_data(k, v):
        if k.startswith('CAM'):
            return f"{v['width']}×{v['height']} JPEG" + (', as recorded' if full else '')
        if k.startswith('RADAR'):
            return 'detections (.pcd, nuScenes radar fields)'
        return 'point cloud (.pcd.bin' + (' + .time.bin)' if full else ')')
    kind = lambda k: 'camera' if k.startswith('CAM') else 'radar' if k.startswith('RADAR') else 'lidar'
    ch_rows = '\n'.join(f"| `{k}` | {kind(k)} | {ch_data(k, v)} | {v['hz']:.0f} Hz | {v['frames']:,} |" for k, v in ch.items())
    rel_rows = '\n'.join(f"| `{r['name']}` | {r['created'][:10]} | {', '.join(_short(x) for x in sorted(r['new_logs'], key=_order)) or '—'} | {r['new_scenes']} | {r['scenes']} | "
                         f"`{r['parser_commit'][:7]}`{' (+local changes)' if r.get('parser_dirty') else ''} |" for r in man['releases'])
    log_rows = '\n'.join(f"| {lg} | {(info.get(lg) or {}).get('date', '')} | {sum(len(x['scene_tokens']) for x in ss)} | {len(ss)} | "
                         f"{sum(x['bytes'] for x in ss) / 1e9:.1f} | `{ss[0]['release']}` |" for lg, ss in sorted(by_log.items(), key=lambda kv: _order(kv[0])))
    first = next((r['name'] for r in man['releases']), rel)
    prev = man['releases'][-2]['name'] if len(man['releases']) > 1 else None
    update_note = (f"Coming from `{prev}`: run the two commands above in the same directories — only the tars new in "
                   f"`{rel}` are downloaded and extracted." if prev else
                   "when a newer release is out, run the same two commands with its name as `--revision`, in the same "
                   "directories: only the new tars are downloaded and extracted.")
    if full:
        title = f"# T-Car nuScenes, full sensor set — release `{rel}`"
        intro = ("Driving data from the T-Car test vehicle with every sensor it records — seven cameras, five LiDARs, "
                 "the front radar, RTK INS / GNSS, vehicle CAN — in the [nuScenes](https://www.nuscenes.org/) "
                 "`v1.0-trainval` format, with the camera and roof-LiDAR calibration.")
        sensors_note = FULL_NOTES
        tar_note = ("the sensor files (`samples/`, `sweeps/` with the `.time.bin` next to each LiDAR frame, and "
                    "`ext/<scene>/`) of a group of whole scenes of one recording")
        meta_note = "the tables (`v1.0-trainval/`), `can_bus/`, `maps/`, `calibration/`, the per-recording `*.import.json`"
        side_note = ''
        maint = f" --repo {repo} --dataroot <full dataroot>"
    else:
        title = f"# T-Car nuScenes — release `{rel}`"
        intro = ("Driving data from the T-Car test vehicle (six cameras, a roof LiDAR, RTK INS) in the\n"
                 "[nuScenes](https://www.nuscenes.org/) `v1.0-trainval` format: load it with the `nuscenes-devkit` as is.")
        sensors_note = STANDARD_NOTES
        tar_note = "the sensor files (`samples/`, `sweeps/`) of a group of whole scenes of one recording"
        meta_note = "the tables (`v1.0-trainval/`), `can_bus/`, `maps/`, the per-recording `*.import.json`"
        side_note = ("\nThe `0923` branch holds the calibration sample recordings of 2026-09-23 (one tar per channel, its own\n"
                     "scene numbering); they are not part of this series.\n")
        maint = ''
    return f"""{title}

{intro}

**{tot['scenes']} scenes** of 20 s ({hours:.2f} h) from **{tot['logs']} recordings** · {tot['samples']:,} key-frame samples ·
{tot['sample_data']:,} sensor frames · {tot['bytes'] / 1e9:.0f} GB to download in {len(shards)} sensor tars + 1 table tar
({tot.get('raw_bytes', tot['bytes']) / 1e9:.0f} GB unpacked).
{('No 3D box annotations (`sample_annotation` is empty).' if not tot.get('annotations') else '')}

## Download and use

1. Request access on this page (the dataset is gated), and log in: `hf auth login`.
2. Download this release and put it together (`assemble.py` checks every tar's sha256 and extracts it):

```bash
pip install -U huggingface_hub
hf download {repo} --repo-type dataset --revision {rel} --local-dir tcar_tars
python tcar_tars/assemble.py --tars tcar_tars --out tcar_nuscenes
```

```python
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
nusc = NuScenes(version="v1.0-trainval", dataroot="tcar_nuscenes")
can = NuScenesCanBus(dataroot="tcar_nuscenes")      # pose, ms_imu, meta per scene
```

**Some recordings only**: download the tables and the recordings you want, and let `assemble.py` skip the rest
(the tables still list every scene; the files of the others are just absent):

```bash
hf download {repo} --repo-type dataset --revision {rel} --local-dir tcar_tars \\
    --include "manifest.json" --include "assemble.py" --include "meta/*" \\
    --include "sensors/A-1_*/*" --include "sensors/B-2_*/*"
python tcar_tars/assemble.py --tars tcar_tars --out tcar_nuscenes --allow-missing
```

**Updating**: {update_note} Nothing already extracted is changed;
the tables are replaced. Add `--delete-tars` to `assemble.py` to remove each tar after extracting it (then
`hf download` fetches it again next time).

## Releases and versioning

Each release is a branch of this repository (`--revision <name>`), a complete dataset on its own; a release
is never changed after it is published. **`main` always holds the latest release** (a copy, stored once), so
`hf download` without `--revision` gets it; give `--revision` to stay on one release. Later releases only
**add** data:

- `sensors/<recording>/<recording>.partNN.tar.gz` — {tar_note}, at most 10 GB unpacked (gzip: the LiDAR files shrink by about a third, the JPEGs not). Once published, a tar never changes and is carried into every
  later release (stored once).
- `meta/TCar_meta.tar.gz` — {meta_note}
  (how each recording was converted) and `curation/`. The only file that changes between releases.
- `manifest.json` — every tar with its recording, scene tokens, files, bytes and sha256, and the release history.
- Scene tokens never change. Scene names (`scene-0001`, …) are the nuScenes train split names in recording
  order; the names of released scenes do not change either. All scenes are in the `train` split.

| release | date | recordings added | scenes added | scenes in total | parser commit |
|---|---|---|---|---|---|
{rel_rows}
{side_note}
## Contents

| course | dates | recordings | scenes | time | GB |
|---|---|---|---|---|---|
{course_rows}

| channel | sensor | data | rate | frames |
|---|---|---|---|---|
{ch_rows}

{sensors_note}
{FRAMES}
## Recordings

| recording | date | scenes | tars | GB | first release |
|---|---|---|---|---|---|
{log_rows}

## Maintainers: making the next release

Releases are made with `scripts/hf_release.py` of the parser repository (`ROSbag_nuscenes_parser`, branch
`tcar-gui`; the commit of each release is in the table above):

```bash
# new recordings: parse them, curate them in the TCAR Parser app, press 최종 확정, then
python scripts/hf_release.py --branch <new> --from {rel}{maint} --token-file <hf token file>            # plan
python scripts/hf_release.py --branch <new> --from {rel}{maint} --token-file <hf token file> --upload   # build + upload
```

It refuses to publish unless every scene is locked and the dataset passes the checks (table references,
devkit, CAN bus), and unless every tar already published still matches the dataset. Only the new recordings'
tars are built and uploaded (one commit each, resumable); the tables, manifest, this card and `assemble.py`
go last. First release of this series: `{first}`.
"""


# ---------------------------------------------------------------- main
RECORD = 'hf_release.json'                        # <dataroot>/selection/: what went out, for the app


def save_record(root, man, repo):
    """The releases and which recording first went out in which one, next to the curation files, so the app can
    tell published data from new data without asking the Hub."""
    first = {}
    for sh in man.get('shards', []):
        first.setdefault(sh['log'], sh['release'])
    rec = {'repo': repo, 'latest': man['release'], 'updated': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
           'releases': [{k: r.get(k) for k in ('name', 'created', 'new_logs', 'new_scenes', 'scenes', 'parser_commit')}
                        for r in man.get('releases', [])],
           'logs': first, 'scene_tokens': sorted({t for sh in man.get('shards', []) for t in sh['scene_tokens']})}
    path = os.path.join(root, 'selection', RECORD)
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    os.replace(path + '.tmp', path)
    log(f'release record: {path}')


def git_state():
    try:
        commit = subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(REPO_DIR), 'status', '--porcelain', '--untracked-files=no'],
                                             text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return 'unknown', True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.join(os.environ.get('TCAR_DATA_ROOT', '/data'), 'parsed', 'tcar_nuscenes'))
    ap.add_argument('--repo', default='shchon11/TCar')
    ap.add_argument('--branch', required=True, help='the release (branch) to write, e.g. 1002')
    ap.add_argument('--from', dest='base', default=None, help='when the branch is new: the release (branch) it starts from')
    ap.add_argument('--stage', default=os.path.expanduser('~/tcar/data/hf_stage'), help='where tars are built (two at a time)')
    ap.add_argument('--part-gb', type=float, default=10.0, help='largest sensor tar (GB)')
    ap.add_argument('--upload', action='store_true', help='build and upload (default: plan only)')
    ap.add_argument('--token-file', default=None, help='text file with the hf_ token (default: $HF_TOKEN / hf login)')
    ap.add_argument('--to-dir', default=None, help='write the repo into this directory instead of Hugging Face')
    ap.add_argument('--skip-check', action='store_true', help='skip the dataset check (tables, devkit, CAN bus)')
    ap.add_argument('--readme-out', default=None, help='with the plan: write the README this release would get here')
    ap.add_argument('--card-only', action='store_true', help='rewrite only README.md of the published release (from its manifest)')
    ap.add_argument('--no-main', action='store_true', help='leave main as it is (default: main becomes a copy of this release)')
    ap.add_argument('--main-only', action='store_true', help='only make main a copy of the published release --branch')
    ap.add_argument('--record-only', action='store_true', help=f'only write selection/{RECORD} from the published release --branch')
    args = ap.parse_args()
    root = os.path.realpath(args.dataroot)

    token = None
    if args.token_file:
        m = re.search(r'hf_[A-Za-z0-9]{30,}', open(os.path.expanduser(args.token_file), encoding='utf-8').read())
        if not m:
            sys.exit(f'no hf_ token in {args.token_file}')
        token = m.group(0)
    remote = DirRemote(args.to_dir, args.branch) if args.to_dir else HfRemote(args.repo, args.branch, token or os.environ.get('HF_TOKEN'))

    if args.card_only or args.main_only or args.record_only:   # nothing of the release's data is touched
        man = remote.read_json('manifest.json')
        if not man:
            sys.exit(f'{args.branch} has no manifest.json')
        save_record(root, man, args.repo)
        if args.record_only:
            return
        if args.card_only:
            remote.commit({'README.md': readme(man, args.repo)}, [], f'{args.branch}: README')
            log(f'README.md of {args.branch} rewritten')
        if args.main_only or not args.no_main:
            log(f"main = release {args.branch}: {remote.mirror(args.branch)} changes")
        return

    # 1. the dataset: finalized and whole
    lock = apply_selection.locked_tokens(os.path.join(root, 'selection'))
    log(f'reading {root}')
    ds = load_dataset(root)
    unlocked = [s['name'] for s in ds['scenes'] if s['token'] not in lock]
    if unlocked:
        sys.exit(f'{len(unlocked)} scenes are not finalized (locked) yet, e.g. {unlocked[:5]}: finish the curation round '
                 '(최종 확정) first; only locked scenes can be published, because only they never change')
    if not args.skip_check:
        problems = apply_selection.check(root, log=log)
        if problems:
            sys.exit(f'the dataset check found {len(problems)} problems; nothing uploaded')

    # 2. what is published already
    exists = remote.has_branch(args.branch)
    if not exists and not args.base:
        sys.exit(f'branch {args.branch} does not exist: give --from <release> to start it from (e.g. main or the last release)')
    src = args.branch if exists else args.base
    prev = (remote.read_json('manifest.json', src) if remote.has_branch(src) else None) or {}
    published = prev.get('shards', [])
    current = {s['token'] for s in ds['scenes']}
    gone = [p['path'] for p in published if not set(p['scene_tokens']) <= current]
    if gone:
        sys.exit(f'published tars whose scenes are no longer in the dataset: {gone[:5]} — the data is append-only; '
                 'a locked scene was removed? (nothing uploaded)')
    ours = ('sensors/', 'meta/', '.gitattributes', 'README.md', 'manifest.json', 'assemble.py')   # an older upload's files go
    legacy = [] if prev or not remote.has_branch(src) else [p for p in remote.files(src) if not p.startswith(ours)]
    sizes = scene_bytes(root, ds)
    parts = plan_parts(ds, sizes, published, int(args.part_gb * 1e9))

    new_logs = sorted({p['log'] for p in parts})
    log(f"release {args.branch} ({'update' if exists else f'new, from {args.base}'}): {len(ds['scenes'])} scenes in "
        f"{len(ds['by_log'])} recordings; published already: {len(published)} tars")
    log(f"new: {len(parts)} sensor tars, {sum(p['bytes_est'] for p in parts) / 1e9:.1f} GB, recordings {new_logs}")
    for p in parts:
        log(f"  {p['path']}  {len(p['scenes'])} scenes  ~{p['bytes_est'] / 1e9:.1f} GB")
    if legacy:
        log(f"the base has no manifest (an older upload): its {len(legacy)} files are removed from this branch "
            f"(they stay on {args.base}): {legacy[:6]}{' …' if len(legacy) > 6 else ''}")
    if not args.upload:
        if args.readme_out:                      # the card as it would be (sizes estimated, no sha256 yet)
            commit, dirty = git_state()
            shards = published + [dict(p, bytes=p['bytes_est'], release=args.branch) for p in parts]
            mine = [x for x in shards if x['release'] == args.branch]
            rels = [r for r in prev.get('releases', []) if r['name'] != args.branch] + [{
                'name': args.branch, 'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                'new_logs': sorted({x['log'] for x in mine}), 'new_scenes': sum(len(x['scene_tokens']) for x in mine),
                'scenes': len(ds['scenes']), 'parser_commit': commit, 'parser_dirty': dirty}]
            man = {'release': args.branch, 'profile': ds['profile'], 'releases': rels, 'shards': shards,
                   'channels': ds['channels'], 'logs': ds['log_info'],
                   'totals': {'scenes': len(ds['scenes']), 'logs': len(ds['by_log']), 'samples': ds['samples'],
                              'sample_data': ds['sample_data'], 'bytes': round(sum(x['bytes'] for x in shards) * 0.86),
                              'raw_bytes': sum(x['bytes'] for x in shards),
                              'seconds': ds['seconds'], 'annotations': ds['annotations']}}
            Path(args.readme_out).write_text(readme(man, args.repo), encoding='utf-8')
            log(f'README preview: {args.readme_out}')
        log('plan only: add --upload to build and upload')
        return

    # 3. upload: branch, then the parts (each its own commit, resumable), then meta + manifest + README
    os.makedirs(args.stage, exist_ok=True)
    state_path = os.path.join(args.stage, f'{args.branch}.state.json')
    state = json.load(open(state_path)) if os.path.isfile(state_path) else {'shards': {}}
    if not exists:
        remote.create_branch(args.base)
        log(f'branch {args.branch} created from {args.base}')
    if legacy:
        remote.commit({'README.md': f'# T-Car nuScenes — release `{args.branch}`\n\nUpload in progress.\n'}, legacy,
                      f'{args.branch}: start the release (the older upload stays on {args.base})')
    have = remote.files()
    lock_ = threading.Lock()

    def build(p):
        dest = Path(args.stage) / os.path.basename(p['path'])
        members = [f for t in p['scene_tokens'] for f in ds['files'].get(t, [])]
        t0 = time.time()
        n, sha, k, raw = write_tar(root, members, dest)
        dt = max(time.time() - t0, 1e-6)
        log(f"built {p['path']}: {raw / 1e9:.2f} GB -> {n / 1e9:.2f} GB gz, {k} files in {dt:.0f} s ({raw / 1e6 / dt:.0f} MB/s)")
        return dest, n, sha, k, raw

    todo = [p for p in parts if not (p['path'] in state['shards'] and have.get(p['path']) == state['shards'][p['path']]['sha256'])]
    log(f'{len(parts) - len(todo)} of {len(parts)} tars are up already; {len(todo)} to go')
    with ThreadPoolExecutor(1) as pool:                # build the next tar while this one uploads
        fut = pool.submit(build, todo[0]) if todo else None
        for i, p in enumerate(todo):
            dest, n, sha, k, raw = fut.result()
            fut = pool.submit(build, todo[i + 1]) if i + 1 < len(todo) else None
            t0 = time.time()
            remote.put(dest, p['path'], f"{args.branch}: {p['path']} ({len(p['scenes'])} scenes)")
            log(f"uploaded [{i + 1}/{len(todo)}] {p['path']} in {time.time() - t0:.0f} s ({n / 1e6 / max(time.time() - t0, 1e-6):.0f} MB/s)")
            with lock_:
                state['shards'][p['path']] = {k_: v for k_, v in p.items() if k_ != 'bytes_est'} | {
                    'bytes': n, 'raw_bytes': raw, 'sha256': sha, 'files': k, 'release': args.branch}
                with open(state_path + '.tmp', 'w') as f:
                    json.dump(state, f, indent=1)
                os.replace(state_path + '.tmp', state_path)
            os.remove(dest)

    # 4. meta, manifest, README, assemble.py: one commit
    meta_dest = Path(args.stage) / os.path.basename(META_PATH)
    mn, msha = write_meta(root, meta_dest)
    log(f'built {META_PATH}: {mn / 1e9:.2f} GB')
    if exists and not parts and (prev.get('meta') or {}).get('sha256') == msha:
        os.remove(meta_dest)
        log(f'release {args.branch} is up to date: nothing to commit')
        return
    commit, dirty = git_state()
    shards = published + [state['shards'][p['path']] for p in parts]
    mine = [s_ for s_ in shards if s_['release'] == args.branch]       # this release's own additions (also on a re-run)
    old = next((r for r in prev.get('releases', []) if r['name'] == args.branch), {})
    rels = [r for r in prev.get('releases', []) if r['name'] != args.branch] + [{
        'name': args.branch, 'created': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'from': old.get('from') or (args.base if not exists else prev.get('release')),
        'new_logs': sorted({s_['log'] for s_ in mine}), 'new_scenes': sum(len(s_['scene_tokens']) for s_ in mine),
        'scenes': len(ds['scenes']), 'parser_commit': commit, 'parser_dirty': dirty,
        'lock_rounds': [r['stamp'] for r in apply_selection.lock_rounds(os.path.join(root, 'selection'))]}]
    man = {'dataset': 'T-Car nuScenes', 'profile': ds['profile'], 'version': VERSION, 'layout': LAYOUT, 'release': args.branch,
           'totals': {'scenes': len(ds['scenes']), 'logs': len(ds['by_log']), 'samples': ds['samples'],
                      'sample_data': ds['sample_data'], 'bytes': sum(s['bytes'] for s in shards),
                      'raw_bytes': sum(s.get('raw_bytes', s['bytes']) for s in shards), 'seconds': round(ds['seconds'], 1),
                      'annotations': ds['annotations']},
           'channels': ds['channels'], 'logs': ds['log_info'],
           'meta': {'path': META_PATH, 'bytes': mn, 'sha256': msha}, 'releases': rels, 'shards': shards}
    remote.commit({META_PATH: meta_dest, 'manifest.json': json.dumps(man, ensure_ascii=False, indent=1),
                   'README.md': readme(man, args.repo), 'assemble.py': (HERE / 'hf_assemble.py').read_text(encoding='utf-8')},
                  [], f'{args.branch}: tables, manifest, README ({len(ds["scenes"])} scenes, +{rels[-1]["new_scenes"]})')
    os.remove(meta_dest)
    log(f"release {args.branch} done: {len(shards)} sensor tars + meta, {len(ds['scenes'])} scenes")
    save_record(root, man, args.repo)
    if not args.no_main and args.branch != 'main':  # main always holds the latest release
        log(f"main = release {args.branch}: {remote.mirror(args.branch)} changes")


if __name__ == '__main__':
    main()
