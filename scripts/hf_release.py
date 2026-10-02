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
    tars of at most --part-gb: sensors/<log>/<log>.partNN.tar. A release uploads only the parts that
    are not in the repo yet.
  - The tables, CAN bus, maps, import records and the curation lists are one tar, meta/TCar_meta.tar,
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
META_PATH = 'meta/TCar_meta.tar'
CURATION_FILES = ('locked.json', 'keep.txt', 'drop.txt', 'deleted.txt', 'filters.json', 'result.json')
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
    files = {}                                    # scene token -> [sensor file names]
    for sd in rd('sample_data'):
        files.setdefault(scene_of[sd['sample_token']], []).append(sd['filename'])
    by_log = {}
    for s in sorted(scenes, key=lambda s: s['name']):
        by_log.setdefault(logfile[s['log_token']], []).append(s)
    return {'scenes': scenes, 'by_log': by_log, 'files': files, 'samples': len(samples), 'seconds': seconds,
            'sample_data': sum(len(v) for v in files.values())}


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
            while f'sensors/{lg}/{lg}.part{k:02d}.tar' in used:
                k += 1
            path = f'sensors/{lg}/{lg}.part{k:02d}.tar'
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


def write_tar(root, members, dest, workers=16, window=96):
    """members: relative paths under root (files). Deterministic headers; returns (bytes, sha256, files).
    Files are read `workers` at a time, up to `window` ahead, so the HDD can order the seeks (one reader
    gets ~35 MB/s from the scattered token-named files); they are written in order."""
    ino = _inodes(root, [m for m in members if '/' in m])
    members = sorted(members, key=lambda m: (ino.get(m, 0), m))

    def read(m):
        p = os.path.join(root, m)
        with open(p, 'rb') as f:
            return os.fstat(f.fileno()), f.read()
    with open(dest, 'wb') as raw, ThreadPoolExecutor(workers) as pool:
        hw = _HashWriter(raw)
        with tarfile.open(fileobj=hw, mode='w', format=tarfile.GNU_FORMAT) as tf:
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
    return hw.n, hw.h.hexdigest(), len(members)


def meta_members(root):
    """(arcname, path): the tables, CAN bus, maps, import records, and selection/'s curation lists as curation/."""
    out = []
    for d in (VERSION, 'can_bus', 'maps'):
        base = os.path.join(root, d)
        if os.path.isdir(base):
            out += [(f'{d}/{n}', os.path.join(base, n)) for n in sorted(os.listdir(base)) if os.path.isfile(os.path.join(base, n))]
    out += [(n, os.path.join(root, n)) for n in sorted(os.listdir(root)) if n.endswith('.import.json')]
    for n in CURATION_FILES:
        p = os.path.join(root, 'selection', n)
        if os.path.isfile(p):
            out.append((f'curation/{n}', p))
    return out


def write_meta(root, dest):
    with open(dest, 'wb') as raw:
        hw = _HashWriter(raw)
        with tarfile.open(fileobj=hw, mode='w', format=tarfile.GNU_FORMAT) as tf:
            tf.copybufsize = 4 << 20
            for arc, p in meta_members(root):
                st = os.stat(p)
                ti = tarfile.TarInfo(arc)
                ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = st.st_size, int(st.st_mtime), 0o644, 0, 0, '', ''
                with open(p, 'rb') as f:
                    tf.addfile(ti, f)
    return hw.n, hw.h.hexdigest()


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
                out[rel] = apply_selection_sha(p) if p.suffix == '.tar' else None
        return out

    def read_json(self, path, rev=None):
        p = self.base / (rev or self.branch) / path
        return json.load(open(p, encoding='utf-8')) if p.is_file() else None

    def create_branch(self, base):
        src, dst = self.base / base, self.base / self.branch
        if not dst.exists():
            shutil.copytree(src, dst, copy_function=os.link) if src.is_dir() else dst.mkdir(parents=True)

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


def readme(man, repo):
    rel, tot = man['release'], man['totals']
    hours = tot['seconds'] / 3600
    rows = '\n'.join(f"| `{r['name']}` | {r['created'][:10]} | {len(r['new_logs'])} | {r['new_scenes']} | {r['scenes']} | "
                     f"{r['parser_commit'][:7]}{' (dirty)' if r.get('parser_dirty') else ''} |" for r in man['releases'])
    logs = {}
    for s in man['shards']:
        logs.setdefault(s['log'], []).append(s)
    log_rows = '\n'.join(f"| {lg} | {sum(len(s['scene_tokens']) for s in ss)} | {len(ss)} | {sum(s['bytes'] for s in ss) / 1e9:.1f} | "
                         f"`{ss[0]['release']}` |" for lg, ss in sorted(logs.items()))
    return f"""# T-Car nuScenes (v1.0-trainval layout) — release `{rel}`

{tot['scenes']} scenes ({hours:.2f} h, 20 s each) from {tot['logs']} recordings · {tot['samples']:,} samples ·
{tot['sample_data']:,} sample_data · {tot['bytes'] / 1e9:.0f} GB in {len(man['shards'])} sensor tars + 1 table tar.
Six cameras, LIDAR_TOP, INSPVA ego pose and CAN bus. Curated: every scene here was reviewed and locked
(see *Curation*).

## Download

```bash
pip install -U huggingface_hub
hf download {repo} --repo-type dataset --revision {rel} --local-dir tcar_tars
python tcar_tars/assemble.py --tars tcar_tars --out tcar_nuscenes
```

```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-trainval", dataroot="tcar_nuscenes")
```

**Updating to a newer release**: run the same two commands with the new `--revision`. `hf download`
fetches only the tars you do not have; `assemble.py` extracts only the new sensor tars and replaces
the tables. Nothing you already extracted changes.

## Layout and how releases grow

- `sensors/<recording>/<recording>.partNN.tar` — the sensor files (`samples/`, `sweeps/`) of a group of
  scenes of one recording. Published once, never changed: curated scenes are locked, so later releases
  only add tars for new recordings.
- `meta/TCar_meta.tar` — `v1.0-trainval/` tables, `can_bus/`, `maps/`, the per-recording `*.import.json`
  and `curation/` (decision lists). Replaced by every release.
- `manifest.json` — every tar with its recording, scene tokens, size and sha256, and the release history.
- `assemble.py` — builds / updates the dataset directory from the tars (Python, standard library).

Each release is its own branch (`--revision <name>`); earlier releases stay as they were.

| release | date | new recordings | new scenes | scenes in total | parser |
|---|---|---|---|---|---|
{rows}

## Curation

Scenes are filtered in a fixed order — (1) same-place stops, keeping the one with the most surrounding
motion, (2) few road users nearby (< 3 per sample, 6 cameras, YOLO), (3) the same road driven the same
way (≥ 60 % of the path) — and a filter only proposes; a person confirmed every deletion. A release
contains locked scenes only. `curation/deleted.txt` lists the scenes taken out, `curation/keep.txt` the
ones kept and why.

{FRAMES}
## Recordings

| recording | scenes | tars | GB | first release |
|---|---|---|---|---|
{log_rows}
"""


# ---------------------------------------------------------------- main
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
    args = ap.parse_args()
    root = os.path.realpath(args.dataroot)

    token = None
    if args.token_file:
        m = re.search(r'hf_[A-Za-z0-9]{30,}', open(os.path.expanduser(args.token_file), encoding='utf-8').read())
        if not m:
            sys.exit(f'no hf_ token in {args.token_file}')
        token = m.group(0)
    remote = DirRemote(args.to_dir, args.branch) if args.to_dir else HfRemote(args.repo, args.branch, token or os.environ.get('HF_TOKEN'))

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
        dest = os.path.join(args.stage, os.path.basename(p['path']))
        members = [f for t in p['scene_tokens'] for f in ds['files'].get(t, [])]
        t0 = time.time()
        n, sha, k = write_tar(root, members, dest)
        log(f"built {p['path']}: {n / 1e9:.2f} GB, {k} files in {time.time() - t0:.0f} s ({n / 1e6 / max(time.time() - t0, 1e-6):.0f} MB/s)")
        return dest, n, sha, k

    todo = [p for p in parts if not (p['path'] in state['shards'] and have.get(p['path']) == state['shards'][p['path']]['sha256'])]
    log(f'{len(parts) - len(todo)} of {len(parts)} tars are up already; {len(todo)} to go')
    with ThreadPoolExecutor(1) as pool:                # build the next tar while this one uploads
        fut = pool.submit(build, todo[0]) if todo else None
        for i, p in enumerate(todo):
            dest, n, sha, k = fut.result()
            fut = pool.submit(build, todo[i + 1]) if i + 1 < len(todo) else None
            t0 = time.time()
            remote.put(dest, p['path'], f"{args.branch}: {p['path']} ({len(p['scenes'])} scenes)")
            log(f"uploaded [{i + 1}/{len(todo)}] {p['path']} in {time.time() - t0:.0f} s ({n / 1e6 / max(time.time() - t0, 1e-6):.0f} MB/s)")
            with lock_:
                state['shards'][p['path']] = {k_: v for k_, v in p.items() if k_ != 'bytes_est'} | {
                    'bytes': n, 'sha256': sha, 'files': k, 'release': args.branch}
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
    man = {'dataset': 'T-Car nuScenes', 'version': VERSION, 'layout': LAYOUT, 'release': args.branch,
           'totals': {'scenes': len(ds['scenes']), 'logs': len(ds['by_log']), 'samples': ds['samples'],
                      'sample_data': ds['sample_data'], 'bytes': sum(s['bytes'] for s in shards), 'seconds': round(ds['seconds'], 1)},
           'meta': {'path': META_PATH, 'bytes': mn, 'sha256': msha}, 'releases': rels, 'shards': shards}
    remote.commit({META_PATH: meta_dest, 'manifest.json': json.dumps(man, ensure_ascii=False, indent=1),
                   'README.md': readme(man, args.repo), 'assemble.py': (HERE / 'hf_assemble.py').read_text(encoding='utf-8')},
                  [], f'{args.branch}: tables, manifest, README ({len(ds["scenes"])} scenes, +{rels[-1]["new_scenes"]})')
    os.remove(meta_dest)
    log(f"release {args.branch} done: {len(shards)} sensor tars + meta, {len(ds['scenes'])} scenes")


if __name__ == '__main__':
    main()
