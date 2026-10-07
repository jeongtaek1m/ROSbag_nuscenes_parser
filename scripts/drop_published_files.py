#!/usr/bin/env python3
"""Free the standard set's disk: delete the sensor files of the scenes a Hugging Face release already holds.

    python scripts/drop_published_files.py --dry-run     # what would go
    python scripts/drop_published_files.py               # /data/parsed/tcar_nuscenes, twins in .._full

A scene loses its samples/ and sweeps/ files only when it is
  - published: in selection/hf_release.json (the release record hf_release.py keeps),
  - locked: an earlier round's 최종 확정 (it never changes again), and
  - in the full set too: a twin with the same name and sample timestamps (the same frames, kept locally).
Its tables, can_bus/ files and the curation state stay, so scene names keep continuing after it, the locks hold
and the next release still finds it. selection/files_dropped.json records which scenes (by token); the check
(apply_selection --check) does not ask for their files and the review shows a note instead of their images.
Get them back with `hf download` of the release, or read the full set.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'curation'))
import apply_selection as A  # noqa: E402

VERSION = 'v1.0-trainval'


def twins(root):
    """{(log, sample timestamps): scene name} of a set."""
    t = os.path.join(root, VERSION)
    rd = lambda n: json.load(open(os.path.join(t, n + '.json'), encoding='utf-8'))
    logfile = {l['token']: l['logfile'] for l in rd('log')}
    ts = {}
    for s in rd('sample'):
        ts.setdefault(s['scene_token'], []).append(s['timestamp'])
    return {(logfile[s['log_token']], tuple(sorted(ts.get(s['token'], ())))): s['name'] for s in rd('scene')}


def main():
    data = os.environ.get('TCAR_DATA_ROOT', '/data')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.join(data, 'parsed', 'tcar_nuscenes'))
    ap.add_argument('--full', default=os.path.join(data, 'parsed', 'tcar_nuscenes_full'))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    sel = os.path.join(a.dataroot, 'selection')
    rec = A._read(os.path.join(sel, 'hf_release.json'), {}) or {}
    published = set(rec.get('scene_tokens') or [])
    if not published:
        sys.exit(f'{sel}/hf_release.json lists no published scenes')
    locked = A.locked_tokens(sel)
    T = A.load_tables(os.path.join(a.dataroot, VERSION))
    logfile = {l['token']: l['logfile'] for l in T['log']}
    ts = {}
    for s in T['sample']:
        ts.setdefault(s['scene_token'], []).append(s['timestamp'])
    full = twins(a.full)
    drop, why_not = [], {'not published': 0, 'not locked': 0, 'no twin in the full set': 0}
    for s in T['scene']:
        key = (logfile[s['log_token']], tuple(sorted(ts.get(s['token'], ()))))
        if s['token'] not in published:
            why_not['not published'] += 1
        elif s['token'] not in locked:
            why_not['not locked'] += 1
        elif full.get(key) != s['name']:
            why_not['no twin in the full set'] += 1
        else:
            drop.append(s)
    tokens = {s['token'] for s in drop}
    samples = {s['token'] for s in T['sample'] if s['scene_token'] in tokens}
    files = [x['filename'] for x in T['sample_data'] if x['sample_token'] in samples]
    print(f"{len(drop)} of {len(T['scene'])} scenes go ({len(files):,} files; release {rec.get('latest')} of {rec.get('repo')}); "
          f"kept: " + ', '.join(f'{v} {k}' for k, v in why_not.items() if v), flush=True)
    if a.dry_run or not drop:
        return
    path = os.path.join(sel, 'files_dropped.json')
    have = A._read(path, {}) or {}
    with A._Locked(a.dataroot, 'drop_published_files'):
        entry = {'when': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                 'repo': rec.get('repo'), 'release': rec.get('latest'), 'full_set': os.path.realpath(a.full),
                 'scenes': sorted(tokens), 'files': len(files), 'done': False}
        A._write(path, {'version': 1, 'rounds': (have.get('rounds') or []) + [entry]}, indent=1)
        gone = freed = 0
        for fn in files:
            p = os.path.join(a.dataroot, fn)
            try:
                freed += os.path.getsize(p)
                os.remove(p)
                gone += 1
            except FileNotFoundError:
                pass
        entry.update(done=True, removed=gone, bytes=freed)
        A._write(path, {'version': 1, 'rounds': (have.get('rounds') or []) + [entry]}, indent=1)
        with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  === 배포된 씬 {len(drop)}개의 센서 파일 삭제 "
                    f"({gone}개, {freed / 1e9:.0f} GB; {rec.get('repo')} {rec.get('latest')}과 full 세트에 있음) ===\n")
    print(f'done: {gone:,} files, {freed / 1e9:.1f} GB freed; selection/files_dropped.json', flush=True)


if __name__ == '__main__':
    main()
