#!/usr/bin/env python3
"""Free the standard set's disk: delete the sensor files of the scenes a Hugging Face release already holds.

    python scripts/drop_published_files.py --plan        # what would go (fast: no file is touched or listed)
    python scripts/drop_published_files.py               # /data/parsed/tcar_nuscenes, twins in .._full

A scene loses its samples/ and sweeps/ files only when it is
  - published: in selection/hf_release.json (the release record hf_release.py keeps),
  - locked: an earlier round's 최종 확정 (it never changes again), and
  - in the full set too: a twin with the same name and sample timestamps (the same frames, kept locally).
Its tables, can_bus/ files and the curation state stay, so scene names keep continuing after it, the locks hold
and the next release still finds it. selection/files_dropped.json records which scenes (by token); the check
(apply_selection --check) does not ask for their files and the review shows a note instead of their images.
Get them back with `hf download` of the release, or read the full set.

Files go in inode order: on the /data SMR disk deleting in table order (random inodes) ran at ~40 GB/h. A round
that stopped before it finished (done: false) is taken up again. Progress lines read "deleted (k/n)".
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
PER_SCENE_GB = 1.8                              # a standard scene's files when selection/scene_bytes.json lacks it


def _rd(root, name):
    return json.load(open(os.path.join(root, VERSION, name + '.json'), encoding='utf-8'))


def _keys(root):
    """{scene token: (log, sample timestamps)}, {key: scene name} of a set (small tables only)."""
    logfile = {l['token']: l['logfile'] for l in _rd(root, 'log')}
    ts = {}
    for s in _rd(root, 'sample'):
        ts.setdefault(s['scene_token'], []).append(s['timestamp'])
    scenes = _rd(root, 'scene')
    key = {s['token']: (logfile[s['log_token']], tuple(sorted(ts.get(s['token'], ())))) for s in scenes}
    return key, {key[s['token']]: s['name'] for s in scenes}, scenes


def plan(dataroot, full):
    """What a run would delete, from the small tables only: {scenes, bytes, open_round, why_not, repo, release}.
    `bytes` is an upper bound for a round left unfinished (some of its files are gone already)."""
    sel = os.path.join(dataroot, 'selection')
    rec = A._read(os.path.join(sel, 'hf_release.json'), {}) or {}
    published = set(rec.get('scene_tokens') or [])
    locked = A.locked_tokens(sel)
    rounds = (A._read(os.path.join(sel, 'files_dropped.json'), {}) or {}).get('rounds') or []
    done = {t for r in rounds if r.get('done') for t in r.get('scenes') or []}
    open_round = next((r for r in reversed(rounds) if not r.get('done')), None)
    key, _, scenes = _keys(dataroot)
    _, full_name, _ = _keys(full) if os.path.isdir(os.path.join(full, VERSION)) else ({}, {}, [])
    why_not = {'not published': 0, 'not locked': 0, 'no twin in the full set': 0, 'dropped already': 0}
    go = []
    for s in scenes:
        if s['token'] in done:
            why_not['dropped already'] += 1
        elif s['token'] not in published:
            why_not['not published'] += 1
        elif s['token'] not in locked:
            why_not['not locked'] += 1
        elif full_name.get(key[s['token']]) != s['name']:
            why_not['no twin in the full set'] += 1
        else:
            go.append(s['token'])
    sizes = (A._read(os.path.join(sel, 'scene_bytes.json'), {}) or {}).get('scenes') or {}
    nbytes = sum((sizes.get(t) or {}).get('bytes') or PER_SCENE_GB * 1e9 for t in go)
    return {'scenes': go, 'bytes': int(nbytes), 'open_round': bool(open_round), 'why_not': why_not,
            'repo': rec.get('repo'), 'release': rec.get('latest')}


def main():
    data = os.environ.get('TCAR_DATA_ROOT', '/data')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.join(data, 'parsed', 'tcar_nuscenes'))
    ap.add_argument('--full', default=os.path.join(data, 'parsed', 'tcar_nuscenes_full'))
    ap.add_argument('--plan', '--dry-run', dest='plan', action='store_true', help='print what would go; touch nothing')
    ap.add_argument('--json', action='store_true', help='with --plan: one JSON line')
    a = ap.parse_args()
    p = plan(a.dataroot, a.full)
    if a.plan:
        if a.json:
            print(json.dumps({k: (len(v) if k == 'scenes' else v) for k, v in p.items()}))
        else:
            print(f"{len(p['scenes'])} scenes go (at most {p['bytes'] / 1e9:.0f} GB; release {p['release']} of {p['repo']})"
                  + (' — finishes an unfinished round' if p['open_round'] else '') + '; kept: '
                  + ', '.join(f'{v} {k}' for k, v in p['why_not'].items() if v))
        return
    if not p['scenes']:
        print('nothing to drop: ' + ', '.join(f'{v} {k}' for k, v in p['why_not'].items() if v))
        return
    sel = os.path.join(a.dataroot, 'selection')
    path = os.path.join(sel, 'files_dropped.json')
    with A._Locked(a.dataroot, 'drop_published_files'):
        rounds = (A._read(path, {}) or {}).get('rounds') or []
        tokens = set(p['scenes'])
        # an unfinished round is replaced by this one (it covers the same scenes and more)
        rounds = [r for r in rounds if r.get('done')]
        entry = {'when': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                 'repo': p['repo'], 'release': p['release'], 'full_set': os.path.realpath(a.full),
                 'scenes': sorted(tokens), 'done': False}
        A._write(path, {'version': 1, 'rounds': rounds + [entry]}, indent=1)
        samples = {s['token'] for s in _rd(a.dataroot, 'sample') if s['scene_token'] in tokens}
        files = [x['filename'] for x in _rd(a.dataroot, 'sample_data') if x['sample_token'] in samples]
        ents = []
        for fn in files:
            q = os.path.join(a.dataroot, fn)
            try:
                st = os.lstat(q)
                ents.append((st.st_ino, st.st_size, q))
            except FileNotFoundError:
                pass
        ents.sort()
        print(f"{len(tokens)} scenes: {len(files):,} files, {len(ents):,} still on disk "
              f"({sum(e[1] for e in ents) / 1e9:.0f} GB)", flush=True)
        freed = 0
        step = max(len(ents) // 200, 1)
        for k, (_, size, q) in enumerate(ents, 1):
            os.remove(q)
            freed += size
            if k % step == 0 or k == len(ents):
                print(f'deleted ({k}/{len(ents)})  {freed / 1e9:.0f} GB', flush=True)
        entry.update(done=True, files=len(files), removed=len(ents), bytes=freed)
        A._write(path, {'version': 1, 'rounds': rounds + [entry]}, indent=1)
        with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  === 배포된 씬 {len(tokens)}개의 센서 파일 삭제 "
                    f"({len(ents)}개, {freed / 1e9:.0f} GB; 배포본({p['repo']} {p['release']})과 full 세트에 있음) ===\n")
    print(f'done: {len(ents):,} files, {freed / 1e9:.1f} GB freed; selection/files_dropped.json', flush=True)


if __name__ == '__main__':
    main()
