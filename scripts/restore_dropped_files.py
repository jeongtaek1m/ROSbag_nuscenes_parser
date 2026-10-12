#!/usr/bin/env python3
"""Bring back the sensor files of scenes that lost them after their release (drop_published_files.py), from
another copy of the same set — e.g. to review a new scene against the earlier round's scene it is compared with.

    python scripts/restore_dropped_files.py --from /data/parsed/tcar_nuscenes --compared
    python scripts/restore_dropped_files.py --from DIR --scenes scene-0349,scene-0350 [--dataroot ...] [--dry-run]

--compared takes the dropped scenes that the removal candidates in selection/result.json are compared with (the
review shows both). The other copy must hold the same scenes (same tokens, hence the same file names): an older
copy of this data root, or the release downloaded and unpacked. Files are copied, never linked. The scenes leave
selection/files_dropped.json (kept under "restored"), so the check asks for their files again and "배포된 파일 정리"
can drop them once more later. Run curate.py (the app: 규칙 적용) afterwards: it refreshes the review's notes.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'curation'))
import apply_selection as A  # noqa: E402

VERSION = 'v1.0-trainval'


def _rd(root, name):
    return json.load(open(os.path.join(root, VERSION, name + '.json'), encoding='utf-8'))


def main():
    data = os.environ.get('TCAR_DATA_ROOT', '/data')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=os.path.join(data, 'parsed', 'tcar_nuscenes'))
    ap.add_argument('--from', dest='src', required=True, help='another copy of this set that still has the files')
    ap.add_argument('--scenes', help='scene names, comma separated')
    ap.add_argument('--compared', action='store_true', help='the dropped scenes the removal candidates are compared with')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    sel = os.path.join(a.dataroot, 'selection')
    path = os.path.join(sel, 'files_dropped.json')
    rec = A._read(path, {}) or {}
    dropped = {t for r in rec.get('rounds') or [] if r.get('done') for t in r.get('scenes') or []}
    scenes = _rd(a.dataroot, 'scene')
    token = {s['name']: s['token'] for s in scenes}
    names = set(filter(None, (a.scenes or '').split(',')))
    if a.compared:
        result = (A._read(os.path.join(sel, 'result.json'), {}) or {}).get('scenes') or {}
        names |= {m for v in result.values() if v.get('status') == 'remove' for m in v.get('compare') or []}
    unknown = sorted(n for n in names if n not in token)
    if unknown:
        sys.exit(f'not in this set: {unknown[:5]}')
    want = {token[n]: n for n in names if token[n] in dropped}
    if not want:
        print('nothing to restore: none of these scenes has dropped files')
        return
    samples = {s['token']: s['scene_token'] for s in _rd(a.dataroot, 'sample') if s['scene_token'] in want}
    files = [x['filename'] for x in _rd(a.dataroot, 'sample_data') if x['sample_token'] in samples]
    files += [f[:-len('.pcd.bin')] + '.time.bin' for f in files
              if f.endswith('.pcd.bin') and os.path.exists(os.path.join(a.src, f[:-len('.pcd.bin')] + '.time.bin'))]
    for n in want.values():                     # a full set's sidecars
        d = os.path.join(a.src, 'ext', n)
        if os.path.isdir(d):
            files += [os.path.join('ext', n, f) for f in os.listdir(d)]
    missing = [f for f in files if not os.path.exists(os.path.join(a.src, f))]
    if missing:
        sys.exit(f'{len(missing):,} of {len(files):,} files are not in {a.src}, e.g. {missing[0]}')
    todo = [f for f in files if not os.path.exists(os.path.join(a.dataroot, f))]
    size = sum(os.path.getsize(os.path.join(a.src, f)) for f in todo)
    print(f"{len(want)} scenes ({', '.join(sorted(want.values()))}): {len(todo):,} of {len(files):,} files to copy, "
          f"{size / 1e9:.1f} GB, from {a.src}", flush=True)
    if a.dry_run:
        return
    with A._Locked(a.dataroot, 'restore_dropped_files'):
        for k, f in enumerate(todo, 1):
            dst = os.path.join(a.dataroot, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(a.src, f), dst + '.part')
            os.replace(dst + '.part', dst)
            if k % 2000 == 0 or k == len(todo):
                print(f'copied ({k}/{len(todo)})', flush=True)
        for r in rec.get('rounds') or []:
            r['scenes'] = [t for t in r.get('scenes') or [] if t not in want]
        rec.setdefault('restored', []).append({
            'when': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
            'from': os.path.realpath(a.src), 'scenes': sorted(want), 'files': len(files)})
        A._write(path, rec, indent=1)
        with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  === 지웠던 씬 {len(want)}개의 센서 파일 복원 "
                    f"({', '.join(sorted(want.values()))}; {len(files)}개, {a.src}에서) ===\n")
    print(f'done: {len(want)} scenes have their files again; run curate.py to refresh the review', flush=True)


if __name__ == '__main__':
    main()
