#!/usr/bin/env python3
"""Put the T-Car nuScenes tars together into one dataset directory, and keep it up to date.

Uploaded next to the tars as `assemble.py` (Python 3.8+, standard library only).

    hf download shchon11/TCar --repo-type dataset --revision <release> --local-dir tcar_tars
    python tcar_tars/assemble.py --tars tcar_tars --out tcar_nuscenes

Run the same two commands again for a newer release: `hf download` fetches only the tars that are new,
and this script extracts only the shards it has not extracted before (it remembers them in
<out>/.tcar_assembled.json), then replaces the tables (meta). Sensor shards never change once
published (curated scenes are locked), so nothing already extracted is touched.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile

STATE = '.tcar_assembled.json'
META_DIRS = ('v1.0-trainval', 'can_bus', 'maps', 'curation')   # replaced as a whole by every release's meta


def sha256(path, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(bufsize), b''):
            h.update(block)
    return h.hexdigest()


def extract(path, out):
    with tarfile.open(path) as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(out, filter='data')
        else:
            for m in tf.getmembers():
                if m.name.startswith('/') or '..' in m.name.split('/'):
                    raise ValueError(f'{path}: unsafe member {m.name}')
            tf.extractall(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tars', required=True, help='the downloaded repository (manifest.json, meta/, sensors/)')
    ap.add_argument('--out', required=True, help='the dataset directory to build or update (nuScenes dataroot)')
    ap.add_argument('--no-verify', action='store_true', help='skip the sha256 check of each tar before extracting')
    ap.add_argument('--delete-tars', action='store_true', help='delete each tar once extracted (saves space; '
                    'hf download then fetches it again next time, so keep them if you will update)')
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.tars, 'manifest.json'), encoding='utf-8'))
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, STATE)
    state = json.load(open(state_path, encoding='utf-8')) if os.path.isfile(state_path) else {'shards': {}, 'meta': None}

    def save():
        with open(state_path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=1)
        os.replace(state_path + '.tmp', state_path)

    todo = [s for s in man['shards'] if state['shards'].get(s['path']) != s['sha256']]
    print(f"release {man['release']}: {len(man['shards'])} sensor shards ({len(todo)} to extract), "
          f"{man['totals']['scenes']} scenes")
    for k, s in enumerate(todo, 1):
        path = os.path.join(args.tars, s['path'])
        if not os.path.isfile(path):
            sys.exit(f'missing {s["path"]}: download the release again (hf download ... --local-dir {args.tars})')
        if not args.no_verify and sha256(path) != s['sha256']:
            sys.exit(f'{s["path"]}: sha256 does not match the manifest (incomplete download?)')
        print(f"  [{k}/{len(todo)}] {s['path']}  ({s['bytes'] / 1e9:.1f} GB, {s['files']} files)", flush=True)
        extract(path, args.out)
        state['shards'][s['path']] = s['sha256']
        save()
        if args.delete_tars:
            os.remove(path)
    meta = man['meta']
    if state.get('meta') != meta['sha256']:
        path = os.path.join(args.tars, meta['path'])
        if not args.no_verify and sha256(path) != meta['sha256']:
            sys.exit(f'{meta["path"]}: sha256 does not match the manifest')
        for d in META_DIRS:                   # scene names (and so CAN bus file names) can change between releases
            shutil.rmtree(os.path.join(args.out, d), ignore_errors=True)
        for f in os.listdir(args.out):
            if f.endswith('.import.json'):
                os.remove(os.path.join(args.out, f))
        print(f"  tables: {meta['path']}", flush=True)
        extract(path, args.out)
        state['meta'], state['release'] = meta['sha256'], man['release']
        save()
    print(f"done: {args.out} is release {man['release']}. Check: "
          f"python -c \"from nuscenes.nuscenes import NuScenes; NuScenes('v1.0-trainval', '{args.out}')\"")


if __name__ == '__main__':
    main()
