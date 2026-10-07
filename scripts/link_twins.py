#!/usr/bin/env python3
"""Store the frames the standard and the full set share once: each standard file becomes a hard link to its twin.

    python scripts/link_twins.py                 # /data/parsed/tcar_nuscenes -> /data/parsed/tcar_nuscenes_full
    python scripts/link_twins.py --dry-run       # hash and count only
    python scripts/link_twins.py --standard <root> --full <root>

The full set carries every LIDAR_TOP and camera frame of the standard set, byte for byte (same bag, same frame
selection; only the tokens and file names differ). A twin is the full frame with the same channel and timestamp.
Both files are hashed (blake2b, read in inode order so the HDD streams), and where size and hash agree the
standard file is replaced by a hard link to the full one (link to a temporary name, then rename over: never a
moment without the file). Tables, names and tokens are not touched; both sets read exactly as before. A file
whose twin differs or is missing stays as it is. Running it again skips what is already linked.

bag2nuscenes.py --full-out links new recordings this way from the start.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

VERSION = 'v1.0-trainval'


def frames(root):
    """{(channel, timestamp): filename} of a set's sample_data."""
    t = os.path.join(root, VERSION)
    rd = lambda n: json.load(open(os.path.join(t, n + '.json'), encoding='utf-8'))
    ch = {s['token']: s['channel'] for s in rd('sensor')}
    cs = {c['token']: ch[c['sensor_token']] for c in rd('calibrated_sensor')}
    return {(cs[x['calibrated_sensor_token']], x['timestamp']): x['filename'] for x in rd('sample_data')}


def hash_files(paths, label, workers):
    """{path: blake2b hex}, read in inode order."""
    order = sorted(paths, key=lambda p: os.stat(p).st_ino)

    def h(p):
        d = hashlib.blake2b(digest_size=20)
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(4 << 20), b''):
                d.update(b)
        return p, d.hexdigest()
    out = {}
    with ThreadPoolExecutor(workers) as pool, tqdm(total=len(order), unit='file', desc=label) as bar:
        for p, d in pool.map(h, order, chunksize=64):
            out[p] = d
            bar.update()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.environ.get('TCAR_DATA_ROOT', '/data')
    ap.add_argument('--standard', default=os.path.join(data, 'parsed', 'tcar_nuscenes'))
    ap.add_argument('--full', default=os.path.join(data, 'parsed', 'tcar_nuscenes_full'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    S, F = frames(a.standard), frames(a.full)
    pairs, n_missing, n_linked, n_size = [], 0, 0, 0
    for k, fn in S.items():
        if k not in F:
            n_missing += 1
            continue
        sp, fp = os.path.join(a.standard, fn), os.path.join(a.full, F[k])
        ss, fs = os.stat(sp), os.stat(fp)
        if ss.st_dev != fs.st_dev:
            sys.exit(f'{a.standard} and {a.full} are on different filesystems: hard links cannot join them')
        if ss.st_ino == fs.st_ino:
            n_linked += 1
        elif ss.st_size != fs.st_size:
            n_size += 1
        else:
            pairs.append((sp, fp, ss.st_size))
    print(f'standard frames {len(S):,}: already linked {n_linked:,}, to check {len(pairs):,} '
          f'({sum(p[2] for p in pairs) / 1e9:.1f} GB), no twin {n_missing:,}, size differs {n_size:,}', flush=True)
    if not pairs:
        return
    hs = hash_files([p[0] for p in pairs], 'standard', a.workers)
    hf = hash_files([p[1] for p in pairs], 'full', a.workers)
    same = [(sp, fp, n) for sp, fp, n in pairs if hs[sp] == hf[fp]]
    print(f'identical {len(same):,} of {len(pairs):,} ({sum(n for *_, n in same) / 1e9:.1f} GB to free); '
          f'bytes differ {len(pairs) - len(same):,}', flush=True)
    if a.dry_run:
        return
    for sp, fp, _ in tqdm(same, unit='file', desc='link'):
        tmp = sp + '.linktmp'
        if os.path.lexists(tmp):
            os.remove(tmp)
        os.link(fp, tmp)
        os.replace(tmp, sp)
    print(f'done: {len(same):,} standard files are hard links to their full twins now', flush=True)


if __name__ == '__main__':
    main()
