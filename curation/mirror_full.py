#!/usr/bin/env python3
"""Give the full set the standard set's curation: the same scenes deleted, the same scenes locked.

    python curation/mirror_full.py                   # /data/parsed/tcar_nuscenes -> /data/parsed/tcar_nuscenes_full
    python curation/mirror_full.py --simulate        # what it would do; nothing is written
    python curation/mirror_full.py --standard <root> --full <root>

The two sets come from the same bags with the same frame selection (bag2nuscenes.py --full-out), so each scene
of one has a twin in the other: same recording, same sample timestamps (the tokens differ). Curation happens on
the standard set (review, 최종 확정 in the app); this brings the full set to the same state:

  - a final deletion the standard set has undone since the last sync is undone here too (its backup);
  - a full scene whose recording is in the standard set but whose twin is gone from it is deleted, with
    apply_selection's deletion: the dry run first, the files into _removed/<stamp>/, the renaming, --undo;
  - selection/locked.json takes the standard set's rounds, each scene token swapped for its twin's;
  - then every twin pair must have the same name, and the full set must pass apply_selection's check
    (references, key-frame files, devkit, CAN bus).

A standard scene without a twin while its recording is in the full set means the two conversions differ: nothing
is changed. A recording in only one of the sets is reported and left alone (convert it into the other first).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop('TCAR_DECISIONS_DIR', None)          # the app's copies of the decision lists are the standard set's
import apply_selection as A  # noqa: E402

VERSION = A.VERSION
MIRROR = 'mirror.json'                              # <full>/selection/: the last sync
DATA_ROOT = os.environ.get('TCAR_DATA_ROOT', '/data')


def twins_index(root):
    """({(log, sample timestamps): scene}, logs the set holds or held: log.json + *.import.json)."""
    tdir = os.path.join(root, VERSION)
    logfile = {l['token']: l['logfile'] for l in A._read(os.path.join(tdir, 'log.json'), [])}
    ts = {}
    for s in A._read(os.path.join(tdir, 'sample.json'), []):
        ts.setdefault(s['scene_token'], []).append(s['timestamp'])
    out = {}
    for sc in A._read(os.path.join(tdir, 'scene.json'), []):
        out[(logfile[sc['log_token']], tuple(sorted(ts.get(sc['token'], ()))))] = sc
    held = set(logfile.values()) | {f[:-len('.import.json')] for f in os.listdir(root) if f.endswith('.import.json')}
    return out, held


def compare(std, full):
    """How the full set stands against the standard set, without changing anything."""
    S, s_logs = twins_index(std)
    F, f_logs = twins_index(full)
    return {
        'S': S, 'F': F,
        'drop': sorted(F[k]['name'] for k in F if k[0] in s_logs and k not in S),
        'no_twin': sorted(S[k]['name'] for k in S if k[0] in f_logs and k not in F),
        'renamed': sorted((F[k]['name'], S[k]['name']) for k in S if k in F and F[k]['name'] != S[k]['name']),
        'only_standard': sorted(s_logs - f_logs), 'only_full': sorted(f_logs - s_logs),
        'twins': {S[k]['token']: F[k]['token'] for k in S if k in F},
    }


def _rounds(std, full, twins, logfile_of, backup=None):
    """The standard set's lock rounds with the full set's tokens. A round keeps the backup it had in the full
    set; one new in this sync gets this sync's backup (when it deleted anything)."""
    had = {r.get('mirrors'): r for r in A.lock_rounds(os.path.join(full, 'selection'))}
    out = []
    std_rounds = A.lock_rounds(os.path.join(std, 'selection'))
    for i, r in enumerate(std_rounds):
        toks = [twins[t] for t in r.get('tokens', []) if t in twins]
        prev = had.get(r['stamp'])
        out.append({'stamp': r['stamp'], 'created': r.get('created'), 'mirrors': r['stamp'],
                    'backup': prev.get('backup') if prev else (backup if i == len(std_rounds) - 1 else None),
                    'deleted': r.get('deleted', 0), 'n': len(toks),
                    'logs': sorted({logfile_of[t] for t in toks}), 'tokens': toks,
                    'standard_tokens_without_twin': sum(t not in twins for t in r.get('tokens', []))})
    return out


def sync(std, full, simulate=False, log=print):
    """Bring the full set to the standard set's curation. Returns the problems ([] = in step)."""
    sel = os.path.join(full, 'selection')
    std_stamps = {r['stamp'] for r in A.lock_rounds(os.path.join(std, 'selection'))}
    # 1. standard rounds undone since the last sync: undo their deletion here too (newest first)
    gone = [r for r in A.lock_rounds(sel) if r.get('mirrors') and r['mirrors'] not in std_stamps]
    for r in reversed(gone):
        if simulate:
            log(f"[mirror] (simulate) 표준에서 되돌린 라운드 {r['mirrors']}: full도 되돌림 ({r.get('backup') or '삭제 없음'})")
            continue
        if r.get('backup'):
            A._undo(r['backup'], full, log)
        rounds = [x for x in A.lock_rounds(sel) if x.get('mirrors') != r['mirrors']]
        A._write(os.path.join(sel, A.LOCK_NAME_JSON), {'version': 1, 'rounds': rounds}, indent=1)
        log(f"[mirror] 표준에서 되돌린 라운드 {r['mirrors']}: full도 되돌렸습니다")

    c = compare(std, full)
    log(f"[mirror] 쌍 {len(c['twins'])}개 · full에서 지울 씬 {len(c['drop'])}개 · 이름 다른 쌍 {len(c['renamed'])}개")
    if c['only_standard']:
        log(f"[mirror] 표준에만 있는 녹화 {len(c['only_standard'])}개 (full로 변환 필요): {', '.join(c['only_standard'])}")
    if c['only_full']:
        log(f"[mirror] full에만 있는 녹화 {len(c['only_full'])}개: {', '.join(c['only_full'])}")
    if c['no_twin']:
        p = [f"표준 씬 {len(c['no_twin'])}개가 full에 짝이 없습니다 (같은 녹화인데 변환 결과가 다름): {', '.join(c['no_twin'][:10])}"]
        log(f'[mirror] 문제: {p[0]} — 아무것도 바꾸지 않았습니다')
        return p
    if simulate:
        if c['drop']:
            return _with_decisions(full, c['drop'], lambda: A.simulate(full, log))
        return []

    # 2. the deletion, through apply_selection (dry run, backup, renaming)
    backup = None
    if c['drop']:
        problems = _with_decisions(full, c['drop'], lambda: A.simulate(full, log))
        if problems:
            return problems
        _write_decisions(full, c['drop'])
        backup = A._apply(full, log).get('backup')
        c = compare(std, full)
    # 3. locks: the standard rounds, in full-set tokens
    F_tok = {sc['token']: k[0] for k, sc in c['F'].items()}
    rounds = _rounds(std, full, c['twins'], F_tok, backup)
    os.makedirs(sel, exist_ok=True)
    A._write(os.path.join(sel, A.LOCK_NAME_JSON), {'version': 1, 'rounds': rounds}, indent=1)
    # 4. in step?
    problems = []
    if c['drop'] or c['no_twin']:
        problems.append(f"정리 후에도 지울 씬 {len(c['drop'])}개, 짝 없는 표준 씬 {len(c['no_twin'])}개")
    if c['renamed']:
        problems.append(f"이름이 다른 쌍 {len(c['renamed'])}개 (예: full {c['renamed'][0][0]} = 표준 {c['renamed'][0][1]}): "
                        "두 세트의 씬 구성이 다릅니다")
    A._write(os.path.join(sel, MIRROR), {
        'standard': os.path.realpath(std), 'synced': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'rounds': [r['stamp'] for r in rounds], 'twins': len(c['twins']), 'backup': backup,
        'only_standard': c['only_standard'], 'only_full': c['only_full'], 'problems': problems}, indent=1)
    for m in problems:
        log(f'[mirror] 문제: {m}')
    log(f"[mirror] full = 표준 큐레이션: 씬 {len(c['F'])}개, 잠금 {sum(r['n'] for r in rounds)}개"
        + (f", 이번에 {backup}로 옮김" if backup else ''))
    return problems


def _write_decisions(full, names):
    """The deletion apply_selection performs: these full scenes as confirmed 버리기 (by this sync)."""
    path = os.path.join(full, 'selection', 'human_decisions.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    now = time.strftime('%Y-%m-%dT%H:%M:%S')
    A._write(path, {'decisions': {n: {'decision': 'drop', 'by': 'mirror_full', 'time': now,
                                      'reason': '표준 세트에서 최종 삭제된 씬'} for n in names}}, indent=1)


def _with_decisions(full, names, fn):
    """Run fn with these drop decisions in place, then put the decisions file back as it was."""
    path = os.path.join(full, 'selection', 'human_decisions.json')
    before = open(path, 'rb').read() if os.path.isfile(path) else None
    _write_decisions(full, names)
    try:
        return fn()
    finally:
        if before is None:
            os.remove(path)
        else:
            with open(path, 'wb') as f:
                f.write(before)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--standard', default=os.path.join(DATA_ROOT, 'parsed', 'tcar_nuscenes'))
    ap.add_argument('--full', default=os.path.join(DATA_ROOT, 'parsed', 'tcar_nuscenes_full'))
    ap.add_argument('--simulate', action='store_true', help='report and dry-run the deletion; write nothing')
    ap.add_argument('--no-check', action='store_true', help='skip the check of the full set afterwards')
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.full, VERSION)):
        sys.exit(f'{args.full}: no full set')
    try:
        with A._Locked(args.full, 'mirror_full'):
            problems = sync(args.standard, args.full, args.simulate)
    except ValueError as e:
        sys.exit(str(e))
    if not problems and not args.simulate and not args.no_check:
        problems = A.check(args.full)
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
