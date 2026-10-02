"""Final deletion: take the scenes a person confirmed as 버리기 out of the dataset and renumber the rest without gaps.

The list is the review's confirmed decisions (selection/human_decisions.json, also written out as
selection/drop.txt). The filters only make candidates (후보: 객체 부족, 정지 중복); a candidate nobody confirmed
is never deleted. Scenes taken out are listed in selection/deleted.txt.

Rounds: --finalize ends a curation round: the confirmed 버리기 scenes go (as --apply), then every scene left is
locked (selection/locked.json, by token, so renumbering does not matter). A locked scene is never deleted and
never a candidate again; when new recordings are added, only their scenes can be. --undo of that round's backup
(or --unlock-last when it deleted nothing) takes the lock back.

    python apply_selection.py                        # plan only (nothing changes)
    python apply_selection.py --apply                # do it
    python apply_selection.py --apply --logs A-1_...,A-2_...   # only those logs' deletions
    python apply_selection.py --undo _removed/<stamp>

Nothing is deleted. Everything taken out goes to <dataroot>/_removed/<stamp>/ and manifest.json there lists
every move, so --undo puts the dataset back exactly:
  1. backup   the tables (v1.0-trainval/*.json), *.import.json and selection/ are copied first
  2. tables   deleted scenes, their samples, sample_data (key frames and sweeps), ego poses and annotations
              are removed; a log / map left without scenes is dropped
  3. files    the deleted scenes' samples/ + sweeps/ files and can_bus/ files are moved into the backup
  4. names    the remaining scenes take the dataset's sorted names in order, so there is no gap: with
              2, 3, 4 and 3 deleted, 4 becomes 3. The names are the first N of the official nuScenes train
              split (what the converter assigned), so the devkit still sees them all as train;
              can_bus files and *.import.json follow the new names
  5. caches   selection/ files keyed by scene name are re-keyed (objects, object counts, stop events, risk, LiDAR, CLIP), and so
              are the human decisions / labels of the scenes that stay; the old result and final list stay in
              the backup (the viewer reruns the rule on the new dataset)
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.basename(HERE)   # curation/ or tools/: the same code in two repositories
VERSION = 'v1.0-trainval'
LOCK_NAME = '.convert.lock'          # converter.LOCK_NAME: one writer per dataroot (conversion, remove_log, this)


class _Locked:
    """Hold the converter's lock on dataroot, so a parse and an apply never write the tables at once."""
    def __init__(self, dataroot, what):
        self.path, self.what = os.path.join(dataroot, LOCK_NAME), what

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            with open(self.path, encoding='utf-8', errors='replace') as f:
                holder = f.read().strip()
            raise ValueError(f'다른 작업이 데이터셋을 쓰는 중입니다 ({holder or LOCK_NAME}). 끝난 뒤 다시 하세요.') from None
        os.write(fd, f'pid {os.getpid()} {self.what}'.encode())
        os.close(fd)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def _read(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write(path, obj, indent=None):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


LOCK_NAME_JSON = 'locked.json'


def lock_rounds(sel_dir):
    return (_read(os.path.join(sel_dir, LOCK_NAME_JSON), {}) or {}).get('rounds') or []


def locked_tokens(sel_dir):
    """Scene tokens earlier rounds locked: never deleted, never candidates again."""
    return {t for r in lock_rounds(sel_dir) for t in r.get('tokens', [])}


def confirmed(sel_dir, want='drop'):
    """Scene names a person confirmed as `want` ('keep' | 'drop') in the review, with the decision record."""
    dec = (_read(os.path.join(sel_dir, 'human_decisions.json'), {}) or {}).get('decisions') or {}
    return {n: d for n, d in dec.items() if (d or {}).get('decision') == want}


CONFIRMED_TXT = (('keep', 'keep.txt', '남기기 확정: 사람이 남기기로 정한 씬과 필터를 모두 통과한 씬 (근거 열; 필터 통과는 규칙을 다시 돌리면 바뀔 수 있음)'),
                 ('drop', 'drop.txt', '버리기 확정: 사람이 검토 화면에서 버리기로 정한 씬. 최종 삭제는 이 씬들만 데이터셋에서 뺍니다'))


def _write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(path + '.tmp', path)


def _publish(sel_dir, names):
    """Copy these selection/ files to $TCAR_DECISIONS_DIR as well (the app sets it, e.g. to the code checkout's
    curation/, so the lists sit next to the code). The dataset's copies stay the ones the tools read."""
    out = os.environ.get('TCAR_DECISIONS_DIR')
    if not out or os.path.realpath(out) == os.path.realpath(sel_dir):
        return
    try:
        os.makedirs(out, exist_ok=True)
        for name in names:
            src = os.path.join(sel_dir, name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(out, name + '.tmp'))
                os.replace(os.path.join(out, name + '.tmp'), os.path.join(out, name))
    except OSError as e:
        print(f'[decisions] could not copy to {out}: {e}')


def write_confirmed(sel_dir, scenes):
    """selection/keep.txt and drop.txt: the confirmed scenes, one per line with log and token (a final deletion
    renumbers names; tokens stay). scenes: dicts with name, log, token. Rewritten on every decision."""
    info = {s['name']: s for s in scenes}
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    result = _read(os.path.join(sel_dir, 'result.json'), {}) or {}
    decided = set(confirmed(sel_dir, 'keep')) | set(confirmed(sel_dir, 'drop'))
    passed = {n: {'by': '필터 통과', 'updated': result.get('created', '')} for n, v in (result.get('scenes') or {}).items()
              if (v or {}).get('status') == 'keep' and n not in decided}     # no filter caught it: kept without a review
    lock = locked_tokens(sel_dir)
    held = {n: {'by': '잠김 (이전 라운드)', 'updated': ''} for n, x in info.items() if x.get('token') in lock}
    for want, fname, title in CONFIRMED_TXT:
        people = {n: dict(d, by='사람') for n, d in confirmed(sel_dir, want).items() if n not in held}
        rows = sorted((n, d) for n, d in {**(passed if want == 'keep' else {}), **people,
                                          **(held if want == 'keep' else {})}.items() if n in info)
        _write_lines(os.path.join(sel_dir, fname),
                     [f'# {title}', f'# 데이터셋: {os.path.realpath(os.path.dirname(sel_dir))}',
                      f"# {now} 기준 {len(rows)}개 (사람 {sum(d['by'] == '사람' for _, d in rows)} · 필터 통과 "
                      f"{sum(d['by'] == '필터 통과' for _, d in rows)} · 잠김 {sum(d['by'].startswith('잠김') for _, d in rows)}) "
                      "· scene\tlog\ttoken\t근거\t정한 시각"]
                     + [f"{n}\t{info[n].get('log', '')}\t{info[n].get('token', '')}\t{d['by']}\t{d.get('updated', '')}" for n, d in rows])
    _publish(sel_dir, ['human_decisions.json', 'keep.txt', 'drop.txt'])


def deleted_scenes(dataroot):
    """Every scene a final deletion took out that is still out: (stamp, name at the time, log, token)."""
    rows = []
    for m in sorted(glob.glob(os.path.join(dataroot, '_removed', '*', 'manifest.json'))):
        man = _read(m, {}) or {}
        if not man.get('done') or os.path.exists(os.path.join(os.path.dirname(m), 'UNDONE')):
            continue
        info = {s['name']: s for s in man.get('scenes', [])}
        rows += [(man.get('stamp', ''), n, info.get(n, {}).get('log', ''), info.get(n, {}).get('token', ''))
                 for n in man.get('deleted', [])]
    return rows


def write_deleted(dataroot):
    """selection/deleted.txt: deleted_scenes(), for people."""
    rows = deleted_scenes(dataroot)
    _write_lines(os.path.join(dataroot, 'selection', 'deleted.txt'),
                 ['# 삭제 완료: 최종 삭제로 데이터셋에서 뺀 씬. 파일은 _removed/<시각>/에 있고 되돌릴 수 있습니다',
                  f'# 데이터셋: {os.path.realpath(dataroot)}',
                  f'# {len(rows)}개 · 시각\tscene (뺄 때 이름)\tlog\ttoken'] + ['\t'.join(r) for r in rows])
    _publish(os.path.join(dataroot, 'selection'), ['deleted.txt'])


def plan(dataroot, logs=None):
    """What --apply would do, without changing anything: the confirmed 버리기 scenes go (candidates stay).
    logs: only the confirmed scenes in these logs (a course or a route); the renumbering is over the whole dataset."""
    sel = os.path.join(dataroot, 'selection')
    drop = confirmed(sel, 'drop')
    decided = set(drop) | set(confirmed(sel, 'keep'))
    rule = (_read(os.path.join(sel, 'result.json'), {}) or {}).get('scenes') or {}
    tdir = os.path.join(dataroot, VERSION)
    scenes = _read(os.path.join(tdir, 'scene.json'))
    names = sorted(s['name'] for s in scenes)
    in_scope = set(names)
    if logs:
        logfile = {l['token']: l['logfile'] for l in _read(os.path.join(tdir, 'log.json'), [])}
        in_scope = {s['name'] for s in scenes if logfile.get(s['log_token']) in set(logs)}
    lock = locked_tokens(sel)
    tok_of = {s['name']: s['token'] for s in scenes}
    held = sorted(n for n in drop if n in in_scope and tok_of.get(n) in lock)   # locked by an earlier round: never
    deleted = sorted(n for n in drop if n in in_scope and tok_of.get(n) not in lock)
    missing = sorted(set(drop) - set(names))          # decisions on names the dataset no longer has: ignored
    remaining = [n for n in names if n not in deleted]
    rename = {old: new for old, new in zip(remaining, names[:len(remaining)]) if old != new}
    sizes = _read(os.path.join(dataroot, 'selection', 'scene_bytes.json'), {}).get('scenes', {})
    tok = {s['name']: s['token'] for s in scenes}
    nbytes = sum((sizes.get(tok[n]) or {}).get('bytes', 0) for n in deleted)
    return {'deleted': deleted, 'missing': missing, 'locked_held': held, 'n_locked': sum(tok_of[n] in lock for n in names),
            'rename': rename, 'n_before': len(names), 'n_after': len(remaining),
            'bytes_moved': nbytes,
            'candidates': sum((rule.get(n) or {}).get('status') == 'remove' and n not in decided and tok_of.get(n) not in lock
                              for n in in_scope),
            'not_reviewed': 0, 'human_decisions': len(decided)}


def load_tables(tdir):
    return {os.path.basename(f)[:-5]: _read(f) for f in glob.glob(os.path.join(tdir, '*.json'))}


def prune_tables(T, deleted, rename):
    """The tables without the deleted scenes and everything only they use; the other scenes renamed. The same
    code for the real apply and the dry run (simulate), so what was checked is what gets written."""
    T = dict(T)
    del_scene = {s['token'] for s in T['scene'] if s['name'] in deleted}
    del_sample = {s['token'] for s in T['sample'] if s['scene_token'] in del_scene}
    del_sd = [x for x in T['sample_data'] if x['sample_token'] in del_sample]
    del_ego = {x['ego_pose_token'] for x in del_sd}
    del_sd_tok = {x['token'] for x in del_sd}
    T['scene'] = [dict(s, name=rename.get(s['name'], s['name'])) for s in T['scene'] if s['token'] not in del_scene]
    T['sample'] = [s for s in T['sample'] if s['token'] not in del_sample]
    T['sample_data'] = [x for x in T['sample_data'] if x['token'] not in del_sd_tok]
    T['ego_pose'] = [e for e in T['ego_pose'] if e['token'] not in del_ego]
    if T.get('sample_annotation'):
        T['sample_annotation'] = [a for a in T['sample_annotation'] if a['sample_token'] not in del_sample]
        ann = {a['token'] for a in T['sample_annotation']}
        T['instance'] = [i for i in T.get('instance', []) if i.get('first_annotation_token') in ann]
    live_logs = {s['log_token'] for s in T['scene']}
    T['log'] = [l for l in T['log'] if l['token'] in live_logs]
    T['map'] = [dict(m, log_tokens=[t for t in m['log_tokens'] if t in live_logs]) for m in T.get('map', [])]
    return T, del_scene, del_sample, del_sd


def integrity(T, most=40):
    """Every reference the nuScenes tables make, checked: what the devkit (and code walking prev / next) relies on.
    Returns the problems found (at most `most`), [] when the tables are whole."""
    out = []
    add = lambda msg: len(out) < most and out.append(msg)
    ix = {k: {r['token']: r for r in T.get(k, [])} for k in
          ('scene', 'sample', 'sample_data', 'ego_pose', 'calibrated_sensor', 'sensor', 'log', 'map', 'instance',
           'sample_annotation', 'category', 'attribute', 'visibility')}
    names = [s['name'] for s in T['scene']]
    if len(set(names)) != len(names):
        add('scene 이름 중복')
    try:
        from nuscenes.utils.splits import create_splits_scenes
        official = {n for v in create_splits_scenes().values() for n in v}
        for n in names:
            if n not in official:
                add(f'{n}: devkit 공식 split에 없는 scene 이름')
    except ImportError:
        pass
    per_scene = {}
    for smp in T['sample']:
        per_scene.setdefault(smp['scene_token'], []).append(smp)
        if smp['scene_token'] not in ix['scene']:
            add(f"sample {smp['token']}: 없는 scene을 가리킴")
        for k in ('prev', 'next'):
            o = smp[k]
            if o and (o not in ix['sample'] or ix['sample'][o][{'prev': 'next', 'next': 'prev'}[k]] != smp['token']):
                add(f"sample {smp['token']}: {k} 연결이 끊김")
    for sc in T['scene']:
        if sc['log_token'] not in ix['log']:
            add(f"{sc['name']}: 없는 log를 가리킴")
        f, l = ix['sample'].get(sc['first_sample_token']), ix['sample'].get(sc['last_sample_token'])
        if not f or not l or f['scene_token'] != sc['token'] or l['scene_token'] != sc['token']:
            add(f"{sc['name']}: 첫/끝 sample이 없거나 다른 scene 것")
            continue
        n, t, seen = 1, f, {f['token']}
        while t['next'] and n <= sc['nbr_samples'] + 1:
            t = ix['sample'].get(t['next'])
            if not t or t['token'] in seen:
                break
            seen.add(t['token']); n += 1
        if n != sc['nbr_samples'] or t is not l or len(per_scene.get(sc['token'], [])) != sc['nbr_samples']:
            add(f"{sc['name']}: sample 사슬이 nbr_samples({sc['nbr_samples']})와 안 맞음")
    for sd in T['sample_data']:
        if sd['sample_token'] not in ix['sample']:
            add(f"sample_data {sd['token']}: 없는 sample을 가리킴")
        if sd['ego_pose_token'] not in ix['ego_pose']:
            add(f"sample_data {sd['token']}: 없는 ego_pose를 가리킴")
        if sd['calibrated_sensor_token'] not in ix['calibrated_sensor']:
            add(f"sample_data {sd['token']}: 없는 calibrated_sensor를 가리킴")
        for k in ('prev', 'next'):
            o = sd[k]
            if o and (o not in ix['sample_data'] or ix['sample_data'][o][{'prev': 'next', 'next': 'prev'}[k]] != sd['token']):
                add(f"sample_data {sd['token']}: {k} 연결이 끊김")
    for cs in T['calibrated_sensor']:
        if cs['sensor_token'] not in ix['sensor']:
            add(f"calibrated_sensor {cs['token']}: 없는 sensor를 가리킴")
    mapped = {t for m in T.get('map', []) for t in m['log_tokens']}
    for lg in T['log']:
        if lg['token'] not in mapped:
            add(f"log {lg['logfile']}: 어느 map에도 없음")
    for t in mapped:
        if t not in ix['log']:
            add(f"map: 없는 log {t}를 가리킴")
    for a in T.get('sample_annotation', []):
        if a['sample_token'] not in ix['sample'] or a['instance_token'] not in ix['instance']:
            add(f"sample_annotation {a['token']}: 없는 sample/instance를 가리킴")
        for k in ('prev', 'next'):
            if a[k] and a[k] not in ix['sample_annotation']:
                add(f"sample_annotation {a['token']}: {k} 연결이 끊김")
    for i in T.get('instance', []):
        if i['first_annotation_token'] not in ix['sample_annotation'] or i['last_annotation_token'] not in ix['sample_annotation']:
            add(f"instance {i['token']}: 첫/끝 annotation이 없음")
    return out


def devkit_load(dataroot, version=VERSION, can_bus=True, log=print):
    """Load the dataset with the nuscenes-devkit (and the CAN bus API for every scene): None when fine, else why."""
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        log('[check] nuscenes-devkit 없음: devkit 불러오기는 건너뜀')
        return None
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        log(f'[check] devkit NuScenes: scene {len(nusc.scene)}, sample {len(nusc.sample)}, sample_data {len(nusc.sample_data)}')
        if can_bus and os.path.isdir(os.path.join(dataroot, 'can_bus')):
            from nuscenes.can_bus.can_bus_api import NuScenesCanBus
            cb = NuScenesCanBus(dataroot=dataroot)
            bad = []
            for sc in nusc.scene:
                try:
                    cb.get_messages(sc['name'], 'pose')
                except Exception as e:           # a scene the API refuses or whose file is missing
                    bad.append(f"{sc['name']}: {type(e).__name__}")
            if bad:
                return 'NuScenesCanBus: ' + ', '.join(bad[:10])
            log(f'[check] devkit NuScenesCanBus: {len(nusc.scene)} scene의 pose 메시지 읽힘')
    except Exception as e:
        return f'devkit 불러오기 실패: {type(e).__name__}: {e}'
    return None


def simulate(dataroot, log=print):
    """The final deletion as it would be, in memory: the pruned tables checked reference by reference, the CAN bus
    files the renaming needs looked up, and the result loaded with the devkit from a scratch copy of the tables.
    Nothing in the dataset changes. Returns the problems ([] = safe to apply)."""
    import shutil as _sh
    import tempfile
    p = plan(dataroot)
    tdir = os.path.join(dataroot, VERSION)
    log(f"[simulate] 삭제 {len(p['deleted'])}개, 이름 당김 {len(p['rename'])}개를 표에만 적용해 봅니다")
    T0 = load_tables(tdir)
    problems = [f'삭제 전 표: {m}' for m in integrity(T0, 10)]
    T, del_scene, del_sample, del_sd = prune_tables(T0, set(p['deleted']), p['rename'])
    log(f"[simulate] 표: scene {len(T0['scene'])} -> {len(T['scene'])}, sample -{len(del_sample)}, sample_data -{len(del_sd)}")
    problems += integrity(T)
    cb = os.path.join(dataroot, 'can_bus')
    if os.path.isdir(cb):                         # every scene that had CAN bus files still has them, under its new name
        have = {f.split('_', 1)[0] for f in os.listdir(cb)}
        back = {v: k for k, v in p['rename'].items()}
        for s in T['scene']:
            old = back.get(s['name'], s['name'])
            if old in have and not glob.glob(os.path.join(cb, old + '_*')):
                problems.append(f"{s['name']}: can_bus 파일({old}_*)이 없음")
    tmp = tempfile.mkdtemp(prefix='tcar-simulate-')
    try:
        os.makedirs(os.path.join(tmp, VERSION))
        if os.path.isdir(os.path.join(dataroot, 'maps')):   # the devkit checks the map masks exist
            os.symlink(os.path.realpath(os.path.join(dataroot, 'maps')), os.path.join(tmp, 'maps'))
        for k, v in T.items():
            _write(os.path.join(tmp, VERSION, k + '.json'), v)
        why = devkit_load(tmp, can_bus=False, log=log)
        if why:
            problems.append(why)
    finally:
        _sh.rmtree(tmp, ignore_errors=True)
    for m in problems:
        log(f'[simulate] 문제: {m}')
    log('[simulate] 이상 없음: 이대로 지워도 devkit이 읽습니다' if not problems else f'[simulate] 문제 {len(problems)}개: 지우지 않습니다')
    return problems


def check(dataroot, log=print):
    """The dataset as it is: every table reference, key-frame files on disk, the devkit and the CAN bus API."""
    T = load_tables(os.path.join(dataroot, VERSION))
    problems = integrity(T)
    miss = [x['filename'] for x in T['sample_data'] if x['is_key_frame'] and not os.path.isfile(os.path.join(dataroot, x['filename']))]
    if miss:
        problems.append(f'key frame 파일 {len(miss)}개가 없음 (예: {miss[0]})')
    why = devkit_load(dataroot, log=log)
    if why:
        problems.append(why)
    for m in problems:
        log(f'[check] 문제: {m}')
    log('[check] 이상 없음' if not problems else f'[check] 문제 {len(problems)}개')
    return problems


def _rekey(d, rename, deleted):
    """dict keyed by scene name -> deleted scenes dropped, the others under their new names."""
    return {rename.get(k, k): v for k, v in d.items() if k not in deleted}


def apply(dataroot, log=print, logs=None):
    with _Locked(dataroot, 'apply_selection'):
        return _apply(dataroot, log, logs)


def finalize(dataroot, log=print, precheck=True):
    """End the round: the confirmed 버리기 scenes go, then every scene left is locked. Refused while candidates
    are still open (they would be locked unseen), and (precheck) when the dry run finds the tables would break."""
    with _Locked(dataroot, 'apply_selection --finalize'):
        p = plan(dataroot)
        if p['candidates']:
            raise ValueError(f"아직 확정하지 않은 후보 {p['candidates']}개가 있습니다. 모두 확정한 뒤 최종 확정하세요")
        if precheck and p['deleted']:
            problems = simulate(dataroot, log)
            if problems:
                raise ValueError(f"미리 검사에서 문제 {len(problems)}개: 지우지 않았습니다 ({problems[0]})")
        r = _apply(dataroot, log) if p['deleted'] else dict(p, backup=None)
        sel, tdir = os.path.join(dataroot, 'selection'), os.path.join(dataroot, VERSION)
        scenes = _read(os.path.join(tdir, 'scene.json'), [])
        logfile = {l['token']: l['logfile'] for l in _read(os.path.join(tdir, 'log.json'), [])}
        before = locked_tokens(sel)
        new = [s for s in scenes if s['token'] not in before]
        rounds = lock_rounds(sel) + [{'stamp': time.strftime('%Y%m%d-%H%M%S'), 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                      'backup': r.get('backup'), 'deleted': len(p['deleted']), 'n': len(new),
                                      'logs': sorted({logfile.get(s['log_token'], '') for s in new}),
                                      'tokens': [s['token'] for s in new]}]
        os.makedirs(sel, exist_ok=True)
        _write(os.path.join(sel, LOCK_NAME_JSON), {'version': 1, 'rounds': rounds}, indent=1)
        with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === 최종 확정: {len(p['deleted'])}개 삭제, 남은 {len(new)}개 잠금 "
                    f"(이전에 잠긴 {len(scenes) - len(new)}개) ===\n")
        log(f"[finalize] deleted {len(p['deleted'])}, locked {len(new)} more scenes ({len(before) + len(new)} in all)")
        return dict(r, locked=len(new))


def unlock_last(dataroot, log=print):
    """Take back the last round's lock when that round deleted nothing (a round that did is undone with --undo)."""
    with _Locked(dataroot, 'apply_selection --unlock-last'):
        sel = os.path.join(dataroot, 'selection')
        rounds = lock_rounds(sel)
        if not rounds:
            raise ValueError('잠긴 라운드가 없습니다')
        if rounds[-1].get('backup'):
            raise ValueError(f"이 라운드는 씬을 지웠습니다: --undo {rounds[-1]['backup']} 로 되돌리세요")
        _write(os.path.join(sel, LOCK_NAME_JSON), {'version': 1, 'rounds': rounds[:-1]}, indent=1)
        with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === 최종 확정 되돌림 (잠금 해제 {rounds[-1]['n']}개) ===\n")
        log(f"[unlock] {rounds[-1]['n']} scenes unlocked")


def _apply(dataroot, log, logs=None):
    p = plan(dataroot, logs)
    if p['missing']:
        log(f"[apply] 데이터셋에 없는 이름의 버리기 결정 {len(p['missing'])}개는 건너뜁니다: {p['missing']}")
    deleted, rename = set(p['deleted']), p['rename']
    if not deleted:
        return dict(p, backup=None)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    B = os.path.join(dataroot, '_removed', stamp)
    tdir, sel = os.path.join(dataroot, VERSION), os.path.join(dataroot, 'selection')
    os.makedirs(B)
    man = {'stamp': stamp, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'deleted': sorted(deleted), 'rename': rename,
           'moves': [], 'done': False}

    def save_manifest():
        _write(os.path.join(B, 'manifest.json'), man, indent=1)

    def move(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        man['moves'].append([src, dst])

    # 1. backup (copies) --------------------------------------------------------------------------------
    shutil.copytree(tdir, os.path.join(B, 'tables'))
    os.makedirs(os.path.join(B, 'import'))
    for f in glob.glob(os.path.join(dataroot, '*.import.json')):
        shutil.copy2(f, os.path.join(B, 'import'))
    if os.path.isdir(sel):                        # models/ (detector weights) is not touched by apply
        shutil.copytree(sel, os.path.join(B, 'selection'), ignore=shutil.ignore_patterns('models'))
    save_manifest()
    log(f'[apply] backup -> {B}')

    # 2. tables -----------------------------------------------------------------------------------------
    T = load_tables(tdir)
    logfile = {l['token']: l['logfile'] for l in T['log']}
    man['scenes'] = [{'name': s['name'], 'log': logfile.get(s['log_token'], ''), 'token': s['token']}
                     for s in T['scene'] if s['name'] in deleted]
    T, del_scene, del_sample, del_sd = prune_tables(T, deleted, rename)
    for k, v in T.items():
        _write(os.path.join(tdir, k + '.json'), v, indent=2)          # same layout as the converter wrote
    log(f"[apply] tables: -{len(del_scene)} scenes, -{len(del_sample)} samples, -{len(del_sd)} sample_data")

    # 3. files ------------------------------------------------------------------------------------------
    for x in del_sd:
        src = os.path.join(dataroot, x['filename'])
        if os.path.isfile(src):
            move(src, os.path.join(B, 'files', x['filename']))
    for n in sorted(deleted):
        for f in glob.glob(os.path.join(dataroot, 'can_bus', n + '_*')):
            move(f, os.path.join(B, 'can_bus', os.path.basename(f)))
        if os.path.isdir(os.path.join(dataroot, 'ext', n)):            # full dataset: per-scene sidecar topics
            move(os.path.join(dataroot, 'ext', n), os.path.join(B, 'ext', n))
    save_manifest()
    log(f"[apply] moved {len(man['moves'])} files into the backup")

    # 4. names: can_bus files in two steps (a new name can be an old name still in use), import.json -------
    cb = os.path.join(dataroot, 'can_bus')
    staged = []
    for old, new in rename.items():
        for f in glob.glob(os.path.join(cb, old + '_*')):
            tmp = os.path.join(cb, '__renaming__' + new + os.path.basename(f)[len(old):])
            move(f, tmp)
            staged.append(tmp)
    ext = os.path.join(dataroot, 'ext')
    for old, new in rename.items():
        if os.path.isdir(os.path.join(ext, old)):
            tmp = os.path.join(ext, '__renaming__' + new)
            move(os.path.join(ext, old), tmp)
            staged.append(tmp)
    for tmp in staged:
        move(tmp, os.path.join(os.path.dirname(tmp), os.path.basename(tmp)[len('__renaming__'):]))
    for f in glob.glob(os.path.join(dataroot, '*.import.json')):
        j = _read(f)
        j['scenes'] = [dict(s, name=rename.get(s['name'], s['name'])) for s in j.get('scenes', []) if s['name'] not in deleted]
        _write(f, j, indent=1)
    save_manifest()
    log(f'[apply] renamed {len(rename)} scenes')

    # 5. selection/ -------------------------------------------------------------------------------------
    for name in ('objects.json', 'stop_events.json', 'risk.json', 'object_counts.json'):
        path = os.path.join(sel, name)
        j = _read(path)
        if j and isinstance(j.get('scenes'), dict):
            j['scenes'] = _rekey(j['scenes'], rename, deleted)
            _write(path, j, indent=1)
    for name in ('lidar_cache.pkl', 'risk_cache.pkl'):
        path = os.path.join(sel, name)
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                c = pickle.load(f)
            with open(path, 'wb') as f:
                pickle.dump(_rekey(c, rename, deleted), f)
    for path in glob.glob(os.path.join(sel, 'image_emb_*.npz')):
        import numpy as np
        z = np.load(path)
        keep = [i for i, n in enumerate(z['names']) if str(n) not in deleted]
        np.savez(path, names=np.array([rename.get(str(z['names'][i]), str(z['names'][i])) for i in keep]), emb=z['emb'][keep])
    for name, key in (('human_labels.json', 'labels'), ('pair_labels.json', 'pairs'), ('human_decisions.json', 'decisions')):
        # decisions on the scenes that stay (e.g. 남기기 against the rule) follow them to their new names, so the
        # rule rerun on the new data cannot silently propose them again
        path = os.path.join(sel, name)
        j = _read(path)
        if j and isinstance(j.get(key), dict):
            if key == 'pairs':
                j[key] = {'|'.join(sorted(rename.get(a, a) for a in k.split('|'))): v for k, v in j[key].items()
                          if not set(k.split('|')) & deleted}
            else:
                j[key] = _rekey(j[key], rename, deleted)
            _write(path, j, indent=1)
    for name in ('lidar_diff.json', 'scene_bytes.json'):          # keyed by token; drop by token, then rename
        path = os.path.join(sel, name)                            # (a new name can equal a deleted scene's old name)
        j = _read(path)
        if j and isinstance(j.get('scenes'), dict):
            j['scenes'] = {t: v for t, v in j['scenes'].items() if t not in del_scene}
            for v in j['scenes'].values():
                if isinstance(v, dict) and v.get('name') in rename:
                    v['name'] = rename[v['name']]
            _write(path, j, indent=1)
    for name in ('result.json', 'final_selection.json', 'pairs.csv', 'scene_features.csv'):
        path = os.path.join(sel, name)                            # old names; the copies stay in the backup
        if os.path.isfile(path):
            os.remove(path)
    with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === 삭제 확정 적용: {len(deleted)}개 scene을 {B} 로 이동, "
                f"{len(rename)}개 이름 당김 ({', '.join(f'{a[-4:]}->{b[-4:]}' for a, b in rename.items())}) ===\n")
    write_confirmed(sel, [{'name': s['name'], 'log': logfile.get(s['log_token'], ''), 'token': s['token']} for s in T['scene']])
    man['done'] = True
    save_manifest()
    write_deleted(dataroot)
    log('[apply] done')
    return dict(p, backup=B)


def undo(backup, dataroot, log=print):
    """Put the dataset back as it was before the --apply that wrote `backup`."""
    with _Locked(dataroot, 'apply_selection --undo'):
        return _undo(backup, dataroot, log)


def _undo(backup, dataroot, log):
    man = _read(os.path.join(backup, 'manifest.json'))
    if not man:
        raise ValueError(f'{backup}/manifest.json 없음')
    for src, dst in reversed(man['moves']):
        if os.path.exists(dst):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            os.replace(dst, src)
    tdir = os.path.join(dataroot, VERSION)
    for f in glob.glob(os.path.join(backup, 'tables', '*.json')):
        shutil.copy2(f, os.path.join(tdir, os.path.basename(f)))
    for f in glob.glob(os.path.join(backup, 'import', '*.json')):
        shutil.copy2(f, os.path.join(dataroot, os.path.basename(f)))
    sel = os.path.join(dataroot, 'selection')
    for f in glob.glob(os.path.join(backup, 'selection', '*')):
        if os.path.isfile(f):                     # backups before models/ was left out hold that directory too
            shutil.copy2(f, os.path.join(sel, os.path.basename(f)))
    with open(os.path.join(sel, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  === 되돌림: {backup} ===\n")
    if not os.path.isfile(os.path.join(backup, 'selection', LOCK_NAME_JSON)):   # this round's lock came after the
        try:                                                                  # backup: nothing was locked before it
            os.remove(os.path.join(sel, LOCK_NAME_JSON))
        except FileNotFoundError:
            pass
    open(os.path.join(backup, 'UNDONE'), 'w').close()
    logfile = {l['token']: l['logfile'] for l in _read(os.path.join(tdir, 'log.json'), [])}
    write_confirmed(sel, [{'name': s['name'], 'log': logfile.get(s['log_token'], ''), 'token': s['token']}
                          for s in _read(os.path.join(tdir, 'scene.json'), [])])
    write_deleted(dataroot)
    log(f"[undo] restored {len(man['moves'])} moved files, tables, import.json and selection/ from {backup}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=None, help='dataset folder (default: sample_viewer.default_dataroot())')
    ap.add_argument('--apply', action='store_true', help='move the confirmed 버리기 scenes out and renumber (default: plan only)')
    ap.add_argument('--undo', metavar='BACKUP_DIR', help='restore from a backup written by --apply')
    ap.add_argument('--finalize', action='store_true',
                    help='end the round: delete the confirmed 버리기 scenes, then lock every scene left (never deleted later)')
    ap.add_argument('--unlock-last', action='store_true', help="take back the last round's lock (a round that deleted nothing)")
    ap.add_argument('--simulate', action='store_true',
                    help='dry run: apply the deletion to the tables in memory, check every reference, load them with the devkit')
    ap.add_argument('--check', action='store_true', help='check the dataset as it is (references, key-frame files, devkit, CAN bus)')
    ap.add_argument('--no-precheck', action='store_true', help='--finalize without its dry run (the app runs --simulate itself first)')
    ap.add_argument('--logs', default='', help='comma separated log names: only the confirmed scenes in them')
    args = ap.parse_args()
    if args.dataroot is None:
        sys.path.insert(0, HERE)
        import sample_viewer
        args.dataroot = sample_viewer.default_dataroot()
    logs = [x for x in args.logs.split(',') if x] or None
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass
    if args.undo:
        return undo(args.undo, args.dataroot)
    if args.unlock_last:
        return unlock_last(args.dataroot)
    if args.check:
        return 1 if check(args.dataroot) else 0
    if args.simulate:
        return 1 if simulate(args.dataroot) else 0
    p = plan(args.dataroot, logs)
    print(f"scene {p['n_before']} -> {p['n_after']}  (버리기 확정 {len(p['deleted'])}개 삭제: "
          f"{', '.join(n[-4:] for n in p['deleted']) or '없음'}, {p['bytes_moved'] / 1e9:.1f} GB 이동)")
    print('이름 당김: ' + (', '.join(f'{a[-4:]}->{b[-4:]}' for a, b in p['rename'].items()) or '없음'))
    if p['candidates']:
        print(f"필터에 걸린 후보 {p['candidates']}개는 아직 확정 전이라 남습니다")
    if p['locked_held']:
        print(f"잠긴 씬의 버리기 결정 {len(p['locked_held'])}개는 무시합니다 (이전 라운드에서 확정): {p['locked_held']}")
    if args.finalize:
        r = finalize(args.dataroot, precheck=not args.no_precheck)
        print(f"최종 확정 완료: {len(p['deleted'])}개 삭제, {r['locked']}개 잠금" + (f" · 되돌리기: --undo \"{r['backup']}\"" if r.get('backup')
                                                                         else ' · 되돌리기: --unlock-last'))
    elif args.apply:
        r = apply(args.dataroot, logs=logs)
        print(f"완료. 되돌리기: python {TOOLS}/apply_selection.py --undo \"{r['backup']}\"" if r['backup'] else '뺄 씬이 없습니다')
    else:
        print('(계획만 표시 · 실제 적용은 --apply)')


if __name__ == '__main__':
    sys.exit(main())
