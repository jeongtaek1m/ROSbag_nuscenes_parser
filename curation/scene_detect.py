"""Road users in every key frame (6 cameras x all samples) with a COCO YOLO detector -> <dataroot>/selection/.

Only the six COCO road-user classes are used: person, bicycle, motorcycle, car, bus, truck. The detector is
closed-set (no open-vocabulary prompts), so nothing ill-defined (debris, cones, "obstacle", ...) comes in.

Detection (GPU, ~25 ms per 1080p image with yolo26l at 1280): raw boxes with confidence >= 0.25 are cached per
scene token in selection/detections_<model>_<imgsz>.npz (tokens survive the renaming apply_selection does), so
the counting below can change without running the detector again. Missing scenes only.

Counting -> selection/object_counts.json (keyed by scene name):
  - confidence >= --conf
  - the ego body is dropped: border box locations present in >= 50 % of all key frames of a camera (IoU >= 0.7;
    the body below the front-left / front-right cameras is in ~95 %, the most frequent real car spot ~20 %),
    plus any vehicle box touching the bottom edge and wider than 40 % of the frame (the rear hood)
  - riders: a person whose box lies >= 30 % on a bicycle / motorcycle box counts as that two-wheeler only
  - near = box height >= 50 px (same as scene_objects.py)
  - per sample, the 6 cameras are summed (overlapping cameras can count one object twice: a relative measure);
    per scene mean / min / max / share of samples with any, for vehicles, pedestrians, two-wheelers, all

    python scene_detect.py                 # detect missing scenes, then count and report
    python scene_detect.py --report        # count and report only (no GPU)
    python scene_detect.py --logs A-1_2026-09-28-14-40-58   # only that log's missing scenes
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_viewer as sv  # noqa: E402

CAMS = sv.CAM_ORDER
COCO = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
GROUP = {'person': 'ped', 'bicycle': 'two', 'motorcycle': 'two', 'car': 'veh', 'bus': 'veh', 'truck': 'veh'}
GROUPS = ('veh', 'ped', 'two', 'all')
GROUP_KO = {'veh': '차량', 'ped': '보행자', 'two': '이륜차', 'all': '전체'}
RAW_CONF, NEAR_PX, RIDER_IOA = 0.25, 50, 0.3
EGO_IOU, EGO_SHARE, EGO_DROP_IOU = 0.7, 0.5, 0.5
COUNTED, LOW, EGO, RIDER = 0, 1, 2, 3            # box flags (the viewer draws them)
W, H = 1920, 1080
VERSION = 1


def cache_path(dataroot, model, imgsz):
    return os.path.join(dataroot, 'selection', f'detections_{os.path.splitext(os.path.basename(model))[0]}_{imgsz}.npz')


def load_cache(path):
    """{scene token: {'frame', 'cam', 'cls', 'conf', 'box'}} from the npz (keys '<token>.<field>')."""
    out = {}
    if os.path.isfile(path):
        z = np.load(path)
        for k in z.files:
            tok, field = k.split('.', 1)
            out.setdefault(tok, {})[field] = z[k]
    return out


def save_cache(path, cache):
    tmp = path + '.tmp.npz'
    np.savez(tmp, **{f'{tok}.{f}': v for tok, d in cache.items() for f, v in d.items()})
    os.replace(tmp, path)


def detect(db, dataroot, model_path, imgsz, path, batch=16, workers=8, logs=None):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from ultralytics import YOLO
    cache = load_cache(path)
    todo = [s for s in db['scenes'] if s['token'] not in cache and (not logs or s['log'] in logs)]
    print(f"[detect] {len(todo)} scenes to detect" + (f" in {len(logs)} log(s)" if logs else ''), flush=True)
    if not todo:
        return cache
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = YOLO(model_path)                        # downloads the weights there on first use
    ex = ThreadPoolExecutor(workers)                # JPEG reads overlap the GPU (cv2 releases the GIL)

    def submit(s):
        jobs = [(k, c, os.path.join(dataroot, db['by_sample'][t][cam]['filename']))
                for k, t in enumerate(s['samples']) for c, cam in enumerate(CAMS)]
        return jobs, [ex.submit(cv2.imread, p) for _, _, p in jobs]
    t0, nxt = time.time(), submit(todo[0])
    try:
        for i, s in enumerate(todo):
            jobs, futs = nxt
            if i + 1 < len(todo):
                nxt = submit(todo[i + 1])           # read the next scene while this one runs
            rows = []
            for b in range(0, len(jobs), batch):
                ims = [f.result() for f in futs[b:b + batch]]
                for (k, c, p), im in zip(jobs[b:b + batch], ims):
                    if im is None:
                        raise IOError(f'cannot read {p}')
                res = model.predict(ims, imgsz=imgsz, half=True, conf=RAW_CONF, classes=sorted(COCO), verbose=False)
                for (k, c, _), r in zip(jobs[b:b + batch], res):
                    bx = r.boxes
                    n = len(bx)
                    if n:
                        rows.append((np.full(n, k, np.int16), np.full(n, c, np.int8), bx.cls.cpu().numpy().astype(np.uint8),
                                     bx.conf.cpu().numpy().astype(np.float16), bx.xyxy.cpu().numpy().astype(np.float32)))
            rows = rows or [(np.zeros(0, np.int16), np.zeros(0, np.int8), np.zeros(0, np.uint8), np.zeros(0, np.float16),
                             np.zeros((0, 4), np.float32))]
            cache[s['token']] = dict(zip(('frame', 'cam', 'cls', 'conf', 'box'), (np.concatenate(x) for x in zip(*rows))))
            el = time.time() - t0
            print(f"  {s['name']} ({i + 1}/{len(todo)}) {len(cache[s['token']]['cls'])} boxes  "
                  f"{el:.0f}s, ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left", flush=True)
            if (i + 1) % 10 == 0:
                save_cache(path, cache)
    finally:
        save_cache(path, cache)
        ex.shutdown(wait=False)
    return cache


def iou_matrix(a, b):
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-6)


def ego_regions(cache, conf, n_images):
    """Per camera index: boxes [k, 4] where the ego vehicle itself gets detected. Candidates are border boxes;
    a location counts when boxes at it (IoU >= EGO_IOU) appear in >= EGO_SHARE of the n_images key frames."""
    toks = sorted(cache)
    need = max(3, EGO_SHARE * n_images)
    out = {}
    for c in range(len(CAMS)):
        bs, sc = [], []
        for si, t in enumerate(toks):
            d = cache[t]
            b = d['box']
            m = (d['cam'] == c) & (d['conf'].astype(np.float32) >= conf) & \
                ((b[:, 3] > 0.95 * H) | (b[:, 0] < 0.01 * W) | (b[:, 2] > 0.99 * W))
            bs.append(b[m]); sc.append(si * 100000 + d['frame'][m])
        b, img = np.concatenate(bs), np.concatenate(sc)
        sup = np.zeros(len(b), np.int64)
        for i in range(0, len(b), 512):                 # support = distinct key frames with a box at the same place
            sup[i:i + 512] = [len(np.unique(img[h])) for h in iou_matrix(b[i:i + 512], b) >= EGO_IOU]
        regions, free = [], np.ones(len(b), bool)
        for i in np.argsort(-sup, kind='stable'):
            if sup[i] < need:
                break
            if not free[i]:
                continue
            member = iou_matrix(b[i:i + 1], b)[0] >= EGO_IOU
            regions.append(np.median(b[member], 0))
            free &= iou_matrix(regions[-1][None], b)[0] < EGO_DROP_IOU
        out[c] = np.array(regions, np.float32).reshape(-1, 4)
    return out


def box_flags(d, conf, ego):
    """(flag per cached box, class names): COUNTED, or why not -- LOW confidence, EGO body, RIDER of a two-wheeler."""
    flag = np.where(d['conf'].astype(np.float32) >= conf, COUNTED, LOW)
    box, cls = d['box'], d['cls']
    x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    name = np.array([COCO[int(c)] for c in cls]) if len(cls) else np.zeros(0, str)
    veh = np.isin(name, ('car', 'bus', 'truck'))
    flag[(flag == COUNTED) & veh & (y2 > 0.97 * H) & ((x2 - x1) > 0.4 * W)] = EGO     # ego body (rear hood)
    for c, reg in ego.items():                                                      # ego body at its recurring places
        m = (d['cam'] == c) & (flag == COUNTED)
        if len(reg) and m.any():
            idx = np.nonzero(m)[0]
            flag[idx[iou_matrix(box[idx], reg).max(1) >= EGO_DROP_IOU]] = EGO
    keep = flag == COUNTED
    # riders: a person mostly on a two-wheeler box of the same image is part of that two-wheeler
    person, two = name == 'person', np.isin(name, ('bicycle', 'motorcycle'))
    img = d['frame'].astype(np.int64) * len(CAMS) + d['cam']
    for i in np.nonzero(keep & person)[0]:
        j = np.nonzero(keep & two & (img == img[i]))[0]
        if len(j):
            iw = np.clip(np.minimum(x2[i], x2[j]) - np.maximum(x1[i], x1[j]), 0, None)
            ih = np.clip(np.minimum(y2[i], y2[j]) - np.maximum(y1[i], y1[j]), 0, None)
            if (iw * ih / max((x2[i] - x1[i]) * (y2[i] - y1[i]), 1e-6)).max() >= RIDER_IOA:
                flag[i] = RIDER
    return flag, name


def count_scene(d, n_samples, conf, ego):
    """Per-sample counts (6 cameras summed) per group, all boxes and near boxes; per-class totals."""
    cnt = {near: {g: np.zeros(n_samples, np.int32) for g in GROUPS} for near in (False, True)}
    flag, name = box_flags(d, conf, ego)
    keep = flag == COUNTED
    near = (d['box'][:, 3] - d['box'][:, 1]) >= NEAR_PX
    for i in np.nonzero(keep)[0]:
        g, k = GROUP[name[i]], d['frame'][i]
        for nr in ((False, True) if near[i] else (False,)):
            cnt[nr][g][k] += 1
            cnt[nr]['all'][k] += 1
    classes = {c: [int((keep & (name == c)).sum()), int((keep & near & (name == c)).sum())] for c in COCO.values()}
    return cnt, classes


def stats(a):
    return {'mean': float(a.mean()), 'min': int(a.min()), 'max': int(a.max()), 'any': float((a > 0).mean())}


def summarise(db, cache, conf, ego):
    out = {}
    for s in db['scenes']:
        d = cache.get(s['token'])
        if d is None:
            continue
        cnt, classes = count_scene(d, len(s['samples']), conf, ego)
        out[s['name']] = {'token': s['token'], 'log': s['log'], 'n_samples': len(s['samples']), 'classes': classes,
                          'all': {g: stats(cnt[False][g]) for g in GROUPS},
                          'near': {g: stats(cnt[True][g]) for g in GROUPS},
                          'per_sample': {g: cnt[True][g].tolist() for g in GROUPS},
                          'per_sample_all': {g: cnt[False][g].tolist() for g in GROUPS}}
    return out


def report(scenes):
    names = sorted(scenes)
    route = lambda n: scenes[n]['log'].split('_')[0]
    print(f'\n[detect] {len(names)} scenes; per sample = 6 cameras summed; near = box height >= {NEAR_PX} px')
    tot = {c: np.sum([scenes[n]['classes'][c] for n in names], 0) for c in COCO.values()}
    print('  boxes per class (all / near): ' + ', '.join(f'{c} {a} / {b}' for c, (a, b) in tot.items()))
    for which in ('near', 'all'):
        print(f'\n  scene mean per sample ({which}):   p10    p25    p50    p75    p90    max')
        for g in GROUPS:
            v = np.array([scenes[n][which][g]['mean'] for n in names])
            q = np.percentile(v, [10, 25, 50, 75, 90])
            print(f"    {GROUP_KO[g]:6s}{'':20s}" + ' '.join(f'{x:6.1f}' for x in q) + f' {v.max():6.1f}')
    v = np.array([scenes[n]['near']['all']['mean'] for n in names])
    edges = [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 1e9]
    print('\n  scenes by near road users per sample (mean):')
    for a, b in zip(edges[:-1], edges[1:]):
        k = int(((v >= a) & (v < b)).sum())
        print(f"    {a:4.0f}-{'' if b > 1e8 else f'{b:.0f}':>3s}  {k:4d}  {'#' * k}")
    print('\n  per route (near, mean per sample):  scenes   veh   ped   two   all')
    for r in sorted({route(n) for n in names}, key=lambda r: int(r.split('-')[1])):
        rn = [n for n in names if route(n) == r]
        print(f'    {r:5s}{"":30s}{len(rn):3d} ' + ' '.join(f"{np.mean([scenes[n]['near'][g]['mean'] for n in rn]):5.1f}" for g in GROUPS))
    print('\n  fewest near road users (mean / min / max per sample; veh ped two):')
    for n in sorted(names, key=lambda n: scenes[n]['near']['all']['mean'])[:15]:
        a = scenes[n]['near']
        print(f"    {n} {route(n):5s} all {a['all']['mean']:5.1f} / {a['all']['min']:2d} / {a['all']['max']:3d}   "
              f"{a['veh']['mean']:5.1f} {a['ped']['mean']:5.1f} {a['two']['mean']:5.1f}  "
              f"samples with none {1 - a['all']['any']:.0%}")
    print('\n  scenes below a minimum (near road users, mean per sample):')
    for t in (1, 2, 3, 5, 8, 10):
        print(f'    < {t:2d}: {int((v < t).sum()):3d}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=sv.default_dataroot())
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--model', default='yolo26l.pt', help='ultralytics COCO weights (downloaded into selection/models/)')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.4, help='confidence for counting (raw cache keeps >= 0.25)')
    ap.add_argument('--report', action='store_true', help='count and report from the cache only')
    ap.add_argument('--logs', default='', help='comma separated log names: detect only their scenes (a course, a route)')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors='replace')
    except AttributeError:
        pass

    db = sv.load_db(args.dataroot, args.version, ego_poses=False)
    sel = os.path.join(args.dataroot, 'selection')
    os.makedirs(sel, exist_ok=True)
    path = cache_path(args.dataroot, args.model, args.imgsz)
    if args.report:
        cache = load_cache(path)
    else:
        weights = args.model if os.path.dirname(args.model) else os.path.join(sel, 'models', args.model)
        cache = detect(db, args.dataroot, weights, args.imgsz, path,
                       logs={x for x in args.logs.split(',') if x} or None)
    ego = ego_regions(cache, args.conf, sum(len(s['samples']) for s in db['scenes'] if s['token'] in cache))
    print('[detect] ego body regions (px): ' + '; '.join(f'{CAMS[c]} {r.round().astype(int).tolist()}' for c, r in ego.items() if len(r)))
    scenes = summarise(db, cache, args.conf, ego)
    out = os.path.join(sel, 'object_counts.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'version': VERSION, 'created': datetime.datetime.now().isoformat(timespec='seconds'),
                   'model': os.path.basename(args.model), 'imgsz': args.imgsz, 'conf': args.conf, 'near_px': NEAR_PX,
                   'cams': CAMS, 'ego_regions': {CAMS[c]: r.tolist() for c, r in ego.items()}, 'scenes': scenes}, f, ensure_ascii=False, indent=1)
    report(scenes)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    sys.exit(main())
