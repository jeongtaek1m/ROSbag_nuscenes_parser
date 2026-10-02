"""nuScenes-format sample viewer (6 cameras + LIDAR_TOP), stdlib only.

    python sample_viewer.py [--dataroot D:/tcar_nuscenes] [--version v1.0-trainval] [--port 8765]

Then open http://localhost:8765 in a browser (simple viewer); http://localhost:8765/full is the earlier
detailed viewer (distribution, risk, labels, selection panels).
"""
import argparse
import bisect
import json
import math
import mimetypes
import os
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

try:                                        # optional: faster tracks and the compact /lidar/ endpoint
    import numpy as np
except ImportError:
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.basename(HERE)   # the same code is curation/ in ROSbag_nuscenes_parser and tools/ in tcar-scene-review


def default_dataroot():
    """The folder above these tools when they sit inside a dataset (<dataroot>/tools/, how tcar-scene-review is
    used), else $TCAR_DATA_ROOT/parsed/tcar_nuscenes (TCAR_DATA_ROOT defaults to /data)."""
    up = os.path.dirname(HERE)
    if os.path.isdir(os.path.join(up, 'v1.0-trainval')):
        return up
    return os.path.join(os.environ.get('TCAR_DATA_ROOT', '/data'), 'parsed', 'tcar_nuscenes')


CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def ll2utm(lat, lon, zone=52):
    """WGS84 lat/lon (deg) -> UTM easting/northing (m) and grid convergence (rad)."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    dlam = lam - math.radians((zone - 1) * 6 - 180 + 3)
    s, c = math.sin(phi), math.cos(phi)
    N = a / math.sqrt(1 - e2 * s * s)
    T, C, A = math.tan(phi) ** 2, ep2 * c * c, c * dlam
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * math.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                       + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y, math.atan(math.tan(dlam) * s)


def _ll2utm_np(np, lat, lon, zone=52):
    """ll2utm on arrays (same formulas); ~10x faster than calling ll2utm per 100 Hz row."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = np.radians(lat), np.radians(lon)
    dlam = lam - math.radians((zone - 1) * 6 - 180 + 3)
    s, c = np.sin(phi), np.cos(phi)
    N = a / np.sqrt(1 - e2 * s * s)
    T, C, A = np.tan(phi) ** 2, ep2 * c * c, c * dlam
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * phi))
    x = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T * T + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    y = k0 * (M + N * np.tan(phi) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
                                     + (61 - 58 * T + T * T + 600 * C - 330 * ep2) * A ** 6 / 720))
    return x, y, np.arctan(np.tan(dlam) * s)


def load_canbus(dataroot, scene_name):
    """Per-scene INS track from can_bus/<scene>_pose.json.

    Position comes from inspva lat/lon (100 Hz) rather than odom `pos`, which drops to
    ~1 Hz in most scenes. Yaw is rotated from true north to the UTM grid.
    """
    path = os.path.join(dataroot, 'can_bus', scene_name + '_pose.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and rows:                   # vectorised path (numpy is optional for the viewer)
        lat, lon = np.array([r['lat'] for r in rows]), np.array([r['lon'] for r in rows])
        x, y, gamma = _ll2utm_np(np, lat, lon)
        q = np.array([r['orientation'] for r in rows])
        yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)) + gamma
        return {'t': [r['utime'] for r in rows], 'x': x.tolist(), 'y': y.tolist(), 'yaw': yaw.tolist(),
                'speed': [math.hypot(r['vel'][0], r['vel'][1]) for r in rows],
                'ax': [r['accel'][0] for r in rows], 'ay': [r['accel'][1] for r in rows],
                'wz': [r['rotation_rate'][2] for r in rows], 'ins': [r['ins_status'] for r in rows],
                'lat': lat.tolist(), 'lon': lon.tolist(), 'h': [r['height'] for r in rows]}
    tr = {k: [] for k in ('t', 'x', 'y', 'yaw', 'speed', 'ax', 'ay', 'wz', 'ins', 'lat', 'lon', 'h')}
    for r in rows:
        x, y, gamma = ll2utm(r['lat'], r['lon'])
        w, qx, qy, qz = r['orientation']
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) + gamma
        tr['t'].append(r['utime']); tr['x'].append(x); tr['y'].append(y); tr['yaw'].append(yaw)
        tr['speed'].append(math.hypot(r['vel'][0], r['vel'][1]))
        tr['ax'].append(r['accel'][0]); tr['ay'].append(r['accel'][1]); tr['wz'].append(r['rotation_rate'][2])
        tr['ins'].append(r['ins_status']); tr['lat'].append(r['lat']); tr['lon'].append(r['lon']); tr['h'].append(r['height'])
    return tr


VIEWER_CACHE_VERSION = 2          # 2: route points carry lat, lon, heading


def _tables_stamp(dataroot, version):
    """Name, mtime and size of every table: any parse, exclusion or apply rewrites them."""
    d = os.path.join(dataroot, version)
    return tuple((n, st.st_mtime_ns, st.st_size) for n in sorted(os.listdir(d)) if n.endswith('.json')
                 for st in [os.stat(os.path.join(d, n))])


def _cache_path(dataroot, version, ego_poses):
    import hashlib
    base = os.environ.get('TCAR_VIEWER_CACHE') or os.path.join(os.path.expanduser('~'), '.cache', 'tcar-parser', 'viewer')
    key = hashlib.sha1(f'{os.path.realpath(dataroot)}|{version}|{int(ego_poses)}'.encode()).hexdigest()[:16]
    return os.path.join(base, key + '.pkl')


def load_db(dataroot, version, ego_poses=True):
    """The dataset as the viewer and the analysis tools use it, from a cache on the local disk when the
    tables have not changed: reading them (632 MB for 334 scenes) and the CAN bus files takes 13 s warm
    and over a minute from a busy HDD; the cache loads in a few seconds."""
    import pickle
    t0 = time.time()
    try:
        stamp, path = _tables_stamp(dataroot, version), _cache_path(dataroot, version, ego_poses)
    except OSError:
        return _load_db(dataroot, version, ego_poses)
    try:
        with open(path, 'rb') as f:
            cached = pickle.load(f)
        if cached.get('v') == VIEWER_CACHE_VERSION and cached.get('stamp') == stamp:
            db = cached['db']
            print(f"[viewer] loaded {len(db['scenes'])} scenes from the cache in {time.time() - t0:.1f}s", flush=True)
            return db
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError, TypeError, ValueError):
        pass
    db = _load_db(dataroot, version, ego_poses)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump({'v': VIEWER_CACHE_VERSION, 'stamp': stamp, 'db': db}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except OSError:
        pass
    return db


def _load_db(dataroot, version, ego_poses=True):
    """ego_poses=False skips ego_pose.json (41 MB): only the viewer shows stored ego poses, the analysis
    tools use the INS track."""
    t0 = time.time()
    d = os.path.join(dataroot, version)

    def load(name):
        with open(os.path.join(d, name + '.json'), encoding='utf-8') as f:
            return json.load(f)

    logs = {l['token']: l for l in load('log')}
    sensors = {s['token']: s for s in load('sensor')}
    calib = {c['token']: c for c in load('calibrated_sensor')}
    samples = {s['token']: s for s in load('sample')}
    scenes = load('scene')

    # Key frames: one sample_data per channel per sample. All frames (sweeps too) are kept per
    # scene/channel as (timestamp, filename) so the viewer can play full-rate video.
    scene_name = {s['token']: s['name'] for s in scenes}
    by_sample, frames = {}, {}
    for sd in load('sample_data'):
        ch = sensors[calib[sd['calibrated_sensor_token']]['sensor_token']]['channel']
        sn = scene_name.get(samples[sd['sample_token']]['scene_token'])
        frames.setdefault(sn, {}).setdefault(ch, []).append((sd['timestamp'], sd['filename']))
        if sd['is_key_frame']:
            by_sample.setdefault(sd['sample_token'], {})[ch] = sd
    for chans in frames.values():
        for lst in chans.values():
            lst.sort()

    ego = {}
    if ego_poses:
        needed = {sd['ego_pose_token'] for chans in by_sample.values() for sd in chans.values()}
        ego = {e['token']: e for e in load('ego_pose') if e['token'] in needed}

    scene_list, tracks, scene_of = [], {}, {}
    for sc in sorted(scenes, key=lambda s: s['name']):
        toks, tok = [], sc['first_sample_token']
        while tok:
            toks.append(tok)
            scene_of[tok] = sc['name']
            tok = samples[tok]['next']
        tr = load_canbus(dataroot, sc['name'])
        tracks[sc['name']] = tr
        scene_list.append({
            'token': sc['token'], 'name': sc['name'], 'description': sc['description'],
            'log': logs[sc['log_token']]['logfile'], 'samples': toks,
            'sample_ts': [samples[t]['timestamp'] for t in toks], 'tr': tr,
        })

    # Route for the map: 10 Hz, in a local frame (origin = min UTM over all scenes) to keep numbers small.
    xs = [v for tr in tracks.values() if tr for v in tr['x']]
    ys = [v for tr in tracks.values() if tr for v in tr['y']]
    origin = [math.floor(min(xs)), math.floor(min(ys))] if xs else [0, 0]
    for s in scene_list:
        tr = s.pop('tr')
        # x, y (local UTM), time, km/h, lat, lon, heading (deg from north) — the map draws on lat/lon over OSM tiles
        s['route'] = [] if not tr else [
            [round(tr['x'][i] - origin[0], 2), round(tr['y'][i] - origin[1], 2), tr['t'][i], round(tr['speed'][i] * 3.6, 2),
             round(tr['lat'][i], 7), round(tr['lon'][i], 7), round((90 - math.degrees(tr['yaw'][i])) % 360, 1)]
            for i in range(0, len(tr['t']), 10)]

    print(f'[viewer] loaded {len(scene_list)} scenes, {len(by_sample)} samples '
          f'in {time.time() - t0:.1f}s', flush=True)
    return {'scenes': scene_list, 'samples': samples, 'by_sample': by_sample, 'frames': frames,
            'ego': ego, 'calib': calib, 'tracks': tracks, 'scene_of': scene_of, 'origin': origin}


def ego_status(db, token, ts, ego_translation):
    tr = db['tracks'].get(db['scene_of'].get(token))
    if not tr:
        return None
    i = bisect.bisect_left(tr['t'], ts)
    if i > 0 and (i == len(tr['t']) or ts - tr['t'][i - 1] < tr['t'][i] - ts):
        i -= 1
    ox, oy = db['origin']
    return {
        'dt_ms': (tr['t'][i] - ts) / 1000.0,
        'x': tr['x'][i] - ox, 'y': tr['y'][i] - oy, 'utm': [tr['x'][i], tr['y'][i]],
        'yaw_deg': math.degrees(tr['yaw'][i]), 'speed_kmh': tr['speed'][i] * 3.6,
        'ax': tr['ax'][i], 'ay': tr['ay'][i], 'wz_dps': math.degrees(tr['wz'][i]),
        'ins_status': tr['ins'][i], 'lat': tr['lat'][i], 'lon': tr['lon'][i], 'height': tr['h'][i],
        # how far the stored ego_pose is from the INS track (large where odom pos froze at 1 Hz)
        'ego_pose_err_m': math.hypot(ego_translation[0] - tr['x'][i], ego_translation[1] - tr['y'][i])
        if ego_translation else None,
    }


def sample_payload(db, token):
    chans = db['by_sample'].get(token)
    if chans is None:
        return None
    ref_ts = chans['LIDAR_TOP']['timestamp'] if 'LIDAR_TOP' in chans else db['samples'][token]['timestamp']
    out = {'token': token, 'timestamp': db['samples'][token]['timestamp'], 'channels': {}}
    for ch, sd in chans.items():
        e = db['ego'][sd['ego_pose_token']]
        c = db['calib'][sd['calibrated_sensor_token']]
        out['channels'][ch] = {
            'token': sd['token'], 'filename': sd['filename'], 'timestamp': sd['timestamp'],
            'dt_ms': (sd['timestamp'] - ref_ts) / 1000.0,
            'ego_translation': e['translation'], 'ego_rotation': e['rotation'],
            'calib_translation': c['translation'], 'calib_rotation': c['rotation'],
            'camera_intrinsic': c['camera_intrinsic'],
        }
    lid = out['channels'].get('LIDAR_TOP')
    out['ego_status'] = ego_status(db, token, ref_ts, lid['ego_translation'] if lid else None)
    return out


LABELS = ('keep', 'must_keep', 'duplicate', '')


def read_json(path, default):
    if not os.path.isfile(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def write_final(sel_dir):
    """selection/final_selection.json = the rule's proposals (result.json) overridden by the viewer's human
    decisions (human_decisions.json): the list to actually keep / delete. Rewritten on every decision and
    after every curate.py run."""
    res = read_json(os.path.join(sel_dir, 'result.json'), None)
    if not res or res.get('missing'):
        return None
    dec = read_json(os.path.join(sel_dir, 'human_decisions.json'), {'decisions': {}}).get('decisions', {})
    scenes = {}
    for n, s in sorted(res['scenes'].items()):
        rule = 'drop' if s.get('status') == 'remove' else 'keep'
        h = (dec.get(n) or {}).get('decision')
        scenes[n] = {'final': h or rule, 'by': 'human' if h else 'rule', 'rule': rule}
    drop = [n for n, v in scenes.items() if v['final'] == 'drop']
    out = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'rule': {'mode': res.get('mode'), 'created': res.get('created')},
           'counts': {'scenes': len(scenes), 'kept': len(scenes) - len(drop), 'deleted': len(drop),
                      'human_decisions': sum(1 for v in scenes.values() if v['by'] == 'human'),
                      'proposals_not_reviewed': sum(1 for v in scenes.values() if v['rule'] == 'drop' and v['by'] == 'rule')},
           'deleted': drop, 'kept': [n for n, v in scenes.items() if v['final'] == 'keep'], 'scenes': scenes}
    path = os.path.join(sel_dir, 'final_selection.json')
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(path + '.tmp', path)
    return out


PLAY_HZ = float(os.environ.get('TCAR_PLAY_HZ', 5))


def play_frames(db, hz=PLAY_HZ):
    """The frames playback reads, per scene and channel: every key frame plus others at most `hz` per second
    (0 = all). The cameras record ~30 Hz and LiDAR 10 Hz, ~3700 files per 20 s scene; at 5 Hz it is ~700, which
    an HDD reads (~40-90 files/s) faster than the scene plays."""
    if not hz:
        return db['frames']
    key = {sd['filename'] for chans in db['by_sample'].values() for sd in chans.values()}
    gap = 0.9e6 / hz                                 # a little under the period, so frames on the grid are not skipped
    out = {}
    for n, chans in db['frames'].items():
        for ch, lst in chans.items():
            kept, last = [], None
            for t, fn in lst:
                if fn in key or last is None or t - last >= gap:
                    kept.append((t, fn)); last = t
            out.setdefault(n, {})[ch] = kept
    return out


class Previews:
    """Playback copies on the local disk: camera frames at half size (JPEG decoded at 1/2 scale, about a
    fifth of the bytes) and LiDAR in the /lidar/ int16 form. The dataset may sit on an HDD where a scene's
    ~3500 frames are as many seeks (8-80 files/s, playback needs ~180); from here a scene plays at full
    rate once read. Scenes asked for with /api/prefetch are read in the background, the newest request
    first, the camera on screen before the others, several files at once so the disk can reorder them."""

    def __init__(self, root, db, workers=None, fill=False):
        workers = workers or int(os.environ.get('TCAR_PREVIEW_WORKERS', 32))   # an HDD reorders many requests: 16 -> 73, 32 -> 90 files/s
        import hashlib
        from concurrent.futures import ThreadPoolExecutor
        base = os.environ.get('TCAR_PREVIEW_CACHE') or os.path.join(os.path.expanduser('~'), '.cache', 'tcar-parser', 'preview')
        self.dir = os.path.join(base, hashlib.sha1(root.encode()).hexdigest()[:16])
        self.root, self.db = root, db
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None
        self.cond = threading.Condition()
        self.queue = []                              # (scene, channel), newest last
        # frames the page is waiting for go first: while it asks, the read-ahead keeps only a few reads in flight
        self.gate = threading.Condition()
        self.reading = 0
        self.last_asked = 0.0
        self.workers = workers
        self.have = set()                            # cached files (rel + ext): status checks never touch the disk
        self.bad = set()                             # files that cannot be cached (missing, undecodable): not waited for
        self.scanned = False                         # until the start-up scan is done, a miss is checked on disk
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix='preview')
        # fill: with nothing asked, keep reading the scenes not yet in the cache (fill_first first: the review's
        # candidates and their kept scenes), so a scene opened later plays without waiting. Asking replaces it.
        self.fill, self.fill_first, self.filled, self._filling = fill, [], set(), None
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._scan, daemon=True).start()

    def _cached(self, rel, ext):
        return os.path.join(self.dir, rel + ext)

    def is_cached(self, rel, ext, disk=False):
        key = rel + ext
        if key in self.have or key in self.bad:
            return True
        if self.scanned and not disk:                # the scan saw every file; new ones are added as they are written
            return False                             # (disk=True: another viewer on the same cache may have written it)
        if os.path.exists(self._cached(rel, ext)):
            self.have.add(key)
            return True
        return False

    def _store(self, path, body):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f'{path}.{threading.get_ident()}.tmp'
            with open(tmp, 'wb') as f:
                f.write(body)
            os.replace(tmp, path)
            self.have.add(os.path.relpath(path, self.dir))
        except OSError:
            pass

    def asked(self):
        """The page is waiting for a file: the read-ahead makes room for a second."""
        self.last_asked = time.monotonic()

    def _slot(self):
        limit = lambda: 4 if time.monotonic() - self.last_asked < 1.0 else self.workers
        with self.gate:
            while self.reading >= limit():
                self.gate.wait(0.05)
            self.reading += 1

    def _release(self):
        with self.gate:
            self.reading -= 1
            self.gate.notify_all()

    def _ahead(self, fetch, rel):
        self._slot()
        try:
            return fetch(rel)
        except Exception:                            # a missing / unreadable file: playback does not wait for it
            self.bad.add(rel + ('.i16' if fetch == self.lidar else '.p.jpg'))
            raise
        finally:
            self._release()

    def image(self, rel):
        path = self._cached(rel, '.p.jpg')
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.have.add(rel + '.p.jpg')            # e.g. written by another viewer on the same cache
            return body
        except FileNotFoundError:
            pass
        with open(os.path.join(self.root, rel), 'rb') as f:
            data = f.read()
        if self.cv2 is None or np is None:
            return data
        im = self.cv2.imdecode(np.frombuffer(data, np.uint8), self.cv2.IMREAD_REDUCED_COLOR_2)
        if im is None:
            self.bad.add(rel + '.p.jpg')
            return data
        body = self.cv2.imencode('.jpg', im, [self.cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        self._store(path, body)
        return body

    def lidar(self, rel):
        path = self._cached(rel, '.i16')
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.have.add(rel + '.i16')
            return body
        except FileNotFoundError:
            pass
        xyz = np.fromfile(os.path.join(self.root, rel), dtype=np.float32).reshape(-1, 5)[:, :3]
        body = np.clip(np.round(xyz * 100), -32767, 32767).astype('<i2').tobytes()
        self._store(path, body)
        return body

    SLICE_US = 2_000_000

    def warm(self, names, chs, log=print):
        """Read these scenes (their playback frames) into the cache and return when done: one pass over the
        dataset, so later playback never waits for the disk."""
        files = [(fn, '.i16' if ch == 'LIDAR_TOP' else '.p.jpg') for n in names for ch in chs
                 for _, fn in self.db['play'].get(n, {}).get(ch, [])]
        t0, idle_n, hist = time.time(), 0, []
        done0 = sum(1 for fn, ext in files if self.is_cached(fn, ext, disk=True))     # already there (e.g. another viewer)
        self.prefetch(names, chs)
        while True:
            done = sum(1 for fn, ext in files if self.is_cached(fn, ext))
            with self.cond:
                idle = not self.queue
            el = time.time() - t0
            hist = [h for h in hist if h[0] > time.time() - 300] + [(time.time(), done)]   # rate over the last 5 min
            rate = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0]) if len(hist) > 1 else 0
            left = (len(files) - done) / rate if rate > 0 else None
            log(f'[warm] {done}/{len(files)} files ({100 * done / max(1, len(files)):.0f}%, {done - done0} read now), '
                f'{el / 60:.1f} min' + (f', {rate:.0f} files/s, ~{left / 60:.0f} min left' if left else ''))
            idle_n = idle_n + 1 if idle and self.reading == 0 else 0
            if done >= len(files) or idle_n >= 2:          # all read, or nothing left to read (twice, 15 s apart)
                return done, len(files)
            time.sleep(15)

    def _next_fill(self):
        """Queue items for the next scene not fully in the cache (all channels), each scene tried once."""
        chs = CAM_ORDER + ['LIDAR_TOP']
        names = [s['name'] for s in self.db['scenes']]
        for n in list(self.fill_first) + names:
            if n in self.filled or n not in self.db['play']:
                continue
            self.filled.add(n)
            if self.buffered(n, chs)['frac'] < 0.999:
                self._filling = n
                return self._items([n], chs)
        self._filling = None
        return []

    def prefetch(self, scenes, chs):
        """Read these scenes' frames of these channels into the cache, one whole scene after another in
        the order asked (the one on screen first), 2 s at a time within a scene. The viewer plays a
        scene once all of it is in the cache. A new request replaces the queue."""
        items = self._items(scenes, chs)
        with self.cond:
            self.queue = items[::-1]                    # popped from the end: first scene, earliest slice first
            if self._filling:                           # the fill was cut short: that scene again later
                self.filled.discard(self._filling)
                self._filling = None
            self.cond.notify()
        return len(items)

    def _items(self, scenes, chs):
        items = []
        for n in scenes:
            fr = self.db['play'].get(n, {})
            ts = [t for ch in chs for t, _ in fr.get(ch, [])[:1]]
            if not ts:
                continue
            t0, t1 = min(ts), max(fr[ch][-1][0] for ch in chs if fr.get(ch))
            items += [(n, tuple(chs), t0 + k * self.SLICE_US, t0 + (k + 1) * self.SLICE_US)
                      for k in range(int((t1 - t0) // self.SLICE_US) + 1)]
        return items

    def buffered(self, scene, chs, disk=False):
        """Seconds from the scene's start that every channel has in the cache without a gap. disk=True also counts
        files another viewer wrote into the same cache since this one started (one stat per missing file)."""
        sc = next((x for x in self.db['scenes'] if x['name'] == scene), None)
        fr = self.db['play'].get(scene, {})
        if not sc:
            return {'buffered': 0.0, 'dur': 0.0, 'frac': 0.0}
        t0 = sc['sample_ts'][0]
        dur = (sc['sample_ts'][-1] - t0) / 1e6
        upto, have, total = dur, 0, 0
        for ch in chs:
            ext = '.i16' if ch == 'LIDAR_TOP' else '.p.jpg'
            lst = fr.get(ch, [])
            total += len(lst)
            gap = None
            for t, fn in lst:
                if self.is_cached(fn, ext, disk):
                    have += 1
                elif gap is None:
                    gap = t
            if gap is not None:
                upto = min(upto, max(0.0, (gap - t0) / 1e6))
        return {'buffered': round(upto, 2), 'dur': round(dur, 2), 'frac': round(have / total, 3) if total else 1.0}

    def _worker(self):
        while True:
            with self.cond:
                while not self.queue:
                    more = self._next_fill() if self.fill else []
                    if more:
                        self.queue = more[::-1]
                        break
                    self.cond.wait(60 if self.fill else None)
                scene, chs, a, b = self.queue.pop()
            todo = sorted((t, ch, fn) for ch in chs for t, fn in self.db['play'].get(scene, {}).get(ch, [])
                          if a <= t < b and not self.is_cached(fn, '.i16' if ch == 'LIDAR_TOP' else '.p.jpg', disk=True))
            futs = [self.pool.submit(self._ahead, self.lidar if ch == 'LIDAR_TOP' else self.image, fn)
                    for _, ch, fn in todo]
            for fut in futs:
                try:
                    fut.result()
                except Exception:                     # a missing file: the viewer reports it when it asks
                    pass

    def _scan(self, limit_gb=None):
        """Once at start: note every cached file (the readiness checks then run in memory; a disk check per
        file made /api/ready take minutes for a dataset of uncached scenes) and keep the cache under
        TCAR_PREVIEW_CACHE_GB (default 80): the oldest files go first."""
        limit = float(os.environ.get('TCAR_PREVIEW_CACHE_GB', limit_gb or 80)) * 1e9
        files, total, gone = [], 0, set()
        for d, _, names in os.walk(self.dir):
            for n in names:
                if n.endswith('.tmp'):
                    continue
                try:
                    st = os.stat(os.path.join(d, n))
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, os.path.join(d, n)))
                total += st.st_size
        if total > limit:
            for _, size, path in sorted(files):
                try:
                    os.remove(path)
                except OSError:
                    continue
                gone.add(os.path.relpath(path, self.dir))
                total -= size
                if total < 0.7 * limit:
                    break
        self.have.update(k for k in (os.path.relpath(p, self.dir) for _, _, p in files) if k not in gone)
        self.have.difference_update(gone)
        self.scanned = True


TILE_URL = os.environ.get('TCAR_TILE_URL', 'https://tile.openstreetmap.org/{z}/{x}/{y}.png')   # the page darkens it
_TILE_FAIL = {'until': 0.0}


def tile(path):
    """/tile/<z>/<x>/<y>.png from the local cache, else from TILE_URL (then kept). None without network:
    the map still draws the routes."""
    import hashlib
    import urllib.request
    try:
        z, x, y = (int(v) for v in path[len('/tile/'):].removesuffix('.png').split('/'))
    except ValueError:
        return None
    if not (0 <= z <= 19 and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
        return None
    base = os.environ.get('TCAR_TILE_CACHE') or os.path.join(os.path.expanduser('~'), '.cache', 'tcar-parser', 'tiles')
    cached = os.path.join(base, hashlib.sha1(TILE_URL.encode()).hexdigest()[:8], str(z), str(x), f'{y}.png')
    try:
        with open(cached, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        pass
    if time.time() < _TILE_FAIL['until']:
        return None
    url = TILE_URL.format(s='abcd'[(x + y) % 4], z=z, x=x, y=y)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'TCAR-Parser/1.0 (dataset curation viewer; tiles cached locally)'})
        body = urllib.request.urlopen(req, timeout=6).read()
    except Exception:
        _TILE_FAIL['until'] = time.time() + 60          # offline: stop asking for a minute
        return None
    try:
        os.makedirs(os.path.dirname(cached), exist_ok=True)
        with open(cached + '.tmp', 'wb') as f:
            f.write(body)
        os.replace(cached + '.tmp', cached)
    except OSError:
        pass
    return body


def make_handler(db, dataroot, version='v1.0-trainval', fill=False):
    root = os.path.realpath(dataroot)
    if 'play' not in db:
        db['play'] = play_frames(db)
    previews = Previews(root, db, fill=fill)
    if fill:                                     # the review opens candidates and their kept scenes: those first
        rule = read_json(os.path.join(root, 'selection', 'result.json'), {}).get('scenes') or {}
        first = [n for n, v in rule.items() if v.get('status') == 'remove']
        previews.fill_first = [x for n in first for x in (rule[n].get('substitute'), n) if x]
    html_path = os.path.join(HERE, 'sample_viewer.html')
    full_path = os.path.join(HERE, 'sample_viewer_full.html')      # the earlier detailed viewer, at /full
    overview_path = os.path.join(HERE, 'overview.html')            # the dataset at a glance, at /
    # scene-selection files: result.json is written by scene_select.py, labels by this viewer
    sel_dir = os.path.join(dataroot, 'selection')
    labels_path = os.path.join(sel_dir, 'human_labels.json')
    pairs_path = os.path.join(sel_dir, 'pair_labels.json')     # human "same situation?" judgements
    decisions_path = os.path.join(sel_dir, 'human_decisions.json')   # 남기기 / 버리기 chosen in the viewer
    result_path = os.path.join(sel_dir, 'result.json')
    scene_names = {s['name'] for s in db['scenes']}
    if os.path.isdir(sel_dir):                   # keep.txt / drop.txt / deleted.txt match the decisions from the start
        try:
            import apply_selection
            apply_selection.write_confirmed(sel_dir, db['scenes'])
            apply_selection.write_deleted(dataroot)
        except OSError as e:
            print(f'[viewer] could not write the confirmed lists: {e}')
    labels_lock = threading.Lock()

    def update_decision(body):
        scene, decision, via = body.get('scene'), body.get('decision'), body.get('via', 'manual')
        if scene not in scene_names or decision not in ('keep', 'drop', None):
            raise ValueError('bad scene or decision')
        import apply_selection                           # a scene an earlier round locked keeps its decision for good
        tok = next((x['token'] for x in db['scenes'] if x['name'] == scene), None)
        if tok in apply_selection.locked_tokens(sel_dir):
            raise ValueError('locked: finalized in an earlier round')   # (HTTP reason: ASCII only)
        with labels_lock:
            data = read_json(decisions_path, {'version': 1, 'decisions': {}})
            if decision is None:
                data['decisions'].pop(scene, None)                 # back to the automatic decision
            else:
                data['decisions'][scene] = {'decision': decision, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            os.makedirs(sel_dir, exist_ok=True)
            tmp = decisions_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, decisions_path)
            write_final(sel_dir)
            import apply_selection                           # selection/keep.txt, drop.txt: the confirmed lists
            apply_selection.write_confirmed(sel_dir, db['scenes'])
            # human-readable history, one line per decision
            r = read_json(result_path, {}).get('scenes', {}).get(scene) or {}
            rule = f"후보 · {r.get('kind') or '기타'}" if r.get('status') == 'remove' else '필터 통과' 
            with open(os.path.join(sel_dir, 'decisions_log.txt'), 'a', encoding='utf-8') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {scene}  "
                        f"{ {'keep': '남기기 확정', 'drop': '버리기 확정', None: '확정 취소 (후보로)'}[decision] }  "
                        f"(규칙: {rule} · {'후보대로 확정' if via == 'rule' else '직접 선택'})\n")
            return data

    det = {'key': None, 'cache': {}}
    det_lock = threading.Lock()

    def boxes_payload(name):
        """2D boxes of one scene from scene_detect.py's cache, flagged the way object_counts.json counted them
        (0 counted, 2 ego body, 3 rider of a two-wheeler; boxes below the counting confidence are left out)."""
        import scene_detect as sd
        oc = read_json(os.path.join(sel_dir, 'object_counts.json'), None)
        sc = next((s for s in db['scenes'] if s['name'] == name), None)
        if np is None or not oc or sc is None:
            return {'missing': True}
        path = sd.cache_path(dataroot, oc['model'], oc['imgsz'])
        if not os.path.isfile(path):
            return {'missing': True}
        with det_lock:                                   # the npz (~5 MB) is read once, again only when it changes
            key = (path, os.path.getmtime(path))
            if det['key'] != key:
                det['cache'], det['key'] = sd.load_cache(path), key
            d = det['cache'].get(sc['token'])
        if d is None:
            return {'missing': True}
        ego = {sd.CAMS.index(c): np.array(r, np.float32).reshape(-1, 4) for c, r in oc.get('ego_regions', {}).items()}
        flag, _ = sd.box_flags(d, oc['conf'], ego)
        boxes = {c: [] for c in sd.CAMS}
        for i in np.nonzero(flag != sd.LOW)[0]:
            boxes[sd.CAMS[d['cam'][i]]].append([int(d['frame'][i])] + [round(float(v), 1) for v in d['box'][i]]
                                               + [int(d['cls'][i]), round(float(d['conf'][i]), 2), int(flag[i])])
        return {'model': oc['model'], 'conf': oc['conf'], 'near_px': oc['near_px'], 'size': [sd.W, sd.H],
                'key_ts': {c: [db['by_sample'][t][c]['timestamp'] for t in sc['samples']] for c in sd.CAMS},
                'boxes': boxes}

    apply_lock = threading.Lock()

    def apply_plan(logs=None):
        import apply_selection
        p = apply_selection.plan(dataroot, logs)
        fin = read_json(os.path.join(sel_dir, 'final_selection.json'), {}).get('scenes', {})
        p['by'] = {n: (fin.get(n) or {}).get('by', 'rule') for n in p['deleted']}
        return p

    def run_apply(body):
        """최종 삭제 (the app runs apply_selection.py itself): move the confirmed 버리기 scenes out, renumber,
        reload, rerun the rule on the new dataset."""
        if body.get('confirm') is not True:
            raise ValueError('confirm required')
        if not apply_lock.acquire(blocking=False):
            raise ValueError('already running')
        try:
            import subprocess
            import apply_selection
            try:
                with labels_lock:
                    r = apply_selection.apply(dataroot, logs=body.get('logs') or None)
            except Exception as e:                                 # partial moves are in <dataroot>/_removed/<stamp>/manifest.json
                return {'ok': False, 'error': f'{type(e).__name__}: {e}',
                        'hint': '중간에 멈췄다면 _removed/ 아래 가장 최근 폴더로 --undo 하면 원래대로 돌아갑니다'}
            fresh = load_db(dataroot, version)                     # the viewer now serves the new dataset
            fresh['play'] = play_frames(fresh, db.get('play_hz', PLAY_HZ))
            db.clear(); db.update(fresh)
            scene_names.clear(); scene_names.update(s['name'] for s in db['scenes'])
            cur = subprocess.run([sys.executable, os.path.join(HERE, 'curate.py'), '--dataroot', dataroot],
                                 capture_output=True, text=True, encoding='utf-8', errors='replace',
                                 env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            return {'ok': True, 'backup': r['backup'], 'deleted': r['deleted'], 'rename': r['rename'],
                    'n_after': r['n_after'], 'curate_ok': cur.returncode == 0,
                    'undo': f'python {TOOLS}/apply_selection.py --undo "{r["backup"]}"'}
        finally:
            apply_lock.release()

    def update_label(body):
        scene, label = body.get('scene'), body.get('label', '')
        dup_of, note = body.get('dup_of') or None, str(body.get('note', ''))[:2000]
        if scene not in scene_names or label not in LABELS:
            raise ValueError('bad scene or label')
        if dup_of is not None and (dup_of not in scene_names or dup_of == scene):
            raise ValueError('bad dup_of')
        with labels_lock:
            data = read_json(labels_path, {'version': 1, 'labels': {}})
            if label or note:
                data['labels'][scene] = {'label': label, 'dup_of': dup_of if label == 'duplicate' else None,
                                         'note': note, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            else:
                data['labels'].pop(scene, None)
            os.makedirs(sel_dir, exist_ok=True)
            tmp = labels_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, labels_path)
            return data

    def update_pair(body):
        a, b, same = body.get('a'), body.get('b'), body.get('same')
        if a not in scene_names or b not in scene_names or a == b or same not in (True, False, None):
            raise ValueError('bad pair')
        key = '|'.join(sorted((a, b)))
        with labels_lock:
            data = read_json(pairs_path, {'version': 1, 'pairs': {}})
            if same is None:
                data['pairs'].pop(key, None)
            else:
                data['pairs'][key] = {'same': same, 'updated': time.strftime('%Y-%m-%dT%H:%M:%S')}
            os.makedirs(sel_dir, exist_ok=True)
            tmp = pairs_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, pairs_path)
            return data

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'          # keep-alive: playback fetches ~70 files/s, one TCP connection each was too many

        def log_message(self, fmt, *args):
            pass

        def send_bytes(self, body, ctype, cache=False):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'max-age=3600' if cache else 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj):
            self.send_bytes(json.dumps(obj).encode('utf-8'), 'application/json')

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            try:
                if path == '/favicon.ico':                 # no icon; answer quietly instead of a 404 in the console
                    self.send_response(204); self.end_headers()
                    return
                if path in ('/', '/index.html', '/review', '/full'):
                    # Re-read on every request so HTML edits apply without a restart. / is the overview; it forwards
                    # links with a #... (scene, review, view) to /review, so older links and the GUI keep working.
                    page = {'/full': full_path, '/review': html_path}.get(path, overview_path)
                    with open(page, 'rb') as f:
                        return self.send_bytes(f.read(), 'text/html; charset=utf-8')
                if path == '/api/page':
                    # when the review page's code last changed: an open page offers a reload
                    return self.send_json({'mtime': os.path.getmtime(html_path)})
                if path == '/api/info':
                    return self.send_json({'dataroot': os.path.realpath(dataroot), 'version': version})
                if path == '/api/objects':
                    # object_counts.json (tools/scene_detect.py) without the per-sample arrays
                    oc = read_json(os.path.join(sel_dir, 'object_counts.json'), None)
                    if not oc:
                        return self.send_json({'missing': True})
                    keep = ('token', 'log', 'classes', 'near', 'all')
                    return self.send_json(dict({k: v for k, v in oc.items() if k != 'scenes'},
                                               scenes={n: {k: v.get(k) for k in keep} for n, v in oc['scenes'].items()}))
                if path == '/api/map':
                    # every scene's GNSS track at 1 Hz, for the app's dataset map (the full routes are in /api/scenes)
                    out = []
                    for sc in db['scenes']:
                        r = sc['route']
                        pts = [[p[4], p[5]] for p in r[::10] + r[-1:] if len(p) > 5 and p[4] is not None]
                        out.append({'name': sc['name'], 'log': sc['log'], 'token': sc['token'], 'pts': pts})
                    return self.send_json({'scenes': out})
                if path == '/api/scenes':
                    return self.send_json({'origin': db['origin'], 'scenes': db['scenes']})
                if path == '/api/labels':
                    with labels_lock:
                        return self.send_json(read_json(labels_path, {'version': 1, 'labels': {}}))
                if path == '/api/pairs':
                    with labels_lock:
                        return self.send_json(read_json(pairs_path, {'version': 1, 'pairs': {}}))
                if path == '/api/selection':
                    return self.send_json(read_json(result_path, {'missing': True}))
                if path == '/api/apply/plan':
                    logs = [x for x in (parse_qs(urlparse(self.path).query).get('logs') or [''])[0].split(',') if x]
                    try:
                        return self.send_json(apply_plan(logs or None))
                    except ValueError as e:
                        return self.send_json({'error': str(e)})
                if path == '/api/decisions':
                    with labels_lock:
                        return self.send_json(read_json(decisions_path, {'version': 1, 'decisions': {}}))
                if path == '/api/risk':
                    return self.send_json(read_json(os.path.join(sel_dir, 'risk.json'), {'missing': True}))
                if path.startswith('/api/boxes/'):
                    return self.send_json(boxes_payload(path.rsplit('/', 1)[-1]))
                if path.startswith('/api/frames/'):
                    # all frames of one channel in a scene: /api/frames/scene-0026?ch=CAM_FRONT
                    ch = (parse_qs(urlparse(self.path).query).get('ch') or ['CAM_FRONT'])[0]
                    lst = db['play'].get(path.rsplit('/', 1)[-1], {}).get(ch)
                    if lst is None:
                        return self.send_error(404, 'unknown scene or channel')
                    return self.send_json({'ts': [t for t, _ in lst], 'fn': [f for _, f in lst]})
                if path.startswith('/api/sample/'):
                    payload = sample_payload(db, path.rsplit('/', 1)[-1])
                    if payload is None:
                        return self.send_error(404, 'unknown sample')
                    return self.send_json(payload)
                if path.startswith('/lidar/') and np is not None:
                    # compact LiDAR for display: x, y, z as int16 centimetres (6 B/point instead of 20 B float32 x 5)
                    rel = path[len('/lidar/'):]
                    full = os.path.realpath(os.path.join(root, rel))
                    if not full.startswith(root + os.sep) or not os.path.isfile(full):
                        return self.send_error(404)
                    previews.asked()
                    return self.send_bytes(previews.lidar(rel), 'application/octet-stream', cache=True)
                if path.startswith('/p/'):
                    # a camera frame for playback: half size, from the local-disk cache
                    rel = path[len('/p/'):]
                    full = os.path.realpath(os.path.join(root, rel))
                    if not full.startswith(root + os.sep) or not os.path.isfile(full):
                        return self.send_error(404)
                    previews.asked()
                    return self.send_bytes(previews.image(rel), 'image/jpeg', cache=True)
                if path == '/api/prefetch':
                    q = parse_qs(urlparse(self.path).query)
                    names = [n for n in (q.get('scenes') or [''])[0].split(',') if n in db['frames']]
                    chs = [c for c in (q.get('chs') or [''])[0].split(',') if c in CAM_ORDER or c == 'LIDAR_TOP']
                    return self.send_json({'queued': previews.prefetch(names, chs or CAM_ORDER + ['LIDAR_TOP'])})
                if path == '/api/ready':
                    # how much of each scene is in the playback cache (the list's readiness dots)
                    q = parse_qs(urlparse(self.path).query)
                    names = [n for n in (q.get('scenes') or [''])[0].split(',') if n in db['frames']]
                    return self.send_json({n: previews.buffered(n, CAM_ORDER + ['LIDAR_TOP'])['frac'] for n in names})
                if path == '/api/buffered':
                    q = parse_qs(urlparse(self.path).query)
                    chs = [c for c in (q.get('chs') or [''])[0].split(',') if c in CAM_ORDER or c == 'LIDAR_TOP']
                    return self.send_json(previews.buffered((q.get('scene') or [''])[0], chs or CAM_ORDER + ['LIDAR_TOP'], disk=True))
                if path.startswith('/tile/'):
                    body = tile(path)
                    if body is None:
                        return self.send_error(404)
                    return self.send_bytes(body, 'image/png', cache=True)
                if path.startswith('/font/'):
                    name = os.path.basename(path)            # the app's Pretendard, so the page looks like the app
                    font = os.path.join(HERE, '..', 'assets', 'fonts', name)
                    if not name.endswith('.otf') or not os.path.isfile(font):
                        return self.send_error(404)
                    with open(font, 'rb') as f:
                        return self.send_bytes(f.read(), 'font/otf', cache=True)
                if path.startswith('/data/'):
                    full = os.path.realpath(os.path.join(root, path[len('/data/'):]))
                    if not full.startswith(root + os.sep) or not os.path.isfile(full):
                        return self.send_error(404)
                    previews.asked()
                    with open(full, 'rb') as f:
                        body = f.read()
                    ctype = mimetypes.guess_type(full)[0] or 'application/octet-stream'
                    return self.send_bytes(body, ctype, cache=True)
                return self.send_error(404)
            except ConnectionError:              # client went away (Broken pipe / reset / aborted, WinError 10053)
                pass

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            handler = {'/api/labels': update_label, '/api/pairs': update_pair, '/api/decisions': update_decision,
                       '/api/apply': run_apply}.get(path)
            if handler is None:
                return self.send_error(404)
            try:
                n = int(self.headers.get('Content-Length', 0))
                if n <= 0 or n > 65536:
                    return self.send_error(400, 'bad body size')
                return self.send_json(handler(json.loads(self.rfile.read(n))))
            except (ValueError, json.JSONDecodeError) as e:
                return self.send_error(400, str(e))
            except ConnectionError:              # client went away (Broken pipe / reset / aborted, WinError 10053)
                pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default=default_dataroot())
    ap.add_argument('--version', default='v1.0-trainval')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1', help='address to listen on (0.0.0.0 = other PCs on the network too)')
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--play-hz', type=float, default=PLAY_HZ,
                    help='frames per second playback reads per camera / LiDAR, key frames always (0 = all; env TCAR_PLAY_HZ)')
    ap.add_argument('--warm', action='store_true', help='read every scene into the playback cache once, then exit')
    ap.add_argument('--fill', action='store_true',
                    help='while nothing is asked, fill the playback cache with the scenes not in it yet (the app does)')
    args = ap.parse_args()

    db = load_db(args.dataroot, args.version)
    db['play'], db['play_hz'] = play_frames(db, args.play_hz), args.play_hz
    if args.warm:
        pv = Previews(os.path.realpath(args.dataroot), db)
        done, total = pv.warm([s['name'] for s in db['scenes']], CAM_ORDER + ['LIDAR_TOP'])
        print(f'[warm] done: {done}/{total} files in the cache {pv.dir}', flush=True)
        return 0

    class Server(ThreadingHTTPServer):
        request_queue_size = 128          # default 5: bursts of parallel image/LiDAR requests got "connection refused"
        daemon_threads = True
    server = Server((args.host, args.port), make_handler(db, args.dataroot, args.version, fill=args.fill))
    url = f'http://127.0.0.1:{args.port}'           # the address it listens on ("localhost" can resolve to ::1 first)
    lan = ''
    if args.host not in ('127.0.0.1', 'localhost'):
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as so:
                so.connect(('192.0.2.1', 9)); lan = f'  ·  LAN: http://{so.getsockname()[0]}:{args.port}'
        except OSError:
            pass
    print(f'[viewer] serving {args.dataroot} at {url}{lan}  (Ctrl+C to stop)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    sys.exit(main())
