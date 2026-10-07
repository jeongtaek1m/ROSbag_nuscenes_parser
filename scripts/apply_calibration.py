#!/usr/bin/env python3
"""Put a T-Car calibration (tcar_calib_<date>/) into a converted dataset, without touching an image.

    python scripts/apply_calibration.py --calib ~/Downloads/tcar_calib_20260923.zip \\
        --dataroot /data/parsed/tcar_nuscenes_full

The calibration folder (README.md, project.py, intrinsic/, windshield/, extrinsic/) is copied to
<dataroot>/calibration/, and every calibrated_sensor record of a channel it covers (the seven cameras and
LIDAR_TOP) takes its values, in every recording:

  translation, rotation   T_ego_cam / T_ego_lidar (sensor -> ego; ego = the NovAtel output point, x forward,
                          y left, z up, the frame ego_pose places), rotation as a w, x, y, z quaternion
  camera_intrinsic        K of the Kannala-Brandt model (fx, fy, cx, cy)
  extra keys (cameras)    camera_model "kannala_brandt", camera_distortion [k1..k4] (OpenCV cv2.fisheye),
                          image_size, time_offset_s, rolling_shutter_readout_s, windshield (the B-spline
                          displacement field file, relative to dataroot)
  calibration             the calibration's name; null on the channels it does not cover (bottom LiDARs, radar),
                          which keep the converter's identity placeholder

Only the full-data converter's dataset (it has ext/) takes it; the standard set is published without
calibration. The images stay as recorded (not rectified): K and the distortion describe the raw JPEGs. Running it again,
e.g. after more recordings were appended, sets the same values on every record again.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'curation'))
import apply_selection  # noqa: E402  (the converter's dataset lock)

VERSION = 'v1.0-trainval'
CALIB_DIRS = ('intrinsic', 'windshield', 'extrinsic')


def quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    return (q if q[0] >= 0 else -q).tolist()


def load(calib):
    """{channel: record fields} from a calibration folder."""
    out = {}
    for p in sorted((calib / 'extrinsic').glob('*.json')):
        ch = p.stem
        e = json.loads(p.read_text())
        T = np.asarray(e['T_ego_lidar' if ch.startswith('LIDAR') else 'T_ego_cam'], dtype=np.float64)
        if T.shape != (4, 4) or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-4:
            sys.exit(f'{p}: not a rigid 4x4 transform')
        rec = {'translation': T[:3, 3].tolist(), 'rotation': quat_wxyz(T[:3, :3]), 'camera_intrinsic': []}
        if ch.startswith('CAM'):
            i = json.loads((calib / 'intrinsic' / f'{ch}.json').read_text())
            if i.get('model') != 'kannala_brandt':
                sys.exit(f'{ch}: intrinsic model {i.get("model")!r}, expected kannala_brandt')
            if not (calib / 'windshield' / f'{ch}.json').is_file():
                sys.exit(f'{ch}: windshield/{ch}.json missing')
            rec.update({
                'camera_intrinsic': [[i['fx'], 0.0, i['cx']], [0.0, i['fy'], i['cy']], [0.0, 0.0, 1.0]],
                'camera_model': 'kannala_brandt',
                'camera_distortion': [i['k1'], i['k2'], i['k3'], i['k4']],
                'image_size': [i['image_width'], i['image_height']],
                'time_offset_s': i['time_offset_s'],
                'rolling_shutter_readout_s': i['rolling_shutter_readout_s'],
                'windshield': f'calibration/windshield/{ch}.json',
            })
        out[ch] = rec
    if not out:
        sys.exit(f'{calib}: no extrinsic/*.json')
    return out


def find_calib(path, tmp):
    """The calibration folder: path itself, or its single subfolder, or the same inside a zip."""
    path = Path(path).expanduser()
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.startswith('/') or '..' in n.split('/'):
                    sys.exit(f'{path}: unsafe member {n}')
            z.extractall(tmp)
        path = Path(tmp)
    if not (path / 'extrinsic').is_dir():
        subs = [d for d in path.iterdir() if d.is_dir() and (d / 'extrinsic').is_dir()]
        if len(subs) != 1:
            sys.exit(f'{path}: no calibration folder (intrinsic/, windshield/, extrinsic/)')
        path = subs[0]
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib', required=True, help='tcar_calib_<date>/ folder or its .zip')
    ap.add_argument('--dataroot', default='/data/parsed/tcar_nuscenes_full')
    ap.add_argument('--name', help='calibration name (default: the folder name, e.g. tcar_calib_20260923)')
    ap.add_argument('--dry-run', action='store_true', help='print what would change, write nothing')
    args = ap.parse_args()
    root = Path(args.dataroot)
    tdir = root / VERSION
    if not (root / 'ext').is_dir():               # the standard (public) set stays without calibration
        sys.exit(f'{root}: not a full-data dataset (no ext/); the calibration goes into the full set only')
    with tempfile.TemporaryDirectory() as tmp:
        calib = find_calib(args.calib, tmp)
        name = args.name or calib.name
        values = load(calib)
        sensors = {s['token']: s for s in json.loads((tdir / 'sensor.json').read_text())}
        cs = json.loads((tdir / 'calibrated_sensor.json').read_text())
        sizes = {}                                  # channel -> {(w, h)} of its sample_data
        for sd in json.loads((tdir / 'sample_data.json').read_text()):
            if sd.get('width'):
                sizes.setdefault(sd['filename'].split('/')[1], set()).add((sd['width'], sd['height']))
        for ch, v in values.items():
            if 'image_size' in v and sizes.get(ch, {tuple(v['image_size'])}) != {tuple(v['image_size'])}:
                sys.exit(f'{ch}: images are {sorted(sizes[ch])}, the calibration is for {v["image_size"]}')
        n = {}
        for r in cs:
            ch = sensors[r['sensor_token']]['channel']
            keep = {'token': r['token'], 'sensor_token': r['sensor_token']}
            if ch in values:
                r.clear()
                r.update(keep, **values[ch], calibration=name)
                n[ch] = n.get(ch, 0) + 1
            else:
                r['calibration'] = None
        missing = sorted(set(values) - set(n))
        print(f'{name}: {sum(n.values())} calibrated_sensor records of {len(cs)} '
              f'({", ".join(f"{ch} {k}" for ch, k in sorted(n.items()))})'
              + (f'; not in the dataset: {", ".join(missing)}' if missing else ''))
        uncal = sorted({sensors[r['sensor_token']]['channel'] for r in cs} - set(values))
        if uncal:
            print(f'  no calibration (identity kept): {", ".join(uncal)}')
        if args.dry_run:
            return
        with apply_selection._Locked(str(root), f'apply_calibration {name}'):
            dst = root / 'calibration'
            tmpdst = root / '.calibration.tmp'
            shutil.rmtree(tmpdst, ignore_errors=True)
            tmpdst.mkdir()
            for f in ('README.md', 'project.py'):
                if (calib / f).is_file():
                    shutil.copy2(calib / f, tmpdst / f)
            for d in CALIB_DIRS:
                shutil.copytree(calib / d, tmpdst / d)
            (tmpdst / 'applied.json').write_text(json.dumps({
                'calibration': name, 'applied': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                'channels': sorted(values), 'records': n, 'uncalibrated': uncal,
                'images': 'as recorded (not rectified)'}, indent=1))
            shutil.rmtree(dst, ignore_errors=True)
            tmpdst.replace(dst)
            out = tdir / '.calibrated_sensor.json.tmp'
            out.write_text(json.dumps(cs, indent=2))       # the converter's layout
            out.replace(tdir / 'calibrated_sensor.json')
        print(f'  -> {tdir / "calibrated_sensor.json"}, {dst}/')


if __name__ == '__main__':
    main()
