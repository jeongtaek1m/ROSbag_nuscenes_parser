#!/usr/bin/env python3
"""Put a T-Car calibration (tcar_calib_<date>/) into a full-set dataset, without touching an image.

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
calibration. The images stay as recorded (not rectified): K and the distortion describe the raw JPEGs.
Recordings the converter appends later get the same values from <dataroot>/calibration/ (tcar_calib.py);
running this again with a newer calibration replaces it on every record.
"""
import argparse
import datetime
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'curation'))
import apply_selection  # noqa: E402  (the converter's dataset lock)
import tcar_calib  # noqa: E402

VERSION = 'v1.0-trainval'
CALIB_DIRS = ('intrinsic', 'windshield', 'extrinsic')


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
        try:
            values = tcar_calib.load_values(calib)
        except (ValueError, KeyError, OSError) as e:
            sys.exit(f'{calib}: {e}')
        channel_of = {s['token']: s['channel'] for s in json.loads((tdir / 'sensor.json').read_text())}
        cs = json.loads((tdir / 'calibrated_sensor.json').read_text())
        sizes = {}                                  # channel -> {(w, h)} of its sample_data
        for sd in json.loads((tdir / 'sample_data.json').read_text()):
            if sd.get('width'):
                sizes.setdefault(sd['filename'].split('/')[1], set()).add((sd['width'], sd['height']))
        for ch, v in values.items():
            if 'image_size' in v and sizes.get(ch, {tuple(v['image_size'])}) != {tuple(v['image_size'])}:
                sys.exit(f'{ch}: images are {sorted(sizes[ch])}, the calibration is for {v["image_size"]}')
        n = tcar_calib.apply_values(cs, channel_of, values, name)
        missing = sorted(set(values) - set(n))
        print(f'{name}: {sum(n.values())} calibrated_sensor records of {len(cs)} '
              f'({", ".join(f"{ch} {k}" for ch, k in sorted(n.items()))})'
              + (f'; not in the dataset: {", ".join(missing)}' if missing else ''))
        uncal = sorted({channel_of[r['sensor_token']] for r in cs} - set(values))
        if uncal:
            print(f'  no calibration (identity kept): {", ".join(uncal)}')
        if args.dry_run:
            return
        with apply_selection._Locked(str(root), f'apply_calibration {name}'):
            dst = root / tcar_calib.DIRNAME
            tmpdst = root / f'.{tcar_calib.DIRNAME}.tmp'
            shutil.rmtree(tmpdst, ignore_errors=True)
            tmpdst.mkdir()
            for f in ('README.md', 'project.py'):
                if (calib / f).is_file():
                    shutil.copy2(calib / f, tmpdst / f)
            for d in CALIB_DIRS:
                shutil.copytree(calib / d, tmpdst / d)
            (tmpdst / tcar_calib.APPLIED).write_text(json.dumps({
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
