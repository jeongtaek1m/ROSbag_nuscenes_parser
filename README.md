# ROSbag → NuScenes parser

Convert ROS1 rosbags from the T-Car platform (Robosense LiDARs + 7 e-CON cameras
+ Novatel INS + ARS548 radar + vehicle CAN) into NuScenes v1.0-trainval datasets.

Two tools, one pipeline:

| | `bag2nuscenes.py` — **standard** | `bag2nuscenes_full.py` — **full** |
|---|---|---|
| For | the nuScenes devkit and nuScenes-based training code, unchanged | our own detailed work |
| LiDAR | `LIDAR_TOP` (`/middle`, Ruby 128) | + `LIDAR_BOTTOM_FRONT/REAR` (M1), `LIDAR_BOTTOM_LEFT/RIGHT` (Bpearl), and per-point times |
| Cameras | the six standard channels, rectified to pinhole | + `CAM_TRAFFIC` (camera_4), rectified |
| Radar | — | `RADAR_FRONT` (ARS548), NuScenes radar `.pcd` |
| GNSS/INS | `can_bus/<scene>_pose.json`, `_ms_imu.json` | same |
| Everything else | — | every other topic as `ext/<scene>/<topic>.json` |

Frame selection, scenes and scene names are identical: the full dataset is the
standard one with more kinds of data, not a different cut.

No ROS installation required — [`rosbags`](https://pypi.org/project/rosbags/)
reads `.bag` files directly.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .              # includes nuscenes-devkit (split lists, validation)
pip install -e '.[plots]'     # matplotlib/jupyter, for diagnostics and notebooks
```

## Convert

```bash
python bag2nuscenes.py      /path/to.bag --calib /path/to/calib --split train   # -> /data/tcar_nuscenes
python bag2nuscenes_full.py /path/to.bag --calib /path/to/calib --split train   # -> /data/tcar_nuscenes_full
python bag2nuscenes.py      /path/to.bag --out /tmp/try                         # no calibration: defaults
```

One pass over the bag. Sensor payloads stream into a staging directory inside
the output root — camera JPEGs are rectified on the way, on a thread pool —
and once frame selection has decided which frames become samples and sweeps,
the staged files are **renamed** into place. Nothing is written twice.

Running it again on another bag **appends**: scene names continue down the
official list, sensor and category tokens are reused, and re-importing the same
bag is refused. Keep the standard and the full dataset in separate roots.

### Frame selection

`LIDAR_TOP` is the anchor. For every LiDAR frame, the converter looks for a
**complete camera set**: one capture instant at which all six standard cameras
delivered a frame, with every one of those frames within `--sync-ms` (25 ms) of
the LiDAR stamp. The cameras share a trigger — frames taken together are
stamped within ~0.04 ms of each other — so a set is chosen as a whole, never
camera by camera; of the (at most two) instants in range, the nearest complete
one wins.

A LiDAR frame without such a set, a missing LiDAR frame, or leaving the
*coverage window* (the interval in which LiDAR, the six cameras and odom are all
live) **breaks the sequence**. Each unbroken run is cut into scenes of exactly
`--scene-dur` (20 s); a remainder shorter than that is not used. In a scene:

- **samples** are every `--keyframe-stride`-th LiDAR frame: 2 Hz, 40 per scene;
- **sweeps** are every frame of every channel between the first and the last
  sample — all 10 Hz LiDAR frames and **all 30 fps camera frames**. As in
  nuScenes, a sweep belongs to the sample that follows it.

Channels beyond `LIDAR_TOP` and the six cameras (full tool only) never break
anything: a sample gets their frame nearest its LiDAR instant when it is within
`--sync-ms` (cameras) or half a period (LiDARs, radar), and none otherwise.

The run prints how many LiDAR frames were lost and which cameras had dropped
their frame there; `<log>.import.json` records it all (`frame_plan`,
`scene_partition`). `scripts/screen_bags.py` shows the same numbers before a
conversion.

### Scene names and splits

Scenes are named from the **official nuScenes train/val lists**
(`nuscenes.utils.splits`), in list order, for the split given by `--split`
(default `train`). Anything that splits a nuScenes dataset by those lists — the
devkit's detection/tracking evaluation, mmdetection3d or BEVFusion info
generation — then puts our scenes in the intended split without modification.
Put every scene of a bag in one split. Scene numbers the CAN bus API refuses
(161–176, 309–314) are skipped, which leaves room for **685 train and 150 val
scenes** (3.8 h and 50 min at 20 s). `scene.description` names the source bag
and the offset into it.

### Cameras are rectified

NuScenes stores a 3×3 `camera_intrinsic` and no distortion, so every consumer
projects with a plain pinhole `K`. The images are therefore undistorted with the
calibration (`cv2.fisheye` for 4 coefficients, plumb_bob for 5) and the **new**
`K` is what `calibrated_sensor.json` holds; the extrinsic is unchanged.
`--rectify-balance 0` (default) keeps only pixels valid across the whole image
(no black border, periphery cropped); `1` keeps the whole field of view.
Rectified frames are re-encoded at `--jpeg-quality 90`, about the size of the
cameras' own JPEGs. `<log>.import.json` records both `K`s per camera.

### CAN bus (GNSS/INS)

GNSS/INS goes in the nuScenes CAN bus expansion layout, so
`NuScenesCanBus(dataroot).get_messages("scene-0001", "pose")` works:

| message | rate | content |
|---|---|---|
| `pose` | 100 Hz | `pos`, `orientation` (same frame as `ego_pose`), `vel`, `accel`, `rotation_rate` (ego frame), plus `lat`, `lon`, `height`, `ins_status` from INSPVA |
| `ms_imu` | 100 Hz | `linear_accel`, `rotation_rate` (ego-aligned axes), `q` |
| `meta` | — | counts, rates, sources and frames of the above |

Accelerations come from CORRIMU and have **gravity removed** — nuScenes'
`ms_imu` includes it. Each file covers its scene plus 0.5 s on both sides.

### Full tool extras

- `LIDAR_BOTTOM_*` are `.pcd.bin` like `LIDAR_TOP`; their frames ride along as
  described above.
- Next to every LiDAR file of the full dataset, `<token>.time.bin` holds each
  point's acquisition time: float32 seconds relative to the frame timestamp,
  same order as the `.pcd.bin` (0–100 ms on the spinning LiDARs).
- `RADAR_FRONT` is NuScenes radar `.pcd` (18 fields), read by
  `RadarPointCloud.from_file` with the default filters. The ARS548 reports only
  radial velocity: `vx, vy` are it resolved along the line of sight, and
  `vx_comp, vy_comp` the same after adding the radar's own motion (from odom and
  CORRIMU), so static targets come out near zero. Fields the ARS548 does not
  report are set to pass the default filters (`dyn_prop` 4 = unknown,
  `ambig_state` 3, `invalid_state` 0).
- `ext/<scene>/<topic>.json` (topic `/` → `__`) is a JSON array of
  `{"utime", "bag_utime", "msg"}` for every other topic — vehicle CAN
  (`/bsw/vehicle_can`, `_extended`), commands and raw frames (`/can_tx0`,
  `/can_rx*`), `/bsw/ad_state`, mode switches, corner and vehicle radar objects,
  radar object list and status, `/app/loc/vehicle_state`, all Novatel logs,
  diagnostics. `utime` is the header stamp (bag time when there is none), in
  the same microseconds as `sample_data`; each file covers the scene ± 0.5 s.
  `_index.json` lists topic and message type per file. Values are the message
  fields as recorded (their own units); constants are omitted. Not exported: the
  `camera_info` topics (placeholder values), `/novatel/oem7/oem7raw` and
  `/radar/DetectionList` (its valid detections are `RADAR_FRONT`).

Useful flags (both tools): `--split`, `--sync-ms`, `--scene-dur`,
`--keyframe-stride`, `--rectify-balance`, `--jpeg-quality`, `--workers`,
`--no-validate`, `--keep-staging`.

### Screen bags first

```bash
python scripts/screen_bags.py /path/to/bags/ --nominal-hz 29.9
```

It measures each bag with the converter's own rules and prints what a conversion
would do with it: per-stream delivery and gaps, the coverage window, odom gaps,
the INS solution status over time, and the **scene yield** per sync tolerance —
how many LiDAR frames are usable, how often the sequence breaks, and how many
seconds end up in 20 s scenes. Each bag gets a PASS / MARGINAL / DROP verdict
with the reasons listed.

## Using the dataset

With the stock nuscenes-devkit. Worked examples live in
[`notebooks/devkit_tutorial.ipynb`](notebooks/devkit_tutorial.ipynb) and
[`notebooks/dataset_validation.ipynb`](notebooks/dataset_validation.ipynb).

**Load and walk.** A scene has 40 samples (2 Hz); its `sample_data` chains also
hold the sweeps — every 30 fps camera frame and every 10 Hz LiDAR frame.

```python
from nuscenes.nuscenes import NuScenes
root = '/data/tcar_nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=root, verbose=True)
nusc.list_scenes()

scene = nusc.scene[0]
sample = nusc.get('sample', scene['first_sample_token'])
sample['data']                                   # channel -> sample_data token (keyframes)

tok = scene['first_sample_token']                # samples of a scene
while tok:
    s = nusc.get('sample', tok); tok = s['next']

sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
while sd['next']:                                # keyframes and sweeps; see sd['is_key_frame']
    sd = nusc.get('sample_data', sd['next'])

path = nusc.get_sample_data_path(sd['token'])
cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])   # sensor in ego, camera_intrinsic
pose = nusc.get('ego_pose', sd['ego_pose_token'])                   # ego at sd['timestamp'], UTM
```

**Point clouds and rendering.** Headless, pass `out_path` to write a file.

```python
import matplotlib; matplotlib.use('Agg')
from nuscenes.utils.data_classes import LidarPointCloud

pc = LidarPointCloud.from_file(nusc.get_sample_data_path(sample['data']['LIDAR_TOP']))  # 4 x N
pc, times = LidarPointCloud.from_file_multisweep(nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=10)

nusc.render_sample(sample['token'], out_path='sample.png')
nusc.render_sample_data(sample['data']['LIDAR_TOP'], nsweeps=5, underlay_map=False, out_path='bev.png')
nusc.render_pointcloud_in_image(sample['token'], camera_channel='CAM_FRONT', out_path='proj.png')
```

**GNSS/INS.** `utime` is in the same microseconds as `sample_data['timestamp']`.

```python
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
can = NuScenesCanBus(dataroot=root)
pose = can.get_messages(scene['name'], 'pose')    # utime, pos, orientation, vel, accel, rotation_rate,
                                                  # lat, lon, height, ins_status
imu = can.get_messages(scene['name'], 'ms_imu')   # utime, linear_accel, rotation_rate, q
```

**Splits.** Scene names are official ones, so the official lists select them.

```python
from nuscenes.utils.splits import create_splits_scenes
splits = create_splits_scenes()
train = [s for s in nusc.scene if s['name'] in splits['train']]
val = [s for s in nusc.scene if s['name'] in splits['val']]
```

**Full dataset extras.** `CAM_TRAFFIC` and `RADAR_FRONT` are best-effort, so a
sample may lack them — use `sample['data'].get(...)`.

```python
import json, numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

if 'RADAR_FRONT' in sample['data']:                  # 18 x N; rows 8, 9 = vx_comp, vy_comp
    rpc = RadarPointCloud.from_file(nusc.get_sample_data_path(sample['data']['RADAR_FRONT']))

pc, _ = LidarPointCloud.from_file_multisweep(        # a bottom LiDAR in the LIDAR_TOP frame
    nusc, sample, 'LIDAR_BOTTOM_FRONT', 'LIDAR_TOP', nsweeps=3)

p = nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])
t = np.fromfile(p.replace('.pcd.bin', '.time.bin'), dtype=np.float32)   # s after sd timestamp

msgs = json.load(open(f"{root}/ext/{scene['name']}/bsw__vehicle_can.json"))
# [{"utime": ..., "bag_utime": ..., "msg": {...}}, ...]; ext/<scene>/_index.json lists the topics
```

**What does not work, or does not mean anything:**

- With default calibration (no `--calib`), anything geometric —
  `render_pointcloud_in_image`, `map_pointcloud_to_image`, box projection —
  is meaningless for the defaulted channels.
- `render_sample` on the full dataset draws every LiDAR channel into
  overlapping axes (devkit behaviour); render the channels one by one with
  `render_sample_data`.
- `render_scene` and `render_scene_channel` open an OpenCV window and need a
  display.
- The map is a blank placeholder: `render_egoposes_on_map` and the map
  expansion (`NuScenesMap`) show nothing useful.
- `NuScenesCanBus` has only `pose`, `ms_imu` and `meta`; `route` and the
  Renault Zoe vehicle messages do not exist, so `plot_baseline_route` and
  friends fail. Vehicle CAN is in `ext/` (full dataset).
- Detection evaluation needs a class/attribute mapping first (known
  limitation 6), and poses must stay float64 (limitation 8).

## Layout

```
bag2nuscenes.py       # CLI: standard dataset
bag2nuscenes_full.py  # CLI: full dataset
converter.py          # the pipeline both CLIs run: read, stage, rectify, materialize
nuscenes_writer.py    # frame selection, scene cutting, scene names, ego pose, 13 tables
rectify.py            # fisheye / plumb_bob -> pinhole
canbus.py             # GNSS/INS -> nuScenes CAN bus files
common.py             # topic map, calibration loading, geometry conventions
msg/                  # ObjectType / MotionType enums — the label taxonomy's source
packet_decoder/       # Robosense RSP128/RSM1/RSBP packet decoders (old packet bags)
scripts/              # diagnostics and QA
notebooks/            # dataset validation (outputs stripped)
docs/                 # design notes, sync reference, labelling handoff
```

`common.py` holds the single source of truth for the topic → channel mapping.
Do not re-declare it in a script; import it. `nuscenes_writer.py` deliberately
knows nothing about rosbags — it takes timestamps and returns tables, so the
sync and scene logic can be developed and tested without touching a bag.

## Topic ↔ channel mapping

Defined once in `common.py`, checked visually on the 2026-09-23 bags with
`scripts/extract_cam_viz.py` (a labelled 7-camera contact sheet).

| Topic | Channel | Tool |
|---|---|---|
| `/camera_3/compressed` | `CAM_FRONT` | both |
| `/camera_6/compressed` | `CAM_FRONT_LEFT` | both |
| `/camera_1/compressed` | `CAM_FRONT_RIGHT` | both |
| `/camera_2/compressed` | `CAM_BACK` | both |
| `/camera_5/compressed` | `CAM_BACK_LEFT` | both |
| `/camera_0/compressed` | `CAM_BACK_RIGHT` | both |
| `/camera_4/compressed` | `CAM_TRAFFIC` (tilted up at traffic lights) | full |
| `/middle/rslidar_points` (or `_packets`) | `LIDAR_TOP` | both |
| `/down_m1_front/rslidar_points` | `LIDAR_BOTTOM_FRONT` | full |
| `/down_m1_rear/rslidar_points` | `LIDAR_BOTTOM_REAR` | full |
| `/down_bp_left/rslidar_points` | `LIDAR_BOTTOM_LEFT` | full |
| `/down_bp_right/rslidar_points` | `LIDAR_BOTTOM_RIGHT` | full |
| `/radar/PointCloudDetection` | `RADAR_FRONT` | full |
| `/novatel/oem7/odom` | `ego_pose`, CAN bus `pose` | both |
| `/novatel/oem7/corrimu`, `/novatel/oem7/inspva` | CAN bus `pose`, `ms_imu` | both |

Before 2026-09 this table had camera_3 and camera_4 the other way round. Bags
recorded before then were not re-checked; run `extract_cam_viz.py` on one first.

## Calibration

**Calibration is not shipped with this repository.** Intrinsics and extrinsics
are specific to one vehicle build; `--calib` points at a snapshot you supply.
The `camera_info` topics in the bags carry a placeholder (`K` = 960/540, `D` = 0)
and are not used.

**Without calibration the tools still run.** Every channel `--calib` does not
cover — all of them when it is omitted — gets a default: identity extrinsic
(the sensor at the ego origin, axes aligned with the ego frame), and for a
camera a 90° pinhole `K` (`f = width / 2`, principal point at the centre) with
no distortion, so its images are copied as recorded instead of rectified. The
run prints which channels were defaulted and `<log>.import.json` lists them in
`calib_defaulted`. Such a dataset loads and renders in the devkit, but anything
geometric — projecting LiDAR into images, fusing sensors, radar ego-motion
compensation — is meaningless for the defaulted channels.

Expected layout — one directory per channel the tool emits:

```
<calib_dir>/
  CAM_FRONT/  CAM_FRONT_LEFT/  CAM_FRONT_RIGHT/
  CAM_BACK/   CAM_BACK_LEFT/   CAM_BACK_RIGHT/
  CAM_TRAFFIC/                           # full tool
      intrinsic.txt    # 3x3 K
      distortion.txt   # 4 coefficients -> OpenCV fisheye, 5 -> plumb_bob
      quat_r.txt       # rotation quaternion, w x y z
      t.txt            # translation
  LIDAR_TOP/
  LIDAR_BOTTOM_FRONT/ LIDAR_BOTTOM_REAR/ LIDAR_BOTTOM_LEFT/ LIDAR_BOTTOM_RIGHT/   # full tool
  RADAR_FRONT/                           # full tool
      r.txt            # rotation quaternion, w x y z
      t.txt            # translation
```

## Conventions

**Calibration convention.** The files use the OpenCV extrinsic convention
(`P_sensor = R · P_ego + t`). NuScenes stores the inverse — the sensor's pose
*in* the ego frame — so `common.opencv_ext_to_nuscenes_pose` inverts it when
writing `calibrated_sensor.json`. The ego frame is `base_link` of
`/novatel/oem7/odom` (x forward, y left, z up).

**Timestamps.** Everything is on header stamps, which are PTP-disciplined on the
2026-09-23 bags. A LiDAR frame's stamp is the **start** of the sweep: its points
were acquired over the following 100 ms (`time.bin` in the full dataset). The
cameras free-run at 29.898 Hz, so their phase against the 10 Hz LiDAR drifts
through a full camera period about every 9.8 s — which is why the set is picked
per frame, not by a fixed offset.

**Files.** Staged payloads are renamed into `samples/`/`sweeps/`, so each byte
is written exactly once. LiDAR frames are NuScenes `.pcd.bin`
(5 × float32 per point: `x, y, z, intensity, ring`). Robosense publishes an
organized cloud, so no-return directions arrive as NaN — about 42% of a
128 × 1800 sweep — and those points are dropped, because NuScenes point clouds
are unorganized and devkit consumers do not expect NaN.

## Diagnostics and QA

| Script | Purpose |
|---|---|
| `scripts/screen_bags.py` | Per-bag verdict: delivery and gaps per stream, coverage window, odom gaps, INS status, scene yield vs `--sync-ms` |
| `scripts/clock_diagnosis.py` | Per-sensor clock offset/skew vs the bag clock, anchored to GPS |
| `scripts/qa_report.py` | Dataset QA: sync (converter rule), ego anomalies, per-scene stats (`--bag` adds raw-timestamp sections) |
| `scripts/lidar2cam_projection.py` | Project LiDAR onto each raw camera image with the full distortion model — calibration check |
| `scripts/extract_cam_viz.py` | 7-camera contact sheet labelled with `common.py`'s mapping — mapping check |
| `scripts/viz_lidar_frame.py` | Single-frame BEV / side view |
| `scripts/rerun_viz.py` | rerun.io: 3D LiDAR + 7 cameras + ego on one timeline |

All of these write into gitignored output directories.

The run ends by loading the result with `NuScenes(version, dataroot)` (the 13
tables, foreign keys, `prev`/`next` chains) and reading one scene's `pose`
through `NuScenesCanBus`.

## Known limitations

These are real and unresolved. Read before trusting the output.

**1. Camera frame loss at record time, and what it costs.** The cameras reach
the bag through `/ros_bridge`; a best-effort bridge silently drops frames, and
dropped frames cannot be recovered. Because every LiDAR frame needs all six
cameras at one instant within 25 ms, a single dropped frame at the wrong phase
breaks the scene sequence. On the 2026-09-23 bags (all INS-good, no LiDAR loss):

| Bag | Length | Breaks | In 20 s scenes | Camera that broke it most |
|---|---|---|---|---|
| A-8 | 344 s | 5 | 280 s (14 scenes) | `CAM_BACK_LEFT` |
| A-9 | 285 s | 9 | 240 s (12 scenes) | `CAM_BACK_LEFT` |
| A-10 | 317 s | 44 | 160 s (8 scenes) | `CAM_BACK_LEFT` (camera_5) |

Record at 10G with reliable QoS; check camera_5 on A-10-like recordings.

**2. Rectification is only as good as the calibration.** No calibration of the
2026-09 build exists yet; the pipeline was verified with a synthetic one.
Without calibration the defaults apply (see *Calibration*) and nothing is
rectified.
`--rectify-balance 0` crops the periphery of the wide cameras. The raw images
are not kept.

**3. The LiDAR stamp is the start of the sweep.** The front of the car is
scanned at the stamp (and again at +100 ms, the frame seam), the right side at
about +25 ms, the rear at +50 ms, the left at +75 ms. The camera set is matched
to the stamp, so in the rear cameras the LiDAR points are on average ~50 ms
younger than the image. nuScenes stamps the end of the sweep and triggers each
camera as the LiDAR passes it; neither is possible here. `time.bin` (full
dataset) is what a motion-compensated consumer needs.

**4. Old recordings.** Bags before 2026-09 had independent per-camera stamps
spread up to ~47 ms (the 2026-03-23 traffic bags); with `--camera-group-ms 5`
their cameras rarely form a complete set. They also used the other camera
mapping (see above). The `/middle/rslidar_packets` path still works for
`LIDAR_TOP`, with frames on the bag receive clock.

**5. Annotations are not emitted — by design.** `sample_annotation.json` and
`instance.json` are written empty — labelling is done externally.
`category`/`attribute`/`visibility` carry the taxonomy the vendor labels against
(see *Annotations* below).

**6. The label taxonomy is not the detection-eval taxonomy.** The devkit's
detection evaluation only knows its ten class names and eight attribute names.
Of our 22 categories only `vehicle.car`, `vehicle.truck`, `vehicle.construction`,
`vehicle.bicycle` and `movable_object.trafficcone` map; `human.pedestrian` and
`vehicle.bus` are silently left out, and a `motion.*` attribute makes
`DetectionBox` raise. Evaluating with the stock devkit needs a mapping step.

**7. `CAM_TRAFFIC` is best-effort** (full dataset only): it is attached when in
tolerance and omitted otherwise, so some samples carry it and some do not.

**8. `ego_pose` is in absolute UTM.** Ego poses come from `/novatel/oem7/odom`,
which the Novatel driver derives from the INS solution in UTM zone 52N, so
translations are of order (3.0e5, 4.1e6) m. The JSON and the devkit are float64
and lose nothing, but float32 has a 0.25 m grid at that magnitude: any consumer
that casts poses to float32 quantizes ego positions and any box positions
produced in that frame. Keep float64 end to end, or subtract a dataset-wide
origin before casting. The converter leaves the frame absolute so that logs
recorded on different days stay in one coordinate system.

**9. INS status is not checked by the converter.** The driver keeps publishing
odom while the INS is aligning or has lost GNSS. `scripts/screen_bags.py`
reports the INSPVA status runs, and every CAN bus `pose` record carries
`ins_status` — screen before converting.

**10. Official scene names are nuScenes' names.** Never put this dataset and
the real nuScenes in one dataroot. Capacity is 685 train and 150 val scenes.

**11. Radar timing and fields.** The ARS548's own clock is not synchronized
(`timestamp_syncstatus` 2); its frames are placed at the ROS header stamp, i.e.
arrival time, whose latency is not measured. Velocities are radial only (see
above).

## Annotations

The dataset ships **schema-complete and value-empty**: everything an annotation
references is populated, and only the two annotation tables are left blank for an
external labelling vendor.

| Table | State |
|---|---|
| `category.json` | **22 classes**, from `msg/ObjectType.msg` |
| `attribute.json` | **5 motion states**, from `msg/MotionType.msg` |
| `visibility.json` | NuScenes' standard four bins, with nuScenes' literal tokens `"1"`–`"4"` (tools filter on them) |
| `sample_annotation.json` | `[]` — vendor fills |
| `instance.json` | `[]` — vendor fills |

Deriving the taxonomy from the `.msg` enums keeps the perception class list and
the labelling class list from drifting apart.

[`docs/labeling_handoff.md`](docs/labeling_handoff.md) is the document to send to
the vendor: required fields and types for both tables, frame conventions, and
the devkit acceptance test.

## Docs

- [`docs/labeling_handoff.md`](docs/labeling_handoff.md) — what the labelling vendor must return
- [`docs/pipeline_overview.md`](docs/pipeline_overview.md) — design rationale
- [`docs/sync_reference.md`](docs/sync_reference.md) — how our sync tolerance compares to public datasets
- [`CHANGELOG.md`](CHANGELOG.md) — what changed, why, and how it was verified
