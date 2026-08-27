# ROSbag → NuScenes parser

Convert ROS1 rosbags from the T-Car platform (Robosense LiDAR + 7 e-CON cameras
+ Novatel INS + perception fusion output) into a NuScenes v1.0-trainval dataset.

No ROS installation required — [`rosbags`](https://pypi.org/project/rosbags/)
reads `.bag` files directly, and the Robosense MSOP/DIFOP packet decoder is
bundled, so neither `roscore` nor the vendor SDK is needed.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e '.[verify]'   # nuscenes-devkit, for output validation + QA report
pip install -e '.[plots]'    # matplotlib/jupyter, for diagnostics and notebooks
```

## Convert

```bash
python bag2nuscenes.py /path/to.bag --out /data/tcar_nuscenes --calib /path/to/calib
```

One pass over the bag. Sensor payloads stream into a staging directory inside
the output root; once synchronization and scene partitioning have decided which
frames become samples and sweeps, the staged files are **renamed** into place —
a metadata operation on the same filesystem. Nothing is written twice and no
intermediate dump survives the run.

LiDAR is read from whichever topic the bag carries: `/middle/rslidar_points`
(PointCloud2, read directly) or `/middle/rslidar_packets` (raw MSOP/DIFOP,
decoded in-process). Only the packet path is slow — roughly 40 minutes for a
20-minute bag, against a couple of minutes for PointCloud2.

Running it again on another bag **appends**: scene numbering continues, sensor
and category tokens are reused, and re-importing the same bag is refused.

Frames are only used inside the **coverage window**: the interval in which the
LiDAR, the six standard cameras and odom are all present, shrunk by `--sync-ms`
at both ends. Sensors start at different times — up to 1.4 s apart on the
2026-08-19 bags — and a keyframe outside that window would have no image or no
pose to attach. The per-stream offsets and the cut are printed and recorded in
`<log>.import.json`; `scripts/screen_bags.py` shows them before converting.

Useful flags: `--sync-ms`, `--keyframe-stride`, `--scene-dur`,
`--no-include-traffic-cam`, `--no-validate`, `--keep-staging`.

### Screen bags first

Not every bag is convertible. Run this before spending an hour on a conversion:

```bash
python scripts/screen_bags.py /path/to/bags/ --nominal-hz 30
```

It measures each bag with the converter's own rules and prints what a conversion
would do with it: per-stream delivery and gaps (cameras, LiDAR, odom), the
*coverage window* and how much would be cut at head and tail, odom gaps, the INS
solution status over time, and keyframe acceptance per sync tolerance (this is
how `--sync-ms` gets chosen). Each bag gets a PASS / MARGINAL / DROP verdict with
the reasons listed. See *Camera frame loss* and *INS status* below.

## Layout

```
bag2nuscenes.py     # CLI: read the bag, stage payloads, materialize the dataset
nuscenes_writer.py  # sync, scene partitioning, ego-pose interpolation, 13 tables
common.py           # topic map, calibration loading, geometry conventions
msg/                # ObjectType / MotionType enums — the label taxonomy's source
packet_decoder/     # Robosense RSP128/RSM1/RSBP packet decoders
scripts/            # diagnostics and QA
notebooks/          # dataset validation (outputs stripped)
docs/               # design notes, sync reference
```

`common.py` holds the single source of truth for the topic → channel mapping.
Do not re-declare it in a script; import it. `nuscenes_writer.py` deliberately
knows nothing about rosbags — it takes timestamps and returns tables, so the
sync and scene logic can be developed and tested without touching a bag.

## Topic ↔ channel mapping

Defined once in `common.py`, verified visually with
`scripts/extract_cam_viz.py` (produces a labelled 7-camera contact sheet).

| Topic | Channel |
|---|---|
| `/camera_4/compressed` | `CAM_FRONT` |
| `/camera_6/compressed` | `CAM_FRONT_LEFT` |
| `/camera_1/compressed` | `CAM_FRONT_RIGHT` |
| `/camera_2/compressed` | `CAM_BACK` |
| `/camera_5/compressed` | `CAM_BACK_LEFT` |
| `/camera_0/compressed` | `CAM_BACK_RIGHT` |
| `/camera_3/compressed` | `CAM_TRAFFIC` (7th channel, non-standard) |
| `/middle/rslidar_packets` or `/middle/rslidar_points` | `LIDAR_TOP` |
| `/novatel/oem7/odom` | ego pose |

## Calibration

**Calibration is not shipped with this repository.** Intrinsics and extrinsics
are specific to one vehicle build, and the labelling vendor produces their own,
so `--calib` is required and must point at a snapshot you supply.

Expected layout:

```
<calib_dir>/
  CAM_FRONT/  CAM_FRONT_LEFT/  CAM_FRONT_RIGHT/
  CAM_BACK/   CAM_BACK_LEFT/   CAM_BACK_RIGHT/
      intrinsic.txt    # 3x3 K
      distortion.txt   # 4 coefficients -> OpenCV fisheye, 5 -> plumb_bob
      quat_r.txt       # rotation quaternion, w x y z
      t.txt            # translation
  LIDAR_TOP/
      r.txt            # rotation quaternion, w x y z
      t.txt            # translation
```

## Conventions

**Calibration convention.** The files use the OpenCV extrinsic convention
(`P_sensor = R · P_ego + t`). NuScenes stores the inverse — the sensor's pose
*in* the ego frame — so `common.opencv_ext_to_nuscenes_pose` inverts it when
writing `calibrated_sensor.json`. Sanity check: the resulting sensor positions
are `LIDAR_TOP (1.56, 0, 1.90)` and `CAM_FRONT (2.04, −0.14, 1.73)` with the
front camera's optical axis along +x.

**Files.** Staged payloads are renamed into `samples/`/`sweeps/`, so each byte
is written exactly once. LiDAR sweeps are stored as NuScenes `.pcd.bin`
(5 × float32 per point). Robosense publishes an organized cloud, so no-return
directions arrive as NaN — about 42% of a 128 × 1800 sweep — and those points
are dropped, because NuScenes point clouds are unorganized and devkit consumers
do not expect NaN.

## Diagnostics and QA

| Script | Purpose |
|---|---|
| `scripts/screen_bags.py` | Per-bag verdict: delivery and gaps per stream, coverage window, odom gaps, INS status, keyframe acceptance vs `--sync-ms` |
| `scripts/clock_diagnosis.py` | Per-sensor clock offset/skew vs the bag clock, anchored to GPS |
| `scripts/qa_report.py` | Dataset QA: sync survival, ego anomalies, per-scene stats (`--bag` adds raw-timestamp sections) |
| `scripts/lidar2cam_projection.py` | Project LiDAR onto each camera — calibration check |
| `scripts/extract_cam_viz.py` | 7-camera contact sheet — topic mapping check |
| `scripts/viz_lidar_frame.py` | Single-frame BEV / side view |
| `scripts/rerun_viz.py` | rerun.io: 3D LiDAR + 7 cameras + ego on one timeline |

All of these write into gitignored output directories.

The run ends by loading the result with `NuScenes(version, dataroot)`, which
validates the 13 tables, foreign keys, and `prev`/`next` chains.

## Known limitations

These are real and unresolved. Read before trusting the output.

**1. Images are not undistorted.** NuScenes has no distortion field, so any
consumer treats `camera_intrinsic` as a pinhole `K`. Five of the six cameras are
OpenCV *fisheye* (4 coefficients); only `CAM_FRONT` is `plumb_bob`. Projecting
with `K` alone displaces points by a median of 10–60 px and up to ~300 px at the
periphery, and 5–33 % of points that belong in frame land outside it. Note that
`scripts/lidar2cam_projection.py` uses the correct model, so its output looks
better than the dataset actually is.

**2. Camera frame loss at record time.** The cameras reach the bag through
`/ros_bridge`; LiDAR and odom are native ROS1 nodes and arrive at ~100 %. A
best-effort bridge silently drops frames. Measured on the 2026-08-19 A/B set:

| Recording | Worst channel | Complete 7-camera sets |
|---|---|---|
| 10G + reliable | 97.9 % | **98.6 %** |
| 10G + best_effort | 87.7 % | 84.0 % |
| 1G + reliable | 24.1 % | — (link saturated at ~840 Mbps) |

Record at **10G with reliable QoS**. Dropped frames cannot be recovered later.

**3. Two camera timestamp regimes.** Newer recordings give all 7 cameras a
bit-identical trigger timestamp, so inter-camera sync error is exactly zero and
frame sets can be grouped by exact equality. Older recordings (e.g. the
2026-03-23 traffic bags) have independent per-camera stamps spread up to ~47 ms.
`nuscenes_writer.sync_keyframes` currently only implements the latter — per-channel
nearest-neighbour matching within `--sync-ms`. On new-regime bags this is
unnecessary work and costs keyframes.

**4. LiDAR frame timestamps are the end of the sweep,** and on the packet path
they come from the bag receive clock (the sensor's own clock is not disciplined
— PTP is not wired up yet). The points in a frame were acquired over the
preceding ~100 ms, so the cloud is on average ~50 ms older than the camera
matched to it. The 0° frame split also puts the sweep seam inside `CAM_FRONT`'s
field of view. The decoder computes per-point timestamps but `lidar_points_to_pcdbin`
discards them, which is what a proper fix would need. `lidar_time_base` in
`<log>.import.json` records which clock a given import used.

**5. Camera clock skew varies per recording.** Measured against the bag clock:
−15 to −26 ppm on one bag, −147 to −180 ppm on another (≈200 ms over 20 min).
It is not a fixed driver property — measure it per bag with
`scripts/clock_diagnosis.py` and remove the *slope*. The intercept mixes clock
offset with genuine capture latency and cannot be separated without
motion-based estimation.

**6. Annotations are not emitted — by design.** `sample_annotation.json` and
`instance.json` are written empty — labelling is done
externally. `category`/`attribute`/`visibility` carry the taxonomy the vendor
labels against (see *Annotations* below).

**7. `CAM_TRAFFIC` has no calibration.** It gets an identity extrinsic and a
fabricated `K`, so it must not be used for geometry. It is also **best-effort**:
only the six standard cameras gate a keyframe, so `CAM_TRAFFIC` is attached when
it is in sync tolerance and omitted otherwise — some samples carry six camera
channels and some seven. Pass `--no-include-traffic-cam` to leave it out
entirely.

**8. `ego_pose` is in absolute UTM.** Ego poses come from `/novatel/oem7/odom`,
which the Novatel driver derives from the INS solution (INSPVA) in UTM zone 52N,
so translations are of order (3.0e5, 4.1e6) m. The JSON and the devkit are
float64 and lose nothing, but float32 has a 0.25 m grid at that magnitude (0.5 m
north of latitude ~37.9°): any consumer that casts poses to float32 — most
training loaders, some annotation tools — quantizes ego positions and, worse,
any box positions produced in that frame. Keep float64 end to end, or subtract a
dataset-wide origin before casting and store the origin alongside. The converter
deliberately leaves the frame absolute so that logs recorded on different days
stay in one coordinate system.

**9. INS status is not checked by the converter.** The driver keeps publishing
odom while the INS is still aligning or has lost GNSS, and those poses look
normal. `scripts/screen_bags.py` reports the INSPVA status runs and odom gaps —
screen before converting.

## Sensor configuration

| | |
|---|---|
| LiDAR | **1** channel, `LIDAR_TOP`, 10 Hz. `.pcd.bin`, 5 × float32 (`x, y, z, intensity, ring`). |
| Cameras | **7** channels, 30 Hz source. Six standard NuScenes channels + `CAM_TRAFFIC`. |
| Samples | 2 Hz keyframes, anchored on a LiDAR sweep. |

The six standard channels gate keyframes and carry real calibration.
`CAM_TRAFFIC` does neither — see limitation 7.

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
the vendor: required fields and types for both tables, frame conventions, the
`CAM_TRAFFIC` caveats, the fisheye warning, and the devkit acceptance test.

## Docs

- [`docs/labeling_handoff.md`](docs/labeling_handoff.md) — what the labelling vendor must return
- [`docs/pipeline_overview.md`](docs/pipeline_overview.md) — design rationale, why a single stage
- [`docs/sync_reference.md`](docs/sync_reference.md) — how our sync tolerance compares to public datasets
