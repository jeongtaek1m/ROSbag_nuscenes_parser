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

## Pipeline

Three stages. The split exists so that the expensive parts are cached: LiDAR
packet decoding runs once per bag, and stage-2 parameters (sync tolerance,
keyframe rate, scene length) can be re-tuned without re-reading the bag.

| Stage | Script | Input → Output | Rough cost (20-min bag) |
|---|---|---|---|
| 1a | `decode_lidar.py` | bag → `intermediate/<bag>/lidar/*.bin.zst` | ~40 min, 22 GB |
| 1b | `bag2raw.py` | bag → cameras, odom, annotations, calib | ~5 min, 88 GB |
| 2 | `raw2nuscenes.py` | intermediate → NuScenes dataset | ~10–15 min, 50 GB |

1a and 1b touch different topics and can run concurrently.

```bash
python decode_lidar.py  /path/to.bag --out /data/intermediate
python bag2raw.py       /path/to.bag --out /data/intermediate --calib calib/2025_6_27
python raw2nuscenes.py  /data/intermediate/<bag_name> --out /data/tcar_nuscenes
```

Stage 2 auto-appends: running it on a second bag continues the scene numbering,
reuses sensor/category tokens, and refuses to import the same bag twice.

### Screen bags first

Not every bag is convertible. Run this before spending an hour on a conversion:

```bash
python scripts/screen_bags.py /path/to/bags/ --nominal-hz 30
```

It reports per-camera delivery rate and flags bags that lost frames at record
time (see *Camera frame loss* below).

## Layout

```
bag2raw.py          decode_lidar.py     raw2nuscenes.py    # the three stages
common.py                                                  # shared topic map, calib, geometry
msg/                                                       # custom .msg for ObjectFusionArray etc.
packet_decoder/                                            # Robosense RSP128/RSM1/RSBP decoders
calib/<date>/                                              # intrinsic/extrinsic snapshots
scripts/                                                   # diagnostics and QA
notebooks/                                                 # dataset validation (outputs stripped)
docs/                                                      # pipeline overview, sync reference
```

`common.py` holds the single source of truth for the topic → channel mapping.
Do not re-declare it in a script; import it.

## Intermediate format

```
intermediate/<bag_basename>/
  cameras/CAM_*/<header_ns>.jpg      # JPEG bytes copied verbatim, no re-encode
  lidar/<frame_ts_ns>.bin.zst        # 5 x float32 (x,y,z,intensity,ring), zstd
  odom.parquet                       # ts_ns, tx,ty,tz, qw,qx,qy,qz
  annotations.parquet                # ts_ns + flattened ObjectFusion fields
  calib.json                         # snapshot of the calibration used
  meta.json / meta_lidar.json        # source bag, topic mapping, counts, time base
```

Every file is inspectable on its own — JPEGs open, parquet reads in pandas or
DuckDB, lidar frames are a `zstd` decompress and a `reshape(-1, 5)` away. When
something looks wrong, that tells you whether the fault is in stage 1 or 2.

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
| `/post_fusion_object` | annotations (extracted, not yet emitted) |

## Conventions

**Calibration.** The `calib/` files use the OpenCV extrinsic convention
(`P_sensor = R · P_ego + t`). NuScenes stores the inverse — the sensor's pose
*in* the ego frame — so `common.opencv_ext_to_nuscenes_pose` inverts it when
writing `calibrated_sensor.json`. Sanity check: the resulting sensor positions
are `LIDAR_TOP (1.56, 0, 1.90)` and `CAM_FRONT (2.04, −0.14, 1.73)` with the
front camera's optical axis along +x.

**Files.** Camera JPEGs are **hard-linked** from the intermediate dump into
`samples/`/`sweeps/`, so the dataset costs no extra disk *and* survives deleting
the intermediate directory. LiDAR is decompressed to real `.pcd.bin` files
because NuScenes requires raw point clouds.

## Diagnostics and QA

| Script | Purpose |
|---|---|
| `scripts/screen_bags.py` | Per-camera delivery rate; which bags are worth converting |
| `scripts/clock_diagnosis.py` | Per-sensor clock offset/skew vs the bag clock, anchored to GPS |
| `scripts/preflight_check.py` | Intermediate integrity before stage 2 |
| `scripts/qa_report.py` | Dataset QA: sync survival, ego anomalies, per-scene stats |
| `scripts/sync_stats.py` | Acceptance rate vs sync tolerance |
| `scripts/lidar2cam_projection.py` | Project LiDAR onto each camera — calibration check |
| `scripts/extract_cam_viz.py` | 7-camera contact sheet — topic mapping check |
| `scripts/viz_lidar_frame.py` | Single-frame BEV / side view |
| `scripts/rerun_viz.py` | rerun.io: 3D LiDAR + 7 cameras + ego on one timeline |

All of these write into gitignored output directories.

Stage 2 ends by loading the result with `NuScenes(version, dataroot)`, which
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
`raw2nuscenes.py` currently only implements the latter — per-channel
nearest-neighbour matching within `--sync-ms`. On new-regime bags this is
unnecessary work and costs keyframes.

**4. LiDAR frame timestamps are the end of the sweep,** and are taken from the
bag receive clock when decoding packets (the sensor's own clock is not
disciplined — PTP is not wired up yet). The points in a frame were acquired over
the preceding ~100 ms, so the cloud is on average ~50 ms older than the camera
matched to it. The 0° frame split also puts the sweep seam inside `CAM_FRONT`'s
field of view. The decoder computes per-point timestamps but `decode_lidar.py`
discards them (`arr[:, :5]`), which is what a proper fix would need.

**5. Camera clock skew varies per recording.** Measured against the bag clock:
−15 to −26 ppm on one bag, −147 to −180 ppm on another (≈200 ms over 20 min).
It is not a fixed driver property — measure it per bag with
`scripts/clock_diagnosis.py` and remove the *slope*. The intercept mixes clock
offset with genuine capture latency and cannot be separated without
motion-based estimation.

**6. Annotations are not emitted.** `bag2raw.py` extracts
`/post_fusion_object` into `annotations.parquet`, but `raw2nuscenes.py` writes
`sample_annotation.json` and `instance.json` as empty — labelling is done
externally. `category`/`attribute`/`visibility` hold one placeholder record
each; **replace them with the real taxonomy** before handing the dataset to a
labelling vendor.

**7. `CAM_TRAFFIC` has no calibration.** It gets an identity extrinsic and a
fabricated `K`. Standard 6-camera pipelines ignore it; do not use it for
geometry. Pass `--no-include-traffic-cam` to leave it out.

## Docs

- [`docs/pipeline_overview.md`](docs/pipeline_overview.md) — design rationale, timings, why three stages
- [`docs/sync_reference.md`](docs/sync_reference.md) — how our sync tolerance compares to public datasets
