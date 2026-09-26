# Labelling handoff spec

What this pipeline delivers to a labelling vendor, and what the vendor must
deliver back.

The converter produces a complete NuScenes v1.0-trainval dataset **except for the
annotations**. `sample_annotation.json` and `instance.json` are written as empty
arrays on purpose. Everything they reference — samples, sample_data, ego poses,
sensor calibration, and the label taxonomy — is already populated, so the vendor
only has to fill in the two annotation tables.

## Sensor configuration

The vendor receives the **standard** dataset (`bag2nuscenes.py`), which has
exactly the sensor set of nuScenes minus the radars.

| | |
|---|---|
| LiDAR | **1** channel, `LIDAR_TOP`. 10 Hz. Files are `.pcd.bin`, 5 × float32 per point: `x, y, z, intensity, ring`. Points are in the LiDAR frame. The frame timestamp is the **start** of the 100 ms sweep. |
| Cameras | The **6** standard channels (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`), 30 Hz, every one present on every sample. |
| Samples | 2 Hz, 40 per scene, anchored on a LiDAR frame. Scenes are exactly 20 s. |
| Radar | none — `num_radar_pts` is always 0. |

All six cameras of a sample were captured at the same instant (the cameras
share a trigger), within 25 ms of the sample's LiDAR frame.

### Images are rectified pinhole images

The source cameras are wide-angle (OpenCV fisheye, and plumb_bob for the front
camera). The converter undistorts every image with the calibration, so
`camera_intrinsic` in `calibrated_sensor.json` is the exact pinhole `K` of the
image as delivered and there are no distortion coefficients to apply. Projecting
3D points with `K` — as `nuscenes-devkit` does — is correct, and 2D boxes may be
derived by projecting 3D boxes.

`calibrated_sensor.json` is populated from whatever calibration snapshot the
conversion ran with. If the vendor is also producing calibration, note that it
applies to the **raw** images, which are not part of the delivery; ask for them
together with the snapshot.

## Taxonomy the vendor labels against

Already written into the dataset; do not invent new names.

- **`category.json`** — 22 classes, derived from `msg/ObjectType.msg`. Names
  follow the NuScenes `family.thing` convention (`vehicle.car`,
  `human.pedestrian`, `movable_object.trafficcone`, …).
- **`attribute.json`** — 5 motion states, derived from `msg/MotionType.msg`
  (`motion.unknown`, `motion.stationary`, `motion.stopped`,
  `motion.moving_slowly`, `motion.moving`).
- **`visibility.json`** — tokens are the literal strings `"1"`, `"2"`, `"3"`, `"4"` exactly as in nuScenes (downstream tools filter on them; do not invent others). NuScenes' standard four bins: `v0-40`, `v40-60`,
  `v60-80`, `v80-100`, measured as the fraction of the object visible across all
  camera channels.

## What the vendor returns

Two JSON files, matching the NuScenes schema exactly.

### `instance.json` — one record per tracked object identity

| Field | Type | Meaning |
|---|---|---|
| `token` | str | 32-hex, unique |
| `category_token` | str | FK into `category.json` — constant for the life of the instance |
| `nbr_annotations` | int | number of `sample_annotation` records for this instance |
| `first_annotation_token` | str | FK, first in the temporal chain |
| `last_annotation_token` | str | FK, last in the temporal chain |

### `sample_annotation.json` — one record per object per sample

| Field | Type | Meaning |
|---|---|---|
| `token` | str | 32-hex, unique |
| `sample_token` | str | FK into `sample.json` |
| `instance_token` | str | FK into `instance.json` |
| `attribute_tokens` | list[str] | FKs into `attribute.json` (may be empty) |
| `visibility_token` | str | one of `"1"`, `"2"`, `"3"`, `"4"` (FK into `visibility.json`) |
| `translation` | [x, y, z] | box centre in the **global** frame, metres |
| `size` | [w, l, h] | width, length, height in metres |
| `rotation` | [w, x, y, z] | box orientation quaternion in the **global** frame |
| `num_lidar_pts` | int | LiDAR points inside the box for this sample |
| `num_radar_pts` | int | 0 — no radar in the standard dataset |
| `prev` | str | previous annotation of the same instance, `""` at the start |
| `next` | str | next annotation of the same instance, `""` at the end |

### Constraints that must hold

- The global frame is the location's **east-north-up frame in metres** (x east,
  y north, z up) at a fixed origin; `log.json`'s `global_frame` names it
  (`enu@37.200000,126.830000,0.000`). Coordinates stay within a few km of the
  origin. Datasets converted before 2026-09-24 were in UTM zone 52N (order
  (3.0e5, 4.1e6)) until rebuilt with `scripts/rebuild_ego_pose.py`: label only a
  dataset whose `log.json` carries `global_frame`, or boxes and poses disagree.
- `translation`/`rotation` are in the **global** frame, not the ego or LiDAR
  frame. Convert with the `ego_pose` referenced by that sample's `LIDAR_TOP`
  `sample_data`, then the `calibrated_sensor` for `LIDAR_TOP`.
- `prev`/`next` form one unbroken chain per instance, ordered by sample
  timestamp, with `""` at both ends.
- `nbr_annotations` equals the length of that chain, and
  `first_annotation_token`/`last_annotation_token` are its endpoints.
- Every FK resolves. `NuScenes(version, dataroot)` must load without error —
  that is the acceptance test.

## Acceptance check

```bash
pip install nuscenes-devkit
python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='<dataroot>', verbose=True)
print(len(nusc.sample_annotation), 'annotations,', len(nusc.instance), 'instances')
nusc.list_categories()
"
```

Loading exercises the full foreign-key graph and the `prev`/`next` chains, so a
clean load is a strong integrity signal.
