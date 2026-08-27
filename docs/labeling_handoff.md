# Labelling handoff spec

What this pipeline delivers to a labelling vendor, and what the vendor must
deliver back.

The converter produces a complete NuScenes v1.0-trainval dataset **except for the
annotations**. `sample_annotation.json` and `instance.json` are written as empty
arrays on purpose. Everything they reference — samples, sample_data, ego poses,
sensor calibration, and the label taxonomy — is already populated, so the vendor
only has to fill in the two annotation tables.

`calibrated_sensor.json` is populated from whatever snapshot the conversion ran
with. If the vendor is also producing calibration, treat those values as
provisional and overwrite them.

## Sensor configuration

| | |
|---|---|
| LiDAR | **1** channel, `LIDAR_TOP`. 10 Hz. Files are `.pcd.bin`, 5 × float32 per point: `x, y, z, intensity, ring`. Points are in the LiDAR frame. |
| Cameras | **7** channels. Six standard NuScenes channels plus `CAM_TRAFFIC`. 30 Hz source. |
| Keyframes (samples) | 2 Hz, anchored on a LiDAR sweep. |

### The seventh camera is different — read this

`CAM_TRAFFIC` is a forward, upward-tilted camera for traffic lights and signs. It
is **not** part of the standard six and is handled differently in three ways:

1. **Its calibration is a placeholder.** `calibrated_sensor` for `CAM_TRAFFIC`
   holds an identity extrinsic and a fabricated intrinsic. **Do not use it for
   any 3D geometry** — no projection, no 3D-to-2D transfer, no cross-camera
   consistency checks.
2. **It never gates a keyframe.** Only the six standard cameras must be in sync
   tolerance for a sample to exist. `CAM_TRAFFIC` is attached when it happens to
   be in tolerance and is simply absent otherwise, so **some samples have six
   camera channels and some have seven**. Handle the missing key.
3. **It is for 2D work only** — traffic light state, sign classification — not
   for 3D box annotation.

The six standard channels (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`,
`CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`) are always present on every sample
and carry real calibration.

### Images are not undistorted

NuScenes has no distortion field, so `camera_intrinsic` is a plain pinhole `K`.
**Five of the six standard cameras are OpenCV fisheye** (equidistant, 4
coefficients, ~96° HFOV); only `CAM_FRONT` is `plumb_bob`. Projecting 3D points
with `K` alone is wrong by a median of 10–60 px and up to ~300 px at the image
edge.

Practical consequence for labelling: **annotate 3D boxes in the LiDAR point
cloud**, and treat camera images as visual context rather than as a source of
precise 2D-3D correspondence. If 2D boxes are required, label them directly in
the image rather than by projecting the 3D box.

The true distortion coefficients and projection model live in the calibration
snapshot used for the conversion, which is not part of the code repository.
Request it if you need it — and if you are producing your own calibration,
`calibrated_sensor.json` in the delivered dataset is provisional and can be
replaced wholesale.

## Taxonomy the vendor labels against

Already written into the dataset; do not invent new names.

- **`category.json`** — 22 classes, derived from `msg/ObjectType.msg`. Names
  follow the NuScenes `family.thing` convention (`vehicle.car`,
  `human.pedestrian`, `movable_object.trafficcone`, …).
- **`attribute.json`** — 5 motion states, derived from `msg/MotionType.msg`
  (`motion.unknown`, `motion.stationary`, `motion.stopped`,
  `motion.moving_slowly`, `motion.moving`).
- **`visibility.json`** — NuScenes' standard four bins: `v0-40`, `v40-60`,
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
| `visibility_token` | str | FK into `visibility.json` |
| `translation` | [x, y, z] | box centre in the **global** frame, metres |
| `size` | [w, l, h] | width, length, height in metres |
| `rotation` | [w, x, y, z] | box orientation quaternion in the **global** frame |
| `num_lidar_pts` | int | LiDAR points inside the box for this sample |
| `num_radar_pts` | int | 0 — no radar in this dataset |
| `prev` | str | previous annotation of the same instance, `""` at the start |
| `next` | str | next annotation of the same instance, `""` at the end |

### Constraints that must hold

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
