# Changelog

## 2026-09-24 — ego_pose를 INSPVA(GPS 시각)로, global 좌표계를 장소별 ENU로

온라인 캘리브레이션 작업 중 `/novatel/oem7/odom` 위치가 1초씩 멈춰 있는 것을 발견해 ego pose 소스를 바꿨다.

### 요약

| 항목 | 전 | 후 |
|---|---|---|
| ego_pose / CAN bus `pose` 소스 | `/novatel/oem7/odom` | **`/novatel/oem7/inspva`** (100 Hz). odom은 INSPVA가 없는 bag에서만 (경고) |
| 시각 | 헤더 스탬프 (도착 시각: 측정보다 중앙값 1.7 ms, 최대 ~10 ms 늦음) | 수신기 **GPS 측정 시각** − 18 s (헤더와 같은 UTC 시계). CORRIMU도 동일 |
| global 좌표계 | UTM 52N 절대좌표 (위치는 UTM 격자, 방향은 진북 ENU — 약 1.3° 어긋남) | nuScenes처럼 장소별 고정 원점의 **동-북-상(ENU)**, `korea-test` = 37.20° N 126.83° E, 타원체고 0 |
| 높이 | odom z = 해수면 높이 | INSPVA 타원체고 기준 ENU up |
| log.json | — | `location`, `global_frame` (`enu@37.200000,126.830000,0.000`; devkit은 무시) |
| 이어쓰기 | 제한 없음 | 좌표계가 다른 데이터셋에는 거부 → `scripts/rebuild_ego_pose.py` 안내 |
| 기존 데이터셋 | — | **`scripts/rebuild_ego_pose.py` (신규)**: ego_pose·CAN bus·log.json만 bag의 INSPVA로 다시 씀 (토큰·타임스탬프·센서 파일 그대로) |

### 근거 (0923 bag, A-8·A-9·A-10)

- odom 위치는 이동 중 샘플의 55–80 %에서 직전 값 그대로 — 1 Hz 위치 소스. 변환된 데이터셋(A-8·A-9·A-10, ego_pose 125,769개)을 다시 계산하니 기존 ego_pose가 수평으로 log별 중앙값 0.77–2.57 m, p95 7.2–10.4 m, 최대 9.3–15.1 m 틀려 있었음.
- odom이 갱신되는 샘플에서는 odom = UTM(INSPVA 위경도)가 0.15 mm로 일치 → 기준점 동일, `common.geodetic_to_utm` 검증. odom 쿼터니언은 INSPVA 자세(`common.novatel_attitude`)와 0.000°로 일치 → 진북 ENU.
- INSPVA 높이는 타원체고: 수신기 로그의 BESTPOS(해수면 높이 + 지오이드고 22.70 m)와 0.86 m(안테나–INS 기준점 높이차) 차이. odom z는 INSPVA 높이 − 22.70 m = 해수면 높이.
- 실시간 해는 이미 RTK 고정해 98 %(BESTPOS σ 수평 2 cm) — INSPVA가 cm급, odom의 문제는 1 Hz 위치 소스뿐.

### 변경 파일

- `common.py`: `LOCATION`, `GLOBAL_ORIGINS`, `global_frame_id`, `geodetic_to_enu`, `geodetic_to_utm`/`utm_to_geodetic`(Krüger 3차, 왕복 0.3 mm), `novatel_gps_ns`, `novatel_attitude`, `GEOID_UNDULATION`(odom 대체용).
- `canbus.py`: `inspva_row`/`odom_row`/`corrimu_row`, `pose_from_inspva`/`pose_from_odom`, `build_ins`(변환기와 재작성 도구가 공유). `InsData`의 `odom_*` → `pose_*`, `pose_source`, `global_frame`. meta에 시각 기준과 좌표계 기록.
- `converter.py`: INSPVA 전 필드를 읽고 `build_ins`로 궤적 생성. 요약 줄에 소스·좌표계·헤더 지연. 좌표계 다른 데이터셋 이어쓰기 거부. `import.json` 통계 키 `odom_max_gap_ms` → `pose_max_gap_ms`, `pose_source`, `global_frame`, `ins_header_lag_ms` 추가.
- `nuscenes_writer.py`: `SensorData.odom_*` → `pose_*`, `location`, `global_frame`; `required_streams`의 `ODOM` → `INS`; log.json에 `global_frame`.
- `scripts/screen_bags.py`: 필수 스트림 `ODOM` → `INS`(INSPVA, 없으면 odom + MARGINAL 플래그). `--max-odom-gap` → `--max-ins-gap`(구 이름도 받음).
- `scripts/rebuild_ego_pose.py` (신규), `README.md`, `docs/pipeline_overview.md`, `docs/labeling_handoff.md`.

### 검증

A-10 20–50 s를 잘라낸 bag(`--scene-dur 10`, scene 2개)으로:
- 구 변환기(HEAD) 출력을 `rebuild_ego_pose.py`로 고친 결과 = 새 변환기 출력: ego_pose 3,611개 최대 10 µm(µs 타임스탬프 반올림), 쿼터니언 2.4e-7, CAN bus pose 동일.
- 이 구간은 odom이 100 Hz로 갱신되던 구간이라 구/신 차이는 타이밍 효과(수평 중앙값 1.2 cm, 최대 11 cm, 회전 최대 0.31°)뿐이고 높이 차 1 mm.
- 구 데이터셋에 새 변환기로 이어쓰기 → 거부. `--in-place` 재작성(하드링크 사본, 원본 불변) 후 이어쓰기 → scene 4개, devkit·`NuScenesCanBus` 로드.

## 2026-09-23 — 두 도구(표준형 / 전체 데이터형), 완전한 카메라 세트 규칙, rectify, 공식 scene 이름, CAN bus

2026-09-23 실측 bag(`0923_Calib_sample`의 A-8·A-9·A-10)으로 파서를 점검한 결과를 반영했다.

### 요약

| 항목 | 전 | 후 |
|---|---|---|
| 도구 | `bag2nuscenes.py` 하나 | 표준형 `bag2nuscenes.py` + 전체 데이터형 `bag2nuscenes_full.py` (공통 `converter.py`) |
| 카메라 매핑 | camera_4 = `CAM_FRONT`, camera_3 = `CAM_TRAFFIC` | **camera_3 = `CAM_FRONT`, camera_4 = `CAM_TRAFFIC`** (0923 bag 이미지로 확인) |
| 카메라 선택 | keyframe에서만, 카메라별 최근접 25 ms | 라이다 프레임마다 **6대가 모두 찍힌 한 시점**이 25 ms 안에 있어야 함 |
| 동기 실패 시 | keyframe만 빠지고 scene은 그대로 (A-10: scene 안에 최대 1.5 s 구멍 8개) | 그 자리에서 **시퀀스가 끊김** |
| 라이다 누락 | 무시 (keyframe을 인덱스로 뽑아 간격이 밀림) | 시퀀스 끊김 |
| scene | 첫 keyframe부터 20 s 창, 마지막 90 % 미만 버림 | 끊기지 않은 구간을 **정확히 20 s**로 자르고 나머지는 버림 (sample 40개) |
| scene 이름 | `scene-0001`부터 순번 | **공식 nuScenes train/val 목록**에서 `--split`에 따라 배정 |
| 이미지 | 원본 JPEG (어안 왜곡 그대로, K만 기록) | **pinhole로 rectify**, 새 K 기록 |
| GNSS/INS | 없음 | `can_bus/<scene>_pose.json`, `_ms_imu.json`, `_meta.json` |
| 추가 센서 | CAM_TRAFFIC(placeholder calib) | 전체 데이터형: CAM_TRAFFIC, 하단 라이다 4개, ARS548 레이더, 포인트별 시각, 나머지 토픽 전부 |

### 점검에서 확인한 사실 (2026-09-23 bag)

- 카메라: camera_3이 전방 도로, camera_4는 위로 기울어진 신호등 카메라. 기존 매핑은 이 bag에서 신호등 카메라를 `CAM_FRONT`로 넣고 있었다.
- `camera_info`는 placeholder(K = 960/540, D = 0). 캘리브 소스로 쓸 수 없다.
- 라이다 header stamp = **스윕 시작**(포인트 `timestamp`가 header+0~100 ms). 중앙 라이다는 0°(전방)에서 시작해 시계 방향. README 한계 4의 "sweep 끝"은 이 bag에 해당하지 않았다.
- 카메라 7대는 같은 시점 stamp가 0.04 ms 이내(트리거 공유, 비트 동일은 아님). 주기 33.448 ms(29.898 Hz)로 자유 동작해 10 Hz 라이다 대비 위상이 약 9.8초마다 한 주기씩 흐른다.
- 라이다 5개는 bag 3개 모두 드랍 0. 카메라 드랍은 대부분 1–2 프레임, camera_5(`CAM_BACK_LEFT`)가 가장 많다(A-10 delivery 97.8 %).
- PTP 정상(`/diag/ptp` 오프셋 ≤ 2 µs), INSPVA 전 구간 SOLUTION_GOOD. ARS548 자체 시계는 미동기(`timestamp_syncstatus` 2).
- CORRIMU는 IMU 샘플(125 Hz)을 `imu_data_count`개 누적한 값, NovAtel 차량 축(x 우, y 전, z 상). `/bsw/vehicle_can` yaw rate, odom 자세 변화율과 대조해 스케일·부호 확인(세 축 상관 0.95, 기울기 1.00/1.01/1.08).
- nuscenes-devkit: `render_scene*`·lidarseg 렌더러·mmdet3d는 표준 6채널 이름을 하드코딩, `ref_chan='LIDAR_TOP'` 고정, 카메라는 pinhole만. `NuScenesCanBus`는 scene 이름 `scene-NNNN`과 고정 메시지 이름만 받고 161–176·309–314번 scene을 거부. 레이더 PCD 파서는 마지막 필드 뒤에 1바이트가 더 있어야 하고(`end_p < len`), 빈 프레임은 NaN 포인트 1개로 표현한다.

### 변환기

- **`converter.py` (신규).** 두 CLI가 공유하는 파이프라인. `Profile`이 싣는 데이터(카메라 채널, 추가 라이다, 레이더, 포인트별 시각, sidecar)를 정한다. 기존 `bag2nuscenes.py`의 PointCloud2 변환과 `_StagingWriter`를 옮기고 일반화했다(작업을 callable로도 받아 워커에서 rectify).
- **`bag2nuscenes.py` / `bag2nuscenes_full.py`.** 각각 `run(STANDARD)` / `run(FULL)`을 부르는 얇은 CLI. `--split {train,val}` 추가, `--include-traffic-cam` 제거(표준형은 항상 제외, 전체형은 항상 포함), `--rectify-balance`, `--jpeg-quality`(기본 90), `--camera-group-ms`, `--workers`.
- **`nuscenes_writer.py`.** `sync_keyframes`를 `camera_instants` + `plan_frames`로 교체(완전한 카메라 세트, 끊김 이유 집계), `partition_scenes`는 segment를 정확한 길이로 자름, `official_scene_names` 추가, `build_tables`는 채널 수·모달리티(lidar/camera/radar)에 무관하게 동작하고 gating이 아닌 채널은 best-effort로 붙인다. `SensorData`는 채널별 `frames`/`modality`/`intrinsic`으로 바뀌었다. prev/next 연결은 레코드 참조로.
- **`rectify.py` (신규).** fisheye(4계수)·plumb_bob(5계수) → pinhole, 크기 유지. 왜곡이 0이면 바이트 그대로 통과.
- **`canbus.py` (신규).** odom + CORRIMU + INSPVA → `pose`(100 Hz; pos/orientation은 ego_pose와 동일, vel/accel/rotation_rate는 ego 프레임, lat/lon/height/ins_status 추가 키), `ms_imu`(100 Hz), `meta`. 가속도는 중력 제거 상태(nuScenes `ms_imu`와 다름, meta에 명시).
- **전체 데이터형**: `LIDAR_BOTTOM_FRONT/REAR/LEFT/RIGHT`(.pcd.bin), 모든 라이다 파일 옆 `<token>.time.bin`(float32, 프레임 시각 기준 초), `RADAR_FRONT`(nuScenes 18필드 radar .pcd, 반경 속도를 시선 방향으로 분해, odom·CORRIMU로 ego-motion 보정), 나머지 토픽은 `ext/<scene>/<topic>.json`(scene ± 0.5 s, 스트리밍으로 분할).
- **`common.py`.** 카메라 매핑 수정, 추가 라이다·레이더·CORRIMU 토픽, `load_calib(calib_dir, channels)`가 임의 채널(카메라 / `r.txt`·`t.txt` 포인트 센서)을 읽고 `missing_calib` 추가. 인자 없이 부르면 예전처럼 6캠 + LIDAR_TOP.
- **여러 bag 한 번에.** 위치 인자로 `.bag` 파일과 디렉토리를 여러 개 받는다(디렉토리는 아래의 `*.bag` 전부, 이름순). 차례로 같은 데이터셋에 변환하고, 이미 들어간 bag(같은 파일명)은 건너뛰며, 실패한 bag은 보고하고 나머지를 계속한 뒤 요약을 출력한다(실패가 있으면 종료 코드 1). devkit 검증은 마지막에 한 번.
- **append 안전장치.** 같은 출력 루트에 변환 두 개가 동시에 돌면 staging을 서로 지우고 테이블을 덮어쓰므로, `<out>/.convert.lock`으로 한 번에 하나만 돌게 했다(기존 테이블 읽기도 잠금 뒤로). `write_tables`는 13개 테이블을 임시 파일에 다 쓴 뒤 한꺼번에 교체한다 — 중간에 멈춰도 이전 import가 보존된다.
- **캘리브 없이도 실행.** `--calib`는 선택. 폴더에 없는 채널(생략하면 전부)은 기본값: extrinsic 항등(원점, ego 축과 일치), 카메라는 90° pinhole K(`f = width/2`)에 왜곡 0이라 rectify 없이 원본 JPEG을 그대로 쓴다(`common.default_calib`, `resolve_calib`). 기본값을 쓴 채널은 실행 로그와 `<log>.import.json`의 `calib_defaulted`에 남는다. 전체형의 `--skip-uncalibrated`는 없앴다.
- `pyproject.toml`: nuscenes-devkit이 기본 의존성으로(scene 이름에 공식 split 목록 사용). `verify` extra 제거. opencv-python `<5`(4.13에서 검증; 5.x는 calib3d 재편). 빌드 요구 setuptools `>=64`(editable 설치, PEP 660). Ubuntu 22.04 기본 pip 22는 이 pyproject를 설치하지 못하므로 README에 `pip install -U pip` 추가.

### 진단 스크립트

- `scripts/screen_bags.py`: keyframe 수용률 대신 **scene 수율** — 허용오차별로 쓸 수 있는 라이다 프레임, 끊김 수, 20 s scene에 들어가는 초(변환기의 `plan_frames`·`partition_scenes` 그대로). `--scene-dur`, `--camera-group-ms`, `--min-scene-frac`(기본 0.6) 추가, `--keyframe-stride` 제거.
- `scripts/qa_report.py`: sync 섹션이 변환기 규칙(`plan_frames`)으로 판정.
- `scripts/extract_cam_viz.py`: 자체 매핑 표를 지우고 `common.py`의 매핑을 쓴다. 타일에 채널과 토픽을 같이 표시.

### 문서

README 재작성(두 도구, 프레임 선택, scene 이름, rectify, CAN bus, 전체형 형식, devkit 사용법("Using the dataset": 로드·순회·포인트클라우드·렌더링·CAN bus·split·전체형 추가 데이터, 안 되는 것), 캘리브 레이아웃, 한계 목록 갱신 — 기존 1(미보정)·7(placeholder calib) 해소, 라이다 stamp·taxonomy/평가 불일치·공식 이름 용량·레이더 시각 추가), `docs/pipeline_overview.md`, `docs/labeling_handoff.md`(벤더에는 표준형: 6캠 pinhole, 레이더 없음), `docs/sync_reference.md`.

### 검증

- 실측 bag 3개의 header stamp로 `plan_frames`: 선택된 세트는 모두 완전하고 25 ms 이내(최대 24.7 ms), 카메라 프레임 중복 사용 0.

  | bag | 길이 | 끊김 | 20 s scene |
  |---|---|---|---|
  | A-8 | 344 s | 5 | 14 (280 s) |
  | A-9 | 285 s | 9 | 12 (240 s) |
  | A-10 | 317 s | 44 | 8 (160 s) |

  `screen_bags.py` 출력과 변환기 로그가 같은 수치.
- 가짜 캘리브(어안 5 + plumb_bob 1, 모든 센서)로 A-9 0–56 s: 표준형·전체형 모두 devkit 로드, `render_sample`, `render_pointcloud_in_image`, `LidarPointCloud.from_file_multisweep`(LIDAR_TOP, 하단 라이다 ref LIDAR_TOP), `RadarPointCloud.from_file`·multisweep·`render_sample_data`, `NuScenesCanBus.get_messages`·`print_message_stats`, `load_gt(nusc, 'train', DetectionBox)`(80 sample 전부 인식) 통과. scene당 sample 40개, 간격 500 ms, 카메라 29.9 Hz·라이다 10 Hz·레이더 20 Hz. 레이더 보정: |v| 중앙값 7.46 m/s → |v_comp| 0.03 m/s.
- A-9 전체(285 s) 표준형: 2분 11초, 최대 RSS 758 MB, 12 scene, 26 GB. 이어서 A-8 일부를 `--split val`로 append: val 이름(`scene-0003`, `scene-0012`) 배정, 센서 토큰 재사용, 같은 bag 재import 거부.
- **실제 캘리브로 변환한 데이터는 없다.** rectify·투영의 기하 정확도는 캘리브가 나온 뒤 `scripts/lidar2cam_projection.py`와 devkit 투영을 비교해 확인해야 한다.

### 기존 데이터셋에 대한 영향

모든 것이 바뀐다(카메라 매핑, 이미지, scene 경계와 이름, sweep 구성). 기존 데이터셋에 append하지 말고 새 루트에 다시 변환할 것.

### 하지 않은 것

차량 CAN(`/bsw/vehicle_can` 등)을 nuScenes CAN bus 메시지(`steeranglefeedback`, `vehicle_monitor`, `zoe_veh_info`)로 옮기는 것 — 단위·의미가 Renault Zoe 기준이라 전체형의 `ext/`에 원본 필드로 두었다. 라벨 taxonomy → devkit detection 평가 이름 매핑. 라이다 포인트 motion compensation(`time.bin`으로 가능). 2026-09 이전 bag의 카메라 매핑 재확인.

## 2026-08-27 — 성능, coverage window, nuScenes 관례, 진단 도구

커밋 `5d9adb7`(변환기·문서), `14fe46d`(진단 스크립트). 9개 파일, +680 / −282.

### 요약

| 항목 | 전 | 후 |
|---|---|---|
| `read_bag` (61 s bag, SSD, warm cache) | 13.6 s | **8.1 s** |
| PointCloud2 → `.pcd.bin` 한 프레임 | 12.2 ms | **5.5 ms** (출력 바이트 동일) |
| `interp_pose` 호출당 (odom 12만 개) | 77.6 ms | **1.1 ms** — 20분 bag 기준 39 s → 0.2 s |
| odom이 카메라보다 늦게 시작하는 bag | pose가 첫 odom 값으로 얼어붙은 프레임이 조용히 저장됨 (테스트에서 133개) | coverage window 밖 프레임은 keyframe 후보에서 제외, 얼어붙은 pose **0개** |
| `visibility.json` 토큰 | uuid | nuScenes와 같은 `"1"`–`"4"` |
| sweep의 `sample_token` | 시간상 가장 가까운 sample | nuScenes와 같이 **뒤에 오는** keyframe |
| `screen_bags.py` | 카메라 delivery만으로 PASS/DROP | 스트림 전체·coverage window·odom gap·INS 상태·sync 수용률, 이유가 붙은 판정 |

HDD에서 bag을 읽으며 같은 HDD에 쓰는 경우는 여전히 I/O 병목이다(같은 bag 181 s, CPU 12 %). `--out`을 다른 디스크(SSD)에 두는 것이 코드 개선보다 효과가 크다.

### 변환기

#### `nuscenes_writer.py`

- **Coverage window.** `required_streams()`, `coverage_window()`, `format_coverage()` 추가. 라이다·표준 카메라 6개·odom의 `[첫 스탬프, 마지막 스탬프]` 교집합을 `--sync-ms`만큼 안쪽으로 줄인 구간. `sync_keyframes(…, window=)`가 keyframe 후보를 이 구간으로 제한한다. scene과 sweep은 keyframe 사이에만 놓이므로 모든 프레임에 이미지와 pose가 있음이 구조적으로 보장된다. 배경: 센서마다 녹화 시작·종료 시점이 다르고(2026-08-19 bag에서 CAM_FRONT_LEFT가 1.411 s 늦게 시작), 카메라는 sync 조건이 간접적으로 걸러 주었지만 odom은 어떤 조건에도 없었다.
- **`interp_pose`.** odom 범위 밖 쿼리를 첫/마지막 값으로 clip하던 것을 `ValueError`로 바꿈. coverage window 덕에 도달 불가능해야 하며, 도달하면 로직 버그다.
- **Slerp 캐시.** `SensorData._slerp`에 보간기를 한 번만 만들어 재사용. 이전에는 (scene × 채널)마다 odom 전체를 재전처리했다.
- **sweep 소속.** `assign_to_nearest_sample` → `assign_to_following_sample`. sweep은 자기 뒤에 오는 첫 keyframe의 sample에 붙고, keyframe 프레임(카메라 포함)은 `sync_keyframes`가 매칭한 sample에 붙는다(카메라 keyframe이 라이다보다 몇 ms 뒤에 찍혀도 다음 sample로 넘어가지 않도록). v1.0-mini 실측: 모든 모달리티의 sweep이 100 % 뒤 keyframe에 붙고, keyframe은 −1…+48 ms 범위에서 자기 sample에 붙는다.
- **visibility 토큰.** 4개 행의 토큰을 `"1"`,`"2"`,`"3"`,`"4"`로. nuScenes에서 유일하게 uuid를 쓰지 않는 표이고, mmdetection3d 등이 이 문자열로 필터한다. 벤더가 라벨에 적는 값이므로 **벤더 전달 전에** 반영돼 있어야 한다.
- 모듈 docstring의 "category/attribute/visibility는 placeholder 하나씩"이라는 옛 설명 제거.

#### `bag2nuscenes.py`

- **PointCloud2 변환.** `pointcloud2_to_xyzir` + `_write_lidar_frame`(6 MB 메시지를 6번 복사, `all(axis=1)` 행 reduce)을 `_pointcloud2_dtype`(필드 offset 기반 dtype, 레이아웃별 캐시) + `pointcloud2_to_pcdbin`(zero-copy view, x·y·z·intensity finite 필터, 살릴 포인트만 컬럼당 1회 복사)으로 교체. ring은 정수라 항상 finite이므로 필터 결과는 이전의 5컬럼 검사와 동일하다(실제 60프레임 + 전체 bag 비교로 바이트 동일 확인). 패킷 경로용 `lidar_points_to_pcdbin`은 구 로직 그대로.
- **`_StagingWriter`.** staging 파일 쓰기를 스레드 4개로 넘겨 deserialize·numpy와 겹친다. 대기 64개 상한(메모리 역압), 완료된 쓰기를 submit마다 회수해 첫 실패를 메인 스레드에서 재전파, `with` 종료 시 전부 대기. 라이다는 변환(살아남은 포인트 수 반환)은 메인에서, 쓰기만 워커에서.
- `read_bag`이 `odom_max_gap_ms`를 통계에 넣고 요약 줄에 찍는다(0.5 s 초과 시 `[!]`). `main`이 coverage window를 출력하고 `<log>.import.json`에 `coverage_window`로 기록하며, 필수 스트림이 겹치지 않으면 명확한 메시지로 중단한다.

### 진단 스크립트

- **`scripts/screen_bags.py` 재작성.** 도착 시각 대신 header.stamp(변환기가 sync하는 시각)를 쓰고, 카메라·라이다·odom에 INSPVA 상태까지 읽는다(deserialize는 view라 비용 없음). bag마다: 스트림별 delivery/gap/시작·끝 오프셋 표, coverage window와 잘릴 길이, 0.5 s 넘는 odom gap 목록, INSPVA `SOLUTION_GOOD` 비율과 비정상 구간, header.stamp가 벽시계가 아닌 스트림(off-clock) 감지, `--sync-ms`별 keyframe 생존율(변환기의 `nearest_ts`·6캠 게이팅 규칙 그대로). 판정 PASS/MARGINAL/DROP에 이유를 전부 나열. 임계값 플래그: `--min-delivery`, `--max-odom-gap`, `--max-lidar-gap`, `--max-ins-bad-run`, `--max-ins-bad-frac`, `--max-cut`, `--sync-ms`, `--keyframe-stride`.
- **`scripts/sync_stats.py` 삭제.** 기능은 `screen_bags`의 keyframe acceptance 섹션으로. 이전 구현은 CAM_TRAFFIC까지 7캠을 게이팅해 변환기(6캠)와 규칙이 달랐다.
- **`scripts/qa_report.py`.** sync 섹션이 자체 nearest-neighbour 대신 `nuscenes_writer.nearest_ts`를 쓰고 표준 6캠만 게이팅. CAM_TRAFFIC은 "best-effort, not gating"으로 표시.
- `notebooks/dataset_validation.ipynb` 주석 한 줄(`sync_stats.py` → `screen_bags.py`).

### 문서

- `README.md`: Convert 섹션에 coverage window 문단; screen_bags 설명과 진단 표 갱신; limitation 6 정정; **limitation 8** (`ego_pose`가 UTM 52N 절대좌표 — float32로 캐스팅하면 0.25 m 격자) 및 **9** (INS 상태를 변환기가 검사하지 않음) 추가; visibility 토큰 명시.
- `docs/pipeline_overview.md`: "Coverage window" 설계 노트.
- `docs/labeling_handoff.md`: 좌표계가 UTM이라 float64 유지 필요, `visibility_token`은 `"1"`–`"4"`.

### 검증

- 성능 패치(Slerp·zero-copy·스레드풀): 토큰 생성을 카운터로 고정해 같은 bag을 패치 전/후로 변환, 페이로드 12,852개 sha256과 13개 JSON 전부 동일. 구/신 코드를 번갈아 3회씩 실행해 타이밍 측정.
- coverage window: odom이 bag 전체를 덮는 경우 출력 동일(`import.json`의 새 키 2개만 추가). odom을 3 s 늦게 시작·2 s 먼저 끝나게 한 경우 구 코드는 얼어붙은 ego pose 133개, 신 코드는 0개.
- nuScenes 포맷: 공식 v1.0-mini와 13개 테이블의 키·타입 대조(populated 테이블 전부 동일), 참조 무결성·체인·단위·파일 레이아웃 검사, devkit API(`get_sample_data`, `map_pointcloud_to_image`, `render_sample_data`, `render_pointcloud_in_image`, `render_sample`, `render_egoposes_on_map`, `list_*`) 양쪽에 동일 호출.
- calib 컨벤션 함수는 합성 배치로 검증(카메라 광축 → ego +x, 10 m 앞 점 → `[0, 0, 10]`). **실제 calib으로 변환한 데이터셋은 없어서 실 calib 값의 검증은 미완.**

### 기존 데이터셋에 대한 영향

이 커밋 이전에 변환한 데이터셋과 비교해 달라지는 것: `visibility.json` 토큰, sweep의 `sample_token`, bag 앞뒤 coverage window 밖 프레임(일반적으로 앞 0.2~1.5 s), `<log>.import.json`의 `coverage_window`·`odom_max_gap_ms` 키. 벤더에 넘기기 전이라면 다시 변환하는 것이 가장 간단하다. append 모드는 기존 `visibility.json`이 있으면 그대로 두므로, uuid 토큰인 기존 데이터셋에 새 bag을 append하면 uuid가 유지된다.

### 손대지 않은 것 (논의만)

scene 이름이 공식 split 목록과 충돌(`scene-0001…`; mmdet3d가 이 목록으로 train/val을 가르므로 자체 이름 체계 권장), 카테고리·속성 이름이 공식 detection eval 목록과 다름(22개 중 5개만 매핑), `rslidar_packets` 경로와 `packet_decoder/` 제거, append 모드의 O(N²) 재작성·재검증, 리더 스레드 분리.
