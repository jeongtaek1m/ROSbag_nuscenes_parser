# tcar_nuscenes 파이프라인 설계 노트

ROS1 rosbag → NuScenes v1.0-trainval 변환기의 설계 배경.

---

## 1. 왜 단일 스테이지인가

초기 버전은 3단계였다: `decode_lidar.py`(패킷 디코딩) → `bag2raw.py`(카메라/odom 덤프)
→ `raw2nuscenes.py`(NuScenes 생성). 중간 산출물(intermediate)을 캐시해서
stage-2 파라미터를 바꿀 때 bag을 다시 안 읽으려는 구조였다.

**그 명분의 핵심은 "라이다 패킷 디코딩 40분"이었는데, 신규 차량 스택에서는
그 단계가 존재하지 않는다.** `/middle/rslidar_packets` 대신
`/middle/rslidar_points`(PointCloud2)를 그대로 발행하므로 디코딩이 필요 없고,
bag 전체를 읽는 데 몇 분이면 된다.

| bag | 라이다 토픽 | 디코딩 비용 |
|---|---|---|
| 2026-03-23 traffic (구) | `rslidar_packets` | ~40분 |
| 2026-08-11 / 08-19 (신) | `rslidar_points` | 없음 |

캐시할 만큼 비싼 단계가 사라지자 중간 포맷은 순수 비용만 남았다:

- 디스크 2배 (카메라 88 GB + 라이다 zstd 22 GB)
- 자체 포맷(parquet, `.bin.zst`, `meta*.json`) 학습 부담
- symlink/hardlink 수명 관리 문제 — 초기 symlink 구현에서는 intermediate를
  지우면 데이터셋의 카메라가 전멸했다
- 스크립트 3개, 실패 지점 3개

그래서 `bag2nuscenes.py` 하나로 합쳤다. bag을 한 번 읽어 센서 페이로드를
출력 루트 안의 staging에 쓰고, 동기화·scene 분할이 끝난 뒤 최종 경로로
**rename**한다. 같은 파일시스템이므로 메타데이터 연산이고, 바이트는 한 번만
쓰인다. staging은 실행이 끝나면 사라진다.

**트레이드오프**: sync tolerance나 scene 길이를 바꾸면 bag을 다시 읽어야 한다.
`rslidar_points` bag 기준으로 몇 분이라 감수할 만하고, 구 packets bag을 다시
튜닝해야 하면 `--keep-staging`으로 staging을 남길 수 있다.

## 2. 두 도구, 한 파이프라인

| | `bag2nuscenes.py` (표준형) | `bag2nuscenes_full.py` (전체 데이터형) |
|---|---|---|
| 목적 | nuScenes devkit·nuScenes 기반 학습 코드를 수정 없이 사용 | 사내 상세 작업용 |
| 센서 | LIDAR_TOP + 표준 카메라 6대 | + CAM_TRAFFIC, 하단 라이다 4개, 전방 레이더, 포인트별 시각 |
| 그 외 | GNSS/INS → CAN bus 확장 파일 | + 나머지 모든 토픽 → `ext/<scene>/<topic>.json` |

둘 다 `converter.run(profile)`을 부르는 얇은 CLI이고, 차이는 `Profile`(어떤 데이터를
싣는가)뿐이다. 프레임 선택·scene 분할·scene 이름이 같으므로, 같은 bag·같은 split이면
두 데이터셋의 scene은 같은 구간을 가리킨다.

```
bag2nuscenes.py / bag2nuscenes_full.py   CLI (Profile만 다름)
converter.py        bag 1회 읽기, staging(카메라 rectify 포함), materialize, sidecar
nuscenes_writer.py  프레임 선택, scene 분할, scene 이름, ego pose 보간, 13개 테이블
rectify.py          fisheye / plumb_bob -> pinhole
canbus.py           GNSS/INS -> nuScenes CAN bus 파일
common.py           토픽 매핑, 캘리브 로딩, 좌표 컨벤션
```

`nuscenes_writer.py`는 **rosbags를 import하지 않는다.** 타임스탬프와 캘리브
dict를 받아 테이블과 materialization plan을 돌려줄 뿐이다. 덕분에 프레임 선택·scene
알고리즘을 bag 없이 개발·테스트할 수 있고, `scripts/screen_bags.py`·`qa_report.py`가
같은 함수(`plan_frames`, `partition_scenes`)로 변환 결과를 미리 계산한다.

## 3. 설계 특징

### ROS 런타임 의존성 0

`rosbags`로 `.bag`을 직접 파싱한다. ROS Noetic 설치, catkin 빌드, roscore 모두
불필요. Robosense RSP128(Ruby 128ch) 패킷 디코더가 순수 Python으로 내장돼 있어
벤더 SDK도 필요 없다. RSM1, RSBP 디코더도 패키지에 있어 다른 모델로 확장 가능.

### 캘리브레이션 컨벤션 자동 변환

원본 calib 파일은 OpenCV 외참 (`P_cam = R · P_ego + t`), NuScenes 표준은
ego 프레임에서의 센서 pose. `common.opencv_ext_to_nuscenes_pose`가 변환한다.
검산: LIDAR_TOP이 (1.56, 0, 1.90), CAM_FRONT가 (2.04, −0.14, 1.73)에 광축 +x.

### 멀티 bag 증분 구축

같은 출력 경로에 다시 돌리면 자동 append — 공식 목록에서 다음 scene 이름을 이어서
배정, 센서/카테고리 토큰 재사용, log 레코드 추가, 같은 bag 중복 import 거부.

### 프레임 선택: 완전한 카메라 세트

LIDAR_TOP이 기준이다. 라이다 프레임마다, 표준 카메라 6대가 **모두** 찍힌 한 촬영
시점(카메라는 트리거를 공유해 같은 시점의 stamp가 0.04 ms 이내)을 찾되 그 프레임들이
전부 `--sync-ms`(25 ms) 안에 있어야 한다. 범위 안의 시점(최대 2개) 중 완전한 것 중
가장 가까운 것을 고른다. 카메라별로 따로 고르지 않는다.

이런 세트가 없는 라이다 프레임, 라이다 프레임 누락, coverage window 이탈은 시퀀스를
**끊는다.** 끊기지 않은 구간을 정확히 20초짜리 scene으로 자르고, 20초가 안 되는
나머지는 버린다. scene 안에서 sample은 라이다 5프레임마다(2 Hz, 40개), sweep은 첫
sample과 마지막 sample 사이의 모든 프레임(라이다 10 Hz, 카메라 30 fps 전부)이다.

카메라는 29.898 Hz로 자유 동작해서 10 Hz 라이다 대비 위상이 약 9.8초마다 한 주기씩
흐른다. 고정 offset이 아니라 프레임마다 세트를 고르는 이유다.

### Coverage window

센서마다 녹화 시작·종료 시점이 다르다(2026-08-19 bag에서 최대 1.4초). 라이다,
표준 카메라 6개, odom이 **모두** 살아 있는 교집합 구간을 `--sync-ms`만큼 안쪽으로
줄인 것이 coverage window이고, 그 밖의 라이다 프레임은 쓰지 않는다. scene과 sweep은
그 안에만 놓이므로 모든 프레임에 이미지와 pose가 있음이 구조적으로 보장되고,
`interp_pose`는 odom 범위 밖 쿼리를 clip하는 대신 에러로 취급한다. 중간에 끊기는
경우(odom gap, INS 상태 불량)는 변환기가 아니라 `scripts/screen_bags.py`가 잡는다.

### 공식 scene 이름

scene 이름은 nuScenes 공식 train/val 목록(`nuscenes.utils.splits`)에서 `--split`에
따라 순서대로 가져온다. 공식 목록으로 split을 나누는 도구(devkit 평가, mmdetection3d,
BEVFusion)가 수정 없이 동작한다. CAN bus API가 거부하는 번호(161–176, 309–314)는
건너뛰며, 용량은 train 685 / val 150 scene이다.

### 카메라 rectify

nuScenes는 3×3 `camera_intrinsic`만 있고 왜곡 계수가 없어 모든 소비자가 pinhole로
투영한다. 그래서 이미지를 캘리브로 undistort하고 새 K를 기록한다(extrinsic 불변).
staging 단계에서 스레드 풀이 decode→remap→encode를 처리한다(OpenCV가 GIL을 놓음).

### 7번째 카메라 (전체 데이터형)

신호등 전용 `CAM_TRAFFIC`(camera_4, 위로 기울어짐)은 전체 데이터형에만 들어간다.
프레임 선택을 막지 않는 best-effort 채널로, sample의 라이다 시각 25 ms 안에 프레임이
있으면 붙고 없으면 빠진다. 다른 추가 센서(하단 라이다, 레이더)도 같은 규칙(반 주기)이다.

### 자동 검증

실행 마지막에 `NuScenes(version, dataroot)`로 직접 로드하고, `NuScenesCanBus`로 한
scene의 `pose`를 읽는다. 13개 테이블, 외래키, `prev`/`next` 체인, ego_pose 참조,
CAN bus 파일 경로가 한 번에 검증된다.
`notebooks/dataset_validation.ipynb`에서 `render_pointcloud_in_image` 등
표준 devkit API가 동작하는 것도 확인할 수 있다.

## 4. 알려진 한계

README의 "Known limitations"를 참조. 요약하면 녹화 시 카메라 프레임 유실(25 ms 완전 세트
규칙에서 곧바로 scene 끊김으로 이어짐), 캘리브 품질에 좌우되는 rectify, 라이다 stamp가
스윕 시작인 점, 구 bag의 카메라 stamp 체계와 매핑, 어노테이션 미출력, 라벨 taxonomy와
devkit detection 평가 이름의 불일치, UTM 절대좌표, 공식 scene 이름의 용량, 레이더 시각.
