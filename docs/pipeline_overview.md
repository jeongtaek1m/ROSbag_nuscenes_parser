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

## 2. 모듈 경계

```
bag2nuscenes.py     CLI, bag 읽기, staging, materialize
nuscenes_writer.py  sync, scene 분할, ego pose 보간, 13개 테이블
common.py           토픽 매핑, 캘리브 로딩, 좌표 컨벤션
```

`nuscenes_writer.py`는 **rosbags를 import하지 않는다.** 타임스탬프와 캘리브
dict를 받아 테이블과 materialization plan을 돌려줄 뿐이다. 덕분에 sync/scene
알고리즘을 bag 없이 개발·테스트할 수 있고, bag 포맷이 바뀌어도 영향을 안 받는다.

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

같은 출력 경로에 다시 돌리면 자동 append — scene 인덱스 연속, 센서/카테고리
토큰 재사용, log 레코드 추가, 같은 bag 중복 import 거부.

### 7번째 카메라

표준 6채널 외에 신호등 전용 `CAM_TRAFFIC`을 등록한다. nuscenes-devkit이 임의
채널 수를 받으므로 표준 6채널 파이프라인은 자동으로 무시한다. 다만 캘리브가
placeholder이므로 기하 용도로 쓰면 안 된다. `--no-include-traffic-cam`으로 제외.

### 자동 검증

실행 마지막에 `NuScenes(version, dataroot)`로 직접 로드한다. 13개 테이블,
외래키, `prev`/`next` 체인, ego_pose 참조가 한 번에 검증된다.
`notebooks/dataset_validation.ipynb`에서 `render_pointcloud_in_image` 등
표준 devkit API가 동작하는 것도 확인할 수 있다.

## 4. 알려진 한계

README의 "Known limitations"를 참조. 요약하면 undistortion 미구현, 브리지
best_effort QoS로 인한 카메라 프레임 유실, 카메라 타임스탬프 2가지 체계,
라이다 프레임 스탬프가 sweep 끝인 점, bag마다 다른 카메라 클럭 skew,
어노테이션 미출력, CAM_TRAFFIC placeholder 캘리브.
