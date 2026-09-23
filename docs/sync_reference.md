# 자율주행 데이터셋 Sync Tolerance 비교

LiDAR ↔ 카메라 (또는 multi-sensor) 동기화 허용 오차. 우리 dataset의 25ms 기준이 어디 위치하는지 비교용 레퍼런스.

## 비교 표

| 데이터셋 | 링크 | Sync 방식 | Cam–LiDAR Tolerance | 출처 |
|---------|------|----------|--------------------|------|
| **Waymo Open Dataset** | [waymo.com/open](https://waymo.com/open/) | Hardware (PTP) | **< 1 ms** | [Sun et al., CVPR'20](https://arxiv.org/abs/1912.04838) |
| **ZOD (Zenseact)** | [zod.zenseact.com](https://zod.zenseact.com/) | Hardware (PTP) | **< 5 ms** | [ZOD whitepaper](https://arxiv.org/abs/2305.02008) |
| **nuScenes** | [nuscenes.org](https://www.nuscenes.org/) | Hardware (FPGA trigger) | **< 27 ms** (도큐 명시) | [nuScenes paper](https://arxiv.org/abs/1903.11027) §3.1 |
| **KITTI** | [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/) | Hardware (Velodyne 0° → cam 동기) | **~ 5–10 ms** | [Geiger IJRR'13](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf) |
| **PandaSet** (Hesai) | [scale.com/open-datasets/pandaset](https://scale.com/open-datasets/pandaset) | Hardware | **~ 10 ms** | [PandaSet docs](https://github.com/scaleapi/pandaset-devkit) |
| **Argoverse 2** | [argoverse.org/av2](https://www.argoverse.org/av2.html) | Hardware | **< 10 ms** | [Wilson et al., NeurIPS Datasets'21](https://arxiv.org/abs/2301.00493) |
| **ONCE (Huawei)** | [once-for-auto-driving.github.io](https://once-for-auto-driving.github.io/) | Hardware | **< 10 ms** | [Mao et al., NeurIPS Datasets'21](https://arxiv.org/abs/2106.11037) |
| **ApolloScape** | [apolloscape.auto](http://apolloscape.auto/) | Hardware | **< 10 ms** | [Huang et al., CVPR'18](https://arxiv.org/abs/1803.06184) |
| **A2D2 (Audi)** | [a2d2.audi](https://www.a2d2.audi/) | Software (host clock) | **~ 50 ms** | [Geyer et al., '20](https://arxiv.org/abs/2004.06320) |
| **Lyft Level 5** | [woven.toyota / level5](https://github.com/woven-planet/l5kit) | Software (ROS time) | **~ 50 ms** | dataset notes |
| **DDAD (TRI)** | [github.com/TRI-ML/DDAD](https://github.com/TRI-ML/DDAD) | Hardware | **< 10 ms** | TRI docs |
| **Cityscapes 3D** | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) | (2D 위주, Lidar 제한적) | n/a | — |

## 정리

- **Hardware-synced (PTP / electrical trigger)**: 1–10 ms 범위. 데이터셋 다수가 여기.
- **Software-synced (NTP / OS clock / ROS time)**: 25–50 ms 범위가 일반적.
- Hardware든 software든 **30Hz 카메라 + 10Hz lidar** 조합이라면 이론적 최소 sync error ≈ camera period / 2 ≈ **16.7 ms**. 그 아래로 내려가려면 hardware trigger 필수.

## 우리 dataset (`tcar_nuscenes`)에서

- **Sync 방식**: software (header.stamp, 2026-09 bag부터 PTP로 시스템 시계 동기). 카메라 7대는
  트리거를 공유해 같은 시점 stamp가 0.04 ms 이내, 라이다와는 위상 고정이 안 됨(카메라
  29.898 Hz → 약 9.8초마다 한 주기씩 위상이 흐름)
- **규칙**: 라이다 프레임마다 표준 카메라 6대가 **모두** 찍힌 한 시점이 25 ms 안에 있어야
  함. 없으면 그 자리에서 scene 시퀀스가 끊김 (`nuscenes_writer.plan_frames`)
- **선택된 세트의 오차** (2026-09-23 bag 3개): p50 8.4 ms, p99 16.6 ms, 최대 24.7 ms
- **이론적 floor**: 카메라 30Hz → 최근접 프레임까지 최대 16.7 ms. 25ms는 floor 위 ~8ms 마진이라,
  최근접 프레임이 드랍되면 위상에 따라 다음 프레임(33.4 ms − 위상)이 허용 범위 밖으로 나간다
- **scene 수율** (20초 scene, `scripts/screen_bags.py`): A-8 344 s 중 280 s, A-9 285 s 중 240 s,
  A-10 317 s 중 160 s. 30 ms면 A-8 320 s, 50 ms면 A-8 340 s / A-10 260 s

## 참고

- nuScenes 원본 페이퍼는 "max 27ms time difference"라고 명시 — hardware-synced임에도 27ms 인정. 우리가 25ms로 가는 게 비합리적이지 않음.
- Detection eval에서 sync error는 보통 **bounding box localization error**보다 작아야 의미가 있음. 박스 1m 정확도 ≈ ego speed × sync error → 30 km/h × 25ms ≈ 0.21m. OK.
- 다른 측 비교 (e.g., kalibr, autoware의 sensor calibration paper)는 **time-shift 추정**으로 sync error 자체를 보정하는 방법도 있음. 우리 dataset에는 적용 안 함 (라벨링 인풋 단계).
