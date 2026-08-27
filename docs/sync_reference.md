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

- **Sync 방식**: software (rosbag receive time, header.stamp)
- **Tolerance 설정**: **25 ms**, 양방향 nearest neighbor
- **위치**: software-synced 데이터셋 중 빡센 편 (50 ms 절반)
- **결과**: keyframe 14.79% drop (1832/11826) — 주로 CAM_FRONT의 dropout 때문 (p99=36ms)
- **이론적 floor**: 카메라 30Hz → 16.7 ms. 25ms는 floor 위 ~8ms 마진. 더 빡세게 가긴 어려움.

## 참고

- nuScenes 원본 페이퍼는 "max 27ms time difference"라고 명시 — hardware-synced임에도 27ms 인정. 우리가 25ms로 가는 게 비합리적이지 않음.
- Detection eval에서 sync error는 보통 **bounding box localization error**보다 작아야 의미가 있음. 박스 1m 정확도 ≈ ego speed × sync error → 30 km/h × 25ms ≈ 0.21m. OK.
- 다른 측 비교 (e.g., kalibr, autoware의 sensor calibration paper)는 **time-shift 추정**으로 sync error 자체를 보정하는 방법도 있음. 우리 dataset에는 적용 안 함 (라벨링 인풋 단계).
