# Changelog

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
