# curation

nuScenes 형식으로 변환한 tcar 주행 데이터를 펼쳐 보고, 규칙이 제안한 삭제 scene을 사람이 눈으로 확인해 결정하는 도구입니다.
규칙은 제안만 하고, 최종 결정은 사람이 합니다.

씬의 상태는 셋입니다.
- **후보**: 필터(객체 부족, 정지 중복)에 걸렸고 아직 사람이 확정하지 않은 씬. 어느 필터에도 걸리지 않은 씬은 후보가 아니라 **필터 통과**입니다.
- **확정**: 검토에서 남기기 / 버리기를 누른 씬. 누를 때마다 `selection/human_decisions.json`과 `selection/keep.txt` · `drop.txt`에 저장됩니다(다시 누르면 후보로).
- **삭제 완료**: 최종 삭제로 데이터셋에서 뺀 씬(`_removed/<시각>/`, `selection/deleted.txt`). 최종 삭제는 **버리기 확정 씬만** 뺍니다.

데스크톱 GUI(`gui.py`)의 **큐레이션** 탭이 데이터셋 전체의 확정 · 후보 · 삭제 완료 비율과 GNSS 경로 지도(OSM)를 보여 주고,
코스별·루트별로 아래 도구를 실행합니다. 필터 카드에 필터마다 기준 · 세는 법 · 이유 · 근거 그래프가 있고, 경로 겹침(같은 길 · 같은 방향으로 지난 씬, 참고 정보)은 지도의 ‘겹침’ 보기로 봅니다. 이미 검출된 씬은 바로 검토할 수 있고, 검토 뷰어도 GUI 창 안에 뜹니다. 최종 삭제는 그 탭에서만 합니다.
앱은 확정 목록(`human_decisions.json`, `keep.txt`, `drop.txt`, `deleted.txt`)을 이 폴더(`curation/`)에도 복사합니다(`$TCAR_DECISIONS_DIR`,
앱 설정 `decisions_dir`). 도구가 읽는 원본은 데이터셋의 `selection/`이고, 이 폴더의 파일은 결정할 때마다 덮어쓰는 사본입니다.
범위를 나누는 옵션: `scene_detect.py --logs`, `apply_selection.py --logs`, 뷰어 주소의 `?logs=<log>,<log>`.
명령줄로 쓸 때는 `--dataroot`를 생략하면 `$TCAR_DATA_ROOT/parsed/tcar_nuscenes`(기본 `/data/parsed/tcar_nuscenes`)를 씁니다.
(폴더가 데이터셋 안에 있으면(`<dataroot>/curation/`) 그 상위 폴더를 씁니다.)

이 폴더가 정본입니다. 2026-09-30에 별도 저장소 `tcar-scene-review`(`tools/`)에서 옮겨 왔고, 그 전 이력은 그 저장소에 있습니다(그 저장소는 더 이상 맞추지 않습니다).

## 흐름

0. **객체 검출** `python curation/scene_detect.py` (GPU, 148 scene 약 15분)
   모든 key frame(카메라 6대)에서 YOLO26l(COCO)로 도로 이용자 6종(person, bicycle, motorcycle, car, bus, truck)만 검출합니다.
   박스 원본은 `selection/detections_yolo26l_1280.npz`에 scene token 기준으로 캐시되고, scene별 개수는 `selection/object_counts.json`에 씁니다.
   이미 검출한 scene은 건너뛰므로, 데이터셋에 녹화를 이어 붙인 뒤에는 새 scene만 검출합니다.
   자차 차체(카메라마다 key frame 50% 이상에서 같은 위치에 나오는 박스)와 이륜차 탑승자는 세지 않습니다. 기준만 바꿔 다시 셀 때는 `--report`를 씁니다.
   torch · ultralytics가 있는 Python이 필요합니다(GUI는 conda 환경 등에서 찾아 기억합니다. 이 PC는 `~/anaconda3/envs/eufs`).
1. **필터 → 후보** `python curation/curate.py` — 세 필터를 늘 이 순서로, 앞에서 정한 씬은 뒤에서 다시 보지 않습니다
   - ① 정지 중복: 같은 루트에서 같은 장소에 멈춘 scene들(시간의 80% 이상 1 km/h 미만, 경로가 서로 80% 이상 10 m·30° 안에서 겹침, 속도 흐름 비슷)은 LiDAR diff(0.5초 간격 프레임 사이 바뀐 칸 수)가 가장 큰 것만 남기고 나머지를 후보로. 그룹에 든 scene은 **남긴 대표까지** ②③으로 넘기지 않습니다(대표가 뒤에서 걸려 그룹이 통째로 빠지지 않게). 짝이 없는 정지 scene은 ②로 갑니다.
   - ② 객체 부족: 가까운 도로 이용자(박스 높이 50px 이상, 카메라 6대 합)가 샘플당 평균 `--min-objects`(기본 3)개 미만. `object_counts.json`이 없으면 아무것도 걸지 않습니다.
   - ③ 경로 겹침: ①②가 정하지 않은 달리는 scene을 모든 루트에 걸쳐 가까운 도로 이용자가 많은 순으로 보며, 먼저 남긴 scene이 이 scene 경로의 `--min-overlap`(기본 60%) 이상을 같은 방향(10 m · 30° 안)으로 지나갔으면 후보.
   - 건너뛰기: `--skip stops,objects,overlap` 또는 `selection/filters.json`의 `"skip"`(앱의 필터 카드 "건너뛰기"가 씀). 건너뛴 필터는 후보를 만들지 않고, 그 씬들은 다음 필터로 갑니다.
   결과: `selection/result.json`(scene마다 `kind` = 걸린 필터, `step`, `substitute` = 남긴 쪽, `path_cover`), `selection/final_selection.json`(참고용)
2. **검토 → 확정** `python curation/sample_viewer.py` → http://localhost:8765 (GUI에서는 창 안, `/review`)
   - 왼쪽 목록에 모든 scene과 상태 표시(탭: 전체 / 후보 / 확정 / 객체 적음, 후보에는 걸린 필터 이름). "다음 후보"(N)와 "건너뛰기"(S, 확정하지 않고 넘김 · 나머지를 다 본 뒤 다시)는 필터 순서대로(① → ② → ③). 경로 겹침은 지도에 먼저 남긴 scene(주황 점선)과 이 scene(파랑)을 옆으로 띄워 그림, 나란히 비교에도 지도 칸. 위의 카드가 후보의 이유를 보여 주고, 남기기 확정(K) · 버리기 확정(D) · 다음 후보(N)
   - 카메라 6대, LiDAR 3D(마우스 회전), 경로 지도(OSM, 자차 위치·방향), 실시간 재생
   - 2D 박스(상단 토글 또는 B 키): 규칙이 센 박스를 카메라 위에 그립니다. 얇은 선은 멀어서 세지 않은 박스, 회색 점선은 제외한 박스(자차 차체, 탑승자)입니다. key frame에만 있어서 멈춰 있을 때 보입니다
   - 나란히 비교: 정지 중복 그룹을 함께 재생(카메라 / LiDAR), scene마다 확정, "후보대로 확정 → 다음"(A)
   - 확정: `selection/human_decisions.json`, 목록 `selection/keep.txt` · `drop.txt`(scene · log · token · 시각), 기록 `selection/decisions_log.txt`
   - 시작: 표에서 뽑은 색인을 로컬 디스크(`~/.cache/tcar-parser/viewer/`)에 캐시합니다. 표가 그대로면 0.5초, 바뀐 뒤 첫 실행만 표를 다시 읽습니다(334 scene, 파일이 메모리에 있으면 13초, 바쁜 HDD에서는 1분 넘게). 검출·규칙 도구도 같은 캐시를 씁니다
   - 재생: 절반 크기 카메라 사본과 LiDAR 표시용 데이터를 `~/.cache/tcar-parser/preview/`에 쌓습니다(`TCAR_PREVIEW_CACHE_GB`, 기본 80 GB). 재생용 프레임은 초당 5장(`TCAR_PLAY_HZ`, key frame은 모두, 0 = 전부). 고른 scene부터 목록 순서대로 한 scene씩 통째로 미리 읽고, 재생은 그 scene이 전부 캐시된 뒤 시작합니다(그동안 준비 진행률). 전체를 미리 채우려면 `--warm`, 요청이 없을 때 뒤에서 채우려면 `--fill`(앱이 켬: 후보와 그 대표 씬 먼저)
3. **최종 삭제** GUI 큐레이션 탭의 "최종 삭제" 또는 `python curation/apply_selection.py --apply`
   - **버리기 확정 scene만** 파일·표 기록을 `_removed/<시각>/`로 옮기고(지우지 않음), 남은 scene 이름을 빈 번호 없이 당깁니다. 후보는 남습니다
   - 뺀 scene은 `selection/deleted.txt`, 남은 scene의 확정은 새 이름으로 따라가고 `keep.txt` · `drop.txt`도 다시 씁니다
   - 변환기와 같은 잠금(`<dataroot>/.convert.lock`)을 잡습니다. 파싱 중에는 기다렸다가 하세요
   - devkit이 깨지지 않게: 지우기 전에 `--simulate`가 삭제를 표에만 적용해 모든 참조(scene/sample/sample_data/ego_pose/calibrated_sensor, prev·next, log/map, 이름 split, can_bus 이름)를 검사하고 devkit으로 불러 봅니다(문제가 있으면 지우지 않음). 지운 뒤 `--check`가 지금 데이터셋을 같은 검사 + key frame 파일 + devkit NuScenes · NuScenesCanBus로 확인합니다. 앱의 최종 확정은 이 둘을 앞뒤로 자동 실행합니다
   - scene이 모두 빠진 log는 `log.json`에서 빠지지만 `<log>.import.json`은 남아, 변환기가 그 bag을 다시 넣지 않습니다
   - 되돌리기: `python curation/apply_selection.py --undo <dataroot>/_removed/<시각>` (GUI: "마지막 최종 삭제 되돌리기…")

## 라운드와 잠금

- **최종 확정**(앱의 큐레이션 탭, 또는 `python curation/apply_selection.py --finalize`)이 한 라운드를 끝냅니다: 버리기 확정 scene을 빼고, 남은 scene을 모두 `selection/locked.json`에 잠급니다(token 기준).
- 잠긴 scene은 **절대 삭제되지 않고** 다시 후보가 되지 않습니다. 새 녹화를 파싱해 데이터셋에 이어 붙이면 새 scene만 필터·검토·삭제 대상입니다. ③ 경로 겹침에서는 잠긴 scene이 먼저 남긴 쪽이 되어, 새 scene이 기존 길과 겹치면 새 scene이 후보가 됩니다.
- 확정하지 않은 후보가 남아 있으면 최종 확정할 수 없습니다. 되돌리기: 그 라운드의 백업으로 `--undo`(삭제가 없던 라운드는 `--unlock-last`), 앱에서는 "마지막 최종 확정 되돌리기".

## 데이터 위치

```
<dataroot>/
  v1.0-trainval/   samples/   sweeps/   can_bus/   maps/   *.import.json
  selection/       # 규칙 결과, 사람 결정, 캐시, 검출 가중치(models/)
  _removed/        # 삭제 확정 백업
```

## 파일

| 파일 | 역할 |
|---|---|
| `sample_viewer.py` / `.html` | 검토 뷰어 서버와 화면 (`/full`: 이전의 자세한 뷰어) |
| `scene_detect.py` | 2D 객체 검출(YOLO, COCO 도로 이용자 6종)과 scene별 개수 |
| `curate.py` | 삭제 제안 규칙 (`--mode stops` 기본: 객체 부족 + 정지 중복, `--mode quota` 이전 2단계 방식) |
| `apply_selection.py` | 최종 삭제(버리기 확정만) / 되돌리기, keep · drop · deleted.txt |
| `scene_select.py` | 장소·자차 상태 특징, LiDAR diff |
| `scene_risk.py` | 위험 이벤트 추정 (실제 주행 경로 위 TTC 등) |
| `scene_objects.py`, `stop_events.py`, `scene_embed.py` | 객체 수, 정차 중 사건, CLIP 임베딩 (quota 모드용) |

필요 패키지: numpy, opencv (LiDAR diff) — 파서와 같은 환경. 객체 검출만 torch · ultralytics(가중치는 처음 실행 때 `selection/models/`에 받음), quota 모드 분석은 torchvision · open_clip.
