# Changelog

## 2026-09-30 — 데스크톱 GUI(TCAR Parser), 기본 출력 `/data/parsed`

IRCV-DM-Ops `parser` 브랜치의 DM Parser GUI를 같은 디자인·기능으로 티카(ROS 1 bag)에 옮겼다.

### 요약

| 항목 | 내용 |
|---|---|
| `gui.py` (신규) | 데이터 폴더(기본 `/data`)의 `raw/<코스>/`, `parsed/tcar_nuscenes`, `logs/`를 관리하는 PyQt5 앱: 가져오기, 파싱, 검증, 수율 검사, 제외·복원 |
| 코스 | bag 이름의 알파벳(`A-1_…bag` → A). DM처럼 코스마다 데이터셋을 두지 않고 모든 코스가 데이터셋 하나로 — scene 이름이 겹치지 않게 |
| `yield_check.py` (신규) | 변환 없이 bag당 scene 수와 나머지가 잘린 이유. 청크 인덱스로 메시지마다 앞 수백 바이트만 읽음 |
| `remove_log.py` (신규) | 데이터셋에서 log 하나를 테이블(map 행 포함)·센서 파일·CAN bus·ext째로 뺌 |
| 기본 출력 | `--out`이 없으면 `<데이터 폴더>/parsed/tcar_nuscenes(_full)`. 데이터 폴더는 `$TCAR_DATA_ROOT`, 없으면 `/data` |
| 배포 | `install_desktop.sh`(앱 메뉴·바탕화면, `--appimage`), `packaging/build_appimage.sh` → Python·의존성까지 든 AppImage 하나(~250 MB) |

### DM GUI와 다른 점

- 녹화 = `.bag` 파일 하나. 옆의 `_integrity.csv`, `_recorder.log`(스탬프가 1초 달라도)를 함께 복사·이동.
- LiDAR 시각 보정 단계 없음(Ouster 전용). 단계는 가져오기 → 파싱 → 검증.
- `fail/` 폴더의 bag과 이미 데이터셋에 있는 bag은 가져오기 목록에서 기본 선택 해제.
- 파싱은 루트 번호 순서(A-2 → A-10)로, 데이터 폴더에 `calib/`가 있으면 `--calib`으로 넘김.
- 원본 bag 없이 데이터셋에만 있는 log도 목록에 표시(제외는 원본을 가져온 뒤).
- 수율 검사 기준은 변환기와 같게: 표준 카메라 6대, 25 ms. INS는 INSPVA 헤더 시각(GPS 시각보다 1–8 ms 늦을 뿐).
- AppImage에는 `opencv-python` 대신 `opencv-python-headless`(PyQt5 옆에 Qt가 하나 더 들어가지 않게).
- 코스·녹화마다 원본 녹화 시간 → 데이터셋 시간, 사용률(마우스를 올리면 20초 미만 자투리·카메라 누락·센서 시작·종료별 손실). `<log>.import.json`에서 계산, 데이터셋 전체 합계도 표시.
- 검증 통과 기록을 `logs/validation.json`(테이블 13개의 mtime·크기)에 남겨 다시 켜도 유지. DM GUI는 세션 동안만 기억. 파싱·제외로 테이블이 바뀌면 무효.

### 큐레이션 통합 (`curation/`)

별도 저장소 `tcar-scene-review`(`tools/`, localhost:8765 뷰어)를 `curation/`으로 옮기고 GUI에 붙였다. 정본은 이 저장소의 `curation/` 하나(`tcar-scene-review`는 옮겨 오기 전 이력만).

- 사이드바의 **파서 | 큐레이션** 전환: 같은 코스들을 큐레이션 페이지로 봄. 코스마다, 그리고 표에서 고른 루트(여러 개 가능)마다 요약(씬·검출·삭제 제안·미검토·삭제 예정)과 검토·검출·삭제 확정. 단계를 강제하지 않음: 검출된 씬은 바로 검토, 검출은 범위의 미검출 씬만 돌리고 끝나면 규칙을 다시 실행.
- 범위 옵션: `scene_detect.py --logs`, `apply_selection.py --logs`(그 log의 삭제만 확정, 이름 당김은 데이터셋 전체), 뷰어 `?logs=`(그 log의 씬만 표시, 화면 안 삭제 확정도 그 범위만; 범위 밖 대표 씬을 가리키는 정지 중복 그룹은 범위 안 씬으로 묶음).
- 검토 뷰어(`sample_viewer.py`)를 앱이 띄우고 창 안(Qt WebEngine)에 표시. 큐레이션 모드에 들어가면 미리 띄워 둠, 데이터셋을 바꾸는 작업 뒤에는 다시 띄움.
- 검토는 아직 결정하지 않은 첫 그룹의 검토 화면으로 바로 열림(대표 씬과 삭제 후보 나란히, 남기기/버리기). 검토 중에는 사이드바와 작업 칸을 숨김, 모든 그룹을 끝내면 "삭제 확정 · N개 빼기" 버튼.
- 뷰어 시작: 표에서 뽑은 색인을 `~/.cache/tcar-parser/viewer/`에 캐시(표 13개의 mtime·크기로 무효화). 334 scene에서 13.4초 → 0.5초. 처음 76초가 걸린 건 HDD가 다른 복사로 100% 쓰이던 때라서.
- 재생: `/data` HDD에서 한 씬의 sweep 파일(이름이 토큰이라 디스크에 흩어짐)은 초당 8–15장밖에 안 읽힘(복사 중), 실시간 6카메라는 180장 필요. 재생 중에는 절반 크기 사본(`/p/`, 로컬 디스크 캐시 `~/.cache/tcar-parser/preview/`), 멈추면 원본. 검토 그룹을 열면 그 그룹과 다음 그룹을 화면의 카메라(그룹의 모든 씬) → LiDAR → 나머지 카메라 순으로 32개씩 동시에 미리 읽음. 복사가 끝난 HDD에서 두 씬의 앞 카메라 1,168장을 13–16초(초당 73–90장; 두 칸 실시간 재생은 60장/s)에 읽음 — 처음 열 때부터 거의 실시간, 한 번 읽은 씬은 끊김 없음.
- 객체 검출은 torch·ultralytics가 있는 Python을 conda 환경 등에서 찾아 기억(이 PC: `envs/eufs`). AppImage는 torch를 싣지 않고 그 Python으로 AppImage 안의 스크립트를 실행.
- `apply_selection.py`: 변환기 잠금(`.convert.lock`)을 잡음, full 데이터셋의 `ext/<scene>`도 옮기고 이름 변경, 되돌리기가 `selection/models/`(검출 가중치 폴더)에서 멈추던 버그 수정(백업에서도 제외).
- 변환기: `<log>.import.json`이 남은 log는 이미 들어간 것으로 봄 → 큐레이션으로 scene이 모두 빠진 bag을 다시 넣지 않음. `remove_log.py`는 그런 log면 import 기록만 지움(복원 후 다시 파싱 가능).
- 코스 화면: 데이터셋 시간·사용률은 큐레이션 뒤 남은 씬 기준, 손실 내역에 "큐레이션으로 뺀 씬", 모두 빠진 녹화는 "큐레이션으로 모두 제외".
- 도구의 `--dataroot` 기본값을 `$TCAR_DATA_ROOT/parsed/tcar_nuscenes`로(원래는 `tools/`의 상위 폴더).
- 검증: 합성 데이터셋(루트 T-1, T-2)으로 T-1만 검출 → T-1에만 삭제 제안, 미검출 T-2도 검토 화면이 열리고 T-2 씬만 표시 → 두 루트 함께 보기 → T-2만 삭제 확정(T-1의 삭제 제안은 남음, 모두 빠진 T-2 log는 재파싱 안 됨) → 되돌리기(씬·이름·결정 원상복구)를 개발 venv와 AppImage 양쪽에서 통과. 실제 데이터셋에는 아무 작업도 돌리지 않음(A 148/148 검출·미검토 23, B 0/117, D 0/69).

### 큐레이션 탭: 후보 → 확정 → 최종 삭제, 전체 비율, 지도

- 상태를 나눔. **후보**: 필터에 걸렸고 아직 확정 안 한 씬, 필터별로(객체 부족 / 정지 중복). **필터 통과**: 어느 필터에도 안 걸린 씬(후보 아님). **확정**: 검토에서 남기기 / 버리기를 누른 씬 — 누를 때마다 `selection/human_decisions.json`과 `keep.txt` · `drop.txt`(scene · log · token · 시각)에 저장, 다시 누르면 후보로. **삭제 완료**: 최종 삭제로 뺀 씬(`_removed/<시각>/`, `deleted.txt`).
- **최종 삭제는 큐레이션 탭에서만, 버리기 확정 씬만.** 예전에는 확정 안 한 삭제 제안도 최종 목록(`final_selection.json`)에 들어가 함께 지워졌음 → 이제 `apply_selection.plan()`이 사람의 버리기 결정만 봄(후보는 남는다고 알림). 검토 화면의 "삭제 확정" 버튼과 루트별 삭제 확정은 없앰. 버튼은 큐레이션 탭 맨 위 카드 오른쪽(아래에 무엇을 빼고 무엇이 남는지 한 줄)과 앱의 검토 화면 위 막대(누르면 검토를 닫고 같은 확인 창)에 있고, 버리기 확정이 0개면 비활성. 검토 중에도 개수가 따라감.
- 큐레이션 탭 위쪽: 데이터셋 **전체**에 대한 확정 · 후보(필터별: 객체 부족, 정지 중복) · 필터 통과 · 삭제 완료 비율(막대 하나, 코스·루트별 비율은 두지 않음), 루트 표도 필터별 후보 열. 상태별 지도는 필터 통과 → 후보 → 확정 순으로 위에 그림. 검토 결정 파일을 1.5초마다 확인해 바로 반영(창 안 검토든 브라우저든). 사이드바의 코스별 시간 막대는 파서 탭에만.
- 지도: 모든 씬의 GNSS 경로(1 Hz, 뷰어의 `/api/map`)를 OSM 위에(타일은 뷰어의 `/tile/` 캐시, 어둡게). 루트별 색(범례: 코스마다 루트 번호) / 상태별 색 전환, 고른 코스·루트 밖은 흐리게. 경로를 누르면 그 루트를 고르고, 두 번 누르면 그 씬부터 검토.
- 검토 화면(`sample_viewer.html`): 칩·목록·카드를 후보 · <필터> / 필터 통과 / 남기기 확정 / 버리기 확정으로, "다음 후보"(N), "확정" 탭(사람이 확정한 씬만). 앱 안에서는(`?app=1`) 밖으로 나가는 링크(개요, 자세히)를 숨김. 앱은 `/review`를 바로 엶(`/`는 다른 세션이 만든 개요 페이지).
- 검증: 합성 데이터셋(씬 3개, 모두 버리기 후보)에서 API로 버리기 1 · 남기기 1 확정 → 3초 안에 탭 비율 · 최종 삭제 버튼 · keep/drop.txt 반영 → 최종 삭제는 버리기 확정 1개만 뺌(후보 1개 남음, 남긴 씬은 새 이름·같은 token으로 keep.txt에) → deleted.txt → 되돌리기. 앱 안 검토 화면에서 버리기 확정 → 닫으면 탭에 반영. 실제 데이터셋은 읽기만(지도 334씬, 타일 표시).
- 확정 목록(`human_decisions.json`, `keep.txt`, `drop.txt`, `deleted.txt`, 첫머리에 데이터셋 경로)을 코드의 `curation/`에도 복사: 앱이 `$TCAR_DECISIONS_DIR`(설정 `decisions_dir`, 기본은 쓰기 가능한 체크아웃의 `curation/`)로 넘기고, 결정 · 최종 삭제 · 되돌리기 때마다 갱신. 원본은 데이터셋 `selection/` 그대로(백업·되돌리기 기준). `tcar-scene-review`로 복사할 때는 이 파일들을 뺌.
- 필터 카드: 필터마다 무엇을 보는지(기준값은 result.json의 규칙 파라미터), 어떻게 세는지, 왜, 근거 수치와 작은 그래프(객체 부족: 씬별 샘플당 가까운 도로 이용자 분포와 기준선 / 정지 중복: 멈춘 씬 → 같은 곳 그룹 → 후보, 그룹 크기 분포), 걸린 씬과 아직 후보인 수.
- 경로 겹침(참고, 후보를 만들지 않음): 모든 루트의 1초 간격 GNSS 경로에서 같은 길 · 같은 방향(10 m · 30°, 정지 중복과 같은 기준)으로 지나간 씬을 셈. 실제 데이터: 달리는 씬 262개 중 서로 80% 이상 겹치는 쌍 1, 50% 이상 18 — 20초 씬은 시작 지점이 어긋나 통째로 겹치는 일이 드물어 필터로 두지 않음. 지도 ‘겹침’ 보기가 경로의 지점마다 지나간 다른 씬 수를 색으로(없음 / 1–2 / 3–4 / 5+), 마우스를 올리면 가장 많이 겹친 씬.
- "전체 검토 · 씬 N개": 코스와 상관없이 데이터셋 전체를 한 목록으로 검토(‘검토 열기’는 고른 코스·루트만).
- 검토 서버 `--fill`(앱이 켬): 요청이 없을 때 아직 캐시에 없는 씬을 채움 — 후보와 그 대표 씬 먼저, 요청이 오면 그쪽이 먼저, 끊긴 씬은 나중에 다시. 실제 데이터에서 30초에 1,100–1,500장.
- **필터 순서 고정: ① 정지 중복 → ② 객체 부족 → ③ 경로 겹침**(전에는 객체 부족 → 정지 중복). 앞에서 정한 씬은 뒤에서 안 봄 — ①의 같은 곳 그룹은 남긴 대표까지 ②③에서 제외(대표가 객체 부족으로 걸려 그룹이 통째로 후보가 되던 가능성 차단; 검증: 그룹 18개 모두 대표 1개, 그룹 씬이 ②③에 걸린 경우 0). ③ 경로 겹침을 필터로: ①②가 정하지 않은 달리는 씬을 모든 루트에 걸쳐 가까운 도로 이용자 많은 순으로 보고, 먼저 남긴 씬이 경로의 60% 이상을 같은 방향(10 m · 30°)으로 지나갔으면 후보(`--min-overlap`). 실제 데이터: 정지 중복 38 · 객체 부족 41 · 경로 겹침 14 = 후보 93.
- 필터 건너뛰기: 필터 카드의 "건너뛰기" → `selection/filters.json`(`skip`, `min_overlap`)에 쓰고 규칙을 다시 돌림(`curate.py --skip`도 가능). 건너뛴 필터의 씬은 다음 필터로 넘어감.
- 검토 화면: "다음 후보"가 필터 순서대로(지금 필터의 후보를 먼저 끝냄, 자기 자신으로 돌아오지 않음). 경로 겹침 후보는 남긴 쪽과 "나란히 비교", 지도에 겹치는 씬을 하늘색으로. 앱 루트 표는 필터별 열(① ② ③).
- 검토의 **건너뛰기(S)**: 메인 카드와 나란히 비교 화면에. 확정하지 않고 다음 후보(그룹)로, 건너뛴 것은 나머지를 다 본 뒤 다시(브라우저에 기억, 확정하면 풀림). 나란히 비교의 그룹 순서도 필터 순서(정지 중복 → 객체 부족 → 경로 겹침).
- 경로 겹침 지도(OSM): 메인 검토 지도가 경로 겹침 씬이면 그 씬과 겹치는 씬에 맞춰 확대하고, 먼저 남긴 씬(이전에 그 길을 지난 경로)은 주황 점선, 이 씬은 파랑 — 같은 길이라 진행 방향 기준 좌우로 7 px 띄워 그림, 진행 방향 꺾쇠, 시작점, 자차 위치. 나란히 비교 화면은 경로 겹침 그룹일 때 지도 칸을 추가해 모든 씬의 경로와 재생 중 현재 위치(번호 색 = 위 칸)를 함께.
- 경로 겹침은 루트·코스를 넘나드므로, 검토 범위(`?logs=`) 밖에 있는 남긴 씬도 불러와 나란히 비교와 지도에 씀(목록·개수에는 안 넣음, 칸에 "검토 범위 밖"과 루트 표시). 전에는 코스 A 검토에서 B-2의 0286이 덮은 A-5의 0072가 혼자 떠서 이유가 안 보였음.
- 검토 페이지가 열려 있는 동안 화면 코드가 바뀌면 위에 "검토 화면이 업데이트됐습니다 · 새로고침" 안내(`/api/page`, 15초마다; 결정은 이미 저장돼 있으니 새로고침해도 잃지 않음).
- 경로 겹침 지도 맞춤: 비교 화면의 지도 띠는 경로가 가로로 눕도록 돌려(주성분 방향, 북쪽 표시) 띠 폭을 다 쓰고, 타일 확대 한계를 넘어(z19 타일을 키워 ~z20.7) 짧은 경로도 크게. 메인 검토 지도도 같은 맞춤(북쪽 위).
- 검토 목록의 "후보" 탭은 아직 확정하지 않은 후보만(확정하면 바로 빠짐, 대표 씬은 그룹에 남은 후보가 있을 때만), 탭 숫자도 남은 후보 수. 다 확정하면 안내 문구. "다음 후보"는 탭과 상관없이 루트 순서로 찾음.
- 필터를 모두 통과한 씬은 확정(남김)으로 봄: keep.txt에 사람이 남긴 씬과 함께(근거 열: 사람 / 필터 통과), 규칙을 다시 돌리면 keep.txt도 다시 씀.
- 큐레이션 탭은 코스(A·B·D)로 나누지 않고 데이터셋 전체만: 사이드바 코스 목록은 파서 탭에만, 루트 표는 모든 루트, 루트를 고르지 않으면 검토·검출이 전체에. 위 카드는 전체(씬 · 시간) / 삭제 예정(버리기 확정) / 삭제 후 예상(개수 · 비율 · 시간) / 후보 / 삭제 완료, 막대는 버리기와 필터별 후보만 색(나머지 회색 = 남는 씬). 시간은 루트마다 import 기록의 씬 길이로.
- 확정한 뒤에도 다시 비교: 검토 목록의 "확정" 탭이 확정한 씬을 그룹째(대표 + 묶인 씬) 보여 주고 그룹마다 "비교"(단독은 "보기") — 나란히 비교에서 다른 쪽을 누르면 바뀜, 같은 쪽을 다시 누르면 후보로.
- 큐레이션 탭 배치: 제목 줄(제목 · **전체 검토** · 규칙 다시 실행)을 스크롤 밖에 고정, "전체 검토"는 늘 데이터셋 전체. 루트만 볼 때는 표에서 고르면 카드에 "고른 루트 검토"(두 번 눌러도 됨). 필터 정의 카드는 맨 아래로. 지도는 우리 데이터 범위(ROI, 여백 8%)까지만 축소·이동 — 경로를 불러올 때마다 범위를 잡고, 창 크기가 바뀌면 최소 축소도 다시 계산.
- **라운드 잠금**: 큐레이션 탭의 "최종 확정"(`apply_selection.py --finalize`)이 버리기 확정 씬을 빼고 남은 씬을 모두 잠금(`selection/locked.json`, token 기준이라 이름 당김과 무관). 잠긴 씬은 이후 절대 삭제되지 않고 후보도 안 됨 — 새로 파싱한 녹화의 씬만 필터·검토·삭제 대상. ③ 경로 겹침에서는 잠긴 씬이 먼저 남긴 씬으로 들어가, 새 씬이 기존 길과 겹치면 새 씬이 후보. 후보가 남아 있으면 최종 확정 불가(잠그면 다시 못 보므로). 검토 화면은 잠긴 씬을 "잠김"으로 표시하고 결정 버튼을 막고, 서버도 거부. keep.txt에 근거 "잠김 (이전 라운드)". "마지막 최종 확정 되돌리기"는 그 라운드의 삭제와 잠금을 함께 되돌림(삭제가 없던 라운드는 `--unlock-last`). 검증(합성): 1라운드 1개 삭제·2개 잠금 → 잠긴 씬 결정 거부 → 같은 길의 새 bag T-3 파싱 → 새 씬만 "경로 겹침" 후보(잠긴 0001이 100% 덮음) → 2라운드는 새 씬만 삭제 → 되돌리기.
- 지도는 코스 구분 대신 **빈도**: 남는 씬(잠김 · 필터 통과 · 사람이 남기기)을 한 색 반투명으로 그려 한 번 지난 길은 옅게, 여러 번 겹칠수록 진하게(범례: 1·2·4·8번 견본). "상태별" 보기는 유지, "루트별"은 뺌. 지도 위 휠은 지도만(페이지가 같이 스크롤되던 문제).
- 검토 목록에 "필터 통과" 탭, 요약의 확정 줄에 필터 통과 수. 다른 루트를 덮은 경로 겹침 후보는 자기 루트에만(두 번 보이던 문제).
- 수정: 이어서 재생이나 캐시를 기다린 뒤 재생이 시작될 때, 늦게 도착한 key frame 2D 박스가 재생 화면에 멈춘 채 남던 문제 — 재생 중에는 박스를 그리지 않음.
- **devkit이 깨지지 않는 최종 삭제**: 표 정리를 `prune_tables()` 하나로 모아 실제 삭제와 미리 검사가 같은 코드를 씀. `apply_selection.py --simulate`: 삭제를 메모리의 표에만 적용해 모든 참조(scene 첫/끝 sample · nbr_samples · sample 사슬, sample/sample_data prev·next 양방향, ego_pose · calibrated_sensor · sensor, log ↔ map, annotation/instance, 씬 이름의 공식 split 포함, 이름 당김 뒤 can_bus 파일)를 검사하고 임시 폴더의 표를 devkit NuScenes로 불러봄 — 문제가 있으면 지우지 않음. `--check`: 지금 데이터셋을 같은 검사 + key frame 파일 + devkit NuScenes + NuScenesCanBus(모든 scene pose). 앱의 최종 확정 = 미리 검사 → 삭제·잠금 → 삭제 후 점검 → 필터, 되돌리기 뒤에도 점검. 변환기가 prev/next를 씬 안에서만 잇고 ego_pose를 sample_data마다 따로 두므로 씬째 빼도 남는 씬의 연결은 끊기지 않음(검사로도 확인). 실제 데이터: 75개 삭제를 미리 적용한 표 → scene 259 · sample 10,360 · sample_data 958,234로 devkit 정상(28초), 지금 데이터셋 334개 --check 통과(25초). 검사기는 일부러 망가뜨린 5가지(씬 중간 sample 삭제, 끊긴 prev, 없는 ego_pose, map에 없는 log, split 밖 이름)를 모두 잡음.
- 수정: 검토 화면의 "← 큐레이션"으로 돌아와도 큐레이션 탭 위 카드(삭제 예정 · 삭제 후 예상)가 검토 전 숫자로 남던 문제 — 버튼의 clicked(bool)가 `_close_review(refresh)`에 들어가 새로 그리지 않았음.
- 실제 데이터 규모 검증(그림자 데이터셋: 센서 파일은 하드링크, 표·selection·can_bus·import는 복사, 원본은 그대로): 최종 확정(66개 삭제, 이름 264개 당김, 센서 파일 244,200개 이동, 268개 잠금)을 CLI와 앱 버튼 경로 양쪽으로 실행 → devkit NuScenes 268 scene · NuScenesCanBus 268, devkit으로 모든 scene의 sample 사슬과 채널별 sample_data 사슬(991,534개 전부) 추적, 파일 991,534개 모두 존재, LIDAR_TOP · CAM_FRONT를 devkit으로 열기, 이름이 바뀐 268개 scene 모두 CAN bus pose가 자기 시간대와 일치. 되돌리기 뒤 표 13개 · import.json · can_bus 이름 · 파일 1,235,734개가 원본과 바이트까지 같음.
- 수정(그 검증에서 발견): 첫 라운드의 최종 확정을 되돌려도 `locked.json`이 남아 268개가 계속 잠겨 있던 문제 — 백업 당시 잠금 파일이 없었으면 되돌릴 때 지움.
- **허깅페이스 배포** `scripts/hf_release.py` / `scripts/hf_assemble.py`: 잠긴(최종 확정된) 데이터만, 녹화별 센서 tar(10 GB 이하, 씬 경계) + 배포마다 바뀌는 표 tar 하나 + manifest(sha256, 파서 커밋) + README + 받는 쪽 assemble.py. 배포마다 브랜치(1002, 다음은 1002에서 만든 1015…), 새 녹화의 tar만 올라감. 올리기 전 데이터셋 점검과 "이미 올린 tar의 씬이 그대로인지" 검사, tar 단위 커밋으로 이어 올리기. 로컬 폴더로 끝까지 시험(1002 → 새 녹화 → 1015, 받는 쪽 1002 조립 → 1015 업데이트에서 새 tar 1개만 풀림, devkit · CAN bus 통과, 센서 파일 집합 동일). 그 시험에서 tar 대신 경로 문자열이 올라가던 버그를 잡음. 실제 데이터 계획: 268개 씬 → 71개 tar(6–9.4 GB, 554 GB), HDD에서 16개 동시 읽기로 74 MB/s(한 개씩은 36 MB/s).
- 허깅페이스 `shchon11/TCar`: 첫 배포 `1002` 업로드(268개 씬, tar 71개 + 표 tar, 476 GB, sha256 전부 manifest와 일치, 표 + D-1만 내려받아 조립 → devkit 정상). `main` = 항상 최신 배포(배포 뒤 서버에서 복사, 다시 올리지 않음), 9/23 캘리브레이션 샘플은 `0923` 브랜치에만.
- 수정: 지도 타일 요청이 응답을 못 받던 문제 — PyQt5에서 `QNetworkReply`를 파이썬 쪽에서 붙잡지 않으면 람다 슬롯 연결이 사라짐. 끝날 때까지 dict에 보관.

### 검증

- 합성 bag(카메라 7대·LiDAR·INSPVA·CORRIMU, 카메라 1초 공백 포함)으로 가져오기 → 파싱 → 수율 검사 → 검증 → 제외 → 복원을 오프스크린 GUI로 실행: 개발 venv와 AppImage 둘 다 통과. 제외 뒤 devkit·`NuScenesCanBus` 로드, 다시 파싱하면 비워진 scene 이름을 재사용.
- 수율 검사 = 변환기(0929 B-2): scene 10개, LiDAR 2,067 프레임, 카메라–LiDAR 차이 p50/p99/max 8.33/16.58/16.74 ms, 자투리 6.4 s 모두 일치. 43 GB bag을 USB 2.0에서 7.5분.

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
