#!/usr/bin/env python3
"""최종 확정 for every volume at once, then the names it freed in the earlier volumes taken by the later volumes'
recordings (as the DM datasets do).

    python curation/finalize_volumes.py [--standard <root>] [--full <root>]          # the plan; nothing changes
    python curation/finalize_volumes.py ... --stage precheck    # every volume: no open candidate, the dry run passes
    python curation/finalize_volumes.py ... --stage finalize    # every volume: delete + renumber + lock, check; full twins
    python curation/move_logs.py --standard ... --full ... --fill --apply        # fill the freed names
    python curation/finalize_volumes.py ... --stage after       # every volume (as they are now) checked, rules again

Review happens per volume (scene names restart in every volume); the round ends for all of them together:

  precheck  refuses when any volume still has an open candidate, and runs the deletion's dry run (references,
            devkit) on every volume that deletes; nothing is changed before all of them pass
  finalize  apply_selection --finalize and --check on each standard volume, then mirror_full for each full twin
  (move)    move_logs.py --fill: the oldest whole recordings of the later volumes that fit go into the earlier
            volumes' free names, in both sets, with their locks
  after     apply_selection --check and mirror_full again on every volume there is now, then curate.py
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import volumes  # noqa: E402
import apply_selection as A  # noqa: E402

VERSION = A.VERSION


def std_volumes(std):
    return [v for v in volumes.volumes(std) if (v / VERSION / "scene.json").exists()]


def twin(full, v):
    """The full set's volume of the same number as the standard volume v (None without a full set)."""
    if full is None:
        return None
    f = volumes.volume(full, volumes.index_of(v))
    return f if (f / VERSION / "scene.json").exists() else None


def plan_all(std, full=None):
    """Per volume: apply_selection's plan; and the recordings the fill would move afterwards (scene counts after
    each volume's deletions), as move_logs.py --fill picks them."""
    import move_logs as M
    vols = std_volumes(std)
    per = []
    for v in vols:
        p = A.plan(str(v))
        T = M.small_tables(v)
        gone = set(p["deleted"])
        logfile = {l["token"]: l["logfile"] for l in T["log.json"]}
        n_by_log = {}
        for s in T["scene.json"]:
            if s["name"] not in gone:
                n_by_log[logfile[s["log_token"]]] = n_by_log.get(logfile[s["log_token"]], 0) + 1
        order = [name for name, _, _ in M.log_order(T)]
        per.append({"volume": v, "plan": p, "n_by_log": n_by_log, "order": order,
                    "published": M.published_logs(v)})
    limit = len(M.official_train())
    cap = os.environ.get("MOVE_LOGS_MAX_SCENES")
    if cap:
        limit = min(limit, int(cap))
    moves, n_after = [], [x["plan"]["n_after"] for x in per]
    left = [dict(x["n_by_log"]) for x in per]
    for i in range(len(per)):
        room = limit - n_after[i]
        for j in range(i + 1, len(per)):
            for name in per[j]["order"]:
                n = left[j].get(name)
                if not n or name in per[j]["published"]:
                    continue
                if n <= room:
                    moves.append((per[j]["volume"].name, per[i]["volume"].name, name, n))
                    room -= n
                    n_after[i] += n
                    n_after[j] -= n
                    left[j].pop(name)
            if not room:
                break
    return {"volumes": per, "moves": moves, "n_final": dict(zip((x["volume"].name for x in per), n_after)),
            "full": {x["volume"].name: str(twin(full, x["volume"]) or "") for x in per}}


def describe(P):
    lines = []
    for x in P["volumes"]:
        p = x["plan"]
        lines.append(f"{x['volume'].name}: 씬 {p['n_before']}개 → {p['n_after']}개 (버리기 {len(p['deleted'])}개, "
                     f"이름 당김 {len(p['rename'])}개), 새로 잠글 씬 {p['n_after'] - p['n_locked']}개"
                     + (f" · 아직 확정 안 한 후보 {p['candidates']}개" if p["candidates"] else ""))
    for src, dst, log, n in P["moves"]:
        lines.append(f"  채우기: {src} → {dst}  {log} ({n}개)")
    lines.append("결과: " + ", ".join(f"{k} {v}개" for k, v in P["n_final"].items()))
    return lines


def run(cmd):
    print("$ " + " ".join(map(str, cmd[1:])), flush=True)
    return subprocess.call([str(c) for c in cmd])


def py(script, *args):
    return [sys.executable, "-u", HERE / script, *args]


def main():
    data = os.environ.get("TCAR_DATA_ROOT", "/data")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--standard", type=Path, default=Path(data) / "parsed" / "tcar_nuscenes")
    ap.add_argument("--full", type=Path, default=Path(data) / "parsed" / "tcar_nuscenes_full")
    ap.add_argument("--stage", choices=("precheck", "finalize", "after"))
    a = ap.parse_args()
    std = volumes.base_of(a.standard)
    full = volumes.base_of(a.full) if (volumes.base_of(a.full) / VERSION).is_dir() else None
    if not a.stage:
        print("\n".join(describe(plan_all(std, full))))
        return 0
    vols = std_volumes(std)
    if a.stage == "precheck":
        P = plan_all(std, full)
        print("\n".join(describe(P)), flush=True)
        open_ = [f"{x['volume'].name} {x['plan']['candidates']}개" for x in P["volumes"] if x["plan"]["candidates"]]
        if open_:
            sys.exit(f"아직 확정하지 않은 후보가 있습니다 ({', '.join(open_)}): 검토에서 모두 확정한 뒤 최종 확정하세요")
        if full and any(twin(full, v) is None for v in vols):
            sys.exit("full 세트의 볼륨이 기본 세트와 짝이 맞지 않습니다")
        for x in P["volumes"]:
            if x["plan"]["deleted"] and run(py("apply_selection.py", "--dataroot", x["volume"], "--simulate")):
                sys.exit(f"{x['volume'].name}: 미리 검사에서 문제가 있어 아무것도 바꾸지 않았습니다")
        print("[finalize_volumes] 미리 검사 통과: 모든 볼륨", flush=True)
        return 0
    if a.stage == "finalize":
        for v in vols:
            if run(py("apply_selection.py", "--dataroot", v, "--finalize", "--no-precheck")):
                sys.exit(f"{v.name}: 최종 확정 실패")
            if run(py("apply_selection.py", "--dataroot", v, "--check")):
                sys.exit(f"{v.name}: 확정 뒤 점검에서 문제")
        for v in vols:
            f = twin(full, v)
            if f and run(py("mirror_full.py", "--standard", v, "--full", f)):
                sys.exit(f"{f.name}: full 세트 반영 · 점검에서 문제")
        print("[finalize_volumes] 모든 볼륨 최종 확정 · 점검 끝", flush=True)
        return 0
    # after the fill: every volume there is now
    for v in vols:
        if run(py("apply_selection.py", "--dataroot", v, "--check")):
            sys.exit(f"{v.name}: 채운 뒤 점검에서 문제")
        f = twin(full, v)
        if f and run(py("mirror_full.py", "--standard", v, "--full", f)):
            sys.exit(f"{f.name}: 채운 뒤 full 세트 점검에서 문제")
    for v in vols:
        run(py("curate.py", "--dataroot", v))
    print("[finalize_volumes] 끝: " + ", ".join(f"{v.name} {len(json.loads((v / VERSION / 'scene.json').read_text()))}개"
                                              for v in vols), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
