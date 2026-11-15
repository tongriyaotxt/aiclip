"""全命令自检：每条子命令跑一次真数据，验证产物存在。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
DEMO = os.path.join(ROOT, "demo", "demo.mp4")
OUT = os.path.join(ROOT, "out", "selftest")
SCRIPT = os.path.join(OUT, "script.txt")

RESULTS: list[tuple[str, bool, str]] = []


def run(args: list[str], label: str, expect: list[str], timeout=900) -> None:
    t0 = time.perf_counter()
    p = subprocess.run([PY, "-m", "aiclip", *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    dt = time.perf_counter() - t0
    missing = [e for e in expect if not os.path.exists(os.path.join(ROOT, e))]
    ok = p.returncode == 0 and not missing
    detail = f"{dt:5.1f}s"
    if missing:
        detail += " missing=" + ",".join(missing)
    elif p.returncode != 0:
        detail += " " + (p.stderr or "")[-160:].replace("\n", " ")
    RESULTS.append((label, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {label:<26} {detail}", flush=True)


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    with open(SCRIPT, "w", encoding="utf-8") as f:
        f.write("这是一段自动配音测试。\n第二句用于验证拼接。\n")

    run(["doctor"], "doctor", [])
    run(["probe", DEMO], "probe", [])
    run(["transcribe", DEMO, "--outdir", "out/selftest/transcribe"],
        "transcribe", ["out/selftest/transcribe/sub.srt", "out/selftest/transcribe/transcript.json"])
    run(["rough", DEMO, "--outdir", "out/selftest/rough"],
        "rough", ["out/selftest/rough/rough_sub.mp4", "out/selftest/rough/edl.json"])
    run(["rough", DEMO, "--outdir", "out/selftest/rough_energy", "--mode", "energy", "--no-burn"],
        "rough --mode energy", ["out/selftest/rough_energy/rough.mp4"])
    run(["subs", DEMO, "--outdir", "out/selftest/subs", "--soft"],
        "subs", ["out/selftest/subs/softsub.mp4", "out/selftest/subs/sub.srt"])
    run(["shorts", DEMO, "--outdir", "out/selftest/shorts", "--n", "2", "--length", "20"],
        "shorts", ["out/selftest/shorts/clip_01/final.mp4", "out/selftest/shorts/clip_02/final.mp4"])
    run(["vertical", DEMO, "--out", "out/selftest/vertical.mp4"],
        "vertical", ["out/selftest/vertical.mp4"])
    run(["beats", DEMO, "--out", "out/selftest/beats.json"], "beats", ["out/selftest/beats.json"])
    run(["index", "demo", "--out", "out/selftest/idx"], "index", ["out/selftest/idx/emb.npy"])
    run(["search", "out/selftest/idx", "蓝色 背景", "--topk", "2",
         "--out", "out/selftest/picks.json"], "search", ["out/selftest/picks.json"])
    run(["assemble", "out/selftest/picks.json", "--out", "out/selftest/assembled.mp4"],
        "assemble", ["out/selftest/assembled.mp4"])
    run(["dub", DEMO, "out/selftest/script.txt", "--out", "out/selftest/dub.mp4"],
        "dub", ["out/selftest/dub.mp4"])
    run(["pipeline", DEMO, "--outdir", "out/selftest/full", "--n", "1", "--length", "20"],
        "pipeline", ["out/selftest/full/summary.json"])

    ok = sum(1 for _, o, _ in RESULTS if o)
    total = len(RESULTS)
    print(f"\n{ok}/{total} 通过", flush=True)
    with open(os.path.join(OUT, "selftest.json"), "w", encoding="utf-8") as f:
        json.dump([{"cmd": c, "ok": o, "detail": d} for c, o, d in RESULTS],
                  f, ensure_ascii=False, indent=1)
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
