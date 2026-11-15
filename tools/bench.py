"""CPU vs GPU 基准：转写与建索引。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiclip import asr, vision  # noqa: E402


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label:<22} {dt:7.2f}s", flush=True)
    return dt, out


def main(media: str, model: str = "small"):
    res = {}
    for dev in ("cpu", "cuda"):
        try:
            d, tr = timed(f"transcribe {model} {dev}",
                          lambda d=dev: asr.transcribe(media, model_size=model, device=d))
            res[f"asr_{dev}"] = round(d, 2)
            res[f"asr_{dev}_segs"] = len(tr["segments"])
        except Exception as exc:
            print(f"transcribe {dev} 失败: {type(exc).__name__}: {exc}", flush=True)
            res[f"asr_{dev}"] = None

    for dev in ("cpu", "cuda"):
        out = f"out/bench_idx_{dev}"
        shutil.rmtree(out, ignore_errors=True)
        try:
            d, idx = timed(f"index {dev}",
                           lambda d=dev: vision.Index.build("demo", out, device=d))
            res[f"index_{dev}"] = round(d, 2)
            res[f"index_{dev}_clips"] = len(idx.meta)
        except Exception as exc:
            print(f"index {dev} 失败: {type(exc).__name__}: {exc}", flush=True)
            res[f"index_{dev}"] = None

    if res.get("asr_cpu") and res.get("asr_cuda"):
        res["asr_speedup"] = round(res["asr_cpu"] / res["asr_cuda"], 2)
    if res.get("index_cpu") and res.get("index_cuda"):
        res["index_speedup"] = round(res["index_cpu"] / res["index_cuda"], 2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open("out/bench.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo/demo.mp4")
