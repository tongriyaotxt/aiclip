"""预热后的纯推理基准（排除模型加载与解码开销）。"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from aiclip import asr, models  # noqa: E402


def decode(media: str):
    return asr._decode(media)


def main(media="demo/demo.mp4", model="small"):
    audio = decode(media)
    print(f"音频 {len(audio)/16000:.1f}s", flush=True)
    res = {}
    for dev in ("cpu", "cuda"):
        if dev == "cuda" and not models.cuda_info().get("cuda_available"):
            continue
        m = asr.load_model(model, device=dev)
        segs, _ = m.transcribe(audio[: 16000 * 3], beam_size=5)  # 预热
        list(segs)
        t0 = time.perf_counter()
        segs, _ = m.transcribe(audio, beam_size=5)
        text = "".join(s.text for s in segs)
        dt = time.perf_counter() - t0
        res[dev] = {"seconds": round(dt, 2),
                    "rtf": round(dt / (len(audio) / 16000), 3),
                    "chars": len(text),
                    "device": str(getattr(m.model, "device", "?")),
                    "compute": str(getattr(m.model, "compute_type", "?"))}
        print(f"{dev:5} {dt:6.2f}s  RTF={dt/(len(audio)/16000):.3f}  设备={res[dev]['device']}/{res[dev]['compute']}", flush=True)
    if res.get("cpu") and res.get("cuda"):
        res["speedup"] = round(res["cpu"]["seconds"] / res["cuda"]["seconds"], 2)
        print(f"加速比 {res['speedup']}x", flush=True)
    with open("out/bench2.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["demo/demo.mp4", "small"]))
