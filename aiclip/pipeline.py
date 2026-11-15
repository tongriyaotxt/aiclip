"""高层成品流水线：粗剪、切片、字幕、竖屏。"""
from __future__ import annotations

import json
import os
import tempfile
from typing import List, Sequence

from . import asr, audio, edit, ff, text, vision


def _ensure(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _report(path: str, data: dict) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return path


# ---------------------------------------------------------------- 口播粗剪
def rough_cut(
    src: str,
    outdir: str = "out/rough",
    model: str = "small",
    lang: str | None = "zh",
    max_gap: float = 0.55,
    pad: float = 0.08,
    min_block: float = 0.40,
    drop_fillers: bool = True,
    burn: bool = True,
    width: int | None = None,
    height: int | None = None,
    fps: int = 30,
    crf: int = 20,
    mode: str = "asr",
    noise_db: float = -34.0,
    device: str = "auto",
    progress=None,
) -> dict:
    _ensure(outdir)
    info = ff.probe(src)
    width = width or info["width"] or 1920
    height = height or info["height"] or 1080
    total = info["duration"]

    wav = os.path.join(outdir, "audio.wav")
    ff.extract_audio(src, wav)

    subs: List[dict] = []
    if mode == "energy":
        if progress:
            progress("能量检测切分（无语音素材模式）…")
        blocks = audio.silence_segments(wav, noise_db=noise_db, min_silence=max_gap, pad=pad)
        edl = [{"start": round(a, 3), "end": round(b, 3)} for a, b in blocks]
        tr = {"mode": "energy", "words": [], "segments": []}
    else:
        if progress:
            progress("转写中…")
        tr = asr.transcribe(src, model_size=model, language=lang, device=device)
        _report(os.path.join(outdir, "transcript.json"), tr)
        edl = asr.speech_edl(
            tr["words"], max_gap=max_gap, pad=pad, min_block=min_block,
            drop_fillers=drop_fillers, total=total,
        )
        subs = asr.segments_from_edl(tr["words"], edl)
    if not edl:
        edl = [{"start": 0.0, "end": total}]
    edit.save_edl(edl, os.path.join(outdir, "edl.json"))

    if progress:
        progress(f"渲染 {len(edl)} 段…")
    raw = os.path.join(outdir, "rough.mp4")
    edit.render_edl(
        src, edl, raw, width=width, height=height, fps=fps, crf=crf,
        audio=info["acodec"] is not None, progress=progress,
    )

    if not subs and mode == "asr":
        subs = asr.segments_from_edl(tr["words"], edl)
    srt = text.write_srt(subs, os.path.join(outdir, "rough.srt"))
    ass = text.write_ass(subs, os.path.join(outdir, "rough.ass"), width, height)

    final = raw
    if burn and subs:
        if progress:
            progress("烧字幕…")
        final = ff.burn_subtitle(raw, os.path.join(outdir, "rough_sub.mp4"), ass, crf=crf)

    new_dur = ff.probe(final)["duration"]
    rep = {
        "input": os.path.abspath(src),
        "output": os.path.abspath(final),
        "duration_in": round(total, 2),
        "duration_out": round(new_dur, 2),
        "cut_ratio": round(1 - new_dur / total, 4) if total else 0,
        "blocks": len(edl),
        "subtitle_lines": len(subs),
        "model": model,
        "files": {
            "video": os.path.abspath(final),
            "preview": os.path.abspath(raw),
            "srt": os.path.abspath(srt),
            "ass": os.path.abspath(ass),
            "edl": os.path.abspath(os.path.join(outdir, "edl.json")),
            "transcript": os.path.abspath(os.path.join(outdir, "transcript.json")),
        },
    }
    _report(os.path.join(outdir, "report.json"), rep)
    return rep


# ---------------------------------------------------------------- 切片选段
def pick_windows(
    segments: Sequence[dict],
    duration: float,
    n: int = 3,
    length: float = 45.0,
    min_len: float = 15.0,
) -> List[dict]:
    if not segments:
        return []
    cands: List[dict] = []
    for i, s in enumerate(segments):
        start = max(0.0, s["start"] - 0.15)
        end = min(duration, start + length)
        if end - start < min_len:
            start = max(0.0, end - length)
        inside = [x for x in segments if x["end"] > start + 0.01 and x["start"] < end - 0.01]
        if not inside:
            continue
        dur = inside[-1]["end"] - inside[0]["start"]
        if dur < min_len:
            continue
        chars = sum(len(asr.norm_word(x["text"])) for x in inside)
        density = chars / max(0.1, dur)
        cands.append(
            {
                "start": round(max(0.0, inside[0]["start"] - 0.15), 3),
                "end": round(min(duration, inside[-1]["end"] + 0.2), 3),
                "score": round(float(density), 3),
                "chars": chars,
                "text": "".join(x["text"] for x in inside)[:120],
            }
        )
    cands.sort(key=lambda c: -c["score"])
    chosen: List[dict] = []
    for c in cands:
        if any(not (c["end"] <= o["start"] or c["start"] >= o["end"]) for o in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= n:
            break
    chosen.sort(key=lambda c: c["start"])
    return chosen


def auto_shorts(
    src: str,
    outdir: str = "out/shorts",
    n: int = 3,
    length: float = 45.0,
    model: str = "small",
    lang: str | None = "zh",
    vertical: str = "pad",
    burn: bool = True,
    crf: int = 20,
    device: str = "auto",
    progress=None,
) -> dict:
    _ensure(outdir)
    info = ff.probe(src)
    tr = asr.transcribe(src, model_size=model, language=lang, device=device)
    _report(os.path.join(outdir, "transcript.json"), tr)
    wins = pick_windows(tr["segments"], info["duration"], n=n, length=length)
    if not wins:
        raise RuntimeError("没有可用的切片窗口（素材太短或没有语音）")

    items = []
    for i, w in enumerate(wins, 1):
        d = _ensure(os.path.join(outdir, f"clip_{i:02d}"))
        if progress:
            progress(f"切片 {i}/{len(wins)}")
        raw = os.path.join(d, "clip.mp4")
        edit.render_edl(
            src, [{"start": w["start"], "end": w["end"]}], raw,
            width=info["width"] or 1920, height=info["height"] or 1080,
            fps=int(info["fps"] or 30), crf=crf, audio=info["acodec"] is not None,
        )
        vpath = raw
        if vertical in ("crop", "pad", "fit"):
            vpath = ff.vertical(raw, os.path.join(d, "vertical.mp4"), mode=vertical, crf=crf)
        subs = asr.segments_from_edl(tr["words"], [{"start": w["start"], "end": w["end"]}])
        srt = text.write_srt(subs, os.path.join(d, "sub.srt"))
        ass = text.write_ass(
            subs, os.path.join(d, "sub.ass"), 1080, 1920,
            max_chars=12, margin_v=300, font_size=64,
        )
        if burn and subs:
            vpath = ff.burn_subtitle(vpath, os.path.join(d, "final.mp4"), ass, crf=crf)
        items.append(
            {
                "index": i,
                "start": w["start"],
                "end": w["end"],
                "score": w["score"],
                "video": os.path.abspath(vpath),
                "srt": os.path.abspath(srt),
            }
        )
    rep = {"input": os.path.abspath(src), "count": len(items), "clips": items}
    _report(os.path.join(outdir, "shorts.json"), rep)
    return rep


# ---------------------------------------------------------------- 素材检索
def build_index(source: str, out: str = "out/idx", threshold: float = 27.0,
                device: str | None = None, progress=None) -> dict:
    idx = vision.Index.build(source, out, threshold=threshold, device=device, progress=progress)
    return {"index": os.path.abspath(out), **idx.stats()}


def search(source_index: str, query: str, topk: int = 8, out: str | None = None) -> List[dict]:
    idx = vision.Index.load(source_index)
    picks = idx.search(query, topk=topk)
    if out:
        vision.save_picks(picks, out)
    return picks


def assemble(picks_path: str, out: str = "out/assembly.mp4", vertical: str | None = None,
             crf: int = 20, progress=None) -> str:
    picks = vision.load_picks(picks_path)
    return edit.render_picks(picks, out, vertical=vertical, crf=crf, progress=progress)


# ---------------------------------------------------------------- 字幕处理
def subtitles(
    video: str,
    outdir: str = "out/subs",
    model: str = "small",
    lang: str | None = "zh",
    burn: bool = True,
    crf: int = 20,
    device: str = "auto",
    progress=None,
) -> dict:
    _ensure(outdir)
    info = ff.probe(video)
    if progress:
        progress("转写中…")
    tr = asr.transcribe(video, model_size=model, language=lang, device=device)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in tr["segments"]]
    srt = text.write_srt(segs, os.path.join(outdir, "sub.srt"))
    text.write_vtt(segs, os.path.join(outdir, "sub.vtt"))
    ass = text.write_ass(segs, os.path.join(outdir, "sub.ass"), info["width"] or 1920,
                         info["height"] or 1080)
    final = video
    if burn:
        final = ff.burn_subtitle(video, os.path.join(outdir, "burned.mp4"), ass, crf=crf)
    else:
        final = ff.mux_soft_subtitle(video, os.path.join(outdir, "softsub.mp4"), srt)
    return {"video": os.path.abspath(final), "srt": os.path.abspath(srt),
            "ass": os.path.abspath(ass), "lines": len(segs)}


# ---------------------------------------------------------------- 配音
def dub(
    video: str,
    script: str,
    out: str = "out/dub.mp4",
    voice: str | None = None,
    rate: int = 0,
    gap: float = 0.25,
    progress=None,
) -> dict:
    """用文稿合成配音，替换视频原声。"""
    from . import tts

    _ensure(os.path.dirname(os.path.abspath(out)) or "out")
    wav = os.path.join(os.path.dirname(os.path.abspath(out)) or ".", "_dub.wav")
    if progress:
        progress("合成配音…")
    tts.script_to_wav(script, wav, voice=voice, rate=rate, gap=gap)
    ff.mix_audio(video, wav, out)
    return {
        "output": os.path.abspath(out),
        "audio": os.path.abspath(wav),
        "duration": round(tts.wav_duration(wav), 2),
        "voice": voice or "系统默认",
    }
