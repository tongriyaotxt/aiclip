"""语音层：faster-whisper 转写 + 基于词级时间戳的剪辑 EDL。"""
from __future__ import annotations

import os
import re
from typing import Iterable, List, Sequence

from .text import CJK

_MODEL_CACHE: dict = {}

FILLERS = {
    "嗯", "呃", "额", "啊", "哦", "唉", "哎", "唔", "呐", "嘛", "呀", "诶",
    "那个", "这个", "就是", "然后", "其实", "反正", "差不多", "怎么说",
    "um", "uh", "erm", "ah", "hmm", "like", "you", "know", "so", "well",
}
PUNCT = re.compile(r"[\s，。、；：！？,\.!\?;:\"'“”‘’（）()\[\]【】—…~`\-]+")


def norm_word(w: str) -> str:
    return PUNCT.sub("", w or "").lower()


def is_filler(w: str) -> bool:
    return norm_word(w) in FILLERS


def load_model(size: str = "small", device: str = "auto", compute: str = "auto"):
    from faster_whisper import WhisperModel

    from . import models

    device = models.pick_device(device)
    if device == "cuda":
        models.enable_cuda_dlls()
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    key = (size, device, compute)
    if key not in _MODEL_CACHE:
        local = models.whisper_dir(size)
        target = local or size
        try:
            _MODEL_CACHE[key] = WhisperModel(target, device=device, compute_type=compute)
        except Exception as exc:
            print(f"[asr] {device}/{compute} 不可用({type(exc).__name__})，回退 CPU int8", flush=True)
            _MODEL_CACHE[key] = WhisperModel(target, device="cpu", compute_type="int8")
    return _MODEL_CACHE[key]


def _decode(media):
    """自己解码成 16k 单声道 float32，绕开 PyAV/faster-whisper 的版本依赖。"""
    import numpy as np
    import tempfile

    from . import audio, ff

    if not isinstance(media, str) or not os.path.exists(media):
        return media
    if media.lower().endswith(".wav"):
        x, sr = audio.load_wav(media, 16000)
        return x.astype(np.float32)
    tmp = os.path.join(tempfile.gettempdir(), f"aiclip_{abs(hash(media)) % 10**8}.wav")
    try:
        ff.extract_audio(media, tmp, sr=16000)
        x, sr = audio.load_wav(tmp, 16000)
        return x.astype(np.float32)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def transcribe(
    media: str,
    model_size: str = "small",
    language: str | None = "zh",
    vad: bool = True,
    device: str = "auto",
    compute: str = "auto",
) -> dict:
    model = load_model(model_size, device, compute)
    seg_iter, info = model.transcribe(
        _decode(media),
        language=language,
        vad_filter=vad,
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,
    )
    segs: List[dict] = []
    for s in seg_iter:
        words = [
            {"w": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3)}
            for w in (s.words or [])
            if w.word and w.word.strip()
        ]
        if not words and s.text.strip():
            words = [{"w": s.text.strip(), "start": round(s.start, 3), "end": round(s.end, 3)}]
        segs.append(
            {
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "text": s.text.strip(),
                "words": words,
            }
        )
    flat = [w for s in segs for w in s["words"]]
    return {
        "language": getattr(info, "language", language),
        "language_probability": round(float(getattr(info, "language_probability", 0) or 0), 4),
        "duration": round(float(getattr(info, "duration", 0) or 0), 3),
        "segments": segs,
        "words": flat,
        "model": model_size,
    }


def speech_edl(
    words: Sequence[dict],
    max_gap: float = 0.55,
    pad: float = 0.08,
    min_block: float = 0.40,
    drop_fillers: bool = True,
    total: float | None = None,
) -> List[dict]:
    """按词级时间戳切出“有话说”的区间，顺手剪掉气口和口头禅。"""
    kept = [w for w in words if not (drop_fillers and is_filler(w["w"]))]
    if not kept:
        return [{"start": 0.0, "end": total or 0.0}] if total else []
    blocks: List[List[float]] = []
    cur = [kept[0]["start"], kept[0]["end"]]
    for w in kept[1:]:
        if w["start"] - cur[1] > max_gap:
            blocks.append(cur)
            cur = [w["start"], w["end"]]
        else:
            cur[1] = max(cur[1], w["end"])
    blocks.append(cur)

    padded = []
    for a, b in blocks:
        a2, b2 = max(0.0, a - pad), b + pad
        if total:
            a2, b2 = min(a2, total), min(b2, total)
        if b2 - a2 >= min_block:
            if padded and a2 - padded[-1][1] < 0.02:
                padded[-1][1] = b2
            else:
                padded.append([a2, b2])
    return [{"start": round(a, 3), "end": round(b, 3)} for a, b in padded]


def _join(a: str, b: str) -> str:
    if not a:
        return b
    if CJK.search(a[-1]) or CJK.search(b[:1]):
        return a + b
    return f"{a} {b}"


def _timed_words(words: Sequence[dict], edl: Sequence[dict]) -> List[dict]:
    """把词映射到剪辑后的新时间轴。"""
    out: List[dict] = []
    offset = 0.0
    for blk in edl:
        a, b = blk["start"], blk["end"]
        for w in words:
            if w["end"] <= a or w["start"] >= b:
                continue
            ws = max(w["start"], a)
            we = min(w["end"], b)
            out.append(
                {
                    "w": w["w"],
                    "start": round(offset + ws - a, 3),
                    "end": round(offset + we - a, 3),
                }
            )
        offset += b - a
    return out


TAIL_PUNCT = "，。、；：！？,.!?;:…"


def segments_from_edl(
    words: Sequence[dict],
    edl: Sequence[dict],
    max_chars: int = 18,
    max_dur: float = 3.8,
    min_chars: int = 6,
) -> List[dict]:
    """把剪辑后的时间线重新映射成字幕：按标点/字数/时长断行。"""
    tw = sorted(_timed_words(words, edl), key=lambda x: x["start"])
    if not tw:
        return []
    out: List[dict] = []
    cur_text, cur_start, cur_end = "", tw[0]["start"], tw[0]["end"]
    for w in tw:
        text = w["w"].strip()
        if not text:
            continue
        cur_text = _join(cur_text, text) if cur_text else text
        cur_end = w["end"]
        n = len(norm_word(cur_text))
        dur = cur_end - cur_start
        ends_sentence = bool(text) and text[-1] in TAIL_PUNCT
        if (ends_sentence and n >= min_chars) or n >= max_chars or dur >= max_dur:
            out.append({"start": cur_start, "end": cur_end, "text": cur_text.strip()})
            cur_text, cur_start = "", cur_end
    if cur_text.strip():
        out.append({"start": cur_start, "end": cur_end, "text": cur_text.strip()})

    # 去掉结尾残留标点、修掉零长度
    fixed = []
    for s in out:
        t = s["text"].strip(" ，,、")
        if not t:
            continue
        if s["end"] - s["start"] < 0.25:
            s["end"] = round(s["start"] + 0.6, 3)
        fixed.append({"start": s["start"], "end": s["end"], "text": t})
    return fixed


def total_duration(media: str) -> float:
    from .ff import probe

    return probe(media)["duration"]


def update(model_size: str = "small") -> None:
    """预热模型（下载权重）。"""
    load_model(model_size)
