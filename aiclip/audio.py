"""音频分析：静音检测、节拍检测、响度曲线。"""
from __future__ import annotations

import json
import math
import os
import wave
from typing import List, Tuple

import numpy as np


def load_wav(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
        rate = w.getframerate()
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if rate != sr:
        # 简易线性重采样（够用于能量分析）
        idx = np.linspace(0, len(data) - 1, int(len(data) * sr / rate))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data, sr


def rms_curve(x: np.ndarray, sr: int, win: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    hop = max(1, int(sr * win))
    frames = max(1, len(x) // hop)
    trimmed = x[: frames * hop].reshape(frames, hop)
    rms = np.sqrt((trimmed ** 2).mean(axis=1) + 1e-12)
    times = np.arange(frames) * hop / sr
    return times, rms


def silence_segments(
    wav_path: str,
    noise_db: float = -34.0,
    min_silence: float = 0.35,
    pad: float = 0.10,
) -> List[Tuple[float, float]]:
    """能量阈值法：返回需要保留的语音区间。"""
    x, sr = load_wav(wav_path)
    if x.size == 0:
        return []
    times, rms = rms_curve(x, sr)
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    thr = noise_db
    # 自适应：取 10 分位与 90 分位调节
    lo, hi = np.percentile(db, 10), np.percentile(db, 90)
    if hi - lo > 6:
        thr = max(-50.0, min(-20.0, lo + (hi - lo) * 0.35))
    voiced = db > thr
    keep: List[Tuple[float, float]] = []
    i = 0
    n = len(voiced)
    total = len(x) / sr
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            a = times[i] - pad
            b = (times[j] if j < n else total) + pad
            keep.append([max(0.0, a), min(total, b)])
            i = j
        else:
            i += 1
    merged: List[Tuple[float, float]] = []
    for a, b in keep:
        if merged and a - merged[-1][1] < min_silence:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= 0.15]
    if not out and total > 0:
        out = [(0.0, round(total, 3))]
    return out


def beats(wav_path: str, start_bpm: float = 120.0) -> dict:
    import librosa

    y, sr = load_wav(wav_path, 22050)
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=start_bpm, units="frames")
    times = librosa.frames_to_time(frames, sr=sr)
    tempo_val = float(np.atleast_1d(tempo)[0])
    return {
        "bpm": round(tempo_val, 2),
        "beats": [round(float(t), 3) for t in times],
        "count": int(len(times)),
        "duration": round(len(y) / sr, 3),
    }


def speech_ratio(wav_path: str, a: float, b: float) -> float:
    x, sr = load_wav(wav_path)
    s, e = int(a * sr), int(min(b * sr, len(x)))
    if e <= s:
        return 0.0
    seg = x[s:e]
    _, rms = rms_curve(seg, sr)
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    return float((db > -34).mean())


def save_json(obj, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return path
