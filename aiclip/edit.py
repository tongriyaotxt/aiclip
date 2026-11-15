"""剪辑层：EDL 渲染、多源拼接、成片封装。"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Iterable, List, Sequence

from . import ff


def save_edl(edl: Sequence[dict], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(edl), f, ensure_ascii=False, indent=1)
    return path


def load_edl(path: str) -> List[dict]:
    return json.load(open(path, encoding="utf-8"))


def render_edl(
    src: str,
    edl: Sequence[dict],
    out: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    crf: int = 20,
    preset: str = "veryfast",
    audio: bool = True,
    progress=None,
) -> str:
    """把一段素材按 EDL 剪出来（单源）。"""
    if not edl:
        raise ValueError("EDL 为空")
    tmp = tempfile.mkdtemp(prefix="aiclip_edl_")
    parts: List[str] = []
    for i, seg in enumerate(edl):
        seg_out = os.path.join(tmp, f"{i:04d}.mp4")
        ff.trim(src, seg_out, seg["start"], seg["end"], width, height, fps, crf, preset, audio)
        parts.append(seg_out)
        if progress:
            progress(f"{i + 1}/{len(edl)} 段已渲染")
    ff.concat(parts, out, tmp)
    return out


def render_picks(
    picks: Sequence[dict],
    out: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    crf: int = 20,
    vertical: str | None = None,
    progress=None,
) -> str:
    """按检索结果（多源）拼接成片；vertical=crop/pad/fit 时成片为竖屏。"""
    if vertical:
        width, height = 1080, 1920
    tmp = tempfile.mkdtemp(prefix="aiclip_pick_")
    parts = []
    for i, p in enumerate(picks):
        seg = os.path.join(tmp, f"{i:04d}.mp4")
        probe = ff.probe(p["video"])
        ff.trim(
            p["video"], seg, float(p["start"]), float(p["end"]),
            width, height, fps, crf, "veryfast", probe["acodec"] is not None,
        )
        parts.append(seg)
        if progress:
            progress(f"{i + 1}/{len(picks)} 段已渲染")
    ff.concat(parts, out, tmp)
    return out


def add_vertical(src: str, out: str, mode: str = "pad") -> str:
    return ff.vertical(src, out, mode=mode)
