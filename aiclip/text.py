"""字幕格式与文本工具（SRT / ASS / VTT）。"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

CJK = re.compile(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]")
PUNCT_TAIL = "，。、；：！？,.!?;: 　"


def srt_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def vtt_time(t: float) -> str:
    return srt_time(t).replace(",", ".")


def clean(text: str) -> str:
    text = text.replace("\u200b", "").strip()
    return re.sub(r"\s+", " ", text)


def wrap_cjk(text: str, max_chars: int = 18) -> str:
    """中文按字数硬换行，英文按词换行；ASS 用 \\N。"""
    text = clean(text)
    if len(text) <= max_chars:
        return text
    if len(CJK.findall(text)) / max(1, len(text)) > 0.3:
        lines, cur = [], ""
        for ch in text:
            cur += ch
            if len(cur) >= max_chars and ch in PUNCT_TAIL + "".join(list(PUNCT_TAIL)):
                lines.append(cur.strip())
                cur = ""
            elif len(cur) >= max_chars + 3:
                lines.append(cur.strip())
                cur = ""
        if cur.strip():
            lines.append(cur.strip())
        return r"\N".join(lines)
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return r"\N".join(lines)


def write_srt(segments: Iterable[dict], path: str) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{srt_time(s['start'])} --> {srt_time(s['end'])}")
        lines.append(clean(s["text"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_vtt(segments: Iterable[dict], path: str) -> str:
    out = ["WEBVTT", ""]
    for s in segments:
        out.append(f"{vtt_time(s['start'])} --> {vtt_time(s['end'])}")
        out.append(clean(s["text"]))
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def write_ass(
    segments: Sequence[dict],
    path: str,
    width: int = 1920,
    height: int = 1080,
    font: str = "Microsoft YaHei",
    font_size: int | None = None,
    color: str = "&H00FFFFFF",
    outline: int = 3,
    shadow: int = 1,
    margin_v: int | None = None,
    align: int = 2,
    max_chars: int = 18,
    bold: int = 0,
) -> str:
    font_size = font_size or max(26, int(min(width, height) * 0.056))
    margin_v = margin_v if margin_v is not None else int(height * 0.075)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{color},&H000000FF,&H00000000,&H80000000,{bold},0,0,0,100,100,0.5,0,1,{outline},{shadow},{align},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = []
    for s in segments:
        txt = wrap_cjk(s["text"], max_chars).replace("\n", r"\N")
        body.append(
            f"Dialogue: 0,{ass_time(s['start'])},{ass_time(s['end'])},Default,,0,0,0,,{txt}"
        )
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(body) + "\n")
    return path
