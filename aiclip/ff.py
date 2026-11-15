"""ffmpeg / ffprobe 封装：所有像素级操作集中在这里。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FFError(RuntimeError):
    pass


def run(args: Sequence[str], quiet: bool = True) -> str:
    cmd = [str(a) for a in args]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").strip().splitlines()[-8:])
        raise FFError(f"命令失败: {' '.join(cmd[:6])} ...\n{tail}")
    return p.stdout if quiet else (p.stdout + p.stderr)


def probe(path: str | os.PathLike) -> dict:
    out = run(
        [
            FFPROBE, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
    info = json.loads(out or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})
    fps = 0.0
    if v and v.get("r_frame_rate", "0/1") != "0/0":
        num, _, den = v["r_frame_rate"].partition("/")
        try:
            fps = float(num) / float(den or 1)
        except ZeroDivisionError:
            fps = 0.0
    return {
        "path": str(path),
        "duration": float(fmt.get("duration") or 0.0),
        "size": int(fmt.get("size") or 0),
        "bitrate": int(fmt.get("bit_rate") or 0),
        "vcodec": (v or {}).get("codec_name"),
        "width": int((v or {}).get("width") or 0),
        "height": int((v or {}).get("height") or 0),
        "fps": round(fps, 3),
        "acodec": (a or {}).get("codec_name"),
        "sample_rate": int((a or {}).get("sample_rate") or 0),
        "channels": int((a or {}).get("channels") or 0),
    }


def extract_audio(src: str, dst: str, sr: int = 16000, mono: bool = True) -> str:
    run([
        FFMPEG, "-y", "-v", "error", "-i", src,
        "-vn", "-ac", "1" if mono else "2", "-ar", str(sr),
        "-c:a", "pcm_s16le", dst,
    ])
    return dst


def _fmt_num(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".") or "0"


def trim(
    src: str,
    dst: str,
    start: float,
    end: float,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    crf: int = 20,
    preset: str = "veryfast",
    has_audio: bool = True,
    vf_extra: str | None = None,
) -> str:
    dur = max(0.05, end - start)
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
    )
    if vf_extra:
        vf = f"{vf},{vf_extra}"
    args = [
        FFMPEG, "-y", "-v", "error",
        "-ss", _fmt_num(start), "-t", _fmt_num(dur), "-i", src,
        "-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        args += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]
    args += ["-movflags", "+faststart", dst]
    run(args)
    return dst


def concat(parts: Iterable[str], dst: str, workdir: str | None = None) -> str:
    parts = [os.path.abspath(p) for p in parts]
    if not parts:
        raise FFError("没有可拼接的片段")
    wd = workdir or tempfile.mkdtemp(prefix="aiclip_cat_")
    os.makedirs(wd, exist_ok=True)
    listfile = os.path.join(wd, "concat.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % p.replace("\\", "/").replace("'", r"'\''"))
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", dst])
    return dst


def burn_subtitle(src: str, dst: str, ass_path: str, crf: int = 20) -> str:
    p = os.path.abspath(ass_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    vf = f"ass='{p}'"
    run([
        FFMPEG, "-y", "-v", "error", "-i", src, "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", dst,
    ])
    return dst


def mux_soft_subtitle(src: str, dst: str, srt_path: str, lang: str = "chi") -> str:
    run([
        FFMPEG, "-y", "-v", "error", "-i", src, "-i", srt_path,
        "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", f"language={lang}", dst,
    ])
    return dst


def vertical(
    src: str,
    dst: str,
    mode: str = "pad",
    width: int = 1080,
    height: int = 1920,
    crf: int = 20,
) -> str:
    """mode: crop=中心裁切 | pad=上下模糊填充 | fit=完整画面加黑边"""
    if mode == "crop":
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},setsar=1"
        )
        args = [FFMPEG, "-y", "-v", "error", "-i", src, "-vf", vf]
    elif mode == "pad":
        vf = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},boxblur=24:2[bg];"
            f"[0:v]scale={width}:-2:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
        args = [FFMPEG, "-y", "-v", "error", "-i", src, "-filter_complex", vf]
    else:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
        )
        args = [FFMPEG, "-y", "-v", "error", "-i", src, "-vf", vf]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", dst,
    ]
    run(args)
    return dst


def mix_audio(video: str, audio: str, dst: str, duck: float = 0.15) -> str:
    """用新音轨替换视频声音（配音成品）。"""
    run([
        FFMPEG, "-y", "-v", "error", "-i", video, "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-shortest", dst,
    ])
    return dst


def thumbnail(src: str, dst: str, at: float = 1.0, width: int = 640) -> str:
    run([FFMPEG, "-y", "-v", "error", "-ss", _fmt_num(at), "-i", src,
         "-frames:v", "1", "-vf", f"scale={width}:-2", dst])
    return dst
