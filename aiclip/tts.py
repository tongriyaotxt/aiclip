"""配音：Windows SAPI 离线 TTS，把文稿变成音轨并可替换视频声音。"""
from __future__ import annotations

import os
import subprocess
import wave

DEFAULT_VOICES = {
    "zh": "Microsoft Huihui Desktop",
    "en": "Microsoft Zira Desktop",
}


def available_voices() -> list[str]:
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60)
        return [v.strip() for v in (out.stdout or "").splitlines() if v.strip()]
    except Exception:
        return []


def _ps_quote(s: str) -> str:
    return s.replace("'", "''")


def speak(text: str, out_wav: str, voice: str | None = None, rate: int = 0,
          volume: int = 100) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)) or ".", exist_ok=True)
    text = " ".join(text.split())
    if not text:
        raise ValueError("配音文本为空")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        + (f"$s.SelectVoice('{_ps_quote(voice)}');" if voice else "")
        + f"$s.Rate={int(rate)};$s.Volume={int(volume)};"
        f"$s.SetOutputToWaveFile('{_ps_quote(os.path.abspath(out_wav))}');"
        f"$s.Speak('{_ps_quote(text)}');$s.Dispose();"
    )
    p = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not os.path.exists(out_wav):
        raise RuntimeError(f"TTS 失败: {(p.stderr or '')[:200]}")
    return out_wav


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def script_to_wav(
    script_path: str,
    out_wav: str,
    voice: str | None = None,
    rate: int = 0,
    gap: float = 0.25,
) -> str:
    """把文稿按行合成，行间留 gap 秒静音。"""
    with open(script_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError("文稿为空")
    tmpdir = os.path.dirname(os.path.abspath(out_wav)) or "."
    parts = []
    for i, ln in enumerate(lines):
        p = os.path.join(tmpdir, f"_tts_{i:03d}.wav")
        speak(ln, p, voice=voice, rate=rate)
        parts.append(p)
    from . import ff

    lst = os.path.join(tmpdir, "_tts_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/"))
            if gap > 0:
                sil = os.path.join(tmpdir, "_sil.wav")
                if not os.path.exists(sil):
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                                    "-i", f"anullsrc=r=24000:cl=mono", "-t", f"{gap}",
                                    sil], check=True)
                f.write("file '%s'\n" % sil.replace("\\", "/"))
    ff.run([ff.FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", lst, "-c", "copy", out_wav])
    return out_wav
