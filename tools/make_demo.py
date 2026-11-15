"""生成演示素材：中文 TTS 配音 + 三个视觉场景，用于端到端验证。"""
from __future__ import annotations

import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.abspath("demo")
FONT = r"C:\Windows\Fonts\msyh.ttc"

PARAS = [
    ("场景一 · 开场", "大家好，今天我们来聊一个很实际的问题。很多人剪视频，一天只能剪一条，为什么？因为大部分时间都花在了找素材和剪气口上。"),
    ("场景二 · 痛点", "你录了四十分钟的口播，里面有大量的停顿、口误、还有嗯、啊、那个这些废话。手动剪一遍，眼睛都花了。"),
    ("场景三 · 解法", "现在换一种做法。让程序先把声音转成文字，标出每个字的起止时间，再把没有声音的片段全部切掉。四十分钟，可以压到十二分钟。"),
]


def tts(text: str, path: str) -> str:
    safe = text.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$s.SelectVoice('Microsoft Huihui Desktop');"
        "$s.Rate=0;"
        f"$s.SetOutputToWaveFile('{path}');"
        f"$s.Speak('{safe}');"
        "$s.Dispose();"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return path


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
        data = np.interp(idx, np.arange(len(data)), data)
    return data.astype(np.float32)


def write_wav(path: str, data: np.ndarray, sr: int = 16000) -> str:
    pcm = np.clip(data, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return path


def silence(sec: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(sec * sr), dtype=np.float32)


def make_image(title: str, idx: int, path: str, size=(1280, 720)) -> str:
    w, h = size
    palettes = [((18, 26, 48), (58, 96, 160)), ((48, 18, 26), (160, 72, 58)), ((14, 40, 30), (36, 140, 96))]
    top, bottom = palettes[idx % len(palettes)]
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(h):
        k = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(top[i] + (bottom[i] - top[i]) * k) for i in range(3)))
    try:
        f1 = ImageFont.truetype(FONT, 86)
        f2 = ImageFont.truetype(FONT, 34)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    d.text((w // 2, h // 2 - 40), title, font=f1, fill=(255, 255, 255), anchor="mm")
    d.text((w // 2, h // 2 + 50), "AIClip demo footage", font=f2, fill=(235, 235, 235), anchor="mm")
    d.rectangle([40, 40, w - 40, h - 40], outline=(255, 255, 255), width=2)
    img.save(path, quality=92)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    sr = 16000
    tracks, images, durs = [], [], []
    for i, (title, body) in enumerate(PARAS):
        wav = os.path.join(OUT, f"p{i}.wav")
        tts(body, wav)
        seg = read_wav(wav)
        tracks.append(seg)
        images.append(make_image(title, i, os.path.join(OUT, f"scene{i}.jpg")))
        durs.append(len(seg) / sr)

    voice = np.concatenate([
        silence(0.9), tracks[0], silence(1.6), tracks[1], silence(1.4), tracks[2], silence(0.8)
    ])
    voice_path = write_wav(os.path.join(OUT, "voice.wav"), voice)

    # 视觉时间轴：首段静音也要有画面
    durs = [0.9] + [durs[0], 1.6, durs[1], 1.4, durs[2], 0.8]
    imgs = [images[0], images[0], images[1], images[1], images[2], images[2], images[2]]
    txt = os.path.join(OUT, "imgs.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for img, d in zip(imgs, durs):
            f.write(f"file '{img}'\nduration {d:.3f}\n")
        f.write(f"file '{imgs[-1]}'\n")

    out = os.path.join(OUT, "demo.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", txt,
        "-i", voice_path, "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out,
    ], check=True)
    print(f"演示素材: {out}")
    print(f"时长: {len(voice) / sr:.1f}s")


if __name__ == "__main__":
    main()
