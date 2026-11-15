"""模型获取：优先本地，其次国内镜像（不走 huggingface.co）。

所有下载都走 HF_ENDPOINT（默认 https://hf-mirror.com），失败自动换节点重试。
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

HF = os.environ.get("AICLIP_HF") or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
HF = HF.rstrip("/")
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = HF

WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}
WHISPER_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
CLIP_REPO = "laion/CLIP-ViT-B-32-laion2B-s34b-b79K"
CLIP_FILE = "open_clip_pytorch_model.bin"


def _say(msg: str) -> None:
    print(msg, flush=True)


def enable_cuda_dlls() -> bool:
    """把 torch 自带的 cuDNN/cuBLAS DLL 目录挂到进程里，
    让 CTranslate2（faster-whisper）也能找到它们。"""
    try:
        import torch
    except Exception:
        return False
    if os.name != "nt":
        return torch.cuda.is_available()
    lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(lib):
        return False
    try:
        os.add_dll_directory(lib)
    except (AttributeError, OSError):
        pass
    os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    return True


def cuda_info() -> dict:
    try:
        import torch

        ok = torch.cuda.is_available()
        info = {"torch": torch.__version__, "cuda_available": ok,
                "cuda_version": torch.version.cuda}
        if ok:
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu": p.name, "vram_gb": round(p.total_memory / 1024**3, 1),
                         "capability": f"{p.major}.{p.minor}"})
            info["cudnn"] = torch.backends.cudnn.version()
        return info
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def pick_device(device: str = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _remote_size(url: str, timeout: int = 25) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "aiclip/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length") or 0)


def fetch(url: str, dst: str, retries: int = 4, timeout: int = 60, quiet: bool = False) -> str:
    """带断点续传的下载：先写 .part，完整后再改名，避免半包被当成可用模型。"""
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    part = dst + ".part"
    for attempt in range(1, retries + 1):
        start = _size(part)
        try:
            total = _remote_size(url)
        except Exception:
            total = 0
        if total and start == total:
            shutil.move(part, dst)
            return dst
        headers = {"User-Agent": "aiclip/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "ab") as f:
                got = 0
                chunk = 1 << 20
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    if not quiet and got % (8 << 20) < chunk:
                        mb = _size(part) / 1048576
                        tot = f"/{total/1048576:.0f}" if total else ""
                        _say(f"  {os.path.basename(dst)} {mb:.1f}{tot} MB")
            if total and _size(part) < total:
                raise IOError(f"不完整 {_size(part)}/{total}")
            shutil.move(part, dst)
            return dst
        except Exception as exc:
            if attempt >= retries:
                raise
            _say(f"  [重试 {attempt}/{retries}] {type(exc).__name__}: {exc}")
            time.sleep(min(2 * attempt, 8))
    return dst


def whisper_dir(size: str = "small", allow_download: bool = True, quiet: bool = False) -> str | None:
    """返回本地 faster-whisper 模型目录；没有则下载。"""
    if os.path.isdir(size):
        return size
    name = size
    dest = os.path.join(MODELS, "faster-whisper", name)
    repo = WHISPER_REPOS.get(name)
    if not repo:
        return None
    have_all = all(_size(os.path.join(dest, f)) > 0 for f in WHISPER_FILES) and \
        os.path.exists(os.path.join(dest, ".ready"))
    if have_all:
        return dest
    if not allow_download:
        return None
    _say(f"[模型] 下载 faster-whisper-{name}（{HF}）")
    ok = True
    for fn in WHISPER_FILES:
        try:
            fetch(f"{HF}/{repo}/resolve/main/{fn}", os.path.join(dest, fn), quiet=quiet)
        except Exception as exc:
            _say(f"[模型] {fn} 失败: {exc}")
            ok = False
            break
    if ok and all(_size(os.path.join(dest, f)) > 0 for f in WHISPER_FILES):
        with open(os.path.join(dest, ".ready"), "w", encoding="utf-8") as f:
            f.write("ok")
        _say(f"[模型] 就绪: {dest}")
        return dest
    return None


def clip_checkpoint(allow_download: bool = True, quiet: bool = False) -> str | None:
    dst = os.path.join(MODELS, "clip", "ViT-B-32-laion2b.bin")
    if _size(dst) > 100 * 1024 * 1024:
        return dst
    if not allow_download:
        return None
    _say(f"[模型] 下载 CLIP ViT-B-32/laion2b（{HF}）")
    part = dst + ".part"
    try:
        fetch(f"{HF}/{CLIP_REPO}/resolve/main/{CLIP_FILE}", part, quiet=quiet)
        if _size(part) > 100 * 1024 * 1024:
            shutil.move(part, dst)
            _say(f"[模型] 就绪: {dst}")
            return dst
    except Exception as exc:
        _say(f"[模型] CLIP 下载失败: {exc}")
    return None


def ensure_all(whisper_size: str = "small", need_clip: bool = False) -> dict:
    return {
        "whisper": whisper_dir(whisper_size),
        "clip": clip_checkpoint() if need_clip else None,
    }
