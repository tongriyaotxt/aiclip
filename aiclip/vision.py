"""视觉层：镜头切分 + CLIP 语义索引/检索。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".flv", ".ts", ".mts"}


def list_videos(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in VIDEO_EXT:
                out.append(os.path.join(dp, fn))
    return out


def scenes(path: str, threshold: float = 27.0, min_len: int = 15) -> List[dict]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(path)
    fps = float(video.frame_rate or 30.0)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len))
    sm.detect_scenes(video, show_progress=False)
    lst = sm.get_scene_list()
    if not lst:
        from .ff import probe

        return [{"start": 0.0, "end": probe(path)["duration"]}]
    return [
        {"start": round(s[0].get_seconds(), 3), "end": round(s[1].get_seconds(), 3)}
        for s in lst
    ]


def _grab(path: str, at: float):
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, at) * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


_CLIP_CACHE: dict = {}


def _clip(device: str | None = None):
    import torch, open_clip

    from . import models

    dev = models.pick_device(device or "auto")
    if dev == "cuda":
        models.enable_cuda_dlls()
    if dev in _CLIP_CACHE:
        return _CLIP_CACHE[dev]
    ckpt = models.clip_checkpoint()
    if ckpt:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=ckpt, device=dev
            )
        except Exception:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev
            )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=dev
        )
    model.eval()
    bundle = (model, preprocess, open_clip.get_tokenizer("ViT-B-32"), dev)
    _CLIP_CACHE[dev] = bundle
    return bundle


class Index:
    def __init__(self, root: str, meta: List[dict], emb: np.ndarray, dim: int):
        self.root = root
        self.meta = meta
        self.emb = emb
        self.dim = dim

    # ---------- build ----------
    @classmethod
    def build(
        cls,
        source: str,
        out: str,
        threshold: float = 27.0,
        max_frames_per_video: int = 400,
        device: str | None = None,
        progress=None,
    ) -> "Index":
        import torch, cv2
        from PIL import Image

        os.makedirs(out, exist_ok=True)
        fdir = os.path.join(out, "frames")
        os.makedirs(fdir, exist_ok=True)
        model, preprocess, _, dev = _clip(device)

        vids = list_videos(source) if os.path.isdir(source) else [source]
        if not vids:
            raise FileNotFoundError(f"没有找到视频: {source}")

        meta: List[dict] = []
        embs: List[np.ndarray] = []
        batch_t, batch_m = [], []

        def flush():
            if not batch_t:
                return
            with torch.no_grad():
                t = torch.cat(batch_t).to(dev)
                e = model.encode_image(t)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.extend(list(e.cpu().numpy().astype("float32")))
            meta.extend(batch_m)
            batch_t.clear()
            batch_m.clear()

        for vi, vpath in enumerate(vids):
            try:
                sc = scenes(vpath, threshold=threshold)
            except Exception as exc:  # 坏文件跳过
                print(f"[skip] {vpath}: {exc}")
                continue
            if len(sc) > max_frames_per_video:
                step = len(sc) / max_frames_per_video
                sc = [sc[int(i * step)] for i in range(max_frames_per_video)]
            for i, s in enumerate(sc):
                frame = _grab(vpath, (s["start"] + s["end"]) / 2)
                if frame is None:
                    continue
                fn = f"v{vi:03d}_{i:04d}.jpg"
                fp = os.path.join(fdir, fn)
                cv2.imwrite(fp, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                try:
                    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    batch_t.append(preprocess(im).unsqueeze(0))
                    batch_m.append(
                        {
                            "video": os.path.abspath(vpath),
                            "start": s["start"],
                            "end": s["end"],
                            "frame": fp,
                        }
                    )
                except Exception:
                    continue
                if len(batch_t) >= 16:
                    flush()
            if progress:
                progress(f"[{vi + 1}/{len(vids)}] {os.path.basename(vpath)} scenes={len(sc)}")
        flush()
        if not embs:
            raise RuntimeError("没有提取到可用镜头")
        emb = np.stack(embs)
        with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        np.save(os.path.join(out, "emb.npy"), emb)
        with open(os.path.join(out, "index.info.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "ViT-B-32/laion2b_s34b_b79k", "count": len(meta),
                       "dim": int(emb.shape[1]), "device": dev}, f, ensure_ascii=False, indent=1)
        return cls(out, meta, emb, int(emb.shape[1]))

    # ---------- io ----------
    @classmethod
    def load(cls, root: str) -> "Index":
        meta = json.load(open(os.path.join(root, "meta.json"), encoding="utf-8"))
        emb = np.load(os.path.join(root, "emb.npy"))
        return cls(root, meta, emb, int(emb.shape[1]))

    # ---------- query ----------
    def _encode_text(self, texts: Sequence[str], device: str | None = None) -> np.ndarray:
        import torch

        model, _, tok, dev = _clip(device)
        with torch.no_grad():
            t = tok(list(texts)).to(dev)
            e = model.encode_text(t)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().astype("float32")

    def search(
        self,
        text: str,
        topk: int = 8,
        min_gap: float = 1.5,
        queries: Sequence[str] | None = None,
    ) -> List[dict]:
        qs = list(queries) if queries else [text]
        if text and text not in qs:
            qs.insert(0, text)
        te = self._encode_text(qs).mean(axis=0)
        te = te / (np.linalg.norm(te) + 1e-9)
        sims = self.emb @ te
        order = np.argsort(-sims)
        picks: List[dict] = []
        for idx in order:
            m = self.meta[int(idx)]
            clash = any(
                m["video"] == p["video"]
                and not (m["end"] < p["start"] - min_gap or m["start"] > p["end"] + min_gap)
                for p in picks
            )
            if clash:
                continue
            picks.append({**m, "score": round(float(sims[int(idx)]), 4)})
            if len(picks) >= topk:
                break
        return picks

    def stats(self) -> dict:
        vids = {m["video"] for m in self.meta}
        return {"clips": len(self.meta), "videos": len(vids), "dim": self.dim}


def save_picks(picks: Iterable[dict], path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(picks), f, ensure_ascii=False, indent=1)
    return path


def load_picks(path: str) -> List[dict]:
    return json.load(open(path, encoding="utf-8"))
