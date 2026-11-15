"""命令行入口：aiclip <command>。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, asr, audio, edit, ff, pipeline, text, vision


def _p(msg: str) -> None:
    print(msg, flush=True)


def _dump(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_doctor(a):
    import platform

    from . import models

    info = {"python": platform.python_version(), "ffmpeg": ff.FFMPEG}
    try:
        info["ffmpeg_version"] = ff.run([ff.FFMPEG, "-version"]).splitlines()[0][:60]
    except Exception as exc:
        info["ffmpeg_version"] = f"ERR {exc}"
    info.update(models.cuda_info())
    for m in ("faster_whisper", "scenedetect", "open_clip", "gradio", "librosa", "cv2", "ctranslate2"):
        try:
            mod = __import__(m)
            info[m] = getattr(mod, "__version__", "ok")
        except Exception as exc:
            info[m] = f"ERR {exc}"
    _dump(info)


def cmd_probe(a):
    _dump(ff.probe(a.file))


def cmd_transcribe(a):
    tr = asr.transcribe(a.file, model_size=a.model, language=a.lang, device=a.device)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump(tr, f, ensure_ascii=False, indent=1)
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in tr["segments"]]
    text.write_srt(segs, os.path.join(a.outdir, "sub.srt"))
    text.write_vtt(segs, os.path.join(a.outdir, "sub.vtt"))
    _p(f"语言={tr['language']} 置信={tr['language_probability']} 句数={len(segs)}")
    _p(f"输出: {os.path.abspath(a.outdir)}")


def cmd_rough(a):
    rep = pipeline.rough_cut(
        a.file, a.outdir, model=a.model, lang=a.lang, max_gap=a.max_gap,
        pad=a.pad, drop_fillers=not a.keep_fillers, burn=not a.no_burn,
        fps=a.fps, crf=a.crf, mode=a.mode, noise_db=a.noise_db, device=a.device,
        progress=_p,
    )
    _dump(rep)


def cmd_dub(a):
    _dump(pipeline.dub(a.video, a.script, a.out, voice=a.voice, rate=a.rate,
                       gap=a.gap, progress=_p))


def cmd_voices(a):
    from . import tts

    for v in tts.available_voices():
        _p(v)


def cmd_subs(a):
    _dump(pipeline.subtitles(a.file, a.outdir, model=a.model, lang=a.lang,
                            burn=not a.soft, crf=a.crf, device=a.device, progress=_p))


def cmd_index(a):
    _dump(pipeline.build_index(a.source, a.out, threshold=a.threshold,
                               device=a.device, progress=_p))


def cmd_search(a):
    picks = pipeline.search(a.index, a.query, topk=a.topk, out=a.out)
    for p in picks:
        _p(f"{p['score']:.3f}  {os.path.basename(p['video'])}  {p['start']:.2f}-{p['end']:.2f}")


def cmd_assemble(a):
    _dump({"output": pipeline.assemble(a.picks, a.out, vertical=a.vertical,
                                      crf=a.crf, progress=_p)})


def cmd_shorts(a):
    _dump(pipeline.auto_shorts(a.file, a.outdir, n=a.n, length=a.length,
                               model=a.model, lang=a.lang, vertical=a.vertical,
                               burn=not a.no_burn, crf=a.crf, device=a.device,
                               progress=_p))


def cmd_vertical(a):
    out = a.out or os.path.splitext(a.file)[0] + f"_{a.mode}.mp4"
    _dump({"output": ff.vertical(a.file, out, mode=a.mode, crf=a.crf)})


def cmd_beats(a):
    wav = a.file
    if not wav.lower().endswith(".wav"):
        wav = os.path.join(os.path.dirname(os.path.abspath(a.file)) or ".", "_aiclip_beat.wav")
        ff.extract_audio(a.file, wav, sr=22050)
    res = audio.beats(wav)
    _dump({k: v for k, v in res.items() if k != "beats"} | {"beats_head": res["beats"][:10]})
    if a.out:
        audio.save_json(res, a.out)
        _p(f"节拍表: {a.out}")


def cmd_pipeline(a):
    out = a.outdir
    os.makedirs(out, exist_ok=True)
    _p("== 1/3 粗剪 ==")
    rep = pipeline.rough_cut(a.file, os.path.join(out, "rough"), model=a.model,
                             lang=a.lang, progress=_p)
    _p("== 2/3 切片 ==")
    try:
        sh = pipeline.auto_shorts(a.file, os.path.join(out, "shorts"), n=a.n,
                                  length=a.length, model=a.model, lang=a.lang, progress=_p)
    except Exception as exc:
        sh = {"error": str(exc)}
    _p("== 3/3 汇总 ==")
    summary = {"rough": rep, "shorts": sh}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    _dump({"rough_output": rep["output"], "shorts": [c["video"] for c in sh.get("clips", [])],
           "summary": os.path.abspath(os.path.join(out, "summary.json"))})


def cmd_ui(a):
    from .ui import launch

    launch(host=a.host, port=a.port, share=a.share, open_browser=not a.no_browser)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="aiclip", description="AIClip — 本地 AI 视频剪辑流水线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"aiclip {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="体检：ffmpeg / torch / 依赖")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("probe", help="媒体信息")
    s.add_argument("file")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("transcribe", help="转写并输出 SRT/VTT/JSON")
    s.add_argument("file")
    s.add_argument("--model", default="small")
    s.add_argument("--lang", default="zh")
    s.add_argument("--outdir", default="out/transcript")
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    s.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("rough", help="口播粗剪：剪气口/去废话 + 字幕")
    s.add_argument("file")
    s.add_argument("--outdir", default="out/rough")
    s.add_argument("--model", default="small")
    s.add_argument("--lang", default="zh")
    s.add_argument("--max-gap", type=float, default=0.55)
    s.add_argument("--pad", type=float, default=0.08)
    s.add_argument("--keep-fillers", action="store_true")
    s.add_argument("--no-burn", action="store_true")
    s.add_argument("--mode", default="asr", choices=["asr", "energy"],
                   help="asr=按语音剪；energy=按音量剪（无语音素材）")
    s.add_argument("--noise-db", type=float, default=-34.0)
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--crf", type=int, default=20)
    s.set_defaults(func=cmd_rough)

    s = sub.add_parser("dub", help="用文稿配音替换原声")
    s.add_argument("video")
    s.add_argument("script", help="文稿 txt，一行一句")
    s.add_argument("--out", default="out/dub.mp4")
    s.add_argument("--voice", default=None)
    s.add_argument("--rate", type=int, default=0, help="-10~10，越大越快")
    s.add_argument("--gap", type=float, default=0.25)
    s.set_defaults(func=cmd_dub)

    s = sub.add_parser("voices", help="列出系统可用配音音色")
    s.set_defaults(func=cmd_voices)

    s = sub.add_parser("subs", help="整片字幕（烧录或软字幕）")
    s.add_argument("file")
    s.add_argument("--outdir", default="out/subs")
    s.add_argument("--model", default="small")
    s.add_argument("--lang", default="zh")
    s.add_argument("--soft", action="store_true")
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    s.add_argument("--crf", type=int, default=20)
    s.set_defaults(func=cmd_subs)

    s = sub.add_parser("index", help="建素材索引（镜头切分 + CLIP）")
    s.add_argument("source")
    s.add_argument("--out", default="out/idx")
    s.add_argument("--threshold", type=float, default=27.0)
    s.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="自然语言检索素材")
    s.add_argument("index")
    s.add_argument("query")
    s.add_argument("--topk", type=int, default=8)
    s.add_argument("--out", default="out/picks.json")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("assemble", help="按 picks.json 拼成片")
    s.add_argument("picks")
    s.add_argument("--out", default="out/assembly.mp4")
    s.add_argument("--vertical", choices=["crop", "pad", "fit"], default=None)
    s.add_argument("--crf", type=int, default=20)
    s.set_defaults(func=cmd_assemble)

    s = sub.add_parser("shorts", help="长视频→N 条竖屏切片")
    s.add_argument("file")
    s.add_argument("--outdir", default="out/shorts")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--length", type=float, default=45.0)
    s.add_argument("--model", default="small")
    s.add_argument("--lang", default="zh")
    s.add_argument("--vertical", default="pad", choices=["crop", "pad", "fit", "none"])
    s.add_argument("--no-burn", action="store_true")
    s.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    s.add_argument("--crf", type=int, default=20)
    s.set_defaults(func=cmd_shorts)

    s = sub.add_parser("vertical", help="竖屏重构")
    s.add_argument("file")
    s.add_argument("--mode", default="pad", choices=["crop", "pad", "fit"])
    s.add_argument("--out", default=None)
    s.add_argument("--crf", type=int, default=20)
    s.set_defaults(func=cmd_vertical)

    s = sub.add_parser("beats", help="BPM/节拍检测")
    s.add_argument("file")
    s.add_argument("--out", default="out/beats.json")
    s.set_defaults(func=cmd_beats)

    s = sub.add_parser("pipeline", help="一条命令出成品（粗剪+切片）")
    s.add_argument("file")
    s.add_argument("--outdir", default="out/full")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--length", type=float, default=45.0)
    s.add_argument("--model", default="small")
    s.add_argument("--lang", default="zh")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("ui", help="启动网页界面")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--share", action="store_true")
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_ui)

    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n已中断")
    except Exception as exc:
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("AICLIP_DEBUG"):
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
