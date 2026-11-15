"""AIClip 网页界面（Gradio）。"""
from __future__ import annotations

import json
import os
import traceback

from . import ff, pipeline, vision

BASE = os.path.abspath("out")


def _wrap(fn):
    def inner(*args):
        try:
            return fn(*args)
        except Exception as exc:
            return f"[错误] {type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
    return inner


def _rough(video, model, gap, burn):
    if not video:
        return None, None, "请先上传视频"
    rep = pipeline.rough_cut(
        video, "out/ui_rough", model=model, max_gap=gap, burn=burn, progress=print
    )
    return (
        rep["files"]["video"],
        rep["files"]["srt"],
        json.dumps(rep, ensure_ascii=False, indent=2),
    )


def _subs(video, model, burn):
    if not video:
        return None, None, "请先上传视频"
    rep = pipeline.subtitles(video, "out/ui_subs", model=model, burn=burn, progress=print)
    return rep["video"], rep["srt"], json.dumps(rep, ensure_ascii=False, indent=2)


def _index(folder, thr):
    if not folder or not os.path.exists(folder):
        return "目录不存在"
    rep = pipeline.build_index(folder, "out/ui_idx", threshold=float(thr), progress=print)
    return json.dumps(rep, ensure_ascii=False, indent=2)


def _search(query, topk):
    if not os.path.exists("out/ui_idx/meta.json"):
        return [], "还没有索引，先建索引"
    picks = pipeline.search("out/ui_idx", query, topk=int(topk), out="out/ui_picks.json")
    rows = [[os.path.basename(p["video"]), f"{p['start']:.2f}", f"{p['end']:.2f}", p["score"]]
            for p in picks]
    return rows, json.dumps(picks, ensure_ascii=False, indent=1)


def _assemble(vertical):
    if not os.path.exists("out/ui_picks.json"):
        return None, "没有 picks.json"
    out = pipeline.assemble(
        "out/ui_picks.json", "out/ui_assembly.mp4",
        vertical=None if vertical == "none" else vertical, progress=print,
    )
    return out, out


def _shorts(video, n, length, vertical, burn):
    if not video:
        return [], "请先上传视频"
    rep = pipeline.auto_shorts(
        video, "out/ui_shorts", n=int(n), length=float(length),
        vertical=vertical, burn=burn, progress=print,
    )
    return [c["video"] for c in rep["clips"]], json.dumps(rep, ensure_ascii=False, indent=2)


def _vertical(video, mode):
    if not video:
        return None
    return ff.vertical(video, "out/ui_vertical.mp4", mode=mode)


def _doctor():
    from .cli import build_parser  # noqa
    import platform

    lines = [f"python {platform.python_version()}"]
    lines.append(f"ffmpeg {ff.FFMPEG}")
    try:
        import torch

        lines.append(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"gpu {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        lines.append(f"torch ERR {exc}")
    for m in ("faster_whisper", "scenedetect", "open_clip", "gradio"):
        try:
            mod = __import__(m)
            lines.append(f"{m} {getattr(mod, '__version__', 'ok')}")
        except Exception as exc:
            lines.append(f"{m} ERR {exc}")
    return "\n".join(lines)


def build():
    import gradio as gr

    with gr.Blocks(title="AIClip · 本地 AI 剪辑") as demo:
        gr.Markdown(
            "# AIClip · 本地 AI 视频剪辑\n"
            "转写 → 剪气口 → 字幕 → 竖屏切片 → 素材检索，全部在本机跑，不上传云端。"
        )
        with gr.Tab("① 口播粗剪"):
            with gr.Row():
                v1 = gr.Video(label="原始素材", sources=["upload"])
                o1 = gr.Video(label="粗剪成品")
            with gr.Row():
                m1 = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"],
                                 value="small", label="Whisper 模型")
                g1 = gr.Slider(0.2, 1.5, value=0.55, step=0.05, label="最大保留停顿(秒)")
                b1 = gr.Checkbox(value=True, label="烧录字幕")
            btn1 = gr.Button("开始粗剪", variant="primary")
            s1 = gr.File(label="字幕 SRT")
            t1 = gr.Textbox(label="报告", lines=12)
            btn1.click(_wrap(_rough), [v1, m1, g1, b1], [o1, s1, t1])

        with gr.Tab("② 字幕"):
            with gr.Row():
                v2 = gr.Video(label="视频", sources=["upload"])
                o2 = gr.Video(label="成品")
            with gr.Row():
                m2 = gr.Dropdown(["tiny", "base", "small", "medium", "large-v3"],
                                 value="small", label="模型")
                b2 = gr.Checkbox(value=True, label="烧录（否则软字幕）")
            btn2 = gr.Button("生成字幕", variant="primary")
            s2 = gr.File(label="SRT")
            t2 = gr.Textbox(label="结果", lines=8)
            btn2.click(_wrap(_subs), [v2, m2, b2], [o2, s2, t2])

        with gr.Tab("③ 素材检索"):
            with gr.Row():
                d3 = gr.Textbox(label="素材目录", value=os.path.abspath("footage"))
                th3 = gr.Slider(10, 60, value=27, step=1, label="切镜灵敏度阈值")
            btn3a = gr.Button("建立索引")
            t3 = gr.Textbox(label="索引状态", lines=6)
            with gr.Row():
                q3 = gr.Textbox(label="用一句话找素材", value="海边日落 航拍 空镜")
                k3 = gr.Number(value=8, label="返回条数", precision=0)
            btn3b = gr.Button("检索", variant="primary")
            gal3 = gr.Dataframe(headers=["素材", "起", "止", "相似度"], label="命中镜头")
            j3 = gr.Textbox(label="picks.json", lines=8)
            with gr.Row():
                vm3 = gr.Dropdown(["none", "crop", "pad", "fit"], value="none", label="竖向")
                btn3c = gr.Button("按命中拼接成片")
            o3 = gr.Video(label="拼接结果")
            f3 = gr.File(label="输出文件")
            btn3a.click(_wrap(_index), [d3, th3], [t3])
            btn3b.click(_wrap(_search), [q3, k3], [gal3, j3])
            btn3c.click(_wrap(_assemble), [vm3], [o3, f3])

        with gr.Tab("④ 竖屏切片"):
            with gr.Row():
                v4 = gr.Video(label="长视频", sources=["upload"])
                gal4 = gr.Gallery(label="切片成品", columns=3, height=420)
            with gr.Row():
                n4 = gr.Slider(1, 10, value=3, step=1, label="条数")
                l4 = gr.Slider(10, 120, value=45, step=5, label="单条时长(秒)")
                vm4 = gr.Dropdown(["crop", "pad", "fit"], value="pad", label="竖向方式")
                b4 = gr.Checkbox(value=True, label="烧字幕")
            btn4 = gr.Button("生成切片", variant="primary")
            t4 = gr.Textbox(label="结果", lines=10)
            btn4.click(_wrap(_shorts), [v4, n4, l4, vm4, b4], [gal4, t4])

        with gr.Tab("⑤ 竖屏转换 / 体检"):
            with gr.Row():
                v5 = gr.Video(label="横屏视频", sources=["upload"])
                o5 = gr.Video(label="竖屏成品")
            vm5 = gr.Dropdown(["pad", "crop", "fit"], value="pad", label="方式")
            btn5 = gr.Button("转竖屏")
            btn5.click(_wrap(_vertical), [v5, vm5], [o5])
            gr.Markdown("---")
            btn6 = gr.Button("环境体检")
            t6 = gr.Textbox(label="环境", lines=14)
            btn6.click(_wrap(_doctor), None, [t6])
    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False,
           open_browser: bool = True) -> None:
    import gradio as gr

    os.makedirs(BASE, exist_ok=True)
    demo = build()
    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share, inbrowser=open_browser,
                show_error=True, theme=gr.themes.Soft())
