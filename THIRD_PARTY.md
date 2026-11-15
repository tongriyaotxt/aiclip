# 第三方组件与许可证 / Third-Party Notices

本项目本体采用 **MIT**（见 `LICENSE`）。以下依赖均通过 pip 安装、以库形式调用，
未修改其源码；模型权重通过 hf-mirror 下载，不随仓库分发。

## Python 依赖

| 组件 | 许可证 | 用途 |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | 张量计算 / CLIP 推理 |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | 图像预处理 |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | MIT | 语音识别封装 |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | MIT | Whisper 推理引擎 |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | BSD-3-Clause | 镜头切分 |
| [open_clip](https://github.com/mlfoundations/open_clip) | MIT | CLIP 模型与分词器 |
| [OpenCV](https://github.com/opencv/opencv-python) | Apache-2.0 | 视频帧读取 |
| [Gradio](https://github.com/gradio-app/gradio) | Apache-2.0 | 网页界面 |
| [librosa](https://github.com/librosa/librosa) | ISC | 节拍检测 |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | 数值计算 |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU | 图像处理 |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause | 音频读写 |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | open_clip 依赖 |

## 模型权重

| 模型 | 许可证 | 说明 |
|---|---|---|
| [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | MIT | CTranslate2 格式的 Whisper 权重 |
| [laion/CLIP-ViT-B-32-laion2B-s34b-b79K](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34b-b79K) | MIT | 图文对比模型 |

## 外部程序

| 程序 | 许可证 | 说明 |
|---|---|---|
| [FFmpeg](https://ffmpeg.org/legal.html) | LGPL-2.1+ / GPL（取决于构建） | **仅以子进程方式调用**，未链接、未随仓库分发。本机使用的是 gyan.dev full build（GPLv3）。若要再分发，请自行确认所用构建的许可条款。 |
| Windows SAPI（`System.Speech`） | 随 Windows 提供 | 配音功能的离线 TTS 引擎 |

## 分发注意事项

1. 若你要打包成 exe / 安装包分发，**不要**把 FFmpeg 二进制直接塞进包里，除非同时满足其 GPL 条款（提供源码等）。
2. 模型权重体积大（461MB + 577MB），建议首次运行时由 `aiclip/models.py` 下载，而不是入库。
3. 本项目 MIT 许可不覆盖上述第三方组件，它们各自遵循上表许可证。
