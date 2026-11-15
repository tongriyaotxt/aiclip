import importlib.metadata as md

PKGS = ["torch", "torchvision", "faster-whisper", "ctranslate2", "scenedetect",
        "open-clip-torch", "gradio", "numpy", "pillow", "librosa", "soundfile", "timm"]

for p in PKGS:
    try:
        meta = md.metadata(p)
        lic = (meta.get("License") or "").strip().replace("\n", " ")[:40]
        cls = [c for c in (meta.get_all("Classifier") or []) if c.startswith("License ::")]
        spdx = cls[0].split("::")[-1].strip() if cls else lic
        print(f"{p:18} {meta['Version']:14} {spdx}")
    except Exception as exc:
        print(f"{p:18} ERR {exc}")
