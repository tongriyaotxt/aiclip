$ErrorActionPreference = 'Continue'
$root = 'F:\shit\for-video-processing'
Set-Location $root
$py = "$root\.venv\Scripts\python.exe"

# 说明：
#  1) 本机全局 Python 已具备 torch / torchvision / opencv / faster-whisper / librosa，
#     venv 用 --system-site-packages 复用；CUDA 版 torch 用 wheels/ 里的本地包，不联网下载。
#  2) 官方 PyPI 与清华源在本网络下不可用，统一走阿里云源。
#  3) 模型权重不走 huggingface.co，走 hf-mirror.com（aiclip/models.py 内自动处理）。

$idx = @(
  '--index-url','https://mirrors.aliyun.com/pypi/simple/',
  '--trusted-host','mirrors.aliyun.com',
  '--retries','8','--timeout','60','--no-warn-script-location'
)

Write-Host "=== 1/5 创建 venv（复用全局包） ==="
if (Test-Path "$root\.venv") { Remove-Item -Recurse -Force "$root\.venv" }
python -m venv --system-site-packages "$root\.venv"

Write-Host "=== 2/5 安装缺失依赖 ==="
& $py -m pip install @idx --no-deps scenedetect
& $py -m pip install @idx open_clip_torch ftfy regex timm
& $py -m pip install @idx gradio

Write-Host "=== 3/4 CUDA torch（用本地 wheels，不重新下载） ==="
$tw = Get-ChildItem "$root\wheels\torch-*-cu128-cp313-cp313-win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
$vw = Get-ChildItem "$root\wheels\torchvision-*-cu128-cp313-cp313-win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($tw -and $vw) {
    & $py -m pip install --no-deps --no-warn-script-location $tw.FullName $vw.FullName
} else {
    Write-Host "（wheels/ 里没有 cu128 包，保持 CPU 版 torch）"
}

Write-Host "=== 4/5 校验 ==="
& $py -c "import torch,torchvision,cv2,faster_whisper,librosa,scenedetect,open_clip,gradio,numpy;print('deps OK');print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('gradio',gradio.__version__)"

Write-Host "=== 5/5 预热模型（已存在则跳过，不重复下载） ==="
& $py -c "from aiclip import models; print(models.ensure_all('small', need_clip=True)); print(models.cuda_info())"

Write-Host "=== 完成，可运行：.\启动界面.cmd 或 .\aiclip.cmd doctor ==="
