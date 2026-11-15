$ErrorActionPreference = 'Continue'
$root = 'F:\shit\for-video-processing'
$wh = "$root\wheels"
New-Item -ItemType Directory -Force -Path $wh | Out-Null

$torch = 'https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl'
$tv    = 'https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp313-cp313-win_amd64.whl'

function Get-Big($url, $out) {
    for ($i = 1; $i -le 15; $i++) {
        Write-Host "=== [$i] $([IO.Path]::GetFileName($out)) ==="
        & curl.exe -L -C - --retry 20 --retry-delay 2 --retry-all-errors --connect-timeout 20 --max-time 1800 -o $out $url
        if ($LASTEXITCODE -eq 0) { break }
        Start-Sleep -Seconds 3
    }
    $len = (Get-Item $out -ErrorAction SilentlyContinue).Length
    Write-Host "size $([math]::Round($len/1MB,1)) MB  $out"
}

Get-Big $torch "$wh\torch-2.11.0-cu128.whl"
Get-Big $tv "$wh\torchvision-0.26.0-cu128.whl"
Write-Host "=== DOWNLOAD DONE ==="
