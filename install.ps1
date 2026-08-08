$ErrorActionPreference = "Stop"
$env:UV_HTTP_TIMEOUT = "300"

function Invoke-Step($msg, $scriptblock) {
    Write-Host "==> $msg" -ForegroundColor Cyan
    & $scriptblock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $msg -- fix the issue above, then just re-run this script." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Invoke-Step "Creating venv (safe if it already exists)" {
    if (-not (Test-Path ".venv")) { uv venv --python 3.12 }
}

Invoke-Step "Torch CPU triad" {
    uv pip install torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
}

Invoke-Step "Core requirements" {
    uv pip install -r requirements_cpu.txt
}

Invoke-Step "Local indextts package" {
    uv pip install -e . -c constraints.txt
}

Invoke-Step "Dependency check" {
    uv pip check
}

Invoke-Step "NLTK data" {
    .\.venv\Scripts\python.exe -m nltk.downloader averaged_perceptron_tagger
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
