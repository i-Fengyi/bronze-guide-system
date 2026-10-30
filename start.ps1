$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到虚拟环境。请先按照 README 创建 .venv 并安装 requirements.txt。"
}

& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
