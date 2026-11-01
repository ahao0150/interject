# 启动 interject 演示服务
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:INTERJECT_MODEL) {
    $env:INTERJECT_MODEL = ".\models\qwen2.5-1.5b"
}
$port = if ($env:INTERJECT_PORT) { $env:INTERJECT_PORT } else { "8100" }

Write-Host "模型: $env:INTERJECT_MODEL"
Write-Host "地址: http://127.0.0.1:$port/chat"
python -m interject.server --port $port