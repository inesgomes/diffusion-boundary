# RunPod S3 wrapper for the network volume (EU-RO-1).
# Loads S3 creds from .env (AWS_S3_USER / AWS_S3_KEY) and injects the endpoint+region,
# so you can run plain `aws s3` subcommands without any flags.
#
# Examples:
#   .\rp-s3.ps1 ls s3://jl3evn485h/dfst_files/logs/
#   .\rp-s3.ps1 sync s3://jl3evn485h/dfst_files/logs ./results
#   .\rp-s3.ps1 cp s3://jl3evn485h/dfst_files/logs/<run-id>/results_synthetic.parquet .
$ErrorActionPreference = "Stop"

$envFile = Join-Path $PSScriptRoot ".env"
$u = ((Select-String -Path $envFile -Pattern '^AWS_S3_USER=').Line -replace '^AWS_S3_USER=','').Trim()
$k = ((Select-String -Path $envFile -Pattern '^AWS_S3_KEY='  ).Line -replace '^AWS_S3_KEY=','' ).Trim()
if (-not $u -or -not $k) { throw "AWS_S3_USER / AWS_S3_KEY not found in $envFile" }
$env:AWS_ACCESS_KEY_ID     = $u
$env:AWS_SECRET_ACCESS_KEY = $k

$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }

& $aws s3 @args --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io
