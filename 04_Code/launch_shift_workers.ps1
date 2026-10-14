# Starts K follow-up workers in the background (Windows). Each worker runs its share of the scenario cells and
# skips cells that already exist, so the script can be re-run after an interruption.
#   powershell -ExecutionPolicy Bypass -File launch_shift_workers.ps1 [-K 6] [-Python python]
param([int]$K = 6, [string]$Python = "python")
$code = $PSScriptRoot
$root = Split-Path $code -Parent
$logs = Join-Path $root "06_Experiments\shift\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
for ($w = 0; $w -lt $K; $w++) {
    Start-Process -FilePath $Python -ArgumentList "run_shift.py curves --worker $w --nworkers $K" `
        -WorkingDirectory $code -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "worker_$w.out") -RedirectStandardError (Join-Path $logs "worker_$w.err")
}
Write-Output "started $K workers; logs in $logs"
