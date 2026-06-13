$ErrorActionPreference = 'Continue'
$base = 'C:\Users\81806\AppData\Local\Programs\WinSCP'
$logDir = Join-Path $base 'logs'
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$runLogDir = Join-Path $logDir 'remote_watch_status'
if (-not (Test-Path $runLogDir)) {
    New-Item -ItemType Directory -Path $runLogDir | Out-Null
}

$ts = Get-Date
$tsLabel = $ts.ToString('yyyy-MM-dd HH:mm:ss')
$runFile = Join-Path $runLogDir ("remote_watch_status_" + $ts.ToString('yyyyMMdd_HHmmss') + ".log")
$latestFile = Join-Path $logDir 'remote_watch_status.latest.log'
$signalFile = Join-Path $logDir 'remote_watch_signal.latest.log'

$tmpOut = [System.IO.Path]::GetTempFileName()
$tmpErr = [System.IO.Path]::GetTempFileName()
$winscpExe = Join-Path $base 'WinSCP.com'
$winscpScript = Join-Path $base 'winscp_remote_watch_status.txt'
$args = @('/ini=nul', "/script=$winscpScript")

# Prevent indefinite hangs by running WinSCP in a child process with timeout.
$p = Start-Process -FilePath $winscpExe -ArgumentList $args -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr -NoNewWindow -PassThru
$finished = $p.WaitForExit(90000)
if (-not $finished) {
    try { $p.Kill() } catch {}
}

# WinSCP.com output is CP932 on this host. Decode accordingly, then save as UTF-8.
$cp932 = [System.Text.Encoding]::GetEncoding(932)
$rawOut = if (Test-Path $tmpOut) { $cp932.GetString([System.IO.File]::ReadAllBytes($tmpOut)) } else { '' }
$rawErr = if (Test-Path $tmpErr) { $cp932.GetString([System.IO.File]::ReadAllBytes($tmpErr)) } else { '' }

$statusLine = if ($finished) {
    "watch_exit_code=" + $p.ExitCode
}
else {
    'watch_timeout=1'
}

$raw = $rawOut
if ($rawErr.Trim().Length -gt 0) {
    $raw += "`r`n[stderr]`r`n" + $rawErr
}

$content = "===== $tsLabel =====`r`n" + $statusLine + "`r`n" + $raw
Set-Content -Path $runFile -Value $content -Encoding utf8
Set-Content -Path $latestFile -Value $content -Encoding utf8

# Extract trading-relevant lines for quick glance.
$signals = ($content -split "`r?`n") | Where-Object {
    $_ -match 'DRY_RUN=|ninibo1127.py|trade_decision|BUY|SELL|ホールド|買い禁止|売り|ループ|ERROR|Traceback'
}
if ($signals.Count -eq 0) {
    $signals = @('no signal lines matched in this run')
}
Set-Content -Path $signalFile -Value ($signals -join "`r`n") -Encoding utf8

try { Remove-Item $tmpOut -Force -ErrorAction SilentlyContinue } catch {}
try { Remove-Item $tmpErr -Force -ErrorAction SilentlyContinue } catch {}
