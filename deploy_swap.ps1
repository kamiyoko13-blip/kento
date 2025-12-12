<<<<<<< HEAD
# deploy_swap.ps1
# PowerShell helper to safely swap a new version of a bot script into place.
#
# Usage (example):
#   .\deploy_swap.ps1 -Source "C:\tmp\ninibo1127.py" -Target "C:\Users\81806\Desktop\notify_project\staging\ninibo1127.py" -DryRun
#
# Behavior:
# - Creates a timestamped backup of the current target (if present).
# - Copies Source -> Target.
# - Runs python -m py_compile on the target; if it fails, restores backup and exits non-zero.
# - If -DryRun is supplied, sets DRY_RUN=1 and attempts a safe import check (no side-effects intended).
# - On success, prints instructions to promote to production (upload / restart service).

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Source,

    [Parameter(Mandatory=$true)]
    [string]$Target,

    [switch]$DryRun,

    [string]$Python = 'python'
)

function Timestamp() { Get-Date -Format yyyyMMddHHmmss }

if (-not (Test-Path $Source)) {
    Write-Error "Source file not found: $Source"
    exit 2
}

$ts = Timestamp()
$backup = "$Target.bak.$ts"
try {
    if (Test-Path $Target) {
        Copy-Item -Path $Target -Destination $backup -Force
        Write-Output "Backup created: $backup"
    } else {
        Write-Output "No existing target to back up."
    }
    Copy-Item -Path $Source -Destination $Target -Force
    Write-Output "Copied $Source -> $Target"

    # Syntax check
    & $Python -m py_compile $Target
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "py_compile failed for $Target. Restoring backup (if exists)"
        if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
        exit 3
    }
    Write-Output "py_compile OK"

    if ($DryRun.IsPresent) {
        Write-Output "Running DRY_RUN import test (DRY_RUN=1) — this should avoid side-effects."
        $env:DRY_RUN = '1'
        try {
            # attempt a safe import check
            & $Python - <<'PY'
import importlib, sys
try:
    import ninibo1127
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_FAIL', e)
    sys.exit(4)
PY
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Import check failed (exit $LASTEXITCODE). Restoring backup."
                if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
                exit 4
            }
            Write-Output "DRY_RUN import succeeded."
        } catch {
            Write-Warning "Exception during import check: $_. Restoring backup."
            if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
            exit 5
        } finally {
            Remove-Item Env:DRY_RUN -ErrorAction SilentlyContinue
        }
    }

    Write-Output "Swap completed successfully. If this is a staging promotion, consider running final smoke tests and then uploading to production."
    Write-Output "Backup retained at: $backup"
    exit 0
} catch {
    Write-Error "Unexpected error: $_"
    if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
    exit 9
}
=======
# deploy_swap.ps1
# PowerShell helper to safely swap a new version of a bot script into place.
#
# Usage (example):
#   .\deploy_swap.ps1 -Source "C:\tmp\ninibo1127.py" -Target "C:\Users\81806\Desktop\notify_project\staging\ninibo1127.py" -DryRun
#
# Behavior:
# - Creates a timestamped backup of the current target (if present).
# - Copies Source -> Target.
# - Runs python -m py_compile on the target; if it fails, restores backup and exits non-zero.
# - If -DryRun is supplied, sets DRY_RUN=1 and attempts a safe import check (no side-effects intended).
# - On success, prints instructions to promote to production (upload / restart service).

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Source,

    [Parameter(Mandatory=$true)]
    [string]$Target,

    [switch]$DryRun,

    [string]$Python = 'python'
)

function Timestamp() { Get-Date -Format yyyyMMddHHmmss }

if (-not (Test-Path $Source)) {
    Write-Error "Source file not found: $Source"
    exit 2
}

$ts = Timestamp()
$backup = "$Target.bak.$ts"
try {
    if (Test-Path $Target) {
        Copy-Item -Path $Target -Destination $backup -Force
        Write-Output "Backup created: $backup"
    } else {
        Write-Output "No existing target to back up."
    }
    Copy-Item -Path $Source -Destination $Target -Force
    Write-Output "Copied $Source -> $Target"

    # Syntax check
    & $Python -m py_compile $Target
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "py_compile failed for $Target. Restoring backup (if exists)"
        if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
        exit 3
    }
    Write-Output "py_compile OK"

    if ($DryRun.IsPresent) {
        Write-Output "Running DRY_RUN import test (DRY_RUN=1) — this should avoid side-effects."
        $env:DRY_RUN = '1'
        try {
            # attempt a safe import check
            & $Python - <<'PY'
import importlib, sys
try:
    import ninibo1127
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_FAIL', e)
    sys.exit(4)
PY
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Import check failed (exit $LASTEXITCODE). Restoring backup."
                if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
                exit 4
            }
            Write-Output "DRY_RUN import succeeded."
        } catch {
            Write-Warning "Exception during import check: $_. Restoring backup."
            if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
            exit 5
        } finally {
            Remove-Item Env:DRY_RUN -ErrorAction SilentlyContinue
        }
    }

    Write-Output "Swap completed successfully. If this is a staging promotion, consider running final smoke tests and then uploading to production."
    Write-Output "Backup retained at: $backup"
    exit 0
} catch {
    Write-Error "Unexpected error: $_"
    if (Test-Path $backup) { Copy-Item -Path $backup -Destination $Target -Force }
    exit 9
}
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
