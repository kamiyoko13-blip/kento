<<<<<<< HEAD
Production preparation checklist for notify_project

This document summarizes the recommended steps to convert the staging bot into a safe production deployment.

Quick summary
- Verify `staging/ninibo1127.py` is the canonical file and passes `python -m py_compile`.
- Ensure `DRY_RUN` gating is present and that top-level import has no side-effects.
- Run `run_dry.py` (or equivalent) with `DRY_RUN=1` to smoke-test behavior.
- Create a virtualenv on the server and install pinned `requirements.txt`.
- Use `deploy_swap.ps1` on Windows or `deploy_swap.sh` on Linux to promote after checks.

Files in this repo useful for production
- `requirements.txt`  -- candidate dependency list (pin versions after testing)
- `deploy_swap.ps1`  -- Windows helper to backup, swap, py_compile, and optionally import under DRY_RUN.
- `tools/convert_prints_to_logs.py` -- AST helper that converts `print()` to `log_info()` where safe.
- `run_dry.py` -- recommended local dry-run launcher (ensure it's present). If absent, create the file with the contents provided by the maintainer.

Recommended production steps (detailed)
1) Prepare a clean virtual environment

PowerShell (Windows):
```powershell
cd C:\Users\81806\Desktop\notify_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux:
```bash
cd ~/notify_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2) Verify staging file
```powershell
python -m py_compile staging\ninibo1127.py
Get-FileHash -Path staging\ninibo1127.py -Algorithm SHA256 | Format-List
```

3) Smoke test with DRY_RUN
```powershell
$env:DRY_RUN = '1'
python run_dry.py
```

4) Package for deployment (on your local machine)
```powershell
# create zip/tar with deterministic timestamping when possible
$pkg = "notify_project_$(Get-Date -Format yyyyMMddHHmmss).zip"
Compress-Archive -Path .\staging\ninibo1127.py, .\requirements.txt -DestinationPath $pkg -Force
Get-FileHash -Path $pkg -Algorithm SHA256 | Format-List
```

5) Upload and swap on server
- Windows: use WinSCP to upload package, extract, then run `deploy_swap.ps1 -Source <newfile> -Target <production path> -DryRun` to test.
- Linux: upload and then use the `deploy_swap.sh` pattern (not provided here) or manually backup and move then run `python3 -m py_compile` and an import under DRY_RUN.

6) Monitor
- After promotion, monitor `logs/notify_bot.log` and system logs.
- Ensure a process supervisor (systemd / NSSM / Task Scheduler) exists and restarts the bot on failure.

Secrets and environment
- Do NOT store API keys in Git. Use environment variables or a server-side secret store.
- Recommended variables: API_KEY, SECRET_KEY, SMTP_PASS, SMTP_USER, TO_EMAIL
- For systemd, store environment variables in an EnvironmentFile (owned by root with 600 permissions).

Support
If you want, I can create:
- a `deploy_swap.sh` for Linux
- a systemd unit file example
- a NSSM / Task Scheduler snippet for Windows

---
Generated: 2025-11-13
=======
Production preparation checklist for notify_project

This document summarizes the recommended steps to convert the staging bot into a safe production deployment.

Quick summary
- Verify `staging/ninibo1127.py` is the canonical file and passes `python -m py_compile`.
- Ensure `DRY_RUN` gating is present and that top-level import has no side-effects.
- Run `run_dry.py` (or equivalent) with `DRY_RUN=1` to smoke-test behavior.
- Create a virtualenv on the server and install pinned `requirements.txt`.
- Use `deploy_swap.ps1` on Windows or `deploy_swap.sh` on Linux to promote after checks.

Files in this repo useful for production
- `requirements.txt`  -- candidate dependency list (pin versions after testing)
- `deploy_swap.ps1`  -- Windows helper to backup, swap, py_compile, and optionally import under DRY_RUN.
- `tools/convert_prints_to_logs.py` -- AST helper that converts `print()` to `log_info()` where safe.
- `run_dry.py` -- recommended local dry-run launcher (ensure it's present). If absent, create the file with the contents provided by the maintainer.

Recommended production steps (detailed)
1) Prepare a clean virtual environment

PowerShell (Windows):
```powershell
cd C:\Users\81806\Desktop\notify_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux:
```bash
cd ~/notify_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2) Verify staging file
```powershell
python -m py_compile staging\ninibo1127.py
Get-FileHash -Path staging\ninibo1127.py -Algorithm SHA256 | Format-List
```

3) Smoke test with DRY_RUN
```powershell
$env:DRY_RUN = '1'
python run_dry.py
```

4) Package for deployment (on your local machine)
```powershell
# create zip/tar with deterministic timestamping when possible
$pkg = "notify_project_$(Get-Date -Format yyyyMMddHHmmss).zip"
Compress-Archive -Path .\staging\ninibo1127.py, .\requirements.txt -DestinationPath $pkg -Force
Get-FileHash -Path $pkg -Algorithm SHA256 | Format-List
```

5) Upload and swap on server
- Windows: use WinSCP to upload package, extract, then run `deploy_swap.ps1 -Source <newfile> -Target <production path> -DryRun` to test.
- Linux: upload and then use the `deploy_swap.sh` pattern (not provided here) or manually backup and move then run `python3 -m py_compile` and an import under DRY_RUN.

6) Monitor
- After promotion, monitor `logs/notify_bot.log` and system logs.
- Ensure a process supervisor (systemd / NSSM / Task Scheduler) exists and restarts the bot on failure.

Secrets and environment
- Do NOT store API keys in Git. Use environment variables or a server-side secret store.
- Recommended variables: API_KEY, SECRET_KEY, SMTP_PASS, SMTP_USER, TO_EMAIL
- For systemd, store environment variables in an EnvironmentFile (owned by root with 600 permissions).

Support
If you want, I can create:
- a `deploy_swap.sh` for Linux
- a systemd unit file example
- a NSSM / Task Scheduler snippet for Windows

---
Generated: 2025-11-13
>>>>>>> 74f1ab306ca4f7cbafdafeccf820148ccd40d52d
