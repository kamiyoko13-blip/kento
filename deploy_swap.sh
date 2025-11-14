#!/usr/bin/env bash
set -euo pipefail
# deploy_swap.sh
# Safely swap a new Python bot file into place on a Linux server.
# Usage: sudo ./deploy_swap.sh --source /tmp/ninibo1127.py --target /opt/notify_project/ninibo1127.py [--dry-run]

show_help() {
    cat <<'EOF'
Usage: deploy_swap.sh --source <path> --target <path> [--dry-run]

Behavior:
 - Creates a timestamped backup of the target (if exists).
 - Copies source -> target (atomic: use install or mv).
 - Runs python3 -m py_compile on the target; if it fails, restores backup and exits non-zero.
 - If --dry-run is supplied, sets DRY_RUN=1 and attempts a safe import check (no side-effects intended).
 - On success, prints the backup path and next steps.
EOF
}

SOURCE=""
TARGET=""
DRYRUN=0
PYTHON=${PYTHON:-python3}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE="$2"; shift 2;;
        --target)
            TARGET="$2"; shift 2;;
        --dry-run)
            DRYRUN=1; shift;;
        -h|--help)
            show_help; exit 0;;
        *) echo "Unknown arg: $1"; show_help; exit 2;;
    esac
done

if [[ -z "$SOURCE" || -z "$TARGET" ]]; then
    echo "source and target are required" >&2
    show_help
    exit 2
fi

if [[ ! -f "$SOURCE" ]]; then
    echo "Source file not found: $SOURCE" >&2
    exit 3
fi

TS=$(date +%Y%m%d%H%M%S)
BACKUP="$TARGET.bak.$TS"

if [[ -f "$TARGET" ]]; then
    echo "Backing up existing target: $TARGET -> $BACKUP"
    cp -p -- "$TARGET" "$BACKUP"
else
    echo "No existing target to back up."
fi

echo "Installing new file: $SOURCE -> $TARGET"
# Use install to preserve permissions and be reasonably atomic
install -m 0644 -- "$SOURCE" "$TARGET"

echo "Running syntax check: $PYTHON -m py_compile $TARGET"
if ! $PYTHON -m py_compile "$TARGET"; then
    echo "py_compile failed. Restoring backup if present." >&2
    if [[ -f "$BACKUP" ]]; then
        cp -p -- "$BACKUP" "$TARGET"
        echo "Restored backup to $TARGET"
    fi
    exit 4
fi

if [[ "$DRYRUN" -eq 1 ]]; then
    echo "Running DRY_RUN import check (DRY_RUN=1)"
    export DRY_RUN=1
    # run a small python snippet to import the module
    if ! $PYTHON - <<'PY'
import importlib, sys
try:
    import ninibo1127
    print('IMPORT_OK')
except Exception as e:
    print('IMPORT_FAIL', e)
    sys.exit(5)
PY
    then
        echo "Import check failed. Restoring backup if present." >&2
        if [[ -f "$BACKUP" ]]; then
            cp -p -- "$BACKUP" "$TARGET"
            echo "Restored backup to $TARGET"
        fi
        exit 5
    fi
    unset DRY_RUN
    echo "DRY_RUN import succeeded."
fi

echo "Swap completed successfully. Backup retained at: $BACKUP"
echo "Next: run smoke tests, then restart supervisor/service."
exit 0
