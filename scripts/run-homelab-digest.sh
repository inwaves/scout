#!/usr/bin/env bash
set -Eeuo pipefail

PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCOUT_ENV_FILE="${SCOUT_ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$SCOUT_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$SCOUT_ENV_FILE"
  set +a
fi

CONFIG_FILE="${SCOUT_CONFIG_FILE:-config.example.yml}"
KB_DIR="${SCOUT_KB_DIR:-$ROOT_DIR/../kb}"
LOCK_FILE="${SCOUT_LOCK_FILE:-/tmp/scout-digest.lock}"
LOG_DIR="${SCOUT_LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${SCOUT_LOG_FILE:-$LOG_DIR/scout-$(date -u +%Y-%m-%d).log}"
PYTHON_BIN="${SCOUT_PYTHON:-$ROOT_DIR/.venv/bin/python}"
MODE="${MODE:-${SCOUT_MODE:-daily}}"
SCOUT_PUSH="${SCOUT_PUSH:-1}"
SCOUT_GIT_PULL="${SCOUT_GIT_PULL:-1}"
SCOUT_INSTALL_DEPS="${SCOUT_INSTALL_DEPS:-0}"
GIT_AUTHOR_NAME="${SCOUT_GIT_AUTHOR_NAME:-Scout Bot}"
GIT_AUTHOR_EMAIL="${SCOUT_GIT_AUTHOR_EMAIL:-scout[bot]@users.noreply.github.com}"

mkdir -p "$LOG_DIR" "$(dirname "$LOCK_FILE")"
exec >> "$LOG_FILE" 2>&1

finish() {
  local status=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Scout runner finished with status $status"
}
trap finish EXIT

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Scout runner starting in $ROOT_DIR"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another Scout run is already active; exiting."
    exit 0
  fi
else
  LOCK_DIR="${LOCK_FILE}.d"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another Scout run appears to be active; exiting."
    exit 0
  fi

  finish_with_lock_dir() {
    local status=$?
    rmdir "$LOCK_DIR" || true
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Scout runner finished with status $status"
  }
  trap finish_with_lock_dir EXIT
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ "$SCOUT_GIT_PULL" == "1" ]]; then
  git pull --ff-only
  if [[ -d "$KB_DIR/.git" ]]; then
    git -C "$KB_DIR" pull --ff-only
  elif [[ -n "${KB_CLONE_URL:-}" ]]; then
    git clone --depth 1 "$KB_CLONE_URL" "$KB_DIR"
  fi
fi

if [[ "$SCOUT_INSTALL_DEPS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

DOW="$(date -u +%u)"
if [[ "$MODE" == "daily" && "$DOW" == "1" ]]; then
  MODE="weekly"
fi
echo "Running in mode: $MODE"

cmd=("$PYTHON_BIN" -m paper_scout --config "$CONFIG_FILE" -v run)
is_dry_run=0
case "$MODE" in
  daily)
    ;;
  weekly)
    cmd+=(--weekly)
    ;;
  daily-dry-run)
    cmd+=(--dry-run)
    is_dry_run=1
    ;;
  weekly-dry-run)
    cmd+=(--weekly --dry-run)
    is_dry_run=1
    ;;
  *)
    echo "Unknown MODE: $MODE"
    exit 2
    ;;
esac

"${cmd[@]}"

if [[ "$is_dry_run" == "1" ]]; then
  echo "Dry run complete; skipping git commits and pushes."
  exit 0
fi

if [[ -d "$KB_DIR/.git" ]]; then
  git -C "$KB_DIR" config user.name "$GIT_AUTHOR_NAME"
  git -C "$KB_DIR" config user.email "$GIT_AUTHOR_EMAIL"
  git -C "$KB_DIR" add papers/ || true
  if ! git -C "$KB_DIR" diff --cached --quiet; then
    git -C "$KB_DIR" commit -m "Scout: new paper notes $(date -u +%Y-%m-%d)"
    if [[ "$SCOUT_PUSH" == "1" ]]; then
      git -C "$KB_DIR" pull --rebase
      git -C "$KB_DIR" push
    fi
  else
    echo "No KB note changes to commit."
  fi
else
  echo "KB repo not present at $KB_DIR; skipping KB push."
fi

git config user.name "$GIT_AUTHOR_NAME"
git config user.email "$GIT_AUTHOR_EMAIL"
git add digests/ last_run.json || true
if ! git diff --cached --quiet; then
  git commit -m "Scout digest $(date -u +%Y-%m-%d)"
  if [[ "$SCOUT_PUSH" == "1" ]]; then
    git pull --rebase
    git push
  fi
else
  echo "No Scout digest/state changes to commit."
fi
