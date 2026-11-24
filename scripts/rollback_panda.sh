#!/usr/bin/env bash
set -euo pipefail

PANDA_HOST="${PANDA_HOST:-panda}"
KILL_SWITCH_ONLY=0
RESTORE_DB=""

usage() {
  cat << 'EOF'
Usage: rollback_panda.sh [--host <host>] [--kill-switch-only] [--restore-db <remote_backup_path>]

Options:
  --host <host>                     SSH host alias or user@host (default: panda or $PANDA_HOST)
  --kill-switch-only                Instant 2s rollback: set ZUS_ACCURACY_V2=false in ~/zus-api/.env
                                    and recreate zus-api without rebuilding
  --restore-db <remote_backup_path> Optional remote .sql.gz or .sql.gz.enc backup path in ~/zus-backups
                                    to restore into zus-db (skipped by default; migrations are additive)
  -h, --help                        Show this help text
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      PANDA_HOST="${2:?--host requires a value}"
      shift 2
      ;;
    --host=*)
      PANDA_HOST="${1#*=}"
      shift
      ;;
    --kill-switch-only)
      KILL_SWITCH_ONLY=1
      shift
      ;;
    --restore-db)
      RESTORE_DB="${2:?--restore-db requires a remote backup file path}"
      shift 2
      ;;
    --restore-db=*)
      RESTORE_DB="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

restore_remote_db() {
  local backup_path="$1"
  echo "🗄️ Restoring PostGIS database on ${PANDA_HOST} from ${backup_path}..."
  ssh "${PANDA_HOST}" "BACKUP_PATH=$(printf '%q' "${backup_path}") bash -s" << 'DB_EOF'
set -euo pipefail
REMOTE_APP="${HOME}/zus-api"
ENV_FILE="${REMOTE_APP}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [ ! -f "${BACKUP_PATH}" ]; then
  echo "❌ Remote backup file not found: ${BACKUP_PATH}" >&2
  exit 1
fi

COMPOSE="docker compose -f ${REMOTE_APP}/docker-compose.yml"
PG_USER="${POSTGRES_USER:-zus}"
PG_DB="${POSTGRES_DB:-zus_telemetry}"

if [[ "${BACKUP_PATH}" == *.enc ]]; then
  if [ -z "${BACKUP_ENCRYPTION_KEY:-}" ]; then
    echo "❌ BACKUP_ENCRYPTION_KEY is not set in ${ENV_FILE}; cannot decrypt ${BACKUP_PATH}" >&2
    exit 1
  fi
  openssl enc -d -aes-256-cbc -pbkdf2 -pass env:BACKUP_ENCRYPTION_KEY -in "${BACKUP_PATH}" \
    | gunzip -c \
    | ${COMPOSE} exec -T zus-db psql -v ON_ERROR_STOP=1 -U "${PG_USER}" -d "${PG_DB}"
else
  gunzip -c "${BACKUP_PATH}" \
    | ${COMPOSE} exec -T zus-db psql -v ON_ERROR_STOP=1 -U "${PG_USER}" -d "${PG_DB}"
fi
echo "✅ PostGIS database restore complete."
DB_EOF
}

verify_remote_health() {
  echo "🩺 Verifying ZüsAPI health on ${PANDA_HOST}..."
  ssh "${PANDA_HOST}" 'bash -s' << 'HEALTH_EOF'
set -euo pipefail
for i in $(seq 1 20); do
  if curl -fsS --max-time 5 http://127.0.0.1:8000/api/v1/health | grep -q healthy; then
    echo "✅ ZüsAPI is UP and HEALTHY!"
    exit 0
  fi
  sleep 1
done
echo "❌ ZüsAPI failed health check after rollback." >&2
cd "${HOME}/zus-api" && docker compose logs --tail=40 zus-api >&2 || true
exit 1
HEALTH_EOF
}

if [ "${KILL_SWITCH_ONLY}" -eq 1 ]; then
  echo "⚡ Activating ZUS_ACCURACY_V2=false kill-switch on ${PANDA_HOST}..."
  ssh "${PANDA_HOST}" 'bash -s' << 'KS_EOF'
set -euo pipefail
REMOTE_APP="${HOME}/zus-api"
ENV_FILE="${REMOTE_APP}/.env"
if [ ! -f "${REMOTE_APP}/docker-compose.yml" ]; then
  echo "❌ ${REMOTE_APP}/docker-compose.yml not found." >&2
  exit 1
fi
touch "${ENV_FILE}"
if grep -q '^ZUS_ACCURACY_V2=' "${ENV_FILE}"; then
  sed -i 's/^ZUS_ACCURACY_V2=.*/ZUS_ACCURACY_V2=false/' "${ENV_FILE}"
else
  printf '\nZUS_ACCURACY_V2=false\n' >> "${ENV_FILE}"
fi
cd "${REMOTE_APP}"
docker compose up -d --no-build zus-api
KS_EOF

  if [ -n "${RESTORE_DB}" ]; then
    restore_remote_db "${RESTORE_DB}"
  fi
  verify_remote_health
  exit 0
fi

echo "⏪ Rolling back ZüsAPI on ${PANDA_HOST} to pre-deploy snapshot..."
ssh "${PANDA_HOST}" 'bash -s' << 'RB_EOF'
set -euo pipefail
REMOTE_APP="${HOME}/zus-api"
ROLLBACK_DIR="${HOME}/zus-api-rollback-prev"

if [ ! -f "${ROLLBACK_DIR}/docker-compose.yml" ]; then
  echo "❌ Pre-deploy snapshot directory missing (${ROLLBACK_DIR}/docker-compose.yml)." >&2
  echo "Run with --kill-switch-only or redeploy a known-good git revision." >&2
  exit 1
fi

if ! docker image inspect zus-api-zus-api:rollback-prev >/dev/null 2>&1; then
  echo "❌ Pre-deploy Docker image tag zus-api-zus-api:rollback-prev not found." >&2
  echo "Run with --kill-switch-only or rebuild from ${ROLLBACK_DIR}." >&2
  exit 1
fi

rsync -a --delete \
  --exclude '.env' \
  --exclude 'data' \
  "${ROLLBACK_DIR}/" "${REMOTE_APP}/"

docker tag zus-api-zus-api:rollback-prev zus-api-zus-api:latest

cd "${REMOTE_APP}"
docker compose up -d --no-build zus-api
RB_EOF

if [ -n "${RESTORE_DB}" ]; then
  restore_remote_db "${RESTORE_DB}"
fi

verify_remote_health
echo "✅ Rollback complete cleanly without rebuilding."
