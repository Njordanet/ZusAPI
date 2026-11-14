#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${HOME}/zus-backups"
mkdir -p "${BACKUP_DIR}/supabase" "${BACKUP_DIR}/zus-db" "${BACKUP_DIR}/push"
chmod 700 "${BACKUP_DIR}"

if [ -f "${HOME}/zus-api/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "${HOME}/zus-api/.env"
    set +a
fi

COMPOSE="docker compose -f ${HOME}/zus-api/docker-compose.yml"
STAMP=$(date -u +"%Y%m%d_%H%M%SZ")

encrypt_or_write() {
    local target="$1"
    if [ -n "${BACKUP_ENCRYPTION_KEY:-}" ]; then
        openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:BACKUP_ENCRYPTION_KEY -out "${target}.enc"
        chmod 600 "${target}.enc"
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum "${target}.enc" > "${target}.enc.sha256" || true
        fi
    else
        cat > "${target}"
        chmod 600 "${target}"
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum "${target}" > "${target}.sha256" || true
        fi
    fi
}

echo "🛡️ Starting Züs Nightly Database & State Backup [${STAMP}]..."

# 1) Supabase Account DB Dump
if [ -n "${SUPABASE_DB_URL:-}" ]; then
    echo "📦 Backing up Supabase Account DB..."
    docker run --rm -i postgres:17-alpine pg_dump \
        --enable-row-security --no-owner --no-acl --clean --if-exists \
        --schema=public \
        "${SUPABASE_DB_URL}" \
        | gzip -9 \
        | encrypt_or_write "${BACKUP_DIR}/supabase/supabase_accounts_${STAMP}.sql.gz"
    echo "✅ Supabase Account DB backup complete."
else
    echo "ℹ️ SUPABASE_DB_URL not set; skipping Supabase pg_dump."
fi

# 2) Local zus-db Dump
echo "📦 Backing up Local zus-db (PostGIS Telemetry)..."
${COMPOSE} exec -T zus-db pg_dump -U "${POSTGRES_USER:-zus}" "${POSTGRES_DB:-zus_telemetry}" \
    | gzip -9 \
    | encrypt_or_write "${BACKUP_DIR}/zus-db/zus_telemetry_${STAMP}.sql.gz"
echo "✅ Local zus-db backup complete."

# 3) Push SQLite Snapshot
echo "📦 Backing up Push Notification SQLite database..."
if ${COMPOSE} exec -T zus-api test -f /app/data/push_subscriptions.db >/dev/null 2>&1; then
    ${COMPOSE} exec -T zus-api python3 -c "import sqlite3, pathlib; p=pathlib.Path('/app/data/push_subscriptions.db'); src=sqlite3.connect(str(p)); dst=sqlite3.connect('/tmp/push_subscriptions_bak.db'); src.backup(dst); dst.close(); src.close()"
    ${COMPOSE} exec -T zus-api cat /tmp/push_subscriptions_bak.db \
        | gzip -9 \
        | encrypt_or_write "${BACKUP_DIR}/push/push_subscriptions_${STAMP}.db.gz"
    ${COMPOSE} exec -T zus-api rm -f /tmp/push_subscriptions_bak.db
    echo "✅ Push notification database snapshot complete."
else
    echo "ℹ️ /app/data/push_subscriptions.db not found; skipping SQLite snapshot."
fi

# 4) 30-day Local Retention
echo "🧹 Pruning backups older than 30 days in ${BACKUP_DIR}..."
find "${BACKUP_DIR}" -type f -mtime +30 -delete
echo "✅ Local retention prune complete."

# 5) Optional Cloudflare R2 Sync
if [ -n "${R2_BUCKET:-}" ] && [ -n "${R2_ENDPOINT:-}" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ]; then
    echo "☁️ Syncing backups to Cloudflare R2 bucket: ${R2_BUCKET}..."
    export RCLONE_CONFIG_R2_TYPE="s3"
    export RCLONE_CONFIG_R2_PROVIDER="Cloudflare"
    export RCLONE_CONFIG_R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
    export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
    export RCLONE_CONFIG_R2_ENDPOINT="${R2_ENDPOINT}"
    export RCLONE_CONFIG_R2_ACL="private"

    rclone copy "${BACKUP_DIR}" "r2:${R2_BUCKET}/zus-backups"
    rclone delete --min-age 30d "r2:${R2_BUCKET}/zus-backups" || true
    echo "✅ Cloudflare R2 backup sync complete."
else
    echo "ℹ️ Cloudflare R2 env vars not set; local panda backup complete."
fi

echo "🛡️ Backup run finished successfully at $(date -u +"%Y-%m-%dT%H:%M:%SZ")."
