#!/usr/bin/env bash
set -e

PANDA_HOST="${PANDA_HOST:-panda}"
REMOTE_DIR="~/zus-api"

echo "⚡️ Deploying ZüsAPI to ${PANDA_HOST}:${REMOTE_DIR}..."

# 1. Ensure remote directory exists
ssh "${PANDA_HOST}" "mkdir -p ${REMOTE_DIR}"

# 2. Sync codebase
rsync -avz --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude 'venv' \
  --exclude '.env' \
  ./ "${PANDA_HOST}:${REMOTE_DIR}/"

# 3. Build & start docker compose
ssh "${PANDA_HOST}" "cd ${REMOTE_DIR} && docker compose up -d --build"

# 4. Configure systemd --user boot service & 2-minute self-healing watchdog timer
echo "⚡️ Configuring systemd --user boot persistence & self-healing watchdog on ${PANDA_HOST}..."
ssh "${PANDA_HOST}" "mkdir -p ~/.config/systemd/user && cat << 'EOF' > ~/.config/systemd/user/zus-stack.service
[Unit]
Description=ZüsAPI & PostGIS Telemetry Stack (Docker Compose)
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=%h/zus-api
ExecStart=/usr/bin/docker compose up -d
ExecReload=/usr/bin/docker compose up -d

[Install]
WantedBy=default.target
EOF

cat << 'EOF' > ~/.config/systemd/user/zus-watchdog.service
[Unit]
Description=ZüsAPI Health Watchdog & Offline Backfill Trigger
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=%h/zus-api
ExecStart=/bin/bash -c 'if ! curl -fsS --max-time 5 http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then /usr/bin/docker compose up -d && sleep 6 && curl -fsS -X POST \"http://127.0.0.1:8000/api/v1/radar/backfill?lookback_hours=24\" >/dev/null 2>&1 || true; fi'
EOF

cat << 'EOF' > ~/.config/systemd/user/zus-watchdog.timer
[Unit]
Description=Run ZüsAPI Watchdog every 2 minutes and on boot

[Timer]
OnBootSec=45s
OnUnitActiveSec=2min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now zus-stack.service zus-watchdog.timer
"

# 5. Check status
echo "⚡️ Checking container & systemd status..."
ssh "${PANDA_HOST}" "cd ${REMOTE_DIR} && docker compose ps && systemctl --user is-active zus-stack.service zus-watchdog.timer"

echo "⚡️ Verifying health endpoint & triggering initial backfill check..."
ssh "${PANDA_HOST}" "curl -s http://localhost:8000/api/v1/health | grep healthy && curl -s -X POST 'http://localhost:8000/api/v1/radar/backfill?lookback_hours=24' && echo '' && echo '✅ ZüsAPI is UP, HEALTHY, and PERSISTENT!'"
