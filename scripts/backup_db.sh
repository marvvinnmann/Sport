#!/usr/bin/env bash
# Back up the database into data/backups/ (keeps the 14 most recent copies).
# SQLite: safe online copy with the sqlite3 backup API. PostgreSQL (docker compose): pg_dump.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/backups
stamp=$(date +%Y%m%d-%H%M%S)

if docker compose ps db >/dev/null 2>&1 && docker compose ps db | grep -q running; then
  docker compose exec -T db pg_dump -U sportpred sportpred | gzip > "data/backups/sportpred-$stamp.sql.gz"
else
  python3 - "$stamp" <<'EOF'
import sqlite3, sys
src = sqlite3.connect("data/sportpred.db")
dst = sqlite3.connect(f"data/backups/sportpred-{sys.argv[1]}.db")
src.backup(dst)
dst.close(); src.close()
EOF
  gzip "data/backups/sportpred-$stamp.db"
fi
ls -1t data/backups/sportpred-* | tail -n +15 | xargs -r rm --
echo "Backup written to data/backups/ ($stamp)"
