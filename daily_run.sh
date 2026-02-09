#!/bin/bash
# daily_run.sh — 매일 크론으로 실행: 수집 → 구글시트 업로드
#
# crontab 예시 (매일 오전 6시):
#   0 6 * * * /Users/grey/Desktop/oliveyoungBot/daily_run.sh >> /Users/grey/Desktop/oliveyoungBot/daily_run.log 2>&1

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PYTHON="$DIR/.venv/bin/python3"
FOLDER_ID="1aDIHdlb322-qj_dDblcCKbxWNCpShLKr"
TODAY=$(date +%Y-%m-%d)

echo "===== [$TODAY] daily_run start ====="

# 1. 이전 데이터 초기화
rm -f state.json pc_ads.csv m_products.csv
echo "[1/3] Cleaned state.json, pc_ads.csv, m_products.csv"

# 2. 수집
echo "[2/3] Running runner.py ..."
$PYTHON runner.py --mode once --sleep 0.8

# 3. 구글시트 업로드
echo "[3/3] Uploading to Google Sheets ..."
$PYTHON upload_sheets.py --folder_id "$FOLDER_ID" --date "$TODAY"

echo "===== [$TODAY] daily_run done ====="
