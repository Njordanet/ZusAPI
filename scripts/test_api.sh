#!/usr/bin/env bash
set -e

HOST="${1:-http://localhost:8000}"
echo "⚡️ Testing ZüsAPI at ${HOST}..."

echo "--- 1. Healthcheck ---"
curl -s "${HOST}/api/v1/health" | python3 -m json.tool

echo -e "\n--- 2. Root Index ---"
curl -s "${HOST}/" | python3 -m json.tool

echo -e "\n--- 3. Weather (Mount Olympus) ---"
curl -s "${HOST}/api/v1/weather?lat=40.0856&lon=22.3585&city=Mount+Olympus" | python3 -m json.tool | head -n 30

echo -e "\n--- 4. Dataset Inventory ---"
curl -s "${HOST}/api/v1/dataset/inventory" | python3 -m json.tool
