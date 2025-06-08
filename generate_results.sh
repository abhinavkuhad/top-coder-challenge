#!/usr/bin/env bash
set -euo pipefail
command -v jq >/dev/null || { echo "jq needed"; exit 1; }

[[ -f private_cases.json ]] || { echo "private_cases.json missing"; exit 1; }

jq -r '.[] | "\(.trip_duration_days) \(.miles_traveled) \(.total_receipts_amount)"' \
   private_cases.json \
| ./predict_stream.py > private_results.txt

echo "✅ private_results.txt written – $(wc -l < private_results.txt) lines"
