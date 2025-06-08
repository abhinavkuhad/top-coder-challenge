#!/usr/bin/env bash
# FAST vectorised evaluator – no awk, BSD-safe

set -euo pipefail

echo "🧾  Black Box Challenge – FAST Evaluation (vectorised)"
echo "======================================================"
echo

# ── sanity checks ─────────────────────────────────────────────────────
command -v jq >/dev/null || { echo "jq required"; exit 1; }
[[ -f public_cases.json ]]  || { echo "public_cases.json missing"; exit 1; }
[[ -x predict_stream.py ]]  || { echo "predict_stream.py missing / not executable"; exit 1; }

# ── flatten JSON once → tmp inputs / expected -------------------------
jq -r '.[] | "\(.input.trip_duration_days) \(.input.miles_traveled) \(.input.total_receipts_amount) \(.expected_output)"' \
        public_cases.json > _cases.txt

cut -d' ' -f1-3 _cases.txt > _inputs.txt
cut -d' ' -f4   _cases.txt > _expected.txt
num_cases=$(wc -l < _inputs.txt)

echo "➡️   $num_cases cases loaded"
echo
echo "⚡  Predicting in a single process …"
TIMEFORMAT='%3R'; pred_time=$( { time ./predict_stream.py < _inputs.txt > _pred.txt ; } 2>&1 )
echo "⏱   Vectorised prediction time: ${pred_time}s"
echo

# ── summary in pure Python -------------------------------------------
python3 - <<'PY'
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR
exp = [float(x) for x in open("_expected.txt")]
prd = [float(x) for x in open("_pred.txt")]

abs_sum = exact = close = 0.0
max_err = -1.0; max_idx = -1
for i, (e, p) in enumerate(zip(exp, prd), 1):
    err = abs(e - p)
    abs_sum += err
    if err < 0.01: exact += 1
    if err < 1.00: close += 1
    if err > max_err:
        max_err, max_idx = err, i

n = len(exp)
avg_err = abs_sum / n
score   = avg_err * 100 + (n - exact) * 0.1      # ← added

print("✅  Evaluation Complete!\n")
print("📈  Results Summary:")
print(f"  Total test cases: {n}")
print(f"  Successful runs:  {n}")
print(f"  Exact matches (±$0.01): {int(exact)} ({exact*100/n:.1f}%)")
print(f"  Close  matches (±$1.00): {int(close)} ({close*100/n:.1f}%)")
print(f"  Average error: ${avg_err:.2f}")
print(f"  Maximum error: ${max_err:.2f}  (case {max_idx})\n")
print(f"🎯 Your Score: {score:.2f} (lower is better)")
PY

