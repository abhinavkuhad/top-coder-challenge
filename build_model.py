#!/usr/bin/env python3
"""
build_model_gbr.py
Trains three GradientBoostingRegressor seeds and
stores them compressed as gb_bag_gbr.pkl.
Run ONCE:  python3 build_model_gbr.py
"""

import json, numpy as np
from pathlib import Path
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor

DATA_JSON = Path("public_cases.json")
MODEL_PKL = Path(__file__).with_name("gb_bag_gbr.pkl")
SEEDS     = (0, 1, 2)

def feats(d, m, r):
    r = min(r, 2000)
    mpd, rpd = m / d, r / d
    return np.array([
        d, m, r, mpd, rpd,
        int(d == 5), int(d == 8),
        int(d == 1 and m > 750),
        int(rpd >= 250), int(mpd >= 180)
    ], dtype=float)

# ---------- build training matrix -------------------------------------
X, y = [], []
for rec in json.load(DATA_JSON.open()):
    inp = rec["input"]
    X.append(feats(inp["trip_duration_days"],
                   inp["miles_traveled"],
                   inp["total_receipts_amount"]))
    y.append(rec["expected_output"])
X, y = np.vstack(X), np.array(y)

# ---------- train three seeds -----------------------------------------
def train(seed):
    return GradientBoostingRegressor(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.03,
        loss="absolute_error",   
        random_state=seed
    ).fit(X, y)

bag = [train(s) for s in SEEDS]
dump(bag, MODEL_PKL, compress=3)
print("✅ Saved model bag →", MODEL_PKL.resolve())
