#!/usr/bin/env python3
import sys, numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from joblib import load

MODEL_PKL = Path(__file__).with_name("gb_bag_gbr.pkl")
BAG = load(MODEL_PKL)                         

def flip(x: float) -> str:
    d = Decimal(str(x))
    t = d.quantize(Decimal("0.001"), ROUND_HALF_UP)
    k = abs(int(t*1000)) % 10
    if k == 4:
        return str((t*100).to_integral_value(ROUND_CEILING)/Decimal(100))
    if k == 9:
        return str((t*100).to_integral_value(ROUND_FLOOR)  /Decimal(100))
    return str(t.quantize(Decimal("0.01"), ROUND_HALF_UP))

def feats(d, m, r):
    r = min(r, 2000); mpd, rpd = m / d, r / d
    return np.array([
        d, m, r, mpd, rpd,
        int(d == 5), int(d == 8),
        int(d == 1 and m > 750),
        int(rpd >= 250), int(mpd >= 180)
    ], dtype=float).reshape(1, -1)

def main(a):
    if len(a) != 3:
        sys.exit("Usage: run.py <days> <miles> <receipts>")
    d, m, r = int(a[0]), float(a[1]), float(a[2])
    x = feats(d, m, r)
    raw = np.mean([m.predict(x)[0] for m in BAG])
    print(flip(raw))

if __name__ == "__main__":
    main(sys.argv[1:])
