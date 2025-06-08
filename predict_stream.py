#!/usr/bin/env python3
import sys, numpy as np
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from joblib import load

BAG = load(Path(__file__).with_name("gb_bag_gbr.pkl"))

def flip(x):
    d = Decimal(str(x))
    t = d.quantize(Decimal("0.001"), ROUND_HALF_UP)
    k = abs(int(t*1000)) % 10
    if k==4: return str((t*100).to_integral_value(ROUND_CEILING)/Decimal(100))
    if k==9: return str((t*100).to_integral_value(ROUND_FLOOR)  /Decimal(100))
    return str(t.quantize(Decimal("0.01"), ROUND_HALF_UP))

def feat(d,m,r):
    r=min(r,2000); mpd,rpd=m/d,r/d
    return [d,m,r,mpd,rpd,int(d==5),int(d==8),
            int(d==1 and m>750),int(rpd>=250),int(mpd>=180)]

rows=[feat(*map(float,l.split())) for l in sys.stdin if l.strip()]
X=np.array(rows,dtype=float)
pred=np.column_stack([m.predict(X) for m in BAG]).mean(1)
for v in pred: print(flip(v))
