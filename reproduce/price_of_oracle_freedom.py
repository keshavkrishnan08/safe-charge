"""The performance cost of using the fail-safe corner instead of a true-theta oracle localizes
to the binding constraint's channel and does not compound with the uncertainty dimension.

The corner filter admits I*(x,theta_bar)=min_c u_c(x,theta_bar) -- provably the largest current safe
for the whole envelope (optimality), so any charge a health oracle could add is bounded by the single
BINDING constraint's channels. Uncertainty in channels off the binding constraint is free.

We widen the envelope 1->4 channels {R, cooling, thermal mass, plating} and, for each plant, compare
delivered SOC under the corner filter (fail-safe) and under a filter given the true cell (oracle).
Here plating binds: the three thermal channels (off it, entering only through temperature) together
cost a tiny fraction of oracle-free SOC, while the one binding plating channel carries the rest.

    python reproduce/price_of_oracle_freedom.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from safe_charge import BatteryROM, project_current

DT, NSTEP, TAMB, T0 = 30.0, 80, 25.0, 25.0
VLIM, TLIM = 4.20, 45.0
dV, dT, dP = 0.03, 0.5, 0.006
Qn = BatteryROM().p["Q_nom"]; MARG = BatteryROM().plating_margin()
ENV = {"R": (1.0, 1.8), "cool": (1.0, 0.75), "Cth": (1.0, 0.80), "plate": (1.0, 1.6)}
CORNER = {k: v[1] for k, v in ENV.items()}
CHANS = ["R", "cool", "Cth", "plate"]

def make_rom(theta):
    r = BatteryROM(); r.scale = dict(R=theta["R"], Q=1.0, plate=theta["plate"])
    r.p = dict(r.p); r.p["hA"] *= theta["cool"]; r.p["C_th"] *= theta["Cth"]
    return r

def deliver(plant, est):
    s = plant.init_state(0.10, T0); maxT = maxV = 0.0; minP = 1e9
    for _ in range(NSTEP):
        I, _ = project_current(est, s, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM,
                               margin=est.plating_margin(), dV=dV, dT=dT, dP=dP, cool_frac=0.0)
        s, o = plant.step(s, float(max(0.0, I)), DT, TAMB)
        maxT, maxV, minP = max(maxT, o["T"]), max(maxV, o["V"]), min(minP, o["phi_an"])
    binder = min([("T", TLIM-maxT), ("V", VLIM-maxV), ("plating", minP-MARG)], key=lambda kv: kv[1])[0]
    return s["soc"], binder

rng = np.random.default_rng(20260721)
print("price of oracle-freedom over the full charge window (mean oracle SOC - corner SOC):")
pens = {}
for d in range(1, len(CHANS)+1):
    active = CHANS[:d]; N = 300
    corner_d = {k: (CORNER[k] if k in active else 1.0) for k in CHANS}   # robust only to active channels
    pen = []; binders = {"T": 0, "V": 0, "plating": 0}
    for _ in range(N):
        theta = {k: 1.0 for k in CHANS}
        for k in active:
            lo, hi = ENV[k]; theta[k] = float(rng.uniform(min(lo, hi), max(lo, hi)))
        plant = make_rom(theta)
        soc_c, binder = deliver(plant, make_rom(corner_d))
        soc_o, _ = deliver(plant, make_rom(theta))
        pen.append(soc_o - soc_c); binders[binder] += 1
    pens[d] = 100*float(np.mean(pen))
    b = max(binders, key=binders.get)
    print(f"  d={d} {str(active):40} penalty={pens[d]:5.2f}% SOC  binding={b} ({int(100*binders[b]/N)}%)")
print(f"\nThree off-binding thermal channels cost {pens[3]:.2f}% total; the binding plating channel "
      f"adds {pens[4]-pens[3]:.2f}% (full envelope {pens[4]:.2f}%). The price tracks the binding "
      f"constraint, not the dimension.")
