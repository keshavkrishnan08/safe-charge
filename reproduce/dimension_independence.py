"""The same fixed 18-iteration certificate certifies safety over a multi-channel uncertainty
envelope, whatever its dimension.

Monotone structure makes the whole envelope's worst case a single corner, so certifying against that
corner certifies every interior plant exactly. We widen the envelope from 1 to 5 aging/fault channels
and show the certificate stays a fixed 18-iteration bisection with zero violations, and the corner
empirically dominates the interior. An identification scheme would add one state per channel.

Channels (each monotone in the operating region, worst-case direction in parentheses):
  R (high), cooling hA (low), thermal mass C_th (low), plating (high), ambient T_amb (high).

    python reproduce/dimension_independence.py           # n=500 on the full envelope
    python reproduce/dimension_independence.py --full    # n=5000, a 10x tighter bound
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from math import comb
from safe_charge import BatteryROM, project_current

DT, NSTEP, TAMB, T0, SOC_TGT = 30.0, 80, 25.0, 25.0, 0.80
VLIM, TLIM = 4.20, 45.0
dV, dT, dP = 0.03, 0.5, 0.006          # CTRL_MARGINS
Qn = BatteryROM().p["Q_nom"]
# The certificate covers S = {T, V}. We additionally SCORE the plating margin below, so a
# run counts as safe only if it also held plating -- enforced and checked, not certified.
MARG = BatteryROM().plating_margin()
ENV = {"R": (1.0, 1.8), "cool": (1.0, 0.75), "Cth": (1.0, 0.80), "plate": (1.0, 1.6),
       "Tamb": (25.0, 35.0)}
CORNER = {k: v[1] for k, v in ENV.items()}
CHANS = ["R", "cool", "Cth", "plate", "Tamb"]

def make_rom(theta):
    r = BatteryROM(); r.scale = dict(R=theta["R"], Q=1.0, plate=theta["plate"])
    r.p = dict(r.p); r.p["hA"] *= theta["cool"]; r.p["C_th"] *= theta["Cth"]
    return r

def drive(plant, est, theta_plant, theta_est):
    """The filter is driven at the CORNER ambient; the plant runs at its true one."""
    s = plant.init_state(0.10, T0); maxT = maxV = 0.0; minP = 1e9
    for _ in range(NSTEP):
        I, _ = project_current(est, s, 3.0*Qn, DT, theta_est["Tamb"], Vlim=VLIM, Tlim=TLIM,
                               margin=est.plating_margin(), dV=dV, dT=dT, dP=dP, cool_frac=0.0)
        s, o = plant.step(s, float(max(0.0, I)), DT, theta_plant["Tamb"])
        maxT, maxV, minP = max(maxT, o["T"]), max(maxV, o["V"]), min(minP, o["phi_an"])
        if s["soc"] >= SOC_TGT: break
    return maxT, maxV, minP

def cp_upper(k, N, alpha=0.05):
    if k == 0: return 1 - alpha**(1.0/N)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        p = 0.5*(lo+hi)
        tail = sum(comb(N, i)*p**i*(1-p)**(N-i) for i in range(k+1))
        hi, lo = (p, lo) if tail < alpha else (hi, p)
    return hi

ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
FULL = ap.parse_args().full
N_DIM, N_ENV = (500, 5000) if FULL else (150, 500)

def nominal():
    return {k: (25.0 if k == "Tamb" else 1.0) for k in CHANS}

est = make_rom(CORNER)                 # the certificate: fixed 18-iter bisection against the corner
rng = np.random.default_rng(20260720)

print(f"dimension sweep (certificate is 18 iterations at every dimension), n={N_DIM}:")
for d in range(1, len(CHANS)+1):
    active = CHANS[:d]; safe = 0; worstT = 0.0
    for _ in range(N_DIM):
        theta = nominal()
        for k in active:
            lo, hi = ENV[k]; theta[k] = float(rng.uniform(min(lo, hi), max(lo, hi)))
        mT, mV, mP = drive(make_rom(theta), est, theta, CORNER); worstT = max(worstT, mT)
        safe += int(mT <= TLIM and mV <= VLIM and mP >= MARG)
    print(f"  d={d}  {str(active):48}  safe={safe}/{N_DIM}  worstT={worstT:.2f}C  cert=18 iters")

safe = 0; worstT = worstV = 0.0
for _ in range(N_ENV):
    theta = {k: float(rng.uniform(min(ENV[k]), max(ENV[k]))) for k in CHANS}
    mT, mV, mP = drive(make_rom(theta), est, theta, CORNER)
    worstT, worstV = max(worstT, mT), max(worstV, mV)
    safe += int(mT <= TLIM and mV <= VLIM and mP >= MARG)
cornerT, _, _ = drive(make_rom(CORNER), est, CORNER, CORNER)
print(f"\nfull {len(CHANS)}-channel envelope: {safe}/{N_ENV} safe, "
      f"viol={100*(N_ENV-safe)/N_ENV:.2f}%, "
      f"CP95 upper={100*cp_upper(N_ENV-safe, N_ENV):.3f}%, worst {worstT:.2f}C / {worstV:.3f}V")
print(f"corner peakT={cornerT:.2f}C dominates interior worst {worstT:.2f}C: {cornerT >= worstT-1e-6}")
print("Same 18-iteration certificate at every dimension; an identifier needs one state per channel.")
if not FULL:
    print("Run with --full for n=5000 on the envelope, a 10x tighter Clopper-Pearson bound.")
