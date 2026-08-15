"""E12 -- the adversary, at a sample size that certifies something.

E11's adequacy audit caught a weakness in E6a. Searching 72 000 candidate request sequences
sounds like a lot, and it is, but the statistical unit of the safety claim is the *cell*, not
the sequence: an adversary that fails on one cell tells you about one cell. With 40 cells the
Clopper-Pearson upper bound is 7.2 %, which certifies nothing anyone would care about.

This re-runs the same attack against 320 independently drawn cells -- past the 299 needed to
certify below 1 % at 95 % confidence with zero failures -- trading depth per cell for breadth
across cells, which is the axis the bound actually depends on.

    python zeroguard/exp/e12_adversary_scale.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM, NSTEP = 30.0, 45.0, 4.20, 80
CM = dict(V=0.03, T=0.5, plate=0.006)
K_TH, F_COOL, SR = 15.0, 0.25, 1.8
DT_EFF = CM["T"] + K_TH * (SR - 1.0)
EST = BatteryROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0))
QN = EST.p["Q_nom"]


def episode(plant, requests, Tamb):
    s = plant.init_state(0.10, Tamb)
    mT = mV = -1e9
    for k in range(len(requests)):
        I, _ = project_current(EST, s, float(requests[k]), DT, Tamb, Vlim=VLIM, Tlim=TLIM,
                               margin=EST.plating_margin(), dV=CM["V"], dT=DT_EFF,
                               dP=CM["plate"], cool_frac=F_COOL)
        s, o = plant.step(s, float(max(0.0, I)), DT, Tamb)
        mT = max(mT, o["T"]); mV = max(mV, o["V"])
    return mT, mV


def attack(plant, Tamb, rng, pop=40, elite=8, gens=20):
    """Cross-entropy search over the whole request sequence, maximising peak temperature."""
    mu = np.full(NSTEP, 0.5 * 3 * QN)
    sd = np.full(NSTEP, 0.35 * 3 * QN)
    best_T = best_V = -1e9
    for _ in range(gens):
        cand = np.clip(rng.normal(mu, sd, size=(pop, NSTEP)), 0.0, 3 * QN)
        res = [episode(plant, c, Tamb) for c in cand]
        scores = np.array([r[0] for r in res])
        idx = np.argsort(scores)[-elite:]
        mu = cand[idx].mean(0); sd = cand[idx].std(0) + 1e-3
        best_T = max(best_T, float(scores.max()))
        best_V = max(best_V, float(max(r[1] for r in res)))
    return best_T, best_V


def main(n_cells=320, seed=91):
    rng = np.random.default_rng(seed)
    viol = 0
    peaks, gains = [], []
    t0 = time.time()
    print(f"E12 -- cross-entropy adversary against {n_cells} independently drawn cells\n")
    for c in range(n_cells):
        soh = float(rng.uniform(0.80, 1.0)); f = (1 - soh) / 0.2
        plant = BatteryROM(cell_scale=dict(R=1 + 0.8 * f, Q=soh, plate=1 + 0.6 * f))
        Tamb = float(rng.uniform(25.0, 38.0))
        aT, aV = attack(plant, Tamb, rng)
        gT, _ = episode(plant, np.full(NSTEP, 3 * QN), Tamb)
        peaks.append(aT); gains.append(aT - gT)
        viol += int(aT > TLIM + 1e-9 or aV > VLIM + 1e-9)
        if (c + 1) % 40 == 0:
            print(f"  {c+1:>4}/{n_cells} cells | worst peak T so far {max(peaks):.3f} C | "
                  f"violations {viol}  [{time.time()-t0:.0f}s]")
    p = np.array(peaks); g = np.array(gains)
    m, lo, hi = stats.bootstrap_ci(g)
    seqs = n_cells * 40 * 20
    out = dict(**stats.summarize_safety("adversary_at_scale", viol, n_cells),
               sequences_searched=int(seqs),
               episodes_simulated=int(seqs + n_cells),
               worst_peak_T=float(p.max()), mean_peak_T=float(p.mean()),
               limit=TLIM, headroom_C=float(TLIM - p.max()),
               mean_gain_over_greedy_C=m, gain_ci=[lo, hi],
               n_required_for_1pct=stats.n_required(0.01),
               certifies_below_1pct=bool(n_cells >= stats.n_required(0.01) and viol == 0),
               seconds=round(time.time() - t0, 1))
    print(f"\n  cells {n_cells} | sequences searched {seqs:,} | violations {viol}")
    print(f"  worst peak T anywhere: {p.max():.3f} C   (limit {TLIM}, headroom {TLIM-p.max():.3f} C)")
    print(f"  adversary beats greedy by {m:+.4f} C  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  CP95 upper bound: {out['cp95_upper_pct']:.4f}%  "
          f"(was 7.2158% at 40 cells) -- certifies below 1%: {out['certifies_below_1pct']}")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e12_adversary_scale.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e12_adversary_scale.json')}")


if __name__ == "__main__":
    main()
