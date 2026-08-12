"""Act IV -- from one cell to a pack, and why the footprint is the architecture.

A series pack shares one current across every cell, so the admissible set is the
intersection of the cells' admissible sets. Because each of those is an interval anchored at
zero, the intersection is too, and its right edge is the minimum over cells:

        I*_pack  =  min_i  I*_i                                              (Lemma)

That is not a convenience. It says the pack certificate is exactly the weakest cell's
certificate, so a filter that runs per cell loses nothing -- and a filter that runs once on
an averaged cell can be wrong about the only cell that matters.

E5a  verifies the lemma numerically from N = 1 to N = 1024 heterogeneous cells
E5b  injects a single faulted cell into a pack of 100 and asks which architecture notices

The second is the experiment behind the claim that 904 bytes is not a performance detail but
an architectural one: enforcement can live on every cell precisely because it is small.

    python zeroguard/exp/e5_pack.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM = 30.0, 45.0, 4.20
CM = dict(V=0.03, T=0.5, plate=0.006)
K_TH, F_COOL = 15.0, 0.25
DT_EFF = CM["T"] + K_TH * 0.8


def make_cell(sR, sQ, sP):
    return BatteryROM(cell_scale=dict(R=sR, Q=sQ, plate=sP))


_EST = BatteryROM(cell_scale=dict(R=1.8, Q=1.0, plate=1.0))


def per_cell_current(cells, states, I_prop, Tambs, est=None, dT=DT_EFF, iters=18):
    """Each cell runs its own filter; the pack current is what every cell can accept."""
    est = _EST if est is None else est
    best = I_prop
    m = est.plating_margin()
    for s, Ta in zip(states, Tambs):
        I, _ = project_current(est, s, I_prop, DT, Ta, Vlim=VLIM, Tlim=TLIM,
                               margin=m, dV=CM["V"], dT=dT,
                               dP=CM["plate"], cool_frac=F_COOL, iters=iters)
        best = min(best, I)
    return best


def pack_direct(cells, states, I_prop, Tambs, est_scale=1.8, iters=18, dT_base=DT_EFF):
    """Bisection run once against the whole pack's constraint set, for comparison."""
    est = BatteryROM(cell_scale=dict(R=est_scale, Q=1.0, plate=1.0))
    m = est.plating_margin()

    def ok(I):
        for s, Ta in zip(states, Tambs):
            dT = dT_base + F_COOL * max(0.0, s["T"] - Ta)
            V, T, phi, _ = est.probe(s, I, DT, Ta)
            if not (V <= VLIM - CM["V"] + 1e-9 and T <= TLIM - dT + 1e-9
                    and phi >= m + CM["plate"] - 1e-9):
                return False
        return True

    cap = min(est.plate_current_cap(s["T"]) for s in states)
    I_prop = min(max(0.0, I_prop), cap)
    if ok(I_prop):
        return I_prop
    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, I_prop
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def e5a_lemma(seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for N in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        reps = 24 if N <= 128 else 8      # cost grows linearly in N; 40 halvings each
        worst = 0.0
        for rep in range(reps):
            sohs = rng.uniform(0.80, 1.00, N)
            cells, states, Tambs = [], [], []
            for soh in sohs:
                f = (1.0 - soh) / 0.20
                c = make_cell(1.0 + 0.8 * f, soh, 1.0 + 0.6 * f)
                cells.append(c)
                states.append(c.init_state(float(rng.uniform(0.1, 0.7)),
                                           float(rng.uniform(20.0, 40.0))))
                Tambs.append(float(rng.uniform(20.0, 33.0)))
            # 40 halvings resolves to 15/2^40 ~ 1e-11, so any disagreement left is the
            # lemma failing rather than the bisection being coarse
            a = per_cell_current(cells, states, 15.0, Tambs, iters=40)
            b = pack_direct(cells, states, 15.0, Tambs, iters=40)
            worst = max(worst, abs(a - b))
        rows.append(dict(N=N, reps=reps, max_abs_disagreement=float(worst)))
        print(f"  N={N:>5}  max |min-over-cells  -  pack bisection| = {worst:.3e}")
    return dict(sweep=rows, lemma_holds=all(r["max_abs_disagreement"] < 1e-9 for r in rows))


def e5b_fault(n_packs=1200, N=100, fault_R=3.4, seed=4, dT_tight=CM["T"]):
    """One cell in a hundred is anomalous. Per-cell enforcement vs one averaged filter."""
    rng = np.random.default_rng(seed)
    viol_percell = viol_avg = 0
    dTs_percell, dTs_avg = [], []
    for _ in range(n_packs):
        sohs = rng.uniform(0.88, 1.00, N)
        k = int(rng.integers(0, N))
        cells, states, Tambs = [], [], []
        for i, soh in enumerate(sohs):
            f = (1.0 - soh) / 0.20
            sR = 1.0 + 0.8 * f
            if i == k:
                sR = fault_R                      # the faulted cell
            c = make_cell(sR, soh, 1.0 + 0.6 * f)
            cells.append(c)
            states.append(c.init_state(0.15, float(rng.uniform(30.0, 38.0))))
            Tambs.append(float(rng.uniform(28.0, 36.0)))

        # architecture A: a filter on every cell
        sA = [dict(s) for s in states]
        mA = -1e9
        for _ in range(60):
            I = per_cell_current(cells, sA, 15.0, Tambs, dT=dT_tight)
            for j, c in enumerate(cells):
                sA[j], o = c.step(sA[j], I, DT, Tambs[j])
                mA = max(mA, o["T"])
        # architecture B: one filter on the pack-average cell
        sB = [dict(s) for s in states]
        mB = -1e9
        avg = make_cell(float(np.mean([1.0 + 0.8 * (1 - s) / 0.2 for s in sohs])),
                        float(np.mean(sohs)), 1.0)
        for _ in range(60):
            savg = dict(soc=float(np.mean([s["soc"] for s in sB])),
                        T=float(np.mean([s["T"] for s in sB])),
                        V1=float(np.mean([s["V1"] for s in sB])),
                        aging={"Qloss": 0.0, "Rfac": 1.0})
            I, _ = project_current(_EST, savg, 15.0, DT, float(np.mean(Tambs)),
                                   Vlim=VLIM, Tlim=TLIM, margin=_EST.plating_margin(),
                                   dV=CM["V"], dT=dT_tight, dP=CM["plate"], cool_frac=F_COOL)
            for j, c in enumerate(cells):
                sB[j], o = c.step(sB[j], I, DT, Tambs[j])
                mB = max(mB, o["T"])
        dTs_percell.append(mA); dTs_avg.append(mB)
        viol_percell += int(mA > TLIM + 1e-9)
        viol_avg += int(mB > TLIM + 1e-9)
    a = stats.summarize_safety("per_cell", viol_percell, n_packs)
    b = stats.summarize_safety("pack_averaged", viol_avg, n_packs)
    w = stats.wilcoxon_paired(np.array(dTs_avg), np.array(dTs_percell), reps=5000)
    return dict(packs=n_packs, cells_per_pack=N, fault_scale=fault_R,
                per_cell=dict(a, worst_T=float(np.max(dTs_percell))),
                pack_averaged=dict(b, worst_T=float(np.max(dTs_avg))),
                peakT_wilcoxon=w)


def main():
    out = {}
    print("Act IV -- pack structure and per-cell enforcement\n")
    print("E5a  the weakest-cell lemma:  I*_pack == min_i I*_i")
    a = e5a_lemma(); out["e5a_lemma"] = a
    print(f"  lemma holds exactly at every N: {a['lemma_holds']}")

    print("\nE5b  one faulted cell in a hundred")
    t0 = time.time()
    b = e5b_fault(); out["e5b_fault"] = b
    print(f"  {b['packs']} packs x {b['cells_per_pack']} cells, one at s_R = {b['fault_scale']}, "
          f"both architectures given the same margin")
    print(f"  per-cell filters : violations {b['per_cell']['violations']:>5}/{b['packs']} "
          f"({b['per_cell']['rate_pct']:.1f}%) | worst T {b['per_cell']['worst_T']:.2f} C")
    print(f"  pack-averaged    : violations {b['pack_averaged']['violations']:>5}/{b['packs']} "
          f"({b['pack_averaged']['rate_pct']:.1f}%) | worst T {b['pack_averaged']['worst_T']:.2f} C")
    print(f"  peak-T difference: Hodges-Lehmann {b['peakT_wilcoxon']['hodges_lehmann']:+.3f} C, "
          f"p={b['peakT_wilcoxon']['p']:.3g}   ({round(time.time()-t0,1)}s)")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e5_pack.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e5_pack.json')}")


if __name__ == "__main__":
    main()
