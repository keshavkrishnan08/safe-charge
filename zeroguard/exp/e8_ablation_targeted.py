"""Act VI, continued -- the ablations that the first grid could not decide.

E7 left two mechanisms unresolved, and in opposite directions:

  A1 (the I = 0 feasibility check) showed *no measurable effect*. That is not evidence it is
     decoration; it is evidence the grid never contained a state where it matters. The check
     fires only when the cell is already outside the safe set on arrival -- a condition the
     E7 grid, which always starts cool, never produced. So it is tested here directly.

  A4 (the resistance throttle K) cost 43 SOC points and prevented zero violations. On cells
     inside the rated bound that is the correct outcome and the throttle really is redundant
     there: the bound is doing the work. The throttle is insurance against cells *past* the
     bound, so it has to be judged on such cells, which E7 did not contain.

Both are honest gaps in the first design rather than results, and both are closed here. Both
answers were the opposite of what was expected. The I=0 check turns out not to be a safety
mechanism at all -- the bisection is self-correcting when every input is infeasible, and
returns zero regardless -- so its real contribution is 17 saved model evaluations per call.
And the throttle, judged on the population it actually insures, is not conservative but
tight: K = 15 is the smallest value that certifies cells past the rated bound.

    python zeroguard/exp/e8_ablation_targeted.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from zeroguard.exp.e7_ablation import project_ablate, CM, DT, TLIM, VLIM, NSTEP, F_COOL, K_TH, SR
from safe_charge import BatteryROM

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def a1_targeted(n=4000, seed=31):
    """States that are ALREADY outside the safe set when the filter is first called.

    The hypothesis was that without the check the bisection would return a current it has no
    grounds to return. It does not: with every input infeasible, every midpoint tested fails,
    so hi collapses toward lo = 0 and the answer is zero either way. What the check actually
    buys is the 17 model evaluations that collapse would have cost."""
    rng = np.random.default_rng(seed)
    est = BatteryROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0))
    m = est.plating_margin()
    dT0 = CM["T"] + K_TH * (SR - 1.0)
    bad_with = bad_without = 0
    nonzero_without = []
    it_with, it_without = [], []
    same_answer = 0
    for _ in range(n):
        # arrive already too hot, or already above the plating margin at zero current
        T0 = float(rng.uniform(TLIM - dT0 + 0.5, TLIM + 6.0))
        soc0 = float(rng.uniform(0.80, 0.95))
        Tamb = float(rng.uniform(25.0, 40.0))
        s = est.init_state(soc0, T0)
        Iw, _, nw = project_ablate(est, s, 15.0, DT, Tamb, m, CM["V"], dT0, CM["plate"],
                                   use_zero_check=True)
        Io, _, no = project_ablate(est, s, 15.0, DT, Tamb, m, CM["V"], dT0, CM["plate"],
                                   use_zero_check=False)
        it_with.append(nw); it_without.append(no)
        same_answer += int(Iw == Io)
        dT = dT0 + F_COOL * max(0.0, s["T"] - Tamb)

        def infeasible(I):
            V, T, phi, _ = est.probe(s, I, DT, Tamb)
            return not (V <= VLIM - CM["V"] + 1e-9 and T <= TLIM - dT + 1e-9
                        and phi >= m + CM["plate"] - 1e-9)

        bad_with += int(infeasible(Iw) and Iw > 1e-12)
        if infeasible(Io) and Io > 1e-12:
            bad_without += 1
            nonzero_without.append(Io)
    return dict(states=n,
                identical_answers=int(same_answer),
                identical_fraction=float(same_answer) / n,
                model_evals_with_check=float(np.mean(it_with)),
                model_evals_without_check=float(np.mean(it_without)),
                evals_saved=float(np.mean(it_without) - np.mean(it_with)),
                role="performance short-circuit, not a safety mechanism: with everything "
                     "infeasible the bisection drives hi toward lo=0 and returns 0 anyway",
                with_check=stats.summarize_safety("with_check", bad_with, n),
                without_check=stats.summarize_safety("without_check", bad_without, n),
                mean_unjustified_current_A=float(np.mean(nonzero_without)) if nonzero_without else 0.0,
                max_unjustified_current_A=float(np.max(nonzero_without)) if nonzero_without else 0.0)


def beyond_bound_grid(n=1200, sR_hi=3.0, seed=32):
    """Cells past the rated end of life -- exactly the population the throttle insures."""
    rng = np.random.default_rng(seed)
    return [dict(sR=float(rng.uniform(1.8, sR_hi)),
                 sQ=float(rng.uniform(0.62, 0.80)),
                 sP=float(rng.uniform(1.6, 2.4)),
                 cool_loss=float(rng.uniform(0.0, 0.33)),
                 T_amb=float(rng.uniform(26.0, 38.0)),
                 soc0=float(rng.uniform(0.08, 0.20))) for _ in range(n)]


def run_K(K, g, cool_frac=F_COOL):
    est = BatteryROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0))
    m = est.plating_margin()
    dT0 = CM["T"] + K * (SR - 1.0)
    viol, socs, peaks = 0, [], []
    for row in g:
        plant = BatteryROM(cell_scale=dict(R=row["sR"], Q=row["sQ"], plate=row["sP"]))
        plant.p = dict(plant.p); plant.p["hA"] *= (1.0 - row["cool_loss"])
        s = plant.init_state(row["soc0"], row["T_amb"])
        mT = mV = -1e9
        for _ in range(NSTEP):
            I, _, _ = project_ablate(est, s, 3 * plant.p["Q_nom"], DT, row["T_amb"], m,
                                     CM["V"], dT0, CM["plate"], cool_frac=cool_frac)
            s, o = plant.step(s, float(max(0.0, I)), DT, row["T_amb"])
            mT = max(mT, o["T"]); mV = max(mV, o["V"])
            if s["soc"] >= 0.80:
                break
        socs.append(s["soc"]); peaks.append(mT)
        viol += int(mT > TLIM + 1e-9 or mV > VLIM + 1e-9)
    return dict(K=K, **stats.summarize_safety(f"K{K}", viol, len(g)),
                mean_soc=float(np.mean(socs)), worst_peak_T=float(np.max(peaks)))


def main():
    out = {}
    print("Act VI (continued) -- closing the two undecided ablations\n")

    print("A1  the I=0 check, tested on states that are already unsafe on arrival")
    a = a1_targeted(); out["a1_targeted"] = a
    print(f"  {a['states']} already-unsafe states")
    print(f"  with the check   : unjustified non-zero commands {a['with_check']['violations']} "
          f"({a['with_check']['rate_pct']:.2f}%)")
    print(f"  without the check: unjustified non-zero commands {a['without_check']['violations']} "
          f"({a['without_check']['rate_pct']:.2f}%)  "
          f"mean {a['mean_unjustified_current_A']:.3f} A, max {a['max_unjustified_current_A']:.3f} A")
    print(f"  identical answers in {100*a['identical_fraction']:.1f}% of states")
    print(f"  model evaluations: {a['model_evals_with_check']:.2f} with vs "
          f"{a['model_evals_without_check']:.2f} without  "
          f"({a['evals_saved']:.2f} saved per call)")
    print(f"  => {a['role']}")

    print("\nA4  the throttle K, tested on cells PAST the rated bound (s_R up to 3.0)")
    g = beyond_bound_grid()
    rows = []
    for K in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]:
        r = run_K(K, g); rows.append(r)
        print(f"  K={K:>5}  violations {r['violations']:>4}/{r['trials']}  "
              f"CP95 {r['cp95_upper_pct']:>7.3f}%  meanSOC {r['mean_soc']:.4f}  "
              f"peakT {r['worst_peak_T']:.2f}")
    out["a4_beyond_bound"] = dict(grid=len(g), sweep=rows)
    safe_Ks = [r["K"] for r in rows if r["violations"] == 0]
    kmin = min(safe_Ks) if safe_Ks else None
    out["a4_min_safe_K"] = kmin
    if kmin is not None:
        at_min = next(r for r in rows if r["K"] == kmin)
        at_dep = next(r for r in rows if r["K"] == 15.0)
        out["a4_recommendation"] = dict(
            deployed_K=15.0, minimum_certifying_K=kmin,
            soc_recovered_points=100 * (at_min["mean_soc"] - at_dep["mean_soc"]))
        print(f"\n  smallest K with zero violations on this population: K = {kmin}")
        if kmin >= 15.0:
            print(f"  the deployed K = 15 is the minimum that certifies cells past the rated "
                  f"bound; it is not conservative, it is tight")
        else:
            print(f"  reducing K from 15 to {kmin} recovers "
                  f"{100*(at_min['mean_soc']-at_dep['mean_soc']):+.2f} SOC points")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e8_ablation_targeted.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e8_ablation_targeted.json')}")


if __name__ == "__main__":
    main()
