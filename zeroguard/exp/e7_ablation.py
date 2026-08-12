"""Act VI -- taking the method apart, one mechanism at a time.

Every component of a method should be load-bearing. The way to find out is to remove each
one and see what breaks; a component whose removal changes nothing is decoration and should
be deleted rather than defended.

Seven ablations, each run on the *same* grid of aged, cooling-faulted, hot-ambient cells so
the comparisons are paired:

  A0  full method                       the reference
  A1  no I=0 feasibility check          the short-circuit that detects already-unsafe states
  A2  no conservative bound             the filter is told the cell is fresh
  A3  no cooling reserve (f = 0)        the channel that covers heat-transfer faults
  A4  no resistance throttle (K = 0)    the channel that covers resistance growth
  A5  no plating current cap            the empirical cap that carries the plating margin
  A6  bound replaced by point estimate  an estimator with realistic error, no bound
  A7  tolerance exit instead of 18 iters  what makes worst-case run time data-dependent

A7 is the only one whose consequence is not a violation count: it is the QP failure mode,
reproduced inside our own method. Removing the fixed iteration count does not make the
filter unsafe, it makes it unbounded, which is exactly the property that cannot be certified.

    python zeroguard/exp/e7_ablation.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM
from safe_charge.filter import _feasible, _eff_margin

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM, NSTEP = 30.0, 45.0, 4.20, 80
CM = dict(V=0.03, T=0.5, plate=0.006)
K_TH, F_COOL, SR = 15.0, 0.25, 1.8


def project_ablate(rom, s, I_prop, dt, T_amb, margin, dV, dT0, dP,
                   use_zero_check=True, cool_frac=F_COOL, use_cap=True,
                   iters=18, tol=None):
    """The published projection with individual mechanisms switchable off."""
    dT = dT0 + cool_frac * max(0.0, s["T"] - T_amb)
    I_prop = max(0.0, float(I_prop))
    if use_cap and hasattr(rom, "plate_current_cap"):
        I_prop = min(I_prop, rom.plate_current_cap(s["T"]))
    if _feasible(rom, s, I_prop, dt, T_amb, VLIM, TLIM, margin, dV, dT, dP):
        return I_prop, False, 0
    lo, hi = 0.0, I_prop
    if use_zero_check and not _feasible(rom, s, 0.0, dt, T_amb, VLIM, TLIM, margin, dV, dT, dP):
        return 0.0, True, 1
    n = 0
    while True:
        if tol is None:
            if n >= iters:
                break
        else:
            if (hi - lo) <= tol or n >= 400:
                break
        mid = 0.5 * (lo + hi)
        if _feasible(rom, s, mid, dt, T_amb, VLIM, TLIM, margin, dV, dT, dP):
            lo = mid
        else:
            hi = mid
        n += 1
    return lo, True, n


def grid(n=1400, seed=21):
    """Aged, cooling-faulted, hot cells -- the deployment corner, fixed once so every
    ablation sees identical conditions."""
    rng = np.random.default_rng(seed)
    g = []
    for _ in range(n):
        soh = float(rng.uniform(0.80, 1.00))
        f = (1 - soh) / 0.20
        g.append(dict(soh=soh, sR=1 + 0.8 * f, sQ=soh, sP=1 + 0.6 * f,
                      cool_loss=float(rng.uniform(0.0, 0.33)),
                      T_amb=float(rng.uniform(26.0, 38.0)),
                      soc0=float(rng.uniform(0.08, 0.20))))
    return g


ABLATIONS = {
    "A0_full":        dict(),
    "A1_no_zero_check": dict(use_zero_check=False),
    "A2_no_bound":    dict(sR_assumed=1.0),
    "A3_no_cooling_reserve": dict(cool_frac=0.0),
    "A4_no_throttle": dict(K=0.0),
    "A5_no_plating_cap": dict(use_cap=False),
    "A6_point_estimate": dict(point_estimate=True),
    "A7_tolerance_exit": dict(tol=1e-4),
}


def run_ablation(cfg, g, seed=0):
    rng = np.random.default_rng(seed)
    sR_assumed = cfg.get("sR_assumed", SR)
    K = cfg.get("K", K_TH)
    viol, socs, iters_used, peaks = 0, [], [], []
    for row in g:
        plant = BatteryROM(cell_scale=dict(R=row["sR"], Q=row["sQ"], plate=row["sP"]))
        plant.p = dict(plant.p); plant.p["hA"] *= (1.0 - row["cool_loss"])
        if cfg.get("point_estimate"):
            # a realistic estimator: unbiased but noisy, so it is sometimes optimistic
            sr = float(np.clip(row["sR"] * (1.0 + rng.normal(0, 0.12)), 0.5, 3.0))
        else:
            sr = sR_assumed
        est = BatteryROM(cell_scale=dict(R=sr, Q=1.0, plate=1.0))
        dT0 = CM["T"] + K * (sr - 1.0)
        s = plant.init_state(row["soc0"], row["T_amb"])
        mT = mV = -1e9
        for _ in range(NSTEP):
            I, _, n = project_ablate(est, s, 3 * plant.p["Q_nom"], DT, row["T_amb"],
                                     est.plating_margin(), CM["V"], dT0, CM["plate"],
                                     use_zero_check=cfg.get("use_zero_check", True),
                                     cool_frac=cfg.get("cool_frac", F_COOL),
                                     use_cap=cfg.get("use_cap", True),
                                     tol=cfg.get("tol"))
            iters_used.append(n)
            s, o = plant.step(s, float(max(0.0, I)), DT, row["T_amb"])
            mT = max(mT, o["T"]); mV = max(mV, o["V"])
            if s["soc"] >= 0.80:
                break
        socs.append(s["soc"]); peaks.append(mT)
        viol += int(mT > TLIM + 1e-9 or mV > VLIM + 1e-9)
    it = np.array(iters_used)
    return dict(**stats.summarize_safety("x", viol, len(g)),
                mean_soc=float(np.mean(socs)), worst_peak_T=float(np.max(peaks)),
                iters_mean=float(it.mean()), iters_max=int(it.max()),
                iters_p999=float(np.percentile(it, 99.9)),
                iters_is_fixed=bool(it.max() == it.min() or set(np.unique(it)) <= {0, 1, 18}))


def main():
    g = grid()
    out = {"grid_size": len(g), "ablations": {}}
    print(f"Act VI -- ablations on a fixed grid of {len(g)} aged, cooling-faulted, hot cells\n")
    print(f"{'ablation':24}{'viol':>7}{'CP95%':>9}{'meanSOC':>10}{'peakT':>9}{'iters mean/max':>17}")
    ref = None
    for name, cfg in ABLATIONS.items():
        t0 = time.time()
        r = run_ablation(cfg, g)
        r["seconds"] = round(time.time() - t0, 1)
        out["ablations"][name] = r
        if name == "A0_full":
            ref = r
        itstr = f"{r['iters_mean']:.1f}/{r['iters_max']}"
        print(f"{name:24}{r['violations']:>7}{r['cp95_upper_pct']:>8.3f}%{r['mean_soc']:>10.4f}"
              f"{r['worst_peak_T']:>9.2f}{itstr:>17}")

    print("\nconsequence of each removal, against the full method:")
    for name, r in out["ablations"].items():
        if name == "A0_full":
            continue
        dv = r["violations"] - ref["violations"]
        ds = 100 * (r["mean_soc"] - ref["mean_soc"])
        verdict = ("SAFETY" if dv > 0 else
                   ("BOUNDEDNESS" if not r["iters_is_fixed"] else
                    ("charge only" if abs(ds) > 0.05 else "no measurable effect")))
        print(f"  {name:24} {dv:+5d} violations  {ds:+7.2f} SOC pts   -> {verdict}")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e7_ablation.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e7_ablation.json')}")


if __name__ == "__main__":
    main()
