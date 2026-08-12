"""Act V -- trying to break it on purpose.

A safety argument that has only been tested against well-behaved policies has not been
tested. Four attacks:

  E6a  an adversary that *optimizes* the request sequence to maximise peak temperature,
       by cross-entropy search over the whole episode rather than by being greedy
  E6b  corrupted sensing: noise, bias, dropout and quantisation on what the filter reads
  E6c  scheduler jitter: the control step is not the step the model assumed
  E6d  reduced arithmetic: the same run in float32

E6b is the one that should not come out clean, and the experiment is designed so that it
cannot be reported as clean. A filter reads sensors; if the sensors lie about temperature in
the optimistic direction, no amount of monotone structure will save it. What the margin buys
is a *quantity* of lying it can absorb, and that quantity is measurable. Reporting where the
certificate breaks is worth more than asserting that it does not.

    python zeroguard/exp/e6_adversarial.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM, NSTEP = 30.0, 45.0, 4.20, 80
CM = dict(V=0.03, T=0.5, plate=0.006)
K_TH, F_COOL = 15.0, 0.25
DT_EFF = CM["T"] + K_TH * 0.8
EST = BatteryROM(cell_scale=dict(R=1.8, Q=1.0, plate=1.0))
QN = EST.p["Q_nom"]


def episode(plant, requests, Tamb, sensor=None, dt=DT, dT=DT_EFF):
    """Run one charge. `sensor` may distort what the filter sees, without touching the plant."""
    s = plant.init_state(0.10, Tamb)
    mT = mV = -1e9
    for k in range(len(requests)):
        s_seen = s if sensor is None else sensor(s, k)
        I, _ = project_current(EST, s_seen, float(requests[k]), dt, Tamb,
                               Vlim=VLIM, Tlim=TLIM, margin=EST.plating_margin(),
                               dV=CM["V"], dT=dT, dP=CM["plate"], cool_frac=F_COOL)
        s, o = plant.step(s, float(max(0.0, I)), dt, Tamb)
        mT = max(mT, o["T"]); mV = max(mV, o["V"])
    return mT, mV, s["soc"]


def e6a_adversary(n_cells=40, pop=60, elite=12, gens=30, seed=8):
    """Cross-entropy search over the request sequence, maximising peak temperature.

    The greedy governor is the request the filter was designed against. This is the request
    it was not: an optimizer with full knowledge of the episode, free to hold back early so
    it can push later, searching 60 x 30 = 1800 sequences per cell."""
    rng = np.random.default_rng(seed)
    best_overall, viol, rows = -1e9, 0, []
    t0 = time.time()
    for c in range(n_cells):
        soh = float(rng.uniform(0.80, 1.0))
        f = (1 - soh) / 0.2
        plant = BatteryROM(cell_scale=dict(R=1 + 0.8 * f, Q=soh, plate=1 + 0.6 * f))
        Tamb = float(rng.uniform(25.0, 38.0))
        mu = np.full(NSTEP, 0.5 * 3 * QN); sd = np.full(NSTEP, 0.35 * 3 * QN)
        best = -1e9
        for g in range(gens):
            cand = np.clip(rng.normal(mu, sd, size=(pop, NSTEP)), 0.0, 3 * QN)
            scores = np.array([episode(plant, c_, Tamb)[0] for c_ in cand])
            idx = np.argsort(scores)[-elite:]
            mu = cand[idx].mean(0); sd = cand[idx].std(0) + 1e-3
            best = max(best, float(scores.max()))
        gT, gV, _ = episode(plant, np.full(NSTEP, 3 * QN), Tamb)   # greedy reference
        rows.append(dict(soh=soh, T_amb=Tamb, adversary_peak_T=best, greedy_peak_T=float(gT),
                         gain_C=float(best - gT)))
        best_overall = max(best_overall, best)
        viol += int(best > TLIM + 1e-9)
    g = np.array([r["gain_C"] for r in rows])
    m, lo, hi = stats.bootstrap_ci(g)
    return dict(cells=n_cells, sequences_searched=int(n_cells * pop * gens),
                worst_peak_T=float(best_overall), limit=TLIM,
                violations=int(viol),
                cp95_upper_pct=100 * stats.cp_upper(viol, n_cells),
                mean_gain_over_greedy_C=m, gain_ci=[lo, hi],
                seconds=round(time.time() - t0, 1), rows=rows)


def e6b_sensors(n=600, seed=9):
    """Four corruption modes, swept to failure. The bias sweep is the informative one."""
    rng = np.random.default_rng(seed)
    out = {}

    def run_mode(make_sensor, label, levels):
        rows = []
        for lv in levels:
            viol = 0
            for _ in range(n):
                soh = float(rng.uniform(0.80, 1.0)); f = (1 - soh) / 0.2
                plant = BatteryROM(cell_scale=dict(R=1 + 0.8 * f, Q=soh, plate=1 + 0.6 * f))
                Tamb = float(rng.uniform(24.0, 36.0))
                mT, mV, _ = episode(plant, np.full(NSTEP, 3 * QN), Tamb,
                                    sensor=make_sensor(lv, rng))
                viol += int(mT > TLIM + 1e-9 or mV > VLIM + 1e-9)
            rows.append(dict(level=lv, **stats.summarize_safety(f"{label}@{lv}", viol, n)))
        first_bad = next((r["level"] for r in rows if r["violations"] > 0), None)
        return dict(sweep=rows, first_failing_level=first_bad)

    # (a) zero-mean noise on temperature
    out["noise_C"] = run_mode(
        lambda lv, r: (lambda s, k: dict(s, T=s["T"] + float(r.normal(0, lv)))),
        "noise", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    # (b) optimistic bias -- the sensor under-reports temperature (the dangerous direction)
    out["bias_C"] = run_mode(
        lambda lv, r: (lambda s, k: dict(s, T=s["T"] - lv)),
        "bias", [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 13.0, 14.0, 16.0, 20.0])
    # (c) dropout -- the filter holds the last reading
    def drop(lv, r):
        held = {}
        def f(s, k):
            if r.random() < lv and held:
                return dict(s, T=held["T"])
            held["T"] = s["T"]
            return s
        return f
    out["dropout_p"] = run_mode(drop, "dropout", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    # (d) quantisation of the temperature channel
    out["quantisation_C"] = run_mode(
        lambda lv, r: (lambda s, k: dict(s, T=(np.floor(s["T"] / lv) * lv) if lv > 0 else s["T"])),
        "quant", [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    return out


def e6c_jitter(n=1500, seed=10):
    """The scheduler does not deliver exactly 30 s. Uses the exact-ZOH ROM so steps longer
    than tau_RC are legitimate rather than an artefact of explicit Euler."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in [0.0, 0.05, 0.1, 0.2, 0.3]:
        viol = 0
        for _ in range(n):
            soh = float(rng.uniform(0.80, 1.0)); f = (1 - soh) / 0.2
            plant = BatteryROM(cell_scale=dict(R=1 + 0.8 * f, Q=soh, plate=1 + 0.6 * f),
                               rc="exact")
            est = BatteryROM(cell_scale=dict(R=1.8, Q=1.0, plate=1.0), rc="exact")
            Tamb = float(rng.uniform(24.0, 36.0))
            s = plant.init_state(0.10, Tamb); mT = mV = -1e9
            for _ in range(NSTEP):
                dt = DT * (1.0 + float(rng.uniform(-j, j)))
                I, _ = project_current(est, s, 3 * QN, dt, Tamb, Vlim=VLIM, Tlim=TLIM,
                                       margin=est.plating_margin(), dV=CM["V"], dT=DT_EFF,
                                       dP=CM["plate"], cool_frac=F_COOL)
                s, o = plant.step(s, float(max(0.0, I)), dt, Tamb)
                mT = max(mT, o["T"]); mV = max(mV, o["V"])
            viol += int(mT > TLIM + 1e-9 or mV > VLIM + 1e-9)
        rows.append(dict(jitter_frac=j, **stats.summarize_safety(f"jitter{j}", viol, n)))
    return dict(sweep=rows, all_safe=all(r["violations"] == 0 for r in rows))


def e6d_float32(n=4000, seed=12):
    """Deployment arithmetic. Re-run the projection in single precision and compare."""
    rng = np.random.default_rng(seed)
    worst = 0.0; viol = 0
    for _ in range(n):
        soh = float(rng.uniform(0.80, 1.0)); f = (1 - soh) / 0.2
        plant = BatteryROM(cell_scale=dict(R=1 + 0.8 * f, Q=soh, plate=1 + 0.6 * f))
        Tamb = float(rng.uniform(24.0, 36.0))
        s = plant.init_state(0.10, Tamb); mT = mV = -1e9
        for _ in range(NSTEP):
            s32 = dict(soc=float(np.float32(s["soc"])), T=float(np.float32(s["T"])),
                       V1=float(np.float32(s["V1"])), aging=s["aging"])
            I, _ = project_current(EST, s32, 3 * QN, DT, Tamb, Vlim=VLIM, Tlim=TLIM,
                                   margin=EST.plating_margin(), dV=CM["V"], dT=DT_EFF,
                                   dP=CM["plate"], cool_frac=F_COOL)
            I64, _ = project_current(EST, s, 3 * QN, DT, Tamb, Vlim=VLIM, Tlim=TLIM,
                                     margin=EST.plating_margin(), dV=CM["V"], dT=DT_EFF,
                                     dP=CM["plate"], cool_frac=F_COOL)
            worst = max(worst, abs(I - I64))
            s, o = plant.step(s, float(max(0.0, I)), DT, Tamb)
            mT = max(mT, o["T"]); mV = max(mV, o["V"])
        viol += int(mT > TLIM + 1e-9 or mV > VLIM + 1e-9)
    return dict(**stats.summarize_safety("float32", viol, n),
                worst_current_deviation_A=float(worst))


def main():
    out = {}
    print("Act V -- adversarial and corrupted-sensing robustness\n")

    print("E6a  cross-entropy adversary maximising peak temperature")
    a = e6a_adversary(); out["e6a_adversary"] = a
    print(f"  {a['cells']} cells x {a['sequences_searched']//a['cells']} sequences each "
          f"= {a['sequences_searched']:,} episodes searched  ({a['seconds']}s)")
    print(f"  worst peak T found: {a['worst_peak_T']:.3f} C (limit {TLIM}) | violations {a['violations']}")
    print(f"  adversary gains {a['mean_gain_over_greedy_C']:+.3f} C over greedy "
          f"95% CI [{a['gain_ci'][0]:+.3f}, {a['gain_ci'][1]:+.3f}]")

    print("\nE6b  corrupted sensing, swept to failure")
    b = e6b_sensors(); out["e6b_sensors"] = b
    for mode, r in b.items():
        fb = r["first_failing_level"]
        print(f"  {mode:16} first failing level: {fb if fb is not None else 'none in range'}")
        print("      " + "  ".join(f"{x['level']}:{x['violations']}/{x['trials']}" for x in r["sweep"]))

    print("\nE6c  scheduler jitter (exact-ZOH ROM)")
    c = e6c_jitter(); out["e6c_jitter"] = c
    for r in c["sweep"]:
        print(f"  +/-{int(100*r['jitter_frac']):>3}%  violations {r['violations']}/{r['trials']}  "
              f"CP95 {r['cp95_upper_pct']:.3f}%")

    print("\nE6d  single-precision arithmetic")
    d = e6d_float32(); out["e6d_float32"] = d
    print(f"  {d['trials']} episodes | violations {d['violations']} | "
          f"worst |I32 - I64| = {d['worst_current_deviation_A']:.3e} A")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e6_adversarial.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e6_adversarial.json')}")


if __name__ == "__main__":
    main()
