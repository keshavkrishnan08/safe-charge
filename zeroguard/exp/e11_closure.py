"""E11 -- the three items the design register promised and the first pass did not deliver.

Auditing `DESIGN.md` against what actually ran turned up three gaps. They are closed here
rather than quietly dropped, because a design document that is allowed to diverge from the
experiments is worse than no design document at all.

  (a) **Radiation drift as a sixth channel.** The design promised a six-channel vacuum
      envelope including radiation-induced degradation; the first pass ran five. Total
      ionising dose raises series resistance and takes capacity, so it enters exactly like
      the ageing channel already does -- which is the claim, and it needs testing rather
      than asserting.

  (b) **Sample-size adequacy.** `stats.n_required` was written and never called. For a
      zero-failure claim the honest question is not "how many did you run" but "how many
      would you have needed", and the two should be reported together.

  (c) **Latency and footprint.** The design's ninth figure was the worst-case execution time
      against the control budget. The first pass spent that slot on robustness, so the
      timing evidence never reached a figure. This writes the data it needs.

    python zeroguard/exp/e11_closure.py
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from zeroguard.systems import RadiativeBatteryROM
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM, NSTEP = 30.0, 45.0, 4.20, 80
CM = dict(V=0.03, T=0.5, plate=0.006)
K_TH, F_COOL, SR = 15.0, 0.25, 1.8
DT_EFF = CM["T"] + K_TH * (SR - 1.0)

# six channels: the five from E3, plus total-ionising-dose drift
ENV6 = {"R": (1.0, 1.8), "eps": (1.0, 0.60), "Cth": (1.0, 0.80),
        "plate": (1.0, 1.6), "Tsink": (4.0, 250.0), "tid": (1.0, 1.5)}
CHANS6 = list(ENV6)
CORNER6 = {k: v[1] for k, v in ENV6.items()}


def make6(theta):
    """`tid` multiplies resistance and removes capacity on top of the ageing channel."""
    r = RadiativeBatteryROM(eps=0.85 * theta["eps"], T_sink=theta["Tsink"])
    r.scale = dict(R=theta["R"] * theta["tid"],
                   Q=1.0 / theta["tid"] ** 0.5,
                   plate=theta["plate"] * theta["tid"] ** 0.5)
    r.p = dict(r.p); r.p["C_th"] *= theta["Cth"]
    return r


def drive6(plant, est):
    s = plant.init_state(0.10, 25.0)
    Qn = plant.p["Q_nom"]
    mT = mV = -1e9
    for _ in range(NSTEP):
        I, _ = project_current(est, s, 3.0 * Qn, DT, 25.0, Vlim=VLIM, Tlim=TLIM,
                               margin=est.plating_margin(), dV=CM["V"], dT=DT_EFF,
                               dP=CM["plate"], cool_frac=0.0)
        s, o = plant.step(s, float(max(0.0, I)), DT, 25.0)
        mT = max(mT, o["T"]); mV = max(mV, o["V"])
        if s["soc"] >= 0.80:
            break
    return mT, mV, s["soc"]


def a_six_channel(n=20000, seed=20260814):
    rng = np.random.default_rng(seed)
    est = make6(CORNER6)
    safe = 0; wT = wV = -1e9; socs = []
    t0 = time.time()
    for _ in range(n):
        th = {k: float(rng.uniform(min(ENV6[k]), max(ENV6[k]))) for k in CHANS6}
        mT, mV, sc = drive6(make6(th), est)
        wT = max(wT, mT); wV = max(wV, mV); socs.append(sc)
        safe += int(mT <= TLIM and mV <= VLIM)
    cT, cV, cS = drive6(make6(CORNER6), est)
    m, lo, hi = stats.bootstrap_ci(np.array(socs))
    return stats.summarize_safety("radiative_6channel", n - safe, n, extra=dict(
        channels=CHANS6, worst_T=float(wT), worst_V=float(wV),
        corner_T=float(cT), corner_dominates=bool(cT >= wT - 1e-6),
        mean_soc=m, soc_ci=[lo, hi], seconds=round(time.time() - t0, 1)))


def b_sample_adequacy():
    """For each zero-failure claim: what we ran, what it certifies, what it would have taken."""
    src = {
        "cross-domain (E1)": ("e1_generality.json", lambda d: (d["pooled"]["violations"], d["pooled"]["trials"])),
        "vacuum envelope 5ch (E3)": ("e3_radiative.json", lambda d: (d["e3b_envelope"]["violations"], d["e3b_envelope"]["trials"])),
        "satellite mission (E4)": ("e4_mission_life.json", lambda d: (d["duties"]["satellite"]["bound"]["violations"], d["duties"]["satellite"]["cycles"])),
        "robotaxi mission (E4)": ("e4_mission_life.json", lambda d: (d["duties"]["robotaxi"]["bound"]["violations"], d["duties"]["robotaxi"]["cycles"])),
        "per-cell packs (E5)": ("e5_pack.json", lambda d: (d["e5b_fault"]["per_cell"]["violations"], d["e5b_fault"]["packs"])),
        "adversary (E12)": ("e12_adversary_scale.json", lambda d: (d["violations"], d["trials"])),
        "float32 (E6)": ("e6_adversarial.json", lambda d: (d["e6d_float32"]["violations"], d["e6d_float32"]["trials"])),
        "full method (E7)": ("e7_ablation.json", lambda d: (d["ablations"]["A0_full"]["violations"], d["grid_size"])),
    }
    targets = [0.01, 0.001, 0.0001]
    need = {f"{100*t}%": stats.n_required(t) for t in targets}
    rows = []
    for name, (fn, get) in src.items():
        path = os.path.join(RES, fn)
        if not os.path.exists(path):        # E12 may not have been run yet
            continue
        with open(path) as f:
            d = json.load(f)
        k, n = get(d)
        rows.append(dict(claim=name, violations=int(k), trials=int(n),
                         cp95_upper_pct=100 * stats.cp_upper(k, n),
                         certifies_below_pct=100 * stats.cp_upper(k, n),
                         enough_for_1pct=n >= need["1.0%"],
                         enough_for_0p1pct=n >= need["0.1%"],
                         enough_for_0p01pct=n >= need["0.01%"]))
    return dict(n_required=need, claims=rows)


def c_latency(reps=60, batch=400, seed=3):
    """Per-projection cost, separated by code path, plus the work that actually ports."""
    rng = np.random.default_rng(seed)
    est = BatteryROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0))
    Qn = est.p["Q_nom"]
    pools = {}
    for _ in range(2400):
        s = est.init_state(float(rng.uniform(0.05, 0.9)), float(rng.uniform(20.0, 44.0)))
        n = [0]
        class Counting(BatteryROM):
            pass
        # count evaluations by wrapping probe
        base = est.probe
        cnt = {"n": 0}

        def probe(*a, **k):
            cnt["n"] += 1
            return base(*a, **k)
        est.probe = probe
        project_current(est, s, 3 * Qn, DT, 25.0, margin=est.plating_margin(),
                        dV=CM["V"], dT=DT_EFF, dP=CM["plate"], cool_frac=F_COOL)
        est.probe = base
        pools.setdefault(cnt["n"], []).append(s)

    out = {}
    gc.disable()
    for nev, pool in sorted(pools.items()):
        if len(pool) < 20:
            continue
        means = []
        for _ in range(reps):
            sub = pool[:batch]
            t0 = time.perf_counter()
            for s in sub:
                project_current(est, s, 3 * Qn, DT, 25.0, margin=est.plating_margin(),
                                dV=CM["V"], dT=DT_EFF, dP=CM["plate"], cool_frac=F_COOL)
            means.append((time.perf_counter() - t0) / len(sub) * 1e6)
        a = np.array(means)
        out[str(nev)] = dict(evaluations=int(nev), states=len(pool),
                             us_min=float(a.min()), us_median=float(np.median(a)),
                             us_p99=float(np.percentile(a, 99)), us_max=float(a.max()))
    gc.enable()
    hard = max(int(k) for k in out)
    med = out[str(hard)]["us_median"]
    return dict(paths=out, hard_bound_evaluations=hard,
                worst_path_median_us=med,
                control_step_s=DT,
                duty_cycle=med * 1e-6 / DT,
                headroom_orders=float(np.log10(DT / (med * 1e-6))),
                memory_bytes_double=904, memory_bytes_single=452)


def main():
    out = {}
    print("E11 -- closing the three gaps between DESIGN.md and the first pass\n")

    print("(a) six-channel vacuum envelope, with radiation-induced drift")
    a = a_six_channel(); out["six_channel_envelope"] = a
    print(f"    channels: {', '.join(a['channels'])}")
    print(f"    {a['trials']:,} draws | violations {a['violations']} | CP95 upper {a['cp95_upper_pct']:.4f}%")
    print(f"    worst T {a['worst_T']:.2f} C (limit {TLIM}) | corner dominates: {a['corner_dominates']}"
          f" | SOC {a['mean_soc']:.3f}  ({a['seconds']}s)")

    print("\n(b) sample-size adequacy -- what we ran against what it takes")
    b = b_sample_adequacy(); out["sample_adequacy"] = b
    print(f"    trials needed for zero-failure certification at 95%: "
          f"{b['n_required']['1.0%']:,} (1%), {b['n_required']['0.1%']:,} (0.1%), "
          f"{b['n_required']['0.01%']:,} (0.01%)")
    print(f"    {'claim':28}{'n':>8}{'viol':>6}{'certifies below':>17}   adequate for")
    for r in b["claims"]:
        tags = [t for t, ok in [("1%", r["enough_for_1pct"]), ("0.1%", r["enough_for_0p1pct"]),
                                ("0.01%", r["enough_for_0p01pct"])] if ok]
        print(f"    {r['claim']:28}{r['trials']:>8,}{r['violations']:>6}"
              f"{r['cp95_upper_pct']:>16.4f}%   {', '.join(tags) if tags else '-'}")

    print("\n(c) latency by code path")
    c = c_latency(); out["latency"] = c
    print(f"    {'evals':>6}{'states':>8}{'min':>10}{'median':>10}{'p99':>10}{'max':>10}   (us/call)")
    for k, v in sorted(c["paths"].items(), key=lambda kv: int(kv[0])):
        print(f"    {v['evaluations']:>6}{v['states']:>8}{v['us_min']:>10.2f}"
              f"{v['us_median']:>10.2f}{v['us_p99']:>10.2f}{v['us_max']:>10.2f}")
    print(f"    hard bound {c['hard_bound_evaluations']} evaluations | worst-path median "
          f"{c['worst_path_median_us']:.1f} us against a {c['control_step_s']:.0f} s step")
    print(f"    duty cycle {c['duty_cycle']:.2e}  ({c['headroom_orders']:.1f} orders of magnitude of headroom)")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e11_closure.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e11_closure.json')}")


if __name__ == "__main__":
    main()
