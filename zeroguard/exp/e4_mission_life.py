"""Act III -- a whole mission, with nobody available to recalibrate.

The title of this work promises long duration, so this is the experiment that has to pay for
it. Two duty cycles are simulated end to end, with the cell ageing continuously underneath:

  satellite   27 000 shallow eclipse cycles   -- 15 per day for five years
  robotaxi     4 000 deep charge cycles       -- roughly two per day for five years

In both, the filter is handed the datasheet end-of-life bound on day one and never told
anything again. No estimator, no recalibration, no telemetry. Against it runs an oracle arm
that is given the cell's true resistance scale every cycle -- the best any diagnostic could
possibly do.

The claim is not that the two perform the same. It is that they are equally *safe*, and that
everything the oracle buys is charge. If that holds, the estimator is a performance
component, not a safety component, and can be omitted from the trusted path entirely.

    python zeroguard/exp/e4_mission_life.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM = 30.0, 45.0, 4.20
SR_EOL = 1.8                     # the rated bound the filter is given, once, for ever
K_TH, F_COOL = 15.0, 0.25
CM = dict(V=0.03, T=0.5, plate=0.006)

DUTIES = {
    "satellite": dict(cycles=27000, steps=26, soc_lo=0.55, soc_hi=0.80, T_amb=32.0,
                      note="15 eclipse cycles/day for 5 years, shallow DoD, sun-side hot case"),
    "robotaxi": dict(cycles=4000, steps=80, soc_lo=0.10, soc_hi=0.80, T_amb=27.0,
                     note="~2 charges/day for 5 years, full depth of discharge"),
}


def soh_at(cycle, total):
    """Square-root capacity fade to exactly 0.80 SOH at end of mission -- the standard SEI
    growth shape, calibrated so the mission ends at the rated end of life."""
    return 1.0 - 0.20 * np.sqrt(cycle / total)


def scales(soh):
    f = (1.0 - soh) / 0.20
    return dict(R=1.0 + 0.8 * f, Q=soh, plate=1.0 + 0.6 * f)


def run_mission(duty, arm, seed=0, record_every=200):
    """arm='bound'   filter told s_R = 1.8 once, never updated
       arm='oracle'  filter told the true s_R every cycle"""
    d = DUTIES[duty]
    rng = np.random.default_rng(seed)
    total = d["cycles"]
    viol = 0
    trace = []
    delivered = np.empty(total)
    peakT = np.empty(total)
    t0 = time.time()
    for c in range(total):
        soh = soh_at(c, total)
        sc = scales(soh)
        plant = BatteryROM(cell_scale=sc)
        # The margin is a function of the scale the filter ASSUMES: a tighter assumption
        # earns a smaller reserve. That is the only channel through which a diagnosis can
        # buy anything, so it must be modelled or the comparison is vacuous.
        sR_assumed = SR_EOL if arm == "bound" else sc["R"]
        est = BatteryROM(cell_scale=dict(R=sR_assumed, Q=1.0, plate=1.0))
        dT = CM["T"] + K_TH * (sR_assumed - 1.0)
        Tamb = d["T_amb"] + float(rng.uniform(-3.0, 3.0))
        s = plant.init_state(d["soc_lo"], Tamb)
        mT = mV = -1e9
        for _ in range(d["steps"]):
            I, _ = project_current(est, s, 3.0 * plant.p["Q_nom"], DT, Tamb,
                                   Vlim=VLIM, Tlim=TLIM, margin=est.plating_margin(),
                                   dV=CM["V"], dT=dT, dP=CM["plate"], cool_frac=F_COOL)
            s, o = plant.step(s, float(max(0.0, I)), DT, Tamb)
            mT = max(mT, o["T"]); mV = max(mV, o["V"])
            if s["soc"] >= d["soc_hi"]:
                break
        delivered[c] = s["soc"]; peakT[c] = mT
        if mT > TLIM + 1e-9 or mV > VLIM + 1e-9:
            viol += 1
        if c % record_every == 0 or c == total - 1:
            trace.append(dict(cycle=c, soh=float(soh), true_sR=float(sc["R"]),
                              peak_T=float(mT), peak_V=float(mV),
                              delivered_soc=float(s["soc"]),
                              margin_C=float(TLIM - mT)))
    return dict(arm=arm, duty=duty, cycles=total, violations=int(viol),
                delivered=delivered, peakT=peakT, trace=trace,
                seconds=round(time.time() - t0, 1))


def main():
    out = {"duties": {}}
    print("Act III -- whole-mission runs with no recalibration\n")
    for duty in DUTIES:
        d = DUTIES[duty]
        print(f"--- {duty}: {d['cycles']:,} cycles | {d['note']}")
        rb = run_mission(duty, "bound", seed=1)
        ro = run_mission(duty, "oracle", seed=1)
        n = d["cycles"]
        sb = stats.summarize_safety(f"{duty}_bound", rb["violations"], n)
        so = stats.summarize_safety(f"{duty}_oracle", ro["violations"], n)
        gap = ro["delivered"] - rb["delivered"]
        w = stats.wilcoxon_paired(ro["delivered"], rb["delivered"], reps=5000)
        mb, lob, hib = stats.bootstrap_ci(rb["delivered"])
        mo, loo, hio = stats.bootstrap_ci(ro["delivered"])
        rec = dict(note=d["note"], cycles=n,
                   bound=dict(sb, mean_soc=mb, soc_ci=[lob, hib],
                              worst_peak_T=float(rb["peakT"].max()),
                              seconds=rb["seconds"], trace=rb["trace"]),
                   oracle=dict(so, mean_soc=mo, soc_ci=[loo, hio],
                               worst_peak_T=float(ro["peakT"].max()),
                               seconds=ro["seconds"], trace=ro["trace"]),
                   soc_gap_mean=float(gap.mean()),
                   soc_gap_ci=list(stats.bootstrap_ci(gap)[1:]),
                   wilcoxon=w,
                   steps_simulated=int(n * d["steps"] * 2))
        out["duties"][duty] = rec
        print(f"  bound  : violations {sb['violations']:>3} | CP95 {sb['cp95_upper_pct']:.4f}% | "
              f"peak T {rb['peakT'].max():.2f} C | SOC {mb:.4f} [{lob:.4f},{hib:.4f}]  ({rb['seconds']}s)")
        print(f"  oracle : violations {so['violations']:>3} | CP95 {so['cp95_upper_pct']:.4f}% | "
              f"peak T {ro['peakT'].max():.2f} C | SOC {mo:.4f} [{loo:.4f},{hio:.4f}]  ({ro['seconds']}s)")
        print(f"  charge the oracle buys: {100*gap.mean():+.3f} SOC points  "
              f"(Wilcoxon p={w['p']:.3g}, rank-biserial {w['rank_biserial']:+.3f})")
        print(f"  safety difference: {sb['violations']} vs {so['violations']} violations\n")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e4_mission_life.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"wrote {os.path.join(RES,'e4_mission_life.json')}")


if __name__ == "__main__":
    main()
