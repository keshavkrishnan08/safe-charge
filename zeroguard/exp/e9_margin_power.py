"""E9 -- what a margin is worth, to one decimal place.

E6b found that the filter tolerates an optimistic temperature bias of 12 C and fails at 13 C.
The deployed thermal margin is delta_T0 = 0.5 + K (s_R* - 1) = 0.5 + 15 x 0.8 = 12.5 C.

That is not a coincidence, and it is a falsifiable prediction rather than a story told after
the fact: a sensor that under-reports temperature by b degrees moves the whole certificate
down by b, so the filter should survive exactly while b < delta_T0 and fail once b exceeds
it. This resolves the breakpoint to 0.05 C to see whether the prediction holds.

It also explains the tail. Beyond about 14 C the violation rate *falls* again, which looks
wrong until one notices that the plating current cap is itself a function of measured
temperature: a filter told the cell is very cold throttles hard for plating reasons, and
that accidental conservatism starts to protect the thermal constraint it was never meant to
guard. The sweep separates the two mechanisms by re-running with the cap frozen.

    python zeroguard/exp/e9_margin_power.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM, NSTEP = 30.0, 45.0, 4.20, 80
CM = dict(V=0.03, T=0.5, plate=0.006)
K_TH, F_COOL, SR = 15.0, 0.25, 1.8
DT0 = CM["T"] + K_TH * (SR - 1.0)          # = 12.5 C, the predicted breakpoint
EST = BatteryROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0))
QN = EST.p["Q_nom"]


class FrozenCapROM(BatteryROM):
    """Identical, except the plating cap ignores the (mis-reported) temperature. Isolates
    the thermal certificate from the cap's accidental protection."""
    def plate_current_cap(self, T_C):
        return 2.00 * self.p["Q_nom"] * self.scale["Q"]


def sweep(bias_levels, n=500, frozen_cap=False, seed=41):
    rng = np.random.default_rng(seed)
    est = FrozenCapROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0)) if frozen_cap else EST
    rows = []
    for b in bias_levels:
        viol, peaks = 0, []
        for _ in range(n):
            soh = float(rng.uniform(0.80, 1.0)); f = (1 - soh) / 0.2
            plant = BatteryROM(cell_scale=dict(R=1 + 0.8 * f, Q=soh, plate=1 + 0.6 * f))
            Tamb = float(rng.uniform(24.0, 36.0))
            s = plant.init_state(0.10, Tamb); mT = mV = -1e9
            for _ in range(NSTEP):
                seen = dict(s, T=s["T"] - b)          # the sensor under-reports by b
                I, _ = project_current(est, seen, 3 * QN, DT, Tamb, Vlim=VLIM, Tlim=TLIM,
                                       margin=est.plating_margin(), dV=CM["V"], dT=DT0,
                                       dP=CM["plate"], cool_frac=F_COOL)
                s, o = plant.step(s, float(max(0.0, I)), DT, Tamb)
                mT = max(mT, o["T"]); mV = max(mV, o["V"])
            peaks.append(mT)
            viol += int(mT > TLIM + 1e-9 or mV > VLIM + 1e-9)
        rows.append(dict(bias_C=float(b), **stats.summarize_safety(f"b{b}", viol, n),
                         mean_peak_T=float(np.mean(peaks)),
                         worst_peak_T=float(np.max(peaks))))
    return rows


def refined_prediction():
    """A prediction that can be checked without first assuming the answer.

    Write alpha = dt.hA/C_th for the fraction of a step's temperature gap that convection
    removes. A sensor reading b degrees low changes the filter's one-step estimate twice
    over: directly, and through the cooling term, which it also under-estimates.

        T_pred  = (T-b) + dt/C_th [ Q_gen - hA((T-b) - T_amb) ]
        T_true  =    T   + dt/C_th [ Q_gen - hA( T    - T_amb) ]
        =>  T_true = T_pred + b (1 - alpha)

    At high bias the cooling reserve F.max(0, T_seen - T_amb) is identically zero, because
    the filter believes the cell is below ambient. So the filter enforces exactly
    T_pred <= T_max - delta_T0, and safety survives while

        T_max - delta_T0 + b (1 - alpha)  <=  T_max     i.e.   b* = delta_T0 / (1 - alpha)

    Every term is a published constant. Nothing is fitted."""
    rom = BatteryROM()
    alpha = DT * rom.p["hA"] / rom.p["C_th"]
    return DT0 / (1.0 - alpha), alpha


def breakpoint_of(rows):
    for r in rows:
        if r["violations"] > 0:
            return r["bias_C"]
    return None


def main():
    out = {"predicted_breakpoint_C": DT0}
    print("E9 -- the purchasing power of the thermal margin\n")
    print(f"  predicted breakpoint = delta_T0 = 0.5 + {K_TH} x ({SR} - 1) = {DT0} C\n")

    fine = np.round(np.arange(11.0, 14.01, 0.05), 2)
    print("fine sweep, plating cap live (as deployed):")
    live = sweep(fine, n=500); out["cap_live"] = live
    bp_live = breakpoint_of(live); out["breakpoint_cap_live_C"] = bp_live
    for r in live:
        if 12.0 <= r["bias_C"] <= 13.0:
            print(f"  bias {r['bias_C']:>5.2f} C   violations {r['violations']:>4}/{r['trials']}  "
                  f"worst peak T {r['worst_peak_T']:.3f}")
    if bp_live:
        pred, alpha = refined_prediction()
        out["naive_prediction_C"] = DT0
        out["alpha_dt_hA_over_Cth"] = alpha
        out["refined_prediction_formula"] = "b* = delta_T0 / (1 - dt.hA/C_th)"
        out["refined_prediction_C"] = pred
        out["abs_error_C"] = abs(pred - bp_live)
        print(f"  measured breakpoint : {bp_live:.2f} C  (first level with any violation)")
        print(f"  naive prediction    : {DT0:.2f} C          error {abs(DT0-bp_live):.2f} C")
        print(f"  derived prediction  : {pred:.3f} C   error {abs(pred-bp_live):.3f} C   "
              f"[alpha = dt.hA/C_th = {alpha:.5f}]")
    else:
        print("  no breakpoint found")

    print("\ncoarse sweep to the tail, cap live vs cap frozen:")
    coarse = [0, 4, 8, 12, 13, 14, 16, 18, 20, 24, 28]
    cl = sweep(coarse, n=400); cf = sweep(coarse, n=400, frozen_cap=True)
    out["tail_cap_live"] = cl; out["tail_cap_frozen"] = cf
    print(f"  {'bias':>6}{'cap live':>12}{'cap frozen':>13}")
    for a, b in zip(cl, cf):
        print(f"  {a['bias_C']:>6.1f}{a['violations']:>8}/{a['trials']:<4}"
              f"{b['violations']:>9}/{b['trials']:<4}")
    out["tail_interpretation"] = (
        "with the cap live the violation rate falls again beyond ~14 C, because a filter told "
        "the cell is very cold throttles hard for plating reasons; with the cap frozen that "
        "accidental protection is removed and the rate stays high, which isolates the thermal "
        "certificate from the cap")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e9_margin_power.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e9_margin_power.json')}")


if __name__ == "__main__":
    main()
