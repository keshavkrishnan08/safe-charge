"""M1 -- a margin that knows how long the operation lasts.

D1 found the certificate recovering 24 points less braking energy than a fixed cap that is
itself safe, and blamed a design choice rather than the theorem. This experiment takes that
diagnosis seriously enough to test it.

**What the thermal margin is actually for.** The throttle is
`delta_T = delta_T0 + K (s_R - 1)` with `K = 15`, calibrated in the original work as the
smallest value that certifies cells past the rated resistance bound. It is natural to read it as
covering the one-step prediction error, and that reading is wrong by more than an order of
magnitude: at 540 A over a 30 s step, resistance being `s_R` times its assumed value adds
**0.85 K**, against a margin of 12.5 K.

The margin is not covering one step. It is covering *accumulated* divergence, because a
one-step filter applied repeatedly lets a model error compound: sixty consecutive steps of a
half-hour charge, each diverging a fraction of a kelvin, is where 12.5 K comes from.

**Which makes applying it to regeneration a category error.** A regenerative braking pulse
lasts seconds and is followed by cooling. It cannot accumulate thirty minutes of divergence,
and charging it thirty minutes' worth of margin is why the certificate gives away recovered
energy.

**The proposed refinement.** Make the margin a function of the duration over which the
operation will be sustained:

    delta_T(tau) = delta_T0 + k (s_R - 1) tau,   k chosen so tau = 1800 s reproduces K = 15

`tau` is not a prediction of the future. It is a property of the *operating mode*, supplied by
whoever integrates the filter: a charging session sustains current for its duration, a
regeneration pulse for the length of a braking event. Getting it wrong in the unsafe direction
means claiming an operation is shorter than it is, which is a modelling error of the same kind
as claiming a resistance bound that is too tight.

**And the whole point is that this must be tested, not asserted.** A smaller margin is a weaker
guarantee unless the accumulated divergence really is smaller. Both are measured below, and if
safety fails the refinement is rejected.

    python zeroguard/exp/m1_duration_margin.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V
from zeroguard.exp.d1_drive_cycles import load_cycle, pack_power, VEH

TAU_REF = 1800.0                      # the charge duration K = 15 was calibrated against
K_RATE = V.K_TH / TAU_REF             # K per second per unit of assumed resistance growth
SEED = 20260816


def margin_for(plat, tau, s_R=V.S_R):
    """Margins with the thermal entry scaled by how long the operation is sustained."""
    dT = V.DT0 + K_RATE * (s_R - 1.0) * float(tau)
    if plat.mode == "charge":
        return (V.DV, dT, V.DP)
    return (dT, V.DV, V.DSOC, 0.02 * max(plat.load_W, 1e-9))


# ---------------------------------------------------------------------------------------
def a_reduces_correctly():
    """At the calibration duration the new margin must be the old one, exactly."""
    plat = V.pessimistic("robotaxi-urban")
    old = V.margins(plat)
    new = margin_for(plat, TAU_REF)
    return dict(tau_ref_s=TAU_REF, k_rate_K_per_s=K_RATE,
                old_thermal_K=old[1], new_thermal_K=new[1],
                agrees=bool(abs(old[1] - new[1]) < 1e-9),
                margin_at_1s=margin_for(plat, 1.0)[1],
                margin_at_5s=margin_for(plat, 5.0)[1],
                margin_at_60s=margin_for(plat, 60.0)[1])


# ---------------------------------------------------------------------------------------
def b_charging_unchanged(n=1500, seed=SEED):
    """A sustained charge must be exactly as safe as before -- the margin there is unchanged."""
    rng = np.random.default_rng(seed)
    est = V.pessimistic("robotaxi-urban", T_amb=30.0)
    marg = margin_for(est, TAU_REF)
    viol = 0; socs = []
    for _ in range(n):
        plant, _ = V.draw_plant("robotaxi-urban", rng, V.ENVELOPE, T_amb=30.0)
        r = V.charge_session(plant, est, est.init(float(rng.uniform(0.05, 0.30)),
                                                  float(rng.uniform(20.0, 40.0))),
                             30.0, 30.0, marg)
        viol += int(not r["ok"]); socs.append(r["soc"])
    return stats.summarize_safety("charge_duration_margin", viol, n,
                                  extra=dict(mean_soc=float(np.mean(socs))))


# ---------------------------------------------------------------------------------------
def c_regen_over_cycles(n=40, seed=SEED + 1, pulse_tau=5.0):
    """The regeneration case, on the EPA schedules, with the margin sized for a pulse.

    The charging side of every run keeps the full sustained margin. Only the braking side is
    given the pulse-duration margin, because only the braking side is a pulse.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for fname, label in (("us06col.txt", "US06"), ("uddscol.txt", "UDDS"),
                         ("hwycol.txt", "HWFET")):
        t, v = load_cycle(fname)
        dt, Pwr = pack_power(t, v)
        acc = {"sustained": dict(rec=[], breach=0), "duration_aware": dict(rec=[], breach=0)}
        for _ in range(n):
            sc = {k: float(rng.uniform(*q)) for k, q in V.ENVELOPE.items()}
            T_amb = float(rng.choice([-5.0, 5.0, 15.0, 25.0, 35.0]))
            soc0 = float(rng.uniform(0.35, 0.85))
            for mode in ("sustained", "duration_aware"):
                est_c = V.pessimistic("robotaxi-urban", T_amb=T_amb)
                plant_c = P.RobotaxiUrban(T_amb=T_amb, scale=sc)
                marg = (margin_for(est_c, TAU_REF) if mode == "sustained"
                        else margin_for(est_c, pulse_tau))
                s = est_c.init(soc0, T_amb + 2.0)
                avail = used = 0.0; breach = 0
                S = est_c.S
                for k in range(len(t) - 1):
                    h = float(max(dt[k], 1e-3))
                    if h > 5.0:
                        continue
                    Pk = float(Pwr[k])
                    if Pk >= 0.0:
                        # discharge the pack along the cycle so the thermal state is realistic
                        Vc = float(plant_c.probe(s, 0.0, h, T_amb)[0])
                        u = -min(Pk / max(S * Vc, 1.0), est_c.u_max)
                        s, _o = plant_c.step(s, u, h, T_amb)
                        continue
                    Vc = float(plant_c.probe(s, 0.0, h, T_amb)[0])
                    want = min(-Pk / max(S * Vc, 1.0), est_c.u_max)
                    avail += -Pk * h
                    _lo, hi, st = A.interval(est_c, s, h, T_amb, marg)
                    u = max(0.0, min(want, hi) if st == "ok" else 0.0)
                    s, o = plant_c.step(s, u, h, T_amb)
                    used += u * S * float(o[0]) * h
                    bad, _e = V.split_breaches(plant_c, V.check(plant_c, o))
                    if bad and st == "ok":
                        breach += 1
                acc[mode]["rec"].append(min(used / max(avail, 1e-9), 1.0))
                acc[mode]["breach"] += breach
        rec = {}
        for mode, a in acc.items():
            r = np.array(a["rec"])
            rec[mode] = dict(recovery_mean=float(r.mean()),
                             breach_while_certified=a["breach"])
        rec["recovery_gain_points"] = 100 * (rec["duration_aware"]["recovery_mean"]
                                             - rec["sustained"]["recovery_mean"])
        out[fname] = dict(label=label, **rec)
    return out


# ---------------------------------------------------------------------------------------
def d_where_it_breaks(n=1400, seed=SEED + 2, beyond=2.4):
    """How short can the assumed duration get before safety actually fails?

    A refinement that shrinks a margin must be shown to have a floor, or it is a knob someone
    will eventually turn too far. The first version of this sweep found no floor at all -- the
    margin could be cut to 0.51 K with zero violations -- and that was a flaw in the test, not a
    property of the method: every plant was drawn with resistance *inside* the bound the
    estimator assumes, so the margin was never asked to absorb anything.

    The margin exists for cells that have gone **past** the datasheet bound, which is exactly
    what the original calibration of `K = 15` was against. So the plants here are drawn with
    resistance beyond the assumed bound while the estimator still assumes s_R -- a genuinely
    wrong model, which is the only condition under which a thermal margin has a job.
    """
    rng = np.random.default_rng(seed)
    rows = []
    env = dict(V.ENVELOPE); env["R"] = (V.S_R, beyond)     # every cell past the assumed bound
    taus = (1800.0, 600.0, 200.0, 60.0, 20.0, 5.0, 1.0)
    for tau in taus:
        est = V.pessimistic("robotaxi-urban", T_amb=30.0)
        marg = margin_for(est, tau)
        viol = 0; socs = []; worstT = -1e9
        for _ in range(n // len(taus)):
            plant, _ = V.draw_plant("robotaxi-urban", rng, env, T_amb=30.0)
            r = V.charge_session(plant, est, est.init(0.10, float(rng.uniform(20.0, 40.0))),
                                 30.0, 30.0, marg)
            viol += int(not r["ok"]); socs.append(r["soc"]); worstT = max(worstT, r["peak_T"])
        rows.append(dict(assumed_tau_s=tau, thermal_margin_K=marg[1],
                         trials=n // len(taus), violations=viol,
                         violation_rate=viol / (n // len(taus)),
                         peak_T=float(worstT), mean_soc=float(np.mean(socs))))
    first_bad = next((r["assumed_tau_s"] for r in rows if r["violations"] > 0), None)
    safe_taus = [r["assumed_tau_s"] for r in rows if r["violations"] == 0]

    # No floor appears along tau at this level of model error, so ask the question the other
    # way round: holding the margin at its smallest (a 5 s pulse), how wrong may the plant be
    # before the guarantee gives way? That is the quantity an integrator actually needs.
    est = V.pessimistic("robotaxi-urban", T_amb=30.0)
    small = margin_for(est, 5.0)
    tol = []
    for mult in (1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0):
        e2 = dict(V.ENVELOPE); e2["R"] = (mult * 0.95, mult)
        viol = 0; worst = -1e9
        for _ in range(200):
            plant, _ = V.draw_plant("robotaxi-urban", rng, e2, T_amb=30.0)
            r = V.charge_session(plant, est, est.init(0.10, float(rng.uniform(20.0, 40.0))),
                                 30.0, 30.0, small)
            viol += int(not r["ok"]); worst = max(worst, r["peak_T"])
        tol.append(dict(plant_R_multiple=mult, over_assumed_bound=mult / V.S_R,
                        trials=200, violations=viol, peak_T=float(worst)))
    breaks_at = next((r["plant_R_multiple"] for r in tol if r["violations"] > 0), None)
    return dict(rows=rows, true_duration_s=TAU_REF,
                plant_resistance_beyond_bound=beyond, assumed_bound=V.S_R,
                first_unsafe_assumed_tau_s=first_bad,
                smallest_safe_assumed_tau_s=min(safe_taus) if safe_taus else None,
                pulse_margin_K=small[1],
                tolerance_sweep=tol, breaks_at_R_multiple=breaks_at,
                breaks_at_factor_over_bound=(breaks_at / V.S_R if breaks_at else None),
                has_a_floor=breaks_at is not None)


# ---------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("M1 -- a margin that knows how long the operation lasts\n" + "=" * 78)
    out = {}

    print("\n(a) does it reduce to the calibrated margin?")
    r = a_reduces_correctly(); out["reduction"] = r
    print(f"    k = {r['k_rate_K_per_s']:.5f} K/s per unit s_R | at tau = {r['tau_ref_s']:.0f} s: "
          f"{r['new_thermal_K']:.3f} K vs the calibrated {r['old_thermal_K']:.3f} K "
          f"-> agrees: {r['agrees']}")
    print(f"    margin at 60 s {r['margin_at_60s']:.2f} K | at 5 s {r['margin_at_5s']:.2f} K | "
          f"at 1 s {r['margin_at_1s']:.2f} K")

    print("\n(b) sustained charging must be unchanged")
    r = b_charging_unchanged(); out["charging"] = r
    print(f"    {r['trials']:,} sessions | violations {r['violations']} | CP95 "
          f"{r['cp95_upper_pct']:.3f}% | SOC {r['mean_soc']:.3f}")

    print("\n(c) regeneration on the EPA schedules, margin sized for a pulse")
    r = c_regen_over_cycles(); out["regen"] = r
    print(f"    {'cycle':8}{'sustained':>12}{'duration-aware':>17}{'gain':>9}"
          f"{'breaches':>10}")
    for fn, d in r.items():
        print(f"    {d['label']:8}{100*d['sustained']['recovery_mean']:>11.1f}%"
              f"{100*d['duration_aware']['recovery_mean']:>16.1f}%"
              f"{d['recovery_gain_points']:>+9.1f}"
              f"{d['duration_aware']['breach_while_certified']:>10}")
    tot = sum(d["duration_aware"]["breach_while_certified"] for d in r.values())
    gain = float(np.mean([d["recovery_gain_points"] for d in r.values()]))
    out["regen_breaches"] = tot
    out["regen_mean_gain_points"] = gain
    print(f"    mean gain {gain:+.1f} points | breaches while certified: {tot}")

    print("\n(d) how far can the assumed duration be pushed before safety fails?")
    r = d_where_it_breaks(); out["floor"] = r
    print(f"    plants drawn with resistance up to {r['plant_resistance_beyond_bound']:.1f}x "
          f"while the estimator still assumes {r['assumed_bound']:.1f}x -- cells past the bound, "
          f"which is what the margin is for")
    print(f"    {'assumed tau':>13}{'margin':>10}{'viol':>7}{'rate':>8}{'peak T':>9}{'SOC':>8}")
    for row in r["rows"]:
        print(f"    {row['assumed_tau_s']:>12.0f}s{row['thermal_margin_K']:>9.2f}K"
              f"{row['violations']:>7}{100*row['violation_rate']:>7.1f}%"
              f"{row['peak_T']:>8.1f}C{row['mean_soc']:>8.3f}")
    print(f"    no floor along tau at this level of model error (peak T reaches "
          f"{r['rows'][-1]['peak_T']:.1f} C of 45). Asking it the other way round:")
    print(f"    holding the margin at its smallest ({r['pulse_margin_K']:.2f} K, a 5 s pulse), "
          f"how wrong may the plant be?")
    print(f"    {'plant R':>10}{'x bound':>10}{'viol':>7}{'peak T':>9}")
    for row in r["tolerance_sweep"]:
        print(f"    {row['plant_R_multiple']:>9.1f}x{row['over_assumed_bound']:>9.2f}x"
              f"{row['violations']:>7}{row['peak_T']:>8.1f}C")
    if r["breaks_at_R_multiple"]:
        print(f"    the pulse margin gives way at {r['breaks_at_R_multiple']:.1f}x resistance, "
              f"{r['breaks_at_factor_over_bound']:.2f}x beyond the assumed bound "
              f"-> the refinement has a floor: {r['has_a_floor']}")
    else:
        print(f"    no floor found even at 6x resistance -> {r['has_a_floor']}")

    out["verdict"] = dict(
        has_a_demonstrated_floor=out["floor"]["has_a_floor"],
        reduces_correctly=out["reduction"]["agrees"],
        charging_still_safe=out["charging"]["violations"] == 0,
        regen_still_safe=out["regen_breaches"] == 0,
        regen_gain_points=out["regen_mean_gain_points"],
        accepted=bool(out["reduction"]["agrees"]
                      and out["charging"]["violations"] == 0
                      and out["regen_breaches"] == 0
                      and out["regen_mean_gain_points"] > 0))
    print(f"\n  refinement accepted: {out['verdict']['accepted']}")

    path = V.save("m1_duration_margin.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
