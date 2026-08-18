"""S1 -- the failure mode, in the units it is measured in.

The paper enforces a plating constraint throughout and is careful never to certify it. What it
has not done is say how much plating that prevents, and "prevents lithium plating" is a claim
that should carry a number or not be made.

The chain worth being precise about is: plating deposits metallic lithium on the anode, the
deposits grow as dendrites, a dendrite that bridges the separator is an internal short, and an
internal short is the initiating event for thermal runaway. This experiment measures the
**first link only**, and the distinction matters:

  * The certificate holds cell temperature at a *service* limit of 45 C. That is far below any
    self-heating threshold, so nothing here prevents runaway by keeping the cell cool -- the
    thermal constraint is about calendar life and comfort, not fire.
  * What it does is keep the anode potential above the deposition onset. That is upstream of
    the entire chain, and it is where the contribution lives.

Nothing below models dendrite growth, separator breach or runaway, and no claim is made about
them. What is measured is capacity lost to plating, in amp-hours, by a Doyle--Fuller--Newman
cell with irreversible plating physics we did not write -- which is also, incidentally, the
dominant degradation mechanism in fast charging, so the same number answers the lifetime
question as well as the safety one.

**Where this has to be measured.** Plating is a cold-charge failure mode: the anode potential
that must stay positive falls as the cell cools, because the kinetics that would intercalate the
lithium slow down while the current does not. Measured on the DFN, charging at 1.5C crosses the
deposition onset at 0 C and does not come close at 25 C. Testing plating at room temperature is
therefore testing the regime where it does not happen, so the sweep runs cold, warm and mild and
the cold end is the one the claim rests on.

**And the obvious metric is the wrong one.** Total lithium plated over a session does not
separate these controllers, and the reason is worth stating because it is not obvious: DFN's
plating variable accumulates a slow background side reaction whenever the anode potential is
low, not only when it is negative, so a *slower* charge can accumulate more of it while never
once entering the regime that grows dendrites. Measured, the fast rate plates less total lithium
than the certificate at 0 C and crosses the deposition onset in a quarter of its sessions, which
the certificate never does. The quantity that matters is therefore not mass but whether the
anode potential goes negative at all -- that crossing is what deposits metallic lithium rather
than consuming it in a film -- and how long it stays there. Both are reported, and so is the
mass, including where the mass goes against us.

**And it has to be measured at matched charge.** The first version of this experiment compared
total plating between controllers that delivered different amounts of charge, and duly found
that the certificate -- which delivers more -- plated more than a slow de-rate. That is not a
result, it is a units error: more charge plates more lithium. Every controller below therefore
runs to the *same* target state of charge, and what is compared is the damage done to deliver
it, plus the time taken.

Three controllers on identical DFN plants and identical initial states:

  **certificate**   the filter, at the worst corner of the envelope
  **CC-CV 0.5C**    the de-rate B3 shows is safe across the deployment fleet
  **CC-CV 1.5C**    what an operator in a hurry actually wants, and what a de-rate exists to
                    forbid -- included because a constraint is only worth enforcing if
                    violating it costs something measurable

    python zeroguard/exp/s1_plating.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V
from zeroguard.exp.n2_dfn import build, observe, K0, DT, T_AMB_C, pessimistic_cell

warnings.filterwarnings("ignore")
SEED = 20260816
# The horizon has to be long enough that every controller actually *reaches* the target, or the
# matched-charge comparison silently stops being one. At 160 steps the certificate reached 70 %
# in 0 of 8 cold sessions -- it was still climbing when the horizon ended -- so its plating
# figure was for less charge than everyone else's, which is the very units error this experiment
# was restructured to avoid. `verify_vehicles.py` checks that every controller reached the
# target, and that check is what caught it.
HORIZON, TARGET = 400, 0.70
AMBIENTS = (0.0, 10.0, 25.0)

PLATED = "Loss of capacity to negative lithium plating [A.h]"
PHI = "X-averaged negative electrode surface potential difference [V]"


def session(sim, q_nom, soc0, control, T_amb, steps=HORIZON, target=TARGET):
    """One charge on the DFN, reporting what the DFN itself says was lost."""
    sol = None
    plated0 = None
    peakT, minphi, delivered = -1e9, 1e9, 0.0
    at_risk_s = 0.0            # seconds with the anode potential at or below the onset
    plated_at_risk = 0.0       # amp-hours deposited while it was
    prev_plated = None
    for k in range(steps):
        obs = observe(sol, soc0, q_nom) if sol is not None else dict(
            V=np.nan, T=T_amb, phi=np.nan, soc=soc0)
        u = float(max(0.0, control(k, obs)))
        try:
            if sol is None:
                sol = sim.solve([0, DT], inputs={"I": -u}, initial_soc=soc0)
            else:
                sol = sim.step(DT, inputs={"I": -u}, starting_solution=sol)
        except Exception as e:
            return None
        o = observe(sol, soc0, q_nom)
        if plated0 is None:
            plated0 = float(sol[PLATED].entries[0])
        peakT = max(peakT, o["T"]); minphi = min(minphi, o["phi"])
        delivered = o["soc"]
        now = float(sol[PLATED].entries[-1])
        if prev_plated is not None and o["phi"] <= 0.0:
            at_risk_s += DT
            plated_at_risk += now - prev_plated
        prev_plated = now
        if o["soc"] >= target:
            break
    return dict(plated_Ah=float(sol[PLATED].entries[-1]) - plated0,
                plated_at_risk_Ah=plated_at_risk, at_risk_min=at_risk_s / 60.0,
                peak_T=peakT, min_phi=minphi, soc=delivered,
                reached=bool(delivered >= target - 1e-9),
                minutes=(k + 1) * DT / 60.0)


def main(n=8, seed=SEED):
    t0 = time.time()
    print("S1 -- capacity lost to lithium plating, measured by the DFN\n" + "=" * 78)
    print(f"  every controller runs to the same {TARGET:.0%} target, so what is compared is "
          f"the damage done\n  to deliver the same charge -- and the time taken to deliver it")
    q = P.single_cell().cell.q_nom()
    out = dict(sessions=n, dt_s=DT, target_soc=TARGET, ambients=list(AMBIENTS),
               plating_variable=PLATED, by_ambient={})

    for amb in AMBIENTS:
        rng = np.random.default_rng(seed)
        est = P.single_cell(scale=dict(R=V.S_R, Q=V.ENVELOPE["Q"][0],
                                       plate=V.ENVELOPE["plate"][1]))
        marg = V.margins(est)

        def certificate():
            fs = {}
            def f(k, o):
                if k == 0:
                    fs["s"] = est.init(o["soc"], amb)
                else:
                    fs["s"] = dict(fs["s"], soc=o["soc"], T=o["T"])
                _lo, hi, st = A.interval(est, fs["s"], DT, amb, marg)
                u = hi if st == "ok" else 0.0
                fs["s"], _ = est.step(fs["s"], u, DT, amb)
                return u
            return f

        def ccv(c_rate):
            st = {"u": c_rate * q}
            def f(k, o):
                if k > 0 and np.isfinite(o["V"]) and o["V"] >= 4.20 - 1e-2:
                    st["u"] *= 0.85       # the constant-voltage taper, as production does it
                return st["u"]
            return f

        rules = {"certificate": certificate,
                 "CC-CV 0.5C": lambda: ccv(0.5), "CC-CV 1.5C": lambda: ccv(1.5)}
        acc = {k: dict(plated=[], peakT=[], minphi=[], mins=[], onset=0,
                       reached=0, fails=0, risk=[], risk_mah=[]) for k in rules}
        for _ in range(n):
            soc0 = float(rng.uniform(0.05, 0.25))
            for name, mk in rules.items():
                sim, _pv = build(T_amb=amb)
                r = session(sim, q, soc0, mk(), amb)
                a = acc[name]
                if r is None:
                    a["fails"] += 1
                    continue
                a["plated"].append(r["plated_Ah"]); a["peakT"].append(r["peak_T"])
                a["minphi"].append(r["min_phi"]); a["mins"].append(r["minutes"])
                a["onset"] += int(r["min_phi"] <= 0.0); a["reached"] += int(r["reached"])
                a["risk"].append(r["at_risk_min"]); a["risk_mah"].append(r["plated_at_risk_Ah"])

        rec = {}
        for name, a in acc.items():
            if not a["plated"]:
                continue
            pl = np.array(a["plated"])
            rec[name] = dict(sessions=len(pl), solver_failures=a["fails"],
                             plated_mAh=float(1000 * pl.mean()),
                             plated_mAh_max=float(1000 * pl.max()),
                             onset_crossed=a["onset"], reached_target=a["reached"],
                             min_phi_mV=float(1000 * np.min(a["minphi"])),
                             peak_T_C=float(np.max(a["peakT"])),
                             median_minutes=float(np.median(a["mins"])),
                             at_risk_min_total=float(np.sum(a["risk"])),
                             at_risk_min_max=float(np.max(a["risk"])),
                             plated_at_risk_mAh=float(1000 * np.sum(a["risk_mah"])))
        base = rec["certificate"]["plated_mAh"]
        for name in rec:
            rec[name]["plating_vs_certificate"] = rec[name]["plated_mAh"] / max(base, 1e-12)
        out["by_ambient"][f"{amb:g}"] = rec

        print(f"\n  ambient {amb:.0f} C   (to {TARGET:.0%} SOC)")
        print(f"    {'controller':16}{'onset':>8}{'min phi':>10}{'min at risk':>13}"
              f"{'mAh at risk':>13}{'total mAh':>11}{'minutes':>9}")
        for name, r in rec.items():
            print(f"    {name:16}{r['onset_crossed']:>5}/{r['sessions']}"
                  f"{r['min_phi_mV']:>9.1f}mV{r['at_risk_min_total']:>13.1f}"
                  f"{r['plated_at_risk_mAh']:>13.2f}{r['plated_mAh']:>11.1f}"
                  f"{r['median_minutes']:>9.1f}")

    # ---- the claim, in one place -------------------------------------------------------
    cold = out["by_ambient"]["0"]
    c, agg, safe = cold["certificate"], cold["CC-CV 1.5C"], cold["CC-CV 0.5C"]
    out.update(cold_cert_plated_mAh=c["plated_mAh"],
               cold_aggressive_plated_mAh=agg["plated_mAh"],
               cold_aggressive_ratio=agg["plating_vs_certificate"],
               cold_aggressive_onset=agg["onset_crossed"],
               cold_cert_onset=c["onset_crossed"],
               cold_cert_min_phi_mV=c["min_phi_mV"],
               cold_safe_plated_mAh=safe["plated_mAh"],
               cold_safe_ratio=safe["plating_vs_certificate"],
               cold_cert_minutes=c["median_minutes"],
               cold_safe_minutes=safe["median_minutes"],
               cold_minutes_saved=safe["median_minutes"] - c["median_minutes"],
               cold_cert_reached=c["reached_target"],
               cold_safe_reached=safe["reached_target"],
               sessions_per_cell=n)
    out.update(cold_agg_at_risk_min=agg["at_risk_min_total"],
               cold_agg_at_risk_mAh=agg["plated_at_risk_mAh"],
               cold_cert_at_risk_min=c["at_risk_min_total"],
               cold_cert_at_risk_mAh=c["plated_at_risk_mAh"],
               cold_safe_at_risk_min=safe["at_risk_min_total"],
               mass_metric_favours_aggressive=bool(
                   agg["plated_mAh"] < c["plated_mAh"]))
    print(f"\n{'=' * 78}")
    print(f"  cold, at matched charge -- which is where plating happens and where a de-rate is")
    print(f"  the only thing standing between an operator and a damaged pack:")
    print(f"    the rate a de-rate forbids drives the anode negative in "
          f"{agg['onset_crossed']}/{agg['sessions']} sessions, spending "
          f"{agg['at_risk_min_total']:.0f} minutes there and depositing "
          f"{agg['plated_at_risk_mAh']:.2f} mAh while it does")
    print(f"    the certificate crosses in {c['onset_crossed']}/{c['sessions']}, spends "
          f"{c['at_risk_min_total']:.0f} minutes at risk, and holds the anode "
          f"{c['min_phi_mV'] - safe['min_phi_mV']:+.1f} mV further from the onset than the "
          f"de-rate does")
    # And the cost of that, cold, is time -- stated rather than left for a reader to notice.
    slower = c["median_minutes"] - safe["median_minutes"]
    print(f"    it pays for that margin in minutes: {c['median_minutes']:.0f} against "
          f"{safe['median_minutes']:.0f} for the de-rate, {abs(slower):.0f} min "
          f"{'slower' if slower > 0 else 'faster'}, because cold is where the plating "
          f"constraint genuinely binds")
    print(f"\n  and that is the whole point of a state-dependent limit rather than a fixed "
          f"one. Across the sweep\n  the certificate holds a larger anode margin than the "
          f"de-rate at every ambient, and is:")
    for amb in AMBIENTS:
        r = out["by_ambient"][f"{amb:g}"]
        cc, ss = r["certificate"], r["CC-CV 0.5C"]
        d = ss["median_minutes"] - cc["median_minutes"]
        print(f"    {amb:>5.0f} C   anode margin {cc['min_phi_mV']:+.1f} vs "
              f"{ss['min_phi_mV']:+.1f} mV, and {abs(d):>4.1f} min "
              f"{'faster' if d > 0 else 'slower'}")
    out["margin_beats_derate_everywhere"] = all(
        out["by_ambient"][f"{a:g}"]["certificate"]["min_phi_mV"]
        > out["by_ambient"][f"{a:g}"]["CC-CV 0.5C"]["min_phi_mV"] for a in AMBIENTS)
    out["faster_when_warm_slower_when_cold"] = bool(
        out["by_ambient"]["0"]["certificate"]["median_minutes"]
        > out["by_ambient"]["0"]["CC-CV 0.5C"]["median_minutes"]
        and out["by_ambient"]["25"]["certificate"]["median_minutes"]
        < out["by_ambient"]["25"]["CC-CV 0.5C"]["median_minutes"])
    if out["mass_metric_favours_aggressive"]:
        print(f"\n  and the metric that does *not* work, reported because it does not: total "
              f"lithium plated\n  favours the aggressive rate ({agg['plated_mAh']:.1f} against "
              f"{c['plated_mAh']:.1f} mAh). It charges in "
              f"{agg['median_minutes']:.0f} min against {c['median_minutes']:.0f}, so it "
              f"accumulates less\n  of the slow background side reaction -- while entering "
              f"the depositing regime the certificate never\n  enters. Mass integrates a film "
              f"that forms harmlessly; the onset crossing is the event.")
    print(f"\n  what this does not show: dendrite growth, separator breach or runaway. No "
          f"controller here\n  approaches a self-heating threshold -- the 45 C limit is a "
          f"service limit, not a fire one. The\n  chain is measured at its first link and "
          f"only there.")

    path = V.save("s1_plating.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
