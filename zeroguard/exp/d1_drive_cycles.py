"""D1 -- the certificate over published drive cycles.

Every mission profile in this work so far was written by us: a synthetic delivery sortie, a
synthetic AUV transit, a synthetic eclipse. That is defensible for domains where no standard
profile exists, and indefensible for the one where three have existed for decades.

This experiment replaces the synthetic ground profile with the EPA's own dynamometer driving
schedules, downloaded from epa.gov:

  **US06**   the Supplemental Federal Test Procedure -- aggressive acceleration, high speed;
             the cycle written specifically because the older ones were too gentle
  **UDDS**   the Urban Dynamometer Driving Schedule -- stop-and-go city driving, which is what
             a robotaxi actually spends its life doing
  **HWFET**  the Highway Fuel Economy Test -- sustained cruise

The inputs are now external. The speed trace is not ours, the vehicle model is textbook
longitudinal dynamics, and the only thing this work contributes to the simulation is the
certificate itself.

A drive cycle also exercises the two-sided certificate in a way no other experiment here does,
because it alternates between the two cases within seconds of each other. Under traction the
pack is discharging and the traction power is a **floor** -- the car must deliver it or it is
not following the cycle. Under braking the pack is charging and regenerative current is a
**cap**. The same projection handles both by swapping which family the constraint lands in.

The baseline is what production electric vehicles actually do with regeneration: cap it at a
fixed current, chosen offline, applied whatever the pack's temperature and state of charge
happen to be. Recovered braking energy is the metric, because that is the one an efficiency
engineer cares about and the one the fixed cap gives away.

    python zeroguard/exp/d1_drive_cycles.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CYCLES = os.path.join(ROOT, "data", "cycles")
MPH = 0.44704

# A compact robotaxi. Textbook longitudinal dynamics; nothing here is novel or tuned.
VEH = dict(mass=1800.0,        # kg, including pack
           crr=0.010,          # rolling resistance coefficient
           cda=0.62,           # m^2, drag area (Cd 0.24 x A 2.6)
           rho=1.225,          # kg/m^3
           eta_drive=0.90,     # battery -> wheel
           eta_regen=0.70,     # wheel -> battery
           regen_cap_kW=60.0)  # driveline limit on regenerative braking

# What a production vehicle does: a fixed regenerative current limit, chosen offline.
FIXED_REGEN_C = 0.3


def load_cycle(name):
    """EPA schedules ship as a two-column tab-separated file: seconds, target speed in mph."""
    path = os.path.join(CYCLES, name)
    t, v = [], []
    for line in open(path, errors="ignore"):
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        t.append(a); v.append(b * MPH)
    t, v = np.array(t), np.array(v)
    keep = np.argsort(t)
    return t[keep], v[keep]


def pack_power(t, v):
    """Battery-terminal power demand over the cycle. Positive discharges, negative regenerates."""
    dt = np.gradient(t)
    a = np.gradient(v, t)
    F = (VEH["mass"] * a
         + VEH["mass"] * 9.81 * VEH["crr"] * (v > 0.1)
         + 0.5 * VEH["rho"] * VEH["cda"] * v ** 2)
    P_wheel = F * v
    P_batt = np.where(P_wheel >= 0, P_wheel / VEH["eta_drive"],
                      np.maximum(P_wheel * VEH["eta_regen"],
                                 -VEH["regen_cap_kW"] * 1000.0))
    return dt, P_batt


def run_cycle(t, v, plant_scale, T_amb, regen_rule, soc0=0.60):
    """Drive the cycle once, deciding regenerative current by one of two rules.

    Three things this function is careful about, each of which the first version got wrong.

    *Energy accounting.* Recovered energy is measured at the pack terminals as `u * V_pack * h`
    with `V_pack` taken from the same step the current was applied in, and available energy is
    the braking power the driveline could actually deliver. Mixing the two conventions produced
    a recovery fraction above 100 %, which is how the error announced itself.

    *Whose violation.* When the certificate reports the envelope closed and the cycle demands
    traction anyway, the vehicle is driven regardless -- a drive schedule does not negotiate --
    and any resulting breach is recorded separately. It is not a failure of the filter to breach
    a constraint after it has said the manoeuvre is outside the envelope; it is a failure only
    if it breaches while reporting feasible.

    *Refusals are per-run.* They are reported as a fraction of that run's traction steps.
    """
    dt, Pwr = pack_power(t, v)
    est_c = V.pessimistic("robotaxi-urban", T_amb=T_amb)          # charge view (regen)
    est_d = P.RobotaxiUrban(mode="discharge", T_amb=T_amb, load_W=1.0,
                            scale=dict(R=V.S_R, Q=0.80, plate=1.6))
    plant_c = P.RobotaxiUrban(T_amb=T_amb, scale=plant_scale)
    plant_d = P.RobotaxiUrban(mode="discharge", T_amb=T_amb, load_W=1.0, scale=plant_scale)
    marg_c, marg_d = V.margins(est_c), V.margins(est_d)
    S = est_c.S

    s = est_c.init(soc0, T_amb + 2.0)
    regen_avail = regen_used = 0.0
    traction_steps = traction_refused = 0
    breach_while_certified = breach_after_refusal = 0
    peakT = -1e9
    for k in range(len(t) - 1):
        h = float(max(dt[k], 1e-3))
        if h > 5.0:
            continue
        Pk = float(Pwr[k])
        if Pk >= 0.0:                                   # traction: power is a floor
            traction_steps += 1
            est_d.set_load(max(Pk, 1.0)); plant_d.set_load(max(Pk, 1.0))
            lo, hi, st = A.interval(est_d, s, h, T_amb, marg_d)
            certified = st == "ok"
            if not certified:
                traction_refused += 1
                u = max(lo, est_d.anchor(s))
            else:
                u = lo
            s, o = plant_d.step(s, float(u), h, T_amb)
            bad, _e = V.split_breaches(plant_d, V.check(plant_d, o))
            if bad:
                if certified:
                    breach_while_certified += 1
                else:
                    breach_after_refusal += 1
        else:                                           # braking: regen current is a cap
            V_cell = float(plant_c.probe(s, 0.0, h, T_amb)[0])
            V_pack = S * max(V_cell, 1e-3)
            want = min(-Pk / V_pack, est_c.u_max)       # current the driveline offers
            regen_avail += -Pk * h
            if regen_rule[0] == "fixed":
                u = min(want, regen_rule[1] * est_c.cell.q_nom() * est_c.P)
                certified = True                        # a blind cap makes no claim
            else:
                _lo, hi, st = A.interval(est_c, s, h, T_amb, marg_c)
                certified = st == "ok"
                u = min(want, hi) if certified else 0.0
            u = max(0.0, float(u))
            s, o = plant_c.step(s, u, h, T_amb)
            # measured at the terminals with the voltage that actually obtained
            regen_used += u * S * float(o[0]) * h
            bad, _e = V.split_breaches(plant_c, V.check(plant_c, o))
            if bad:
                if certified:
                    breach_while_certified += 1
                else:
                    breach_after_refusal += 1
        peakT = max(peakT, float(s["T"]))
    return dict(regen_available_J=regen_avail, regen_recovered_J=regen_used,
                recovery_fraction=min(regen_used / max(regen_avail, 1e-9), 1.0),
                traction_steps=traction_steps, traction_refused=traction_refused,
                refusal_fraction=traction_refused / max(traction_steps, 1),
                breach_while_certified=breach_while_certified,
                breach_after_refusal=breach_after_refusal,
                ok=breach_while_certified == 0, peak_T=float(peakT),
                soc_end=float(s["soc"]))


def main(n=40, seed=20260815):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    print("D1 -- the certificate over the EPA's own driving schedules\n" + "=" * 78)
    caps = (0.1, 0.2, 0.3, 0.5, 1.0)
    out = {"vehicle": VEH, "regen_cap_sweep": list(caps), "repeats": n, "cycles": {}}

    for fname, label in (("us06col.txt", "US06 (aggressive)"),
                         ("uddscol.txt", "UDDS (city)"),
                         ("hwycol.txt", "HWFET (highway)")):
        t, v = load_cycle(fname)
        dt, Pw = pack_power(t, v)
        rules = [("fixed", c) for c in caps] + [("certificate", None)]
        acc = {r: dict(rec=[], cert=0, after=0, ref=[], peak=-1e9) for r in
               [f"fixed_{c:g}C" for c in caps] + ["certificate"]}
        for _ in range(n):
            sc = {k: float(rng.uniform(*q)) for k, q in V.ENVELOPE.items()}
            T_amb = float(rng.choice([-5.0, 5.0, 15.0, 25.0, 35.0]))
            soc0 = float(rng.uniform(0.35, 0.85))
            for rule in rules:
                key = "certificate" if rule[0] == "certificate" else f"fixed_{rule[1]:g}C"
                r = run_cycle(t, v, sc, T_amb, rule, soc0=soc0)
                a = acc[key]
                a["rec"].append(r["recovery_fraction"])
                a["cert"] += r["breach_while_certified"]
                a["after"] += r["breach_after_refusal"]
                a["ref"].append(r["refusal_fraction"])
                a["peak"] = max(a["peak"], r["peak_T"])
        rec = dict(label=label, seconds=float(t[-1]),
                   distance_km=float((np.trapezoid if hasattr(np, "trapezoid")
                                      else np.trapz)(v, t) / 1000.0),
                   peak_power_kW=float(np.max(Pw) / 1000.0),
                   peak_regen_kW=float(-np.min(Pw) / 1000.0), runs=n, rules={})
        for key, a in acc.items():
            r = np.array(a["rec"])
            rec["rules"][key] = dict(recovery_mean=float(r.mean()),
                                     breach_while_certified=a["cert"],
                                     breach_after_refusal=a["after"],
                                     refusal_fraction=float(np.mean(a["ref"])),
                                     peak_T=float(a["peak"]))
        # iso-safety: the largest fixed cap that never breaches
        safe = [c for c in caps
                if rec["rules"][f"fixed_{c:g}C"]["breach_while_certified"] == 0]
        rec["safe_fixed_caps"] = safe
        rec["best_safe_fixed_C"] = max(safe) if safe else None
        cert = rec["rules"]["certificate"]
        if safe:
            bf = rec["rules"][f"fixed_{max(safe):g}C"]
            rec["recovery_gain_points"] = 100 * (cert["recovery_mean"] - bf["recovery_mean"])
        out["cycles"][fname] = rec

        print(f"\n{label}  --  {rec['seconds']:.0f} s, {rec['distance_km']:.1f} km, "
              f"peak {rec['peak_power_kW']:.0f} kW traction / "
              f"{rec['peak_regen_kW']:.0f} kW regen")
        print(f"  {'regen rule':16}{'recovered':>11}{'breach|certified':>18}"
              f"{'breach after no':>17}")
        for key in [f"fixed_{c:g}C" for c in caps] + ["certificate"]:
            d = rec["rules"][key]
            print(f"  {key:16}{100*d['recovery_mean']:>10.1f}%"
                  f"{d['breach_while_certified']:>18}{d['breach_after_refusal']:>17}")
        print(f"  largest fixed cap that never breaches: {rec['best_safe_fixed_C']}C"
              + (f" | certificate recovers {rec['recovery_gain_points']:+.1f} points vs it"
                 if safe else ""))
        print(f"  the certificate refuses "
              f"{100*cert['refusal_fraction']:.1f}% of traction steps -- the cycle demands "
              f"power the envelope does not allow")

    # A fixed cap is chosen once and must survive every schedule the car meets. Which caps are
    # safe on ALL three, and what does the certificate cost against the best of those?
    per = {c: [] for c in caps}
    for cyc in out["cycles"].values():
        for c in caps:
            per[c].append(cyc["rules"][f"fixed_{c:g}C"]["breach_while_certified"] == 0)
    universal = [c for c in caps if all(per[c])]
    out["caps_safe_on_every_schedule"] = universal
    out["best_universal_cap_C"] = max(universal) if universal else None
    if universal:
        bu = max(universal)
        gaps, tuned_fail = [], {}
        for fn, cyc in out["cycles"].items():
            gaps.append(100 * (cyc["rules"]["certificate"]["recovery_mean"]
                               - cyc["rules"][f"fixed_{bu:g}C"]["recovery_mean"]))
        out["gain_vs_universal_cap_points"] = float(np.mean(gaps))
        # and the transfer failure: a cap tuned on the gentlest schedule, run on the others
        per_cycle_best = {fn: cyc["best_safe_fixed_C"] for fn, cyc in out["cycles"].items()}
        out["per_cycle_best_safe_cap"] = per_cycle_best
        tuned = max(v for v in per_cycle_best.values() if v is not None)
        out["cap_tuned_on_easiest_schedule_C"] = tuned
        out["cap_tuned_elsewhere_breaches"] = {
            fn: cyc["rules"][f"fixed_{tuned:g}C"]["breach_while_certified"]
            for fn, cyc in out["cycles"].items()}

    out["certificate_breaches_while_certified"] = sum(
        out["cycles"][c]["rules"]["certificate"]["breach_while_certified"]
        for c in out["cycles"])
    out["mean_refusal_fraction"] = float(np.mean(
        [out["cycles"][c]["rules"]["certificate"]["refusal_fraction"] for c in out["cycles"]]))
    gains = [out["cycles"][c].get("recovery_gain_points") for c in out["cycles"]]
    gains = [g for g in gains if g is not None]
    out["mean_recovery_gain_points"] = float(np.mean(gains)) if gains else None
    print(f"\n  across all three schedules the certificate breached "
          f"{out['certificate_breaches_while_certified']} times while reporting feasible")
    if gains:
        print(f"  and recovered {out['mean_recovery_gain_points']:+.1f} points of braking "
              f"energy relative to the largest per-schedule-safe fixed cap")
    if out.get("best_universal_cap_C") is not None:
        print(f"  the only fixed caps safe on EVERY schedule are "
              f"{out['caps_safe_on_every_schedule']}; against the best of them the certificate "
              f"is {out['gain_vs_universal_cap_points']:+.1f} points")
        print(f"  a cap tuned on the gentlest schedule ({out['cap_tuned_on_easiest_schedule_C']}C) "
              f"breaches elsewhere: {out['cap_tuned_elsewhere_breaches']}")

    path = V.save("d1_drive_cycles.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
