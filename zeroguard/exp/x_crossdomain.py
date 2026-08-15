"""X -- what has to be true across all four domains for any of this to be one method.

Eleven experiments per domain establish that the certificate holds in each. That is not the
same as establishing that it is the *same* certificate. Four domains that each work for their
own reasons would be four papers, and the claim here is one.

So five cross-cutting checks, each of which would fail if the domains were only superficially
related: the admissible set has the same structure everywhere (X1); the generalisation reduces
to the original theorem exactly where the original theorem applies (X2); the cost is fixed and
the same everywhere (X3); the reserve means the same thing everywhere (X4); and every
zero-failure claim above is backed by enough samples to certify something (X5).

    python zeroguard/exp/x_crossdomain.py
"""
import os, sys, json, time, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V
from safe_charge import BatteryROM, project_current

SEED = 20260815
ENV = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# a temperature each domain's vehicles actually sit at
T0 = dict(ground=(10.0, 44.0), aerial=(-25.0, 50.0), underwater=(-1.8, 20.0),
          space=(-40.0, 38.0), reference=(0.0, 44.0))
W0 = dict(ground=25.0, aerial=15.0, underwater=2.0, space=4.0, reference=25.0)


def _states(plat, rng, k):
    lo_T, hi_T = T0[plat.domain]
    lo_s = 0.02 if plat.mode == "charge" else max(plat.soc_floor + 0.02, 0.2)
    return [plat.init(float(rng.uniform(lo_s, 0.99)), float(rng.uniform(lo_T, hi_T)))
            for _ in range(k)]


# ---------------------------------------------------------------------------------------
def x1_structure(per_case=4000, seed=SEED):
    """Is the admissible set a single interval in every domain, every mode, every state?

    The two-bisection construction is only valid if feasibility is one connected run in `u`.
    A single disconnected scan anywhere would mean the upper bisection can converge into a
    hole and return a number that is not the edge of anything -- so this is scanned densely
    rather than argued from monotonicity, because monotonicity is a hypothesis and this is the
    check on it."""
    rng = np.random.default_rng(seed)
    per_domain = {}
    rows = []
    total = {"single-interval": 0, "disconnected": 0, "empty": 0}
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        marg = V.margins(plat)
        census = {"single-interval": 0, "disconnected": 0, "empty": 0}
        for s in _states(plat, rng, per_case):
            _g, ok = A.scan(plat, s, plat.dt_nominal, W0[plat.domain], marg, n=180)
            census[A.structure(ok)] += 1
        for k, v in census.items():
            total[k] += v
        d = per_domain.setdefault(plat.domain, {"single-interval": 0, "disconnected": 0,
                                                "empty": 0})
        for k, v in census.items():
            d[k] += v
        rows.append(dict(case=case, domain=plat.domain, mode=plat.mode, census=census))
    return dict(cases=rows, per_domain=per_domain, total=total,
                scans=sum(total.values()), grid=180,
                disconnected=total["disconnected"],
                all_single_interval=total["disconnected"] == 0)


# ---------------------------------------------------------------------------------------
def x2_reduction(n=60000, seed=SEED + 1):
    """Where the original theorem applies, is the generalisation the same function?

    A generalisation that changes the answer on the original problem is a replacement, not a
    generalisation. Every charge-mode platform has `u_a = 0` and no floor family, so Anchored
    Collapse must degenerate to Null-Input Collapse -- and on the one-cell reference platform
    that means agreeing with the published `project_current` bit for bit."""
    rng = np.random.default_rng(seed)
    ref_mismatch = 0; worst = 0.0
    plat = P.single_cell()
    rom = BatteryROM()
    marg = (0.03, 0.5, 0.006)
    for _ in range(n // 2):
        s = dict(soc=float(rng.uniform(0.02, 0.98)), T=float(rng.uniform(-10.0, 44.0)),
                 V1=float(rng.uniform(-0.1, 0.1)), aging={"Qloss": 0.0, "Rfac": 1.0})
        w = float(rng.uniform(-20.0, 45.0))
        a, _ = project_current(rom, s, plat.u_max, 30.0, w, margin=rom.plating_margin(),
                               dV=marg[0], dT=marg[1], dP=marg[2], cool_frac=0.0)
        b, _ = A.project_anchored(plat, s, plat.u_max, 30.0, w, marg)
        if a != b:
            ref_mismatch += 1; worst = max(worst, abs(a - b))
    # on every charge platform, is the lower edge exactly zero and is the anchor feasible?
    per_case = []
    for case in P.ALL_CASES:
        pl = V.pessimistic(case)
        if pl.mode != "charge":
            continue
        m = V.margins(pl)
        lows, infeas = [], 0
        for s in _states(pl, rng, n // 20):
            lo, hi, st = A.interval(pl, s, pl.dt_nominal, W0[pl.domain], m)
            if st != "ok":
                infeas += 1
            else:
                lows.append(lo)
        per_case.append(dict(case=case, states=n // 20, infeasible=infeas,
                             max_lower_edge=float(max(lows)) if lows else None,
                             lower_edge_is_zero=bool(lows and max(lows) == 0.0)))
    return dict(reference_states=n // 2, reference_mismatches=ref_mismatch,
                reference_worst_abs=worst, bit_exact=ref_mismatch == 0,
                charge_platforms=per_case,
                all_lower_edges_zero=all(c["lower_edge_is_zero"] for c in per_case))


# ---------------------------------------------------------------------------------------
def x3_cost(reps=25, batch=300, seed=SEED + 2):
    """Is the cost fixed, and is it the same fixed cost everywhere?

    The original method's run-time argument is that the projection spends a bounded number of
    one-step evaluations regardless of the data -- unlike a QP, whose iteration count is not
    something a safety task slot can be sized around. The generalisation adds a second
    bisection on discharge platforms, so the bound goes from 20 evaluations to 40 -- two probes to
    bracket each family plus 18 halvings each -- and the claim is that it goes there and stops."""
    rng = np.random.default_rng(seed)
    rows = []
    gc.disable()
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        marg = V.margins(plat)
        states = _states(plat, rng, 600)
        counts = {}
        for s in states:
            c = V.Counted(plat)
            A.project_anchored(c, s, plat.u_max, plat.dt_nominal, W0[plat.domain], marg)
            counts[c.n] = counts.get(c.n, 0) + 1
        sub = states[:batch]
        means = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for s in sub:
                A.project_anchored(plat, s, plat.u_max, plat.dt_nominal, W0[plat.domain], marg)
            means.append((time.perf_counter() - t0) / len(sub) * 1e6)
        a = np.array(means)
        rows.append(dict(case=case, domain=plat.domain, mode=plat.mode,
                         eval_counts=counts, max_evals=max(counts),
                         us_median=float(np.median(a)), us_p99=float(np.percentile(a, 99)),
                         us_max=float(a.max())))
    gc.enable()
    charge = [r for r in rows if r["mode"] == "charge"]
    disch = [r for r in rows if r["mode"] == "discharge"]
    return dict(rows=rows,
                charge_max_evals=max(r["max_evals"] for r in charge),
                discharge_max_evals=max(r["max_evals"] for r in disch),
                theoretical_charge=20, theoretical_discharge=2 * (18 + 2),
                worst_p99_us=max(r["us_p99"] for r in rows),
                worst_case=max(rows, key=lambda r: r["us_p99"])["case"],
                bounded=(max(r["max_evals"] for r in charge) <= 20
                         and max(r["max_evals"] for r in disch) <= 40))


# ---------------------------------------------------------------------------------------
def x4_reserve_transfers(n=400, seed=SEED + 3):
    """Does the reserve mean the same thing in a drone, a submarine and a satellite?

    The width of the interval is proposed as a *reserve*: how much room the vehicle has before
    the envelope closes. If that is a real physical quantity rather than a unit-dependent
    artefact, the initial width should predict the time to closure.

    It does, but not everywhere, and the exception is the interesting part. An interval can
    close for two different reasons. **Envelope-driven** closure is the upper edge falling onto
    the lower one as the pack heats and sags -- there the width is exactly the distance still to
    travel, and it predicts the clock. **Energy-driven** closure is the state of charge reaching
    its floor with the interval still wide open; there the clock is set by how much charge was
    in the pack to begin with, and the width has nothing to say about it.

    Both are correctly reported as closures, because both are the certificate saying the load
    can no longer be served. But only the first is a *reserve* in the sense of a countdown, and
    a pooled correlation of +0.9 that quietly averages over a domain where the relation is
    absent would be the most flattering way to present this and the least honest.

    So the certificate carries **two clocks**, and it already has the ingredients for both:

        t_envelope  ~  the interval width, which shrinks to zero as the edges converge
        t_energy    =  (soc - soc_floor) * Q  /  u_lo

    the second being simply the charge above the floor divided by the current the load is
    drawing. Neither dominates: on a quadrotor in hover the envelope closes long before the
    charge runs out, and on an AUV in transit the interval stays wide while the pack empties.
    What a vehicle should act on is the **minimum** of the two, and this experiment measures
    each one's predictive power in each domain rather than choosing a favourite."""
    rng = np.random.default_rng(seed)
    per, pooled_x, pooled_y = {}, [], []
    pooled_env_x, pooled_env_y = [], []
    # Horizons are set so the episode ends by closing rather than by running out of steps. An
    # earlier version gave the GEO satellite 300 steps against a ~290-step endurance, so most
    # episodes were right-censored at the horizon, endurance was a near-constant, and both
    # clocks scored badly on what was really a measurement artefact. Censored episodes are now
    # excluded from the correlations as well, because a censored duration is a lower bound and
    # correlating lower bounds is not correlating durations.
    for case, dt, hor in (("delivery-quadrotor", 2.0, 900),
                          ("evtol-air-taxi", 1.0, 1500),
                          ("survey-auv", 60.0, 2400),
                          ("under-ice-auv", 60.0, 2400),
                          ("geo-comsat", 60.0, 900),
                          ("lunar-night-lander", 300.0, 1200)):
        est = V.pessimistic(case)
        marg = V.margins(est)
        xs, ys, rel, tE = [], [], [], []
        env_x, env_y, eng_x, eng_y = [], [], [], []
        censored = 0
        Q_pack = est.cell.q_nom() * est.P
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.build(case, scale=sc)
            lo_s = max(est.soc_floor + 0.05, 0.25)
            s0 = est.init(float(rng.uniform(lo_s, 0.99)),
                          float(rng.uniform(*T0[est.domain])))
            lo, hi, st = A.interval(est, s0, dt, W0[est.domain], marg)
            w = (hi - lo) if st == "ok" else 0.0
            r = V.discharge_mission(plant, est, s0, dt, W0[est.domain], marg, horizon=hor,
                                    fly_past_closure=False)
            if r["closure_step"] < 0:
                censored += 1
                continue                      # a lower bound, not a duration
            xs.append(w); ys.append(r["endurance_s"])
            rel.append(w / max(est.anchor(s0), 1e-9))
            # the other clock: charge above the floor, divided by what the load draws
            tE.append((s0["soc"] - est.soc_floor) * Q_pack * 3600.0 / max(lo, 1e-9))
            # energy-driven if the pack ran down to its floor; envelope-driven otherwise
            if r["soc"] <= est.soc_floor + 0.02:
                eng_x.append(w); eng_y.append(r["endurance_s"])
            else:
                env_x.append(w); env_y.append(r["endurance_s"])
        rho, p = V.spearman(xs, ys, reps=5000)
        rho_n, p_n = V.spearman(rel, ys, reps=5000)
        tr, tp = V.spearman(tE, ys, reps=5000)
        er, ep = (V.spearman(env_x, env_y, reps=5000) if len(env_x) >= 20
                  else (float("nan"), float("nan")))
        gr, gp = (V.spearman(eng_x, eng_y, reps=5000) if len(eng_x) >= 20
                  else (float("nan"), float("nan")))
        per[case] = dict(trials=n, uncensored=len(ys), censored=censored,
                         width_vs_endurance=dict(rho=rho, p=p),
                         normalised_vs_endurance=dict(rho=rho_n, p=p_n),
                         envelope_driven=len(env_x), energy_driven=len(eng_x),
                         envelope_driven_frac=len(env_x) / n,
                         width_vs_endurance_envelope=dict(rho=er, p=ep),
                         width_vs_endurance_energy=dict(rho=gr, p=gp),
                         energy_clock_vs_endurance=dict(rho=tr, p=tp),
                         better_clock=("energy" if (tr == tr and tr > rho) else "envelope"),
                         mean_width_A=float(np.mean(xs)),
                         median_endurance_s=float(np.median(ys)))
        pooled_x.extend(rel); pooled_y.extend(ys)
        pooled_env_x.extend([w / max(np.mean(rel) if False else 1.0, 1e-9) for w in env_x])
        pooled_env_y.extend(env_y)
    rho, p = V.spearman(pooled_x, pooled_y, reps=5000)
    env_ok = [v for v in per.values() if v["envelope_driven"] >= 20]
    def _best(v):
        a, b = v["width_vs_endurance"], v["energy_clock_vs_endurance"]
        return a if a["rho"] >= b["rho"] else b
    best = {k: _best(v)["rho"] for k, v in per.items()}
    best_p = {k: _best(v)["p"] for k, v in per.items()}
    weakest = min(best, key=best.get)
    return dict(per_case=per, pooled=dict(rho=rho, p=p, n=len(pooled_x)),
                cases_with_envelope_closures=len(env_ok),
                best_clock_rho=best, best_clock_p=best_p,
                weakest_case=weakest, weakest_rho=best[weakest],
                every_case_has_a_predictive_clock=all(
                    best[k] > 0.3 and best_p[k] < 0.001 for k in best),
                raw_all_positive=all(v["width_vs_endurance"]["rho"] > 0
                                     for v in per.values()),
                note=("two clocks: the interval width predicts time-to-closure where the "
                      "envelope is what closes, and (soc - soc_floor) Q / u_lo predicts it "
                      "where the pack simply empties. Every platform has one of them "
                      "significantly positive, the GEO satellite most weakly. A vehicle should "
                      "act on the minimum of the two, which is conservative whether or not "
                      "either is individually tight."))


# ---------------------------------------------------------------------------------------
CLAIMS = [
    ("V1-2 robotaxi 350 kW", "v1_ground.json", ("v1_2_fast_charge",), "violations", "trials"),
    ("V1-6 robotaxi 5 y duty", "v1_ground.json", ("v1_6_duty_cycle", "duties", "robotaxi"),
     "violations", "trials"),
    ("V1-8 regen pulses", "v1_ground.json", ("v1_8_regen",), "violations", "trials"),
    ("V2-1 anchored flight", "v2_aerial.json", ("v2_1_anchor_vs_null",),
     "anchored_unsafe_while_certified", "trials"),
    ("V2-3 reserve warns", "v2_aerial.json", ("v2_3_reserve_lead",),
     "breaches_with_no_warning", "trials"),
    ("V2-9 turnaround charging", "v2_aerial.json", ("v2_9_turnaround",), "violations", "trials"),
    ("V2-11 aerial adversary", "v2_aerial.json", ("v2_11_adversary",), "violations", "trials"),
    ("V3-1 sealed hull", "v3_underwater.json", ("v3_1_sealed_hull",), "violations", "trials"),
    ("V3-4 6-month deployment", "v3_underwater.json", ("v3_4_deployment",),
     "violations", "trials"),
    ("V3-7 under-ice warning", "v3_underwater.json", ("v3_7_under_ice",),
     "breaches_with_no_warning", "trials"),
    ("V4-1 LEO 5 y", "v4_space.json", ("v4_1_leo",), "violations", "trials"),
    ("V4-3 deep-space cruise", "v4_space.json", ("v4_3_deep_space",), "violations", "trials"),
]


def x5_adequacy():
    """For every zero-failure claim: what it certifies, and what it would have taken.

    E11 wrote `stats.n_required` and never called it, and when it was finally called it found
    that the adversarial claim rested on 40 cells and bounded nothing below 7.2 %. That is the
    failure mode this exists to prevent: a claim of "zero violations" whose sample size makes
    it compatible with a violation rate no one would accept."""
    need = {f"{100*t}%": stats.n_required(t) for t in (0.01, 0.001, 0.0001)}
    rows = []
    for label, fn, path, kk, nk in CLAIMS:
        f = os.path.join(RES, fn)
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for p in path:
            if p not in d:
                d = None
                break
            d = d[p]
        if d is None or kk not in d or nk not in d:
            continue
        k, n = int(d[kk]), int(d[nk])
        rows.append(dict(claim=label, violations=k, trials=n,
                         certifies_below_pct=100 * stats.cp_upper(k, n),
                         enough_for_1pct=n >= need["1.0%"],
                         enough_for_0p1pct=n >= need["0.1%"],
                         enough_for_0p01pct=n >= need["0.01%"]))
    return dict(n_required=need, claims=rows,
                claims_found=len(rows), claims_expected=len(CLAIMS),
                all_certify_below_1pct=all(r["enough_for_1pct"] and r["violations"] == 0
                                           for r in rows),
                total_trials=sum(r["trials"] for r in rows),
                total_violations=sum(r["violations"] for r in rows))


# ---------------------------------------------------------------------------------------
def main():
    out = {}
    t0 = time.time()
    print("X -- cross-domain connectedness\n" + "=" * 78)

    print("\nX1  is the admissible set a single interval everywhere?")
    r = x1_structure(); out["x1_structure"] = r
    for d, c in r["per_domain"].items():
        print(f"      {d:11s} single {c['single-interval']:>7,}  disconnected "
              f"{c['disconnected']:>4}  empty {c['empty']:>6,}")
    print(f"      {r['scans']:,} dense scans on a {r['grid']}-point grid | disconnected "
          f"{r['disconnected']} | all single-interval: {r['all_single_interval']}")

    print("\nX2  does the generalisation reduce to the original theorem?")
    r = x2_reduction(); out["x2_reduction"] = r
    print(f"      reference cell, {r['reference_states']:,} states vs project_current: "
          f"mismatches {r['reference_mismatches']} | bit-exact {r['bit_exact']}")
    for c in r["charge_platforms"]:
        print(f"        {c['case']:24s} lower edge max {c['max_lower_edge']} "
              f"({'zero' if c['lower_edge_is_zero'] else 'NONZERO'})")
    print(f"      every charge platform anchors at zero: {r['all_lower_edges_zero']}")

    print("\nX3  is the cost fixed, and the same fixed cost everywhere?")
    r = x3_cost(); out["x3_cost"] = r
    print(f"      {'case':24s}{'mode':>10}{'max evals':>11}{'median us':>11}{'p99 us':>9}")
    for row in r["rows"]:
        print(f"      {row['case']:24s}{row['mode']:>10}{row['max_evals']:>11}"
              f"{row['us_median']:>11.2f}{row['us_p99']:>9.2f}")
    print(f"      charge bound {r['charge_max_evals']}/{r['theoretical_charge']} | discharge "
          f"{r['discharge_max_evals']}/{r['theoretical_discharge']} | bounded: {r['bounded']}")
    print(f"      worst p99 {r['worst_p99_us']:.1f} us ({r['worst_case']})")

    print("\nX4  does the reserve mean the same thing in all four media?")
    r = x4_reserve_transfers(); out["x4_reserve"] = r
    print(f"      {'case':20}{'n':>6}{'width clock':>13}{'energy clock':>14}{'better':>9}")
    for k, v in r["per_case"].items():
        print(f"      {k:20}{v['uncensored']:>6}{v['width_vs_endurance']['rho']:>+13.3f}"
              f"{v['energy_clock_vs_endurance']['rho']:>+14.3f}{v['better_clock']:>9}")
    print(f"      pooled (width/anchor vs endurance): rho={r['pooled']['rho']:+.3f} "
          f"p={r['pooled']['p']:.4f} n={r['pooled']['n']:,}")
    print(f"      every platform has a significantly positive clock "
          f"(rho > 0.3, p < 0.001): {r['every_case_has_a_predictive_clock']}")
    print(f"      weakest is {r['weakest_case']} at rho={r['weakest_rho']:+.3f}; a vehicle "
          f"should act on min(t_envelope, t_energy), which is conservative either way")
    print(f"      (the width alone is positive everywhere: {r['raw_all_positive']} -- which is "
          f"why both are reported)")

    print("\nX5  is every zero-failure claim backed by enough samples?")
    r = x5_adequacy(); out["x5_adequacy"] = r
    print(f"      trials needed at 95 % for zero failures: "
          f"{r['n_required']['1.0%']:,} (1%), {r['n_required']['0.1%']:,} (0.1%), "
          f"{r['n_required']['0.01%']:,} (0.01%)")
    print(f"      {'claim':26}{'n':>9}{'viol':>6}{'certifies below':>17}   adequate for")
    for row in r["claims"]:
        tags = [t for t, ok in (("1%", row["enough_for_1pct"]),
                                ("0.1%", row["enough_for_0p1pct"]),
                                ("0.01%", row["enough_for_0p01pct"])) if ok]
        print(f"      {row['claim']:26}{row['trials']:>9,}{row['violations']:>6}"
              f"{row['certifies_below_pct']:>16.4f}%   {', '.join(tags) if tags else '-'}")
    print(f"      {r['total_violations']} violations in {r['total_trials']:,} pooled trials")

    path = V.save("x_crossdomain.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
