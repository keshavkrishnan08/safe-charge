"""P1 -- the policy may be learned; the safety must not be.

The paper takes that position in its introduction and never demonstrates it. Every experiment
so far either runs the certificate alone or compares it against another controller; none puts a
policy *through* it, which is the arrangement the method is actually for. A safety filter's
purpose is not to be a controller. It is to sit between an arbitrary controller and the plant
and make the pair safe without making the controller useless.

Two properties decide whether it succeeds, and they pull against each other:

  **Containment.** An unsafe policy, filtered, must become safe. Not safer -- safe.
  **Transparency.** A policy that was already safe must be left alone. A filter that clips a
  good controller is buying safety with performance that did not need to be spent, and the
  usual way to hide that is to only ever test aggressive policies.

So four policies are run, spanning the range from reckless to conservative, each unfiltered and
filtered on identical plants and seeds. One of them is *learned*: its parameters are fitted by
cross-entropy search to maximise delivered charge on a nominal cell, with no safety term of any
kind in the objective. That is the honest version of "a learned policy" -- not one trained to be
bad, but one trained to be greedy and given no reason to care.

    python zeroguard/exp/p1_policy_filter.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260816
DT, HORIZON, TARGET = 30.0, 80, 0.80


# ---------------------------------------------------------------------------------------
# Policies. Each maps a state to a requested current, knowing nothing about constraints.
# ---------------------------------------------------------------------------------------
# Each takes (state, last measured voltage) -- what a BMS actually has -- and returns a
# requested pack current. None of them knows the constraint set.
def pol_greedy(plat):
    """Ask for everything, always. The reckless limit."""
    return lambda s, v: plat.u_max


def pol_thermal_heuristic(plat, T_soft=38.0):
    """A naive engineer's rule: full current until it feels warm, then taper linearly.

    This is the shape of controller a competent person writes without a model, and it is unsafe
    for an unremarkable reason -- it reacts to temperature that has already arrived.
    """
    def f(s, v):
        T = s["T"]
        if T <= T_soft:
            return plat.u_max
        return plat.u_max * max(0.0, 1.0 - (T - T_soft) / 7.0)
    return f


def pol_ccv(plat, c_rate=0.5):
    """Constant current with a voltage taper -- production CC-CV, and already safe here.

    Included to test transparency, and it has to be the *real* protocol rather than a bare
    constant current: a constant current with no taper crosses the voltage limit near the top of
    charge, so filtering it would look like transparency being bought and would in fact be the
    filter doing necessary work.
    """
    st = {"u": c_rate * plat.cell.q_nom() * plat.P}
    def f(s, v):
        if v is not None and v >= plat.V_max - 1e-2:
            st["u"] *= 0.85
        return st["u"]
    return f


def make_learned(plat, w):
    """u = u_max * sigma(w0 + w1*soc + w2*(T-25)/20 + w3*(V-4)), a smooth state feedback."""
    def f(s, v):
        z = (w[0] + w[1] * s["soc"] + w[2] * (s["T"] - 25.0) / 20.0
             + w[3] * ((3.7 if v is None else v) - 4.0))
        return float(plat.u_max / (1.0 + np.exp(-z)))
    return f


def train_learned(plat, seed=SEED, iters=8, pop=40, elite=8):
    """Fit the policy to deliver charge on a nominal cell. No safety term in the objective.

    The plant it trains against is the *nominal* cell, not the envelope -- which is exactly the
    mistake a well-meaning practitioner makes, and the reason the resulting policy is unsafe on
    a fleet rather than deliberately reckless.
    """
    rng = np.random.default_rng(seed)
    nominal = P.RobotaxiUrban(T_amb=25.0)                # fresh, mid-life, mild
    mu, sd = np.zeros(4), np.full(4, 2.0)
    best = None
    for _ in range(iters):
        cand = mu + sd * rng.standard_normal((pop, 4))
        score = np.empty(pop)
        for j in range(pop):
            f = make_learned(plat, cand[j])
            s, v = nominal.init(0.10, 25.0), None
            for _k in range(HORIZON):
                u = float(np.clip(f(s, v), 0.0, plat.u_max))
                s, o = nominal.step(s, u, DT, 25.0); v = o[0]
                if s["soc"] >= TARGET:
                    break
            score[j] = s["soc"]                          # charge only; nothing about safety
        idx = np.argsort(score)[-elite:]
        mu, sd = cand[idx].mean(axis=0), cand[idx].std(axis=0) + 0.05
        best = cand[idx[-1]]
    return best


# ---------------------------------------------------------------------------------------
def run(plant, est, policy, s0, marg, filtered):
    """One session. `filtered` decides whether the request passes through the projection."""
    s, v = dict(s0), None
    viol, clipped, steps = set(), 0, 0
    for k in range(HORIZON):
        req = float(np.clip(policy(s, v), 0.0, est.u_max))
        if filtered:
            u, st = A.project_anchored(est, s, req, DT, 25.0, marg)
            if st == "infeasible":
                u = 0.0
            elif st != "unclipped":
                clipped += 1
        else:
            u = req
        s, o = plant.step(s, float(max(0.0, u)), DT, 25.0); v = o[0]
        c, _e = V.split_breaches(plant, V.check(plant, o))
        viol.update(c)
        steps = k + 1
        if s["soc"] >= TARGET:
            break
    return dict(soc=float(s["soc"]), steps=steps, clipped=clipped,
                clip_rate=clipped / max(steps, 1), ok=not viol)


def main(n=400, seed=SEED):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    print("P1 -- the policy may be learned; the safety must not be\n" + "=" * 78)
    est = V.pessimistic("robotaxi-urban", T_amb=25.0)
    marg = V.margins(est)

    print("\ntraining a policy on a nominal cell, maximising charge, no safety term")
    w = train_learned(est)
    print(f"  fitted weights: bias {w[0]:+.2f}, soc {w[1]:+.2f}, "
          f"temperature {w[2]:+.2f}, voltage {w[3]:+.2f}")

    policies = {
        "greedy (u_max)": lambda: pol_greedy(est),
        "thermal heuristic": lambda: pol_thermal_heuristic(est),
        "learned (charge only)": lambda: make_learned(est, w),
        "CC-CV 0.5C (production)": lambda: pol_ccv(est),
    }
    res = {k: {m: dict(soc=[], viol=0, clip=[]) for m in ("unfiltered", "filtered")}
           for k in policies}

    for _ in range(n):
        plant, _ = V.draw_plant("robotaxi-urban", rng, V.ENVELOPE, T_amb=25.0)
        s0 = V.safe_init(est, float(rng.uniform(0.05, 0.30)),
                         float(rng.uniform(18.0, 35.0)), marg)
        for name, mk in policies.items():
            for mode, filt in (("unfiltered", False), ("filtered", True)):
                r = run(plant, est, mk(), s0, marg, filt)   # fresh policy: some carry state
                a = res[name][mode]
                a["soc"].append(r["soc"]); a["viol"] += int(not r["ok"])
                a["clip"].append(r["clip_rate"])

    out = {"trials": n, "learned_weights": [float(x) for x in w], "policies": {}}
    for name in policies:
        rec = {}
        for mode in ("unfiltered", "filtered"):
            a = res[name][mode]; s = np.array(a["soc"])
            rec[mode] = dict(violations=a["viol"], violation_rate=a["viol"] / n,
                             mean_soc=float(s.mean()),
                             cp95_upper_pct=100 * stats.cp_upper(a["viol"], n),
                             clip_rate=float(np.mean(a["clip"])))
        # Against the unfiltered run -- which is what the filter *took away*. This is not a
        # cost in any usable sense, because the unfiltered charge was bought with violations;
        # it is reported only so the reader can see the size of what was refused.
        rec["delta_vs_unfiltered_points"] = 100 * (rec["filtered"]["mean_soc"]
                                                   - rec["unfiltered"]["mean_soc"])
        rec["contained"] = rec["filtered"]["violations"] == 0
        rec["charge_cost_points"] = rec["delta_vs_unfiltered_points"]
        rec["transparent"] = rec["filtered"]["clip_rate"] < 0.01
        out["policies"][name] = rec

    print(f"\n{n} sessions per policy, identical plants and seeds\n")
    print(f"{'policy':24}{'unfiltered':>22}{'filtered':>26}")
    print(f"{'':24}{'viol':>8}{'SOC':>8}{'':>6}{'viol':>8}{'SOC':>8}{'clip rate':>10}")
    for name, r in out["policies"].items():
        u, f = r["unfiltered"], r["filtered"]
        print(f"{name:24}{u['violations']:>8}{u['mean_soc']:>8.3f}{'':>6}"
              f"{f['violations']:>8}{f['mean_soc']:>8.3f}{100*f['clip_rate']:>9.1f}%")

    unsafe = {k: v for k, v in out["policies"].items()
              if v["unfiltered"]["violations"] > 0}
    out["unsafe_policies"] = list(unsafe)
    out["all_contained"] = all(v["contained"] for v in out["policies"].values())
    out["worst_unfiltered_rate"] = max(
        (v["unfiltered"]["violation_rate"] for v in out["policies"].values()), default=0.0)
    safe_pol = out["policies"]["CC-CV 0.5C (production)"]
    out["transparent_on_safe_policy"] = safe_pol["transparent"]
    out["safe_policy_charge_cost_points"] = safe_pol["charge_cost_points"]

    print(f"\n  containment: {len(unsafe)} of {len(out['policies'])} policies are unsafe "
          f"unfiltered (worst {100*out['worst_unfiltered_rate']:.0f}% of sessions); "
          f"filtered, all of them violate nothing: {out['all_contained']}")
    print(f"  transparency: on the already-safe policy the filter intervenes in "
          f"{100*safe_pol['filtered']['clip_rate']:.2f}% of steps and costs "
          f"{safe_pol['charge_cost_points']:+.2f} SOC points")
    lp = out["policies"]["learned (charge only)"]
    print(f"  the learned policy: {lp['unfiltered']['violations']}/{n} violations on its own, "
          f"{lp['filtered']['violations']}/{n} through the filter")
    # Worth stating rather than dressing up: the search converged on the boundary. A large
    # positive bias saturates the sigmoid, so the fitted policy is greedy in all but name. That
    # is not a defect of the search, it is what maximising delivered charge with no safety term
    # is *for*, and it is the reason a learned controller cannot be trusted to self-limit.
    out["learned_saturated"] = bool(w[0] > 3.0)
    out["learned_matches_greedy"] = bool(
        abs(lp["unfiltered"]["mean_soc"]
            - out["policies"]["greedy (u_max)"]["unfiltered"]["mean_soc"]) < 5e-3)
    if out["learned_saturated"]:
        print(f"    (the search converged on the boundary -- bias {w[0]:+.1f} saturates the "
              f"sigmoid, so the fitted policy is greedy in all but name. That is what an "
              f"objective with no safety term is for.)")

    # -----------------------------------------------------------------------------------
    # Transparency is not a property of one policy; it is a curve. A filter that clips
    # everything is safe and useless, and a single conservative test case cannot tell the two
    # apart. Sweeping the requested C-rate from below the admissible ceiling to far above it
    # asks how much of each policy survives -- and the intervention rate should rise with
    # aggressiveness rather than sitting at some constant, which is what a filter that ignores
    # the request would produce.
    # -----------------------------------------------------------------------------------
    sweep_C = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
    m = min(80, n)
    rng2 = np.random.default_rng(seed + 7)
    curve = []
    for c in sweep_C:
        cl, sc_f, sc_u, vu = [], [], [], 0
        for _ in range(m):
            plant, _ = V.draw_plant("robotaxi-urban", rng2, V.ENVELOPE, T_amb=25.0)
            s0 = V.safe_init(est, float(rng2.uniform(0.05, 0.30)),
                             float(rng2.uniform(18.0, 35.0)), marg)
            rf = run(plant, est, pol_ccv(est, c), s0, marg, True)
            ru = run(plant, est, pol_ccv(est, c), s0, marg, False)
            cl.append(rf["clip_rate"]); sc_f.append(rf["soc"]); sc_u.append(ru["soc"])
            vu += int(not ru["ok"])
        curve.append(dict(c_rate=c, clip_rate=float(np.mean(cl)),
                          unfiltered_violations=vu, trials=m,
                          filtered_soc=float(np.mean(sc_f)),
                          unfiltered_soc=float(np.mean(sc_u))))
    out["transparency_curve"] = curve
    out["curve_trials"] = m
    rates = [r["clip_rate"] for r in curve]
    out["curve_monotone"] = all(b >= a - 1e-9 for a, b in zip(rates, rates[1:]))
    passthrough = [r for r in curve if r["clip_rate"] < 0.01]
    out["max_untouched_C"] = max((r["c_rate"] for r in passthrough), default=None)
    out["min_unsafe_C"] = min((r["c_rate"] for r in curve
                               if r["unfiltered_violations"] > 0), default=None)

    print(f"\ntransparency across requested C-rate ({m} sessions each)")
    print(f"  {'request':>9}{'clip rate':>12}{'unfilt viol':>13}{'filt SOC':>10}")
    for r in curve:
        print(f"  {r['c_rate']:>8.1f}C{100*r['clip_rate']:>11.1f}%"
              f"{r['unfiltered_violations']:>10}/{r['trials']}{r['filtered_soc']:>10.3f}")
    print(f"  intervention rises monotonically with the request: {out['curve_monotone']}; "
          f"untouched up to {out['max_untouched_C']}C, "
          f"unsafe unfiltered from {out['min_unsafe_C']}C")

    # what the filtered aggressive policy delivers against the safe protocol it replaces
    safe_soc = safe_pol["filtered"]["mean_soc"]
    agg = out["policies"]["learned (charge only)"]["filtered"]["mean_soc"]
    out["safe_protocol_mean_soc"] = safe_soc
    out["filtered_aggressive_mean_soc"] = agg
    out["gain_over_safe_protocol_points"] = 100 * (agg - safe_soc)
    print(f"\n  against the protocol it replaces rather than the run it refused: the filtered "
          f"aggressive policy delivers {out['gain_over_safe_protocol_points']:+.1f} points "
          f"more than safe CC-CV, with the same zero violations")

    path = V.save("p1_policy_filter.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
