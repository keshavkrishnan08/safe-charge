"""B4 -- against the control-barrier-function filter, which is the method's own family.

\\S2.1 places this work among predictive safety filters and control barrier functions and then
benchmarks against CC--CV, a de-rate sweep and MPC. That is a gap a control-theory reader will
find immediately, because the CBF quadratic program is *the* standard safety filter and this
method claims to replace the optimisation inside it. Prose is not an answer.

The discrete-time CBF condition \\cite{agrawal2017dcbf} asks, for each constraint written as
`h(x) >= 0`,

    h(x_{k+1}(u)) >= (1 - gamma) h(x_k),        gamma in (0, 1],

and the filter is the projection `min (u - u_ref)^2` subject to those. In practice `h(x_{k+1})`
is not available in closed form for a cell model, so the constraint is **linearised** in the
input,

    h(x_k) + (dh/du) u >= (1 - gamma) h(x_k),

which turns the problem into a QP. That linearisation is where the two methods actually differ,
and this experiment measures the difference rather than arguing it.

Three things are asked, in increasing order of what they settle:

  **Is the linearised filter safe?** It is the one people deploy, so the question is empirical.

  **Can gamma fix it?** If some gamma is both safe and non-conservative, the CBF-QP is simply the
  better tool and this work should say so. If the safe gamma has to be found by sweeping against
  the population -- which is what tuning a de-rate is -- then it inherits exactly the problem
  \\S11.1 was written about.

  **What happens if the CBF constraint is enforced exactly?** This is the interesting one. For a
  constraint monotone in a scalar input, `{u : h(x_{k+1}(u)) >= (1-gamma) h(x_k)}` is an
  interval, its edges are the roots, and bisection finds them. So the exact discrete-time CBF
  filter for a monotone system *is* this method -- with the margin playing the role of gamma. If
  that holds numerically then the contribution is not a rival to CBF but a statement of when the
  CBF-QP can be solved exactly at fixed cost, which is a stronger claim than beating it.

    python zeroguard/exp/b4_cbf.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260816
DT, HORIZON, TARGET = 30.0, 80, 0.80
GAMMAS = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
FD_STEP = 1e-3           # finite-difference step for dh/du, relative to u_max


def barriers(plat, s, u, dt, w, marg):
    """`h_i >= 0` for every declared limit, margins folded in, at input `u`.

    This is the same constraint set the certificate enforces, written in barrier form, so the
    two filters are held to an identical definition of safe. Nothing here is a softer test.
    """
    vals = plat.probe(s, u, dt, w)
    h = []
    for i, (idx, sense, val, _side) in enumerate(plat.limits):
        m = marg[i]
        h.append((val - m) - vals[idx] if sense == "<=" else vals[idx] - (val + m))
    return np.array(h)


def cbf_qp(plat, s, u_ref, dt, w, marg, gamma, counter=None):
    """The linearised discrete-time CBF filter, as deployed.

    Two model evaluations give `h` at the anchor and a forward difference for `dh/du`; the QP is
    then scalar and its solution is a clip, so no iterative solver is involved and the cost is
    not what separates the methods. Soundness is.
    """
    lo_b, hi_b = A._bounds(plat, s)
    h0 = barriers(plat, s, lo_b, dt, w, marg)
    du = FD_STEP * plat.u_max
    h1 = barriers(plat, s, lo_b + du, dt, w, marg)
    if counter is not None:
        counter[0] += 2
    grad = (h1 - h0) / du

    # h0 + grad * (u - lo_b) >= (1 - gamma) * h0   <=>   grad * (u - lo_b) >= -gamma * h0
    lo, hi = lo_b, hi_b
    for g, hh in zip(grad, h0):
        rhs = -gamma * hh
        if abs(g) < 1e-12:
            if rhs > 1e-9:
                return lo_b, "infeasible"
            continue
        edge = lo_b + rhs / g
        if g > 0:
            lo = max(lo, edge)
        else:
            hi = min(hi, edge)
    if lo > hi:
        return lo_b, "infeasible"
    return float(min(max(u_ref, lo), hi)), "ok"


def exact_cbf(plat, s, dt, w, marg, gamma, iters=40):
    """The same CBF condition enforced on the *true* model, by bisection.

    The condition `h(x_{k+1}(u)) >= (1-gamma) h(x_k)` is the certificate's feasibility test with
    the margin replaced by a gamma-scaled shortfall, so this is `A.interval` with a different
    margin vector -- which is the point being made, not a coincidence.
    """
    h0 = barriers(plat, s, A._bounds(plat, s)[0], dt, w, marg)
    shifted = tuple(float(m + (1.0 - gamma) * max(hh, 0.0))
                    for m, hh in zip(marg, h0))
    return A.interval(plat, s, dt, w, shifted, iters=iters)


def session(plant, est, s0, marg, rule, gamma=None, seed_u=None):
    """One 350 kW charging session under one of the filters."""
    s = dict(s0)
    viol, plated, ev = set(), set(), [0]
    lin_err = []
    for k in range(HORIZON):
        if rule == "cbf":
            c = V.Counted(est)
            cnt = [0]
            u, st = cbf_qp(c, s, est.u_max, DT, 25.0, marg, gamma, cnt)
            ev.append(cnt[0])
            # how wrong the linearisation was, measured against the true model at the same u
            h_true = barriers(est, s, u, DT, 25.0, marg)
            h0 = barriers(est, s, A._bounds(est, s)[0], DT, 25.0, marg)
            du = FD_STEP * est.u_max
            grad = (barriers(est, s, A._bounds(est, s)[0] + du, DT, 25.0, marg) - h0) / du
            h_lin = h0 + grad * (u - A._bounds(est, s)[0])
            lin_err.append(float(np.max(h_lin - h_true)))
        else:
            c = V.Counted(est)
            _lo, hi, st = A.interval(c, s, DT, 25.0, marg)
            ev.append(c.n)
            u = hi if st == "ok" else 0.0
        s, o = plant.step(s, float(max(0.0, u)), DT, 25.0)
        bad, enf = V.split_breaches(plant, V.check(plant, o))
        viol.update(bad); plated.update(enf)
        if s["soc"] >= TARGET:
            break
    return dict(soc=float(s["soc"]), ok=not viol, plated=bool(plated),
                evals=float(np.mean(ev[1:])) if len(ev) > 1 else 0.0,
                lin_err=float(np.max(lin_err)) if lin_err else 0.0)


def main(n=400, seed=SEED):
    t0 = time.time()
    print("B4 -- against the control-barrier-function filter\n" + "=" * 78)
    est = V.pessimistic("robotaxi-urban", T_amb=25.0)
    marg = V.margins(est)

    # ---------------------------------------------------------------------------------
    print(f"\nthe linearised CBF-QP across gamma, and the certificate, on identical draws "
          f"({n} sessions each)\n")
    rows = []
    for gamma in GAMMAS:
        rng = np.random.default_rng(seed)
        viol = plate = 0
        socs, evs, errs = [], [], []
        for _ in range(n):
            plant, _ = V.draw_plant("robotaxi-urban", rng, V.ENVELOPE, T_amb=25.0)
            s0 = V.safe_init(est, float(rng.uniform(0.05, 0.30)),
                             float(rng.uniform(18.0, 35.0)), marg)
            r = session(plant, est, s0, marg, "cbf", gamma)
            viol += int(not r["ok"]); plate += int(r["plated"])
            socs.append(r["soc"]); evs.append(r["evals"]); errs.append(r["lin_err"])
        rows.append(dict(gamma=gamma, violations=viol, plating=plate, trials=n,
                         violation_rate=viol / n, mean_soc=float(np.mean(socs)),
                         evals=float(np.mean(evs)),
                         cp95_upper_pct=100 * stats.cp_upper(viol, n),
                         max_lin_error=float(np.max(errs))))

    rng = np.random.default_rng(seed)
    zv = zp = 0
    zsoc, zev = [], []
    for _ in range(n):
        plant, _ = V.draw_plant("robotaxi-urban", rng, V.ENVELOPE, T_amb=25.0)
        s0 = V.safe_init(est, float(rng.uniform(0.05, 0.30)),
                         float(rng.uniform(18.0, 35.0)), marg)
        r = session(plant, est, s0, marg, "zg")
        zv += int(not r["ok"]); zp += int(r["plated"])
        zsoc.append(r["soc"]); zev.append(r["evals"])
    zg = dict(violations=zv, plating=zp, trials=n, violation_rate=zv / n,
              mean_soc=float(np.mean(zsoc)), evals=float(np.mean(zev)),
              cp95_upper_pct=100 * stats.cp_upper(zv, n))

    print(f"  {'filter':22}{'breach':>9}{'plating':>9}{'CP95 %':>9}{'SOC':>8}{'evals':>8}")
    for r in rows:
        print(f"  {'CBF-QP  gamma=' + format(r['gamma'], '.2f'):22}"
              f"{r['violations']:>6}/{n}{r['plating']:>9}{r['cp95_upper_pct']:>9.2f}"
              f"{r['mean_soc']:>8.3f}{r['evals']:>8.1f}")
    print(f"  {'ZEROGUARD':22}{zg['violations']:>6}/{n}{zg['plating']:>9}"
          f"{zg['cp95_upper_pct']:>9.2f}{zg['mean_soc']:>8.3f}{zg['evals']:>8.1f}")

    safe_g = [r["gamma"] for r in rows if r["violations"] == 0]
    best_safe = max((r for r in rows if r["violations"] == 0),
                    key=lambda r: r["mean_soc"], default=None)
    out = dict(trials=n, gammas=list(GAMMAS), cbf=rows, zeroguard=zg,
               safe_gammas=safe_g, any_gamma_safe=bool(safe_g),
               worst_cbf_rate=max(r["violation_rate"] for r in rows),
               max_lin_error_K=max(r["max_lin_error"] for r in rows))
    if best_safe is not None:
        out["best_safe_gamma"] = best_safe["gamma"]
        out["best_safe_cbf_soc"] = best_safe["mean_soc"]
        out["gain_over_best_safe_cbf_points"] = 100 * (zg["mean_soc"] - best_safe["mean_soc"])
        print(f"\n  the best gamma that is safe here is {best_safe['gamma']:g}, and against it "
              f"the certificate delivers "
              f"{out['gain_over_best_safe_cbf_points']:+.1f} SOC points")

    print(f"\n  the linearisation is optimistic by up to {out['max_lin_error_K']:.2f} "
          f"in barrier units -- that is the gap the QP does not see")

    # ---------------------------------------------------------------------------------
    # The claim that matters: enforce the same CBF condition on the true model and the
    # admissible set is an interval whose edges bisection finds. If that holds, this method is
    # the exact CBF filter for a monotone system rather than a competitor to it.
    # ---------------------------------------------------------------------------------
    print("\nthe exact CBF condition, enforced on the true model\n")
    rng = np.random.default_rng(seed + 5)
    disc = checked = 0
    worst_dev = 0.0
    for _ in range(600):
        s = V.safe_init(est, float(rng.uniform(0.05, 0.78)), float(rng.uniform(-5.0, 40.0)), marg)
        gamma = float(rng.choice(GAMMAS))
        h0 = barriers(est, s, A._bounds(est, s)[0], DT, 25.0, marg)
        shifted = tuple(float(m + (1.0 - gamma) * max(hh, 0.0)) for m, hh in zip(marg, h0))
        lo, hi, st = A.interval(est, s, DT, 25.0, shifted, iters=40)
        g, ok = A.scan(est, s, DT, 25.0, shifted, n=400)
        checked += 1
        if A.structure(ok) == "disconnected":
            disc += 1
        if st == "ok" and ok.any():
            worst_dev = max(worst_dev, abs(float(g[ok][-1]) - hi),
                            abs(float(g[ok][0]) - lo))
    res = float(est.u_max - A._bounds(est, est.init(0.5, 25.0))[0]) / (2 ** 40)
    out["exact"] = dict(states=checked, disconnected=disc, worst_deviation_A=worst_dev,
                        scan_step_A=float((est.u_max) / 399),
                        within_scan_resolution=worst_dev <= float(est.u_max) / 399,
                        bisection_resolution_A=res)
    print(f"  {checked} states, gamma drawn from the same sweep")
    print(f"  the admissible set was one interval in {checked - disc} of {checked}")
    print(f"  bisection edges agree with a dense scan to {worst_dev:.4g} A, against a scan "
          f"step of {out['exact']['scan_step_A']:.4g} A")
    if disc == 0 and out["exact"]["within_scan_resolution"]:
        print("  so for these constraints the exact discrete-time CBF filter and this method "
              "are the same filter: gamma enters as a margin, and the QP is the linearised "
              "approximation of a problem that can be solved exactly at fixed cost")

    # -----------------------------------------------------------------------------------
    # Both filters were safe above, and it would be easy -- and wrong -- to read that as the
    # linearisation being harmless. It is not what is doing the work. Both filters were handed a
    # *pessimistic* estimator sitting on the worst corner of the envelope, and both carried the
    # 12.5 K margin on top of it; between them there is enough slack to swallow a linearisation
    # error many times larger than the one measured. Shrinking the margin does not isolate
    # anything either, because the pessimism alone is sufficient -- tested to zero margin,
    # neither filter broke.
    #
    # The only way to see the approximation is to remove every other source of conservatism:
    # give both filters the *exact* plant, with no envelope pessimism and no margin at all. A
    # filter that evaluates the true one-step model is then exactly safe by construction, and
    # anything that breaks is breaking on its own approximation and nothing else.
    # -----------------------------------------------------------------------------------
    # gamma is swept here rather than fixed at whichever value looked best above, because
    # gamma < 1 keeps a fraction of the barrier in reserve and that reserve is exactly what
    # would hide the linearisation. gamma = 1 is the apples-to-apples case: the CBF condition is
    # then h(x_{k+1}) >= 0, which is the invariance condition the certificate enforces, and the
    # only difference left between the two filters is that one linearises it.
    print("\nperfect model, zero margin: what is left is the approximation\n")
    zero = tuple(0.0 for _ in marg)
    iso = {}
    for rule, label, best_g in ([(f"cbf@{g:g}", f"CBF-QP  gamma={g:.2f}", g) for g in GAMMAS]
                                + [("zg", "ZEROGUARD", 1.0)]):
        rng = np.random.default_rng(seed + 11)
        vv = pp = 0
        socs = []
        for _ in range(600):
            plant, sc = V.draw_plant("robotaxi-urban", rng, V.ENVELOPE, T_amb=25.0)
            exact = P.RobotaxiUrban(T_amb=25.0, scale=sc)      # the filter's model IS the plant
            s0 = V.safe_init(exact, float(rng.uniform(0.05, 0.30)),
                             float(rng.uniform(18.0, 35.0)), marg)
            r = session(plant, exact, s0, zero, rule.split("@")[0], best_g)
            vv += int(not r["ok"]); pp += int(r["plated"]); socs.append(r["soc"])
        iso[rule] = dict(violations=vv, plating=pp, trials=600, violation_rate=vv / 600,
                         cp95_upper_pct=100 * stats.cp_upper(vv, 600),
                         mean_soc=float(np.mean(socs)))
        print(f"  {label:24}{vv:>5}/600 breached   CP95 {iso[rule]['cp95_upper_pct']:>6.2f}%"
              f"   SOC {iso[rule]['mean_soc']:.3f}")
    out["isolated"] = iso
    one = iso["cbf@1"]
    out["isolated_matched_gamma"] = 1.0
    out["linearisation_breaks"] = one["violations"] > 0
    out["linearisation_breach_rate"] = one["violation_rate"]
    out["linearisation_cp95"] = one["cp95_upper_pct"]
    out["exact_stays_safe"] = iso["zg"]["violations"] == 0
    out["iso_safe_gammas"] = [g for g in GAMMAS if iso[f"cbf@{g:g}"]["violations"] == 0]
    # against the best *safe* gamma, never against gamma = 1, whose charge was bought with
    # breaches and is not a baseline anyone may claim against
    _best = max(out["iso_safe_gammas"], default=None)
    out["iso_best_safe_gamma"] = _best
    out["iso_charge_gap_points"] = (
        100 * (iso["zg"]["mean_soc"] - iso[f"cbf@{_best:g}"]["mean_soc"])
        if _best is not None else None)
    if out["linearisation_breaks"] and out["exact_stays_safe"]:
        print(f"\n  at the matched condition -- gamma = 1, which is invariance itself -- the "
              f"linearised filter breaches in {100*one['violation_rate']:.1f}% of sessions and "
              f"the bisection in none, on the same plants with a perfect model and no margin. "
              f"The QP is not unsafe because CBF is unsound; it is unsafe because it solves a "
              f"linearised problem.")
        print(f"  it can be made safe by lowering gamma, and that is the trade being made: "
              f"safe at gamma <= {_best:g}, and there it delivers "
              f"{out['iso_charge_gap_points']:.1f} SOC points less than the exact filter. The "
              f"reserve gamma holds back is charge the vehicle does not get, bought to cover an "
              f"approximation that need not have been made.")
    elif not out["linearisation_breaks"]:
        print("\n  the linearised filter held even here; on this model the linearisation error "
              "is genuinely small, and the case against the QP rests on conservatism alone")

    path = V.save("b4_cbf.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
