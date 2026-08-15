"""B2 -- against the rules each domain actually uses.

B1 compared the certificate against CC-CV and MPC on a charging pack. That is the right
comparison for the ground domain and it is the wrong one for every other, because the decision
a flying, swimming or orbiting vehicle makes is not "how much current" but **"when do I stop"**.

Each of those domains has an incumbent rule, and each rule is a fixed fraction:

  aerial       land when state of charge falls below a fixed reserve (20-30 % is typical)
  underwater   turn back when state of charge falls below a fixed reserve (30 % is typical)
  space        do not discharge past a fixed depth of discharge (50-60 % is typical in GEO)

All three are the same design: a constant chosen offline for the worst case, applied whatever
the vehicle's actual state. They are chosen that way for exactly the reason the ground de-rate
is -- the quantity that actually matters is not observable. A fixed SOC reserve is
simultaneously too cautious in warm conditions with a fresh pack and not cautious enough in cold
conditions with an aged one, and nothing on board tells the vehicle which it is in.

The certificate replaces the constant with an evaluation. The question this experiment asks is
whether that is worth anything: **does it deliver more mission at the same safety, or does it
just relabel the same trade-off?**

Safety here is not "no violation" -- it is *did the vehicle get home*. A rule that flies longer
and arrives dead is not better. So every controller is scored on both: mission delivered, and
the fraction of episodes that retained enough envelope to complete a stated recovery manoeuvre.

    python zeroguard/exp/b2_domain_baselines.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260815
ENV = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))


def _episode(case, rng, dt, horizon, recovery_steps, soc_rules, width_rules, **kw):
    """One mission, scored under several stopping rules at once.

    Every rule sees the identical plant, the identical disturbance sequence and the identical
    trajectory -- they differ only in when they call a halt. Running them on one trajectory
    rather than on separate draws removes sampling noise from the comparison entirely.
    """
    sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
    est = V.pessimistic(case, **kw)
    plant = P.build(case, scale=sc, **kw)
    marg = V.margins(est)
    lo_s = max(est.soc_floor + 0.05, 0.30)
    s = est.init(float(rng.uniform(lo_s, 0.99)), float(rng.uniform(*{
        "aerial": (5.0, 35.0), "underwater": (-1.0, 12.0), "space": (-20.0, 25.0),
    }[est.domain])))
    w = est.w_nominal

    # roll the mission out once, recording what every rule needs to decide
    soc_tr, width_tr, feas_tr = [], [], []
    st = dict(s)
    for k in range(horizon):
        u_lo, u_hi, status = A.interval(est, st, dt, w, marg)
        ok = status == "ok"
        soc_tr.append(float(st["soc"]))
        width_tr.append((u_hi - u_lo) if ok else 0.0)
        feas_tr.append(ok)
        if not ok:
            break
        st, _o = plant.step(st, float(u_lo), dt, w)
    n = len(soc_tr)

    def score(stop_at):
        """Mission delivered, and whether enough envelope remained for the recovery."""
        if stop_at is None:
            stop_at = n - 1
        stop_at = int(min(max(stop_at, 0), n - 1))
        # a recovery is possible only if the envelope was still open that many steps later
        made_it = all(feas_tr[j] for j in range(stop_at,
                                                min(stop_at + recovery_steps, n)))
        made_it = made_it and (stop_at + recovery_steps) <= n
        return dict(mission_s=stop_at * dt, made_it=bool(made_it))

    out = {}
    for r in soc_rules:
        idx = next((i for i, v in enumerate(soc_tr) if v <= r), None)
        out[f"soc<={int(100*r)}%"] = score(idx)
    for th in width_rules:
        idx = next((i for i, v in enumerate(width_tr) if v <= th), None)
        out[f"reserve<={th:g}A"] = score(idx)
    # the rule the paper actually recommends: act on whichever clock fires first
    for r in soc_rules:
        for th in width_rules:
            idx = next((i for i in range(n) if soc_tr[i] <= r or width_tr[i] <= th), None)
            out[f"min(soc<={int(100*r)}%, reserve<={th:g}A)"] = score(idx)
    out["run_to_closure"] = score(None)
    return out


def domain(case, dt, horizon, recovery_steps, soc_rules, width_rules, n, seed, **kw):
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n):
        for k, v in _episode(case, rng, dt, horizon, recovery_steps,
                             soc_rules, width_rules, **kw).items():
            a = acc.setdefault(k, dict(mission=[], made=0, n=0))
            a["mission"].append(v["mission_s"]); a["made"] += int(v["made_it"]); a["n"] += 1
    rows = {}
    for k, a in acc.items():
        m = np.array(a["mission"])
        rows[k] = dict(rule=k, trials=a["n"],
                       mission_median_s=float(np.median(m)),
                       mission_mean_s=float(m.mean()),
                       recovered=a["made"], recovery_rate=a["made"] / a["n"])
    # the fairest headline: among rules that recover at least as reliably as the incumbent,
    # which delivers the most mission?
    return rows


def compare(rows, target=0.95):
    """Iso-safety comparison: at a matched recovery target, which family delivers most mission?

    Comparing each rule against the incumbent's own recovery rate is the wrong test, and the
    first version of this experiment made that mistake. On a rotorcraft the incumbent SOC rule
    recovers in 0 % of episodes, so "at least as safe as the incumbent" is satisfied by every
    rule including running to closure, and the comparison returns nonsense.

    The right question fixes the safety level first and asks what each family buys at it. A rule
    that cannot reach the target at any threshold is reported as unable to -- which is itself
    the finding in one of the three domains.
    """
    fams = {"fixed SOC reserve": [], "certificate reserve": [], "both (min of two clocks)": []}
    for k, r in rows.items():
        if k == "run_to_closure":
            continue
        if k.startswith("soc"):
            fams["fixed SOC reserve"].append(r)
        elif k.startswith("reserve"):
            fams["certificate reserve"].append(r)
        else:
            fams["both (min of two clocks)"].append(r)
    best = {}
    for name, rs in fams.items():
        ok = [r for r in rs if r["recovery_rate"] >= target]
        best[name] = (max(ok, key=lambda r: r["mission_median_s"]) if ok else None)
    soc, res, both = (best["fixed SOC reserve"], best["certificate reserve"],
                      best["both (min of two clocks)"])
    out = dict(target_recovery=target,
               best_per_family={k: (v["rule"] if v else None) for k, v in best.items()},
               mission_per_family={k: (v["mission_median_s"] if v else None)
                                   for k, v in best.items()},
               recovery_per_family={k: (v["recovery_rate"] if v else None)
                                    for k, v in best.items()},
               soc_rule_can_reach_target=soc is not None,
               reserve_rule_can_reach_target=res is not None)
    if soc and res:
        out["reserve_gain_s"] = res["mission_median_s"] - soc["mission_median_s"]
        out["reserve_gain_pct"] = (100 * out["reserve_gain_s"] / soc["mission_median_s"]
                                   if soc["mission_median_s"] > 0 else None)
    if both and soc:
        out["combined_gain_s"] = both["mission_median_s"] - soc["mission_median_s"]
        out["combined_gain_pct"] = (100 * out["combined_gain_s"] / soc["mission_median_s"]
                                    if soc["mission_median_s"] > 0 else None)
    return out


# ---------------------------------------------------------------------------------------
def main(n=300, seed=SEED):
    t0 = time.time()
    print("B2 -- against the stopping rule each domain actually uses\n" + "=" * 78)
    out = {"trials_per_domain": n}

    specs = [
        ("aerial: delivery quadrotor", "delivery-quadrotor", 2.0, 900, 15,
         [0.45, 0.40, 0.35, 0.30, 0.25, 0.20], [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0], {}),
        ("underwater: under-ice AUV", "under-ice-auv", 60.0, 1500, 20,
         [0.50, 0.40, 0.30, 0.25, 0.20], [2.0, 4.0, 8.0, 12.0, 16.0, 20.0], {}),
        ("space: GEO comsat in eclipse", "geo-comsat", 60.0, 300, 10,
         [0.60, 0.55, 0.50, 0.45, 0.40], [4.0, 8.0, 16.0, 24.0, 32.0], {}),
    ]
    for label, case, dt, hor, rec, socs, widths, kw in specs:
        rows = domain(case, dt, hor, rec, socs, widths, n, seed, **kw)
        cmp = compare(rows)
        curve = {f"{int(100*tg)}": compare(rows, tg) for tg in (0.80, 0.90, 0.95, 0.99)}
        out[case] = dict(label=label, rules=rows, comparison=cmp, curve=curve,
                         recovery_steps=rec, recovery_s=rec * dt)
        print(f"\n{label}   (recovery = {rec*dt/60:.0f} min of remaining envelope; "
              f"iso-safety target {100*cmp['target_recovery']:.0f} %)")
        print(f"  {'family':28}{'best rule at target':>26}{'mission':>12}")
        for fam in ("fixed SOC reserve", "certificate reserve", "both (min of two clocks)"):
            rule = cmp["best_per_family"][fam]
            if rule is None:
                print(f"  {fam:28}{'CANNOT REACH TARGET':>26}{'--':>12}")
            else:
                print(f"  {fam:28}{rule:>26}"
                      f"{cmp['mission_per_family'][fam]/60:>9.1f} min")
        if cmp.get("reserve_gain_pct") is not None:
            print(f"    certificate reserve vs fixed SOC: "
                  f"{cmp['reserve_gain_s']/60:+.1f} min ({cmp['reserve_gain_pct']:+.1f}%)")
        if cmp.get("combined_gain_pct") is not None:
            print(f"    both clocks vs fixed SOC:         "
                  f"{cmp['combined_gain_s']/60:+.1f} min ({cmp['combined_gain_pct']:+.1f}%)")
        print(f"  mission delivered at each safety target (min), '--' = unreachable:")
        print(f"    {'target':>8}{'fixed SOC':>12}{'reserve':>10}{'both':>10}")
        for tg, cv in curve.items():
            mp = cv["mission_per_family"]
            def f(x): return f"{x/60:.1f}" if x is not None else "--"
            print(f"    {tg+' %':>8}{f(mp['fixed SOC reserve']):>12}"
                  f"{f(mp['certificate reserve']):>10}"
                  f"{f(mp['both (min of two clocks)']):>10}")

    gains = [out[c]["comparison"].get("combined_gain_pct") for _l, c, *_ in specs]
    gains = [g for g in gains if g is not None]
    soc_fails = [c for _l, c, *_ in specs
                 if not out[c]["comparison"]["soc_rule_can_reach_target"]]
    out["domains_where_soc_rule_cannot_reach_target"] = soc_fails
    out["all_domains_improved"] = bool(gains and all(g >= 0 for g in gains))
    out["median_combined_gain_pct"] = float(np.median(gains)) if gains else None
    print(f"\n  domains where NO fixed SOC reserve reaches the safety target: "
          f"{soc_fails if soc_fails else 'none'}")
    print(f"  median gain of both-clocks over the best fixed SOC rule: "
          f"{out['median_combined_gain_pct']:+.1f}%")

    path = V.save("b2_domain_baselines.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
