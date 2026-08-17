"""B5 -- the same standard applied to every domain, and the metric an operator pays for.

Two things are wrong with the per-domain impact numbers as they stand, and both are our doing.

**The standard was not applied evenly.** \\S11.1 caught a methodological error in B1 -- a fixed
de-rate tuned and evaluated on the same population -- and fixed it for the ground domain by
tuning on a fleet a manufacturer could characterise in advance and deploying on the fleet a
vehicle actually meets. B2 then compared against air, water and space incumbents *without* that
correction: every fixed SOC threshold there was swept against the very population it was scored
on. So the ground baseline was held to a stricter standard than the other three, which is
backwards, and the underwater and GEO results -- where the fixed rule won -- are exactly the
ones that correction would move. Reporting them without it is not conservative, it is
inconsistent.

**And the ground metric is the wrong quantity.** B3 reports mean state of charge after a fixed
number of steps. No fleet operator buys state of charge. A robotaxi that is charging is a
robotaxi not earning, so the quantity is **minutes to a usable charge**, and the two are not the
same comparison: a controller that is faster early and tapers late can deliver identical SOC at
the horizon while returning the vehicle to service sooner.

Neither of these is a search for a better number. The first makes the comparison consistent and
may well go against us -- if a fixed threshold survives its own transfer test, that is the
result. The second is what the application claim was always about.

    python zeroguard/exp/b5_domain_transfer.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260816

# What a manufacturer can characterise before shipping, and what the vehicle then meets.
CHAR = dict(R=(1.0, 1.4), Q=(0.90, 1.0), plate=(1.0, 1.2))
DEPLOY = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))

# Ambient bands, narrow for characterisation and wide for deployment. A rule tuned in a test
# tank at 10 C meets the Arctic later; a satellite qualified at room temperature meets eclipse.
BANDS = {
    "aerial":     dict(char=(15.0, 25.0), deploy=(5.0, 35.0)),
    "underwater": dict(char=(6.0, 12.0), deploy=(-1.0, 12.0)),
    "space":      dict(char=(0.0, 25.0), deploy=(-20.0, 25.0)),
}


def episode(case, rng, dt, horizon, recovery_steps, soc_rules, width_rules, env, band, **kw):
    """One mission, scored under every stopping rule at once, on a drawn plant.

    Structurally B2's episode with the population made an argument instead of a constant, which
    is the whole point: the same code has to be able to run the characterisation fleet and the
    deployment fleet, or the comparison is not the one being claimed.
    """
    sc = {k: float(rng.uniform(*v)) for k, v in env.items()}
    est = V.pessimistic(case, **kw)
    plant = P.build(case, scale=sc, **kw)
    marg = V.margins(est)
    lo_s = max(est.soc_floor + 0.05, 0.30)
    s = V.safe_init(est, float(rng.uniform(lo_s, 0.99)), float(rng.uniform(*band)), marg)
    w = est.w_nominal

    soc_tr, width_tr, feas_tr = [], [], []
    st = dict(s)
    for _k in range(horizon):
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
        if stop_at is None:
            stop_at = n - 1
        stop_at = int(min(max(stop_at, 0), n - 1))
        made = all(feas_tr[j] for j in range(stop_at, min(stop_at + recovery_steps, n)))
        made = made and (stop_at + recovery_steps) <= n
        return dict(mission_s=stop_at * dt, made_it=bool(made))

    out = {}
    for r in soc_rules:
        idx = next((i for i, v in enumerate(soc_tr) if v <= r), None)
        out[f"soc<={int(100*r)}%"] = score(idx)
    for th in width_rules:
        idx = next((i for i, v in enumerate(width_tr) if v <= th), None)
        out[f"reserve<={th:g}A"] = score(idx)
    return out


def population(case, dt, horizon, rec, socs, widths, env, band, n, seed, **kw):
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(n):
        for k, v in episode(case, rng, dt, horizon, rec, socs, widths,
                            env, band, **kw).items():
            a = acc.setdefault(k, dict(mission=[], made=0, n=0))
            a["mission"].append(v["mission_s"]); a["made"] += int(v["made_it"]); a["n"] += 1
    return {k: dict(rule=k, trials=a["n"], mission_median_s=float(np.median(a["mission"])),
                    recovered=a["made"], recovery_rate=a["made"] / a["n"])
            for k, a in acc.items()}


def transfer(label, case, dt, horizon, rec, socs, widths, domain_key, n, seed,
             targets=(0.95, 0.90, 0.80), **kw):
    """Tune the incumbent where it can be characterised; deploy it where the vehicle goes.

    The safety target is not fixed across domains, because in one of them nothing reaches 95 %.
    A rotorcraft flying a delivery sortie has so little envelope left at the end that no fixed
    SOC threshold -- and no certificate threshold either -- recovers in 95 % of episodes; fixing
    the target there produces no comparison at all rather than a result. So the target is the
    strictest of the three that *both* families can meet in characterisation, and it is reported
    alongside the numbers, because a gain at 80 % recovery is not a gain at 95 %.
    """
    band = BANDS[domain_key]
    tune = population(case, dt, horizon, rec, socs, widths, CHAR, band["char"],
                      n, seed, **kw)
    dep = population(case, dt, horizon, rec, socs, widths, DEPLOY, band["deploy"],
                     n, seed + 1, **kw)

    def best_at(rows, prefix, tg):
        ok = [r for k, r in rows.items()
              if k.startswith(prefix) and r["recovery_rate"] >= tg]
        return max(ok, key=lambda r: r["mission_median_s"]) if ok else None

    target = None
    for tg in targets:
        if best_at(tune, "soc<=", tg) and best_at(tune, "reserve<=", tg):
            target = tg
            break
    if target is None:
        return dict(label=label, case=case, target=None, trials=n,
                    reachable=False, targets_tried=list(targets))

    def best(rows, prefix):
        return best_at(rows, prefix, target)

    chosen = best(tune, "soc<=")
    cert_chosen = best(tune, "reserve<=")
    rec_out = dict(label=label, case=case, target=target, trials=n, reachable=True,
                   tune=tune, deploy=dep,
                   chosen_soc_rule=chosen["rule"] if chosen else None,
                   chosen_reserve_rule=cert_chosen["rule"] if cert_chosen else None)

    for tag, ch in (("soc", chosen), ("reserve", cert_chosen)):
        if ch is None:
            rec_out[f"{tag}_deployed"] = None
            continue
        d = dep[ch["rule"]]
        rec_out[f"{tag}_deployed"] = dict(
            rule=ch["rule"], tuned_recovery=ch["recovery_rate"],
            deployed_recovery=d["recovery_rate"],
            deployed_mission_s=d["mission_median_s"],
            holds=bool(d["recovery_rate"] >= target),
            shortfall=float(ch["recovery_rate"] - d["recovery_rate"]))

    # And what each family would have had to be, had the deployment fleet been known
    for tag, prefix in (("soc", "soc<="), ("reserve", "reserve<=")):
        b = best(dep, prefix)
        rec_out[f"{tag}_required"] = (
            dict(rule=b["rule"], mission_s=b["mission_median_s"]) if b else None)

    sr, rr = rec_out["soc_required"], rec_out["reserve_required"]
    if sr and rr:
        rec_out["gain_s"] = rr["mission_s"] - sr["mission_s"]
        rec_out["gain_pct"] = 100 * (rr["mission_s"] / max(sr["mission_s"], 1e-9) - 1)
    return rec_out


# =======================================================================================
def ground_time_to_charge(n=1200, seed=SEED + 5):
    """Minutes back on the road, and where the advantage actually lives.

    B3 reports mean state of charge after a fixed horizon, averaged over a fleet spanning -10 to
    40 C. That average turned out to be hiding a sign change, so it is the wrong summary twice
    over: no operator buys state of charge, and no fleet experiences the mean ambient.
    """
    from zeroguard.exp.b1_baselines import run_ccv, run_filter, DT, HORIZON
    AMBIENTS = (-10.0, 0.0, 15.0, 25.0, 35.0, 40.0)
    TARGETS = (0.50, 0.60, 0.70)
    print("\nB5b  ground: minutes back on the road, and where the advantage lives")
    rng = np.random.default_rng(seed)
    per = {a: dict(zg=[], cc=[], zv=0, cv=0, zt=[], ct=[], n=0) for a in AMBIENTS}
    for _ in range(n):
        amb = float(rng.choice(AMBIENTS))
        sc = {k: float(rng.uniform(*v)) for k, v in DEPLOY.items()}
        est = V.pessimistic("robotaxi-urban", T_amb=amb)
        plant = P.RobotaxiUrban(T_amb=amb, scale=sc)
        marg = V.margins(est)
        s0 = V.safe_init(est, float(rng.uniform(0.05, 0.30)),
                         float(np.clip(rng.uniform(-10.0, 40.0), amb - 5.0, 44.0)), marg)
        z = run_filter(plant, est, s0, amb, marg)
        c = run_ccv(plant, est, s0, amb, 0.5, marg)      # the rate B3 showed is fleet-safe
        a = per[amb]
        a["n"] += 1
        a["zg"].append(z["soc"]); a["cc"].append(c["soc"])
        a["zv"] += int(not z["ok"]); a["cv"] += int(not c["ok"])
        a["zt"].append(z["steps"] * DT / 60.0); a["ct"].append(c["steps"] * DT / 60.0)

    zg_all = np.concatenate([per[a]["zg"] for a in AMBIENTS])
    cc_all = np.concatenate([per[a]["cc"] for a in AMBIENTS])
    out = dict(trials=n, horizon_min=HORIZON * DT / 60.0, ambients=list(AMBIENTS),
               baseline="CC-CV 0.5C, the rate B3 shows is safe across the whole fleet",
               zg_violations=int(sum(per[a]["zv"] for a in AMBIENTS)),
               ccv_violations=int(sum(per[a]["cv"] for a in AMBIENTS)),
               mean_gain_points=float(100 * (zg_all.mean() - cc_all.mean())))

    # (1) how many sessions come back fit for service inside the window
    print(f"  sessions reaching a usable charge within the {out['horizon_min']:.0f} min window")
    print(f"    {'target':>8}{'certificate':>14}{'CC-CV 0.5C':>13}{'ratio':>9}")
    out["reach"] = {}
    for tg in TARGETS:
        zr = float((zg_all >= tg).mean()); cr = float((cc_all >= tg).mean())
        out["reach"][f"{int(100*tg)}"] = dict(
            target=tg, zg=zr, ccv=cr, points=100 * (zr - cr),
            ratio=(zr / cr) if cr > 1e-9 else None)
        rt = f"{zr/cr:.1f}x" if cr > 1e-9 else "n/a"
        print(f"    {tg:>7.0%}{zr:>13.0%}{cr:>13.0%}{rt:>9}")

    # (2) and the decomposition the average was hiding
    print(f"\n  by ambient -- the mean is a fleet average over a sign change, not a fact "
          f"about any vehicle")
    print(f"    {'ambient':>9}{'certificate':>13}{'CC-CV':>9}{'delta':>10}{'breaches':>12}")
    rows = []
    for a in AMBIENTS:
        d = per[a]
        z, c = float(np.mean(d["zg"])), float(np.mean(d["cc"]))
        rows.append(dict(ambient_C=a, trials=d["n"], zg_soc=z, ccv_soc=c,
                         gain_points=100 * (z - c),
                         zg_violations=d["zv"], ccv_violations=d["cv"]))
        print(f"    {a:>+8.0f}C{z:>13.3f}{c:>9.3f}{100*(z-c):>+9.1f} pts"
              f"{d['zv']:>7}/{d['cv']}")
    out["by_ambient"] = rows
    win = [r["ambient_C"] for r in rows if r["gain_points"] > 0]
    lose = [r["ambient_C"] for r in rows if r["gain_points"] <= 0]
    out["wins_at"] = win
    out["loses_at"] = lose
    out["best_gain_points"] = max(r["gain_points"] for r in rows)
    out["worst_gain_points"] = min(r["gain_points"] for r in rows)
    out["crossover_between_C"] = [max(win), min(lose)] if win and lose else None

    # (3) name the mechanism rather than leaving the reversal unexplained
    est25 = V.pessimistic("robotaxi-urban", T_amb=25.0)
    ceiling = est25.T_max - V.margins(est25)[1]
    out["effective_ceiling_C"] = float(ceiling)
    print(f"\n  the reversal is not a mystery and it is not the theorem. The thermal margin "
          f"puts the certificate's effective ceiling at {ceiling:.1f} C; above that ambient "
          f"the cell can never sit below it, so the filter refuses -- correctly, and at a "
          f"cost. That is \\S9's passive-charging ceiling appearing as a performance number.")
    print(f"  it delivers {out['best_gain_points']:+.0f} points at its best ambient and "
          f"{out['worst_gain_points']:+.0f} at its worst, with "
          f"{out['zg_violations']}/{n} breaches against the baseline's "
          f"{out['ccv_violations']}/{n}. The fleet mean of "
          f"{out['mean_gain_points']:+.1f} points is the average of those, and quoting it "
          f"alone would hide both halves.")
    return out


def main(n=300, seed=SEED):
    t0 = time.time()
    print("B5 -- the same standard in every domain\n" + "=" * 78)
    specs = [
        ("aerial: delivery quadrotor", "delivery-quadrotor", 2.0, 900, 15, "aerial",
         [0.45, 0.40, 0.35, 0.30, 0.25, 0.20], [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0], {}),
        ("underwater: under-ice AUV", "under-ice-auv", 60.0, 1500, 20, "underwater",
         [0.50, 0.40, 0.30, 0.25, 0.20], [2.0, 4.0, 8.0, 12.0, 16.0, 20.0], {}),
        ("space: GEO comsat in eclipse", "geo-comsat", 60.0, 300, 10, "space",
         [0.60, 0.55, 0.50, 0.45, 0.40], [4.0, 8.0, 16.0, 24.0, 32.0], {}),
    ]
    out = {"trials_per_domain": n, "target": 0.95,
           "char_envelope": CHAR, "deploy_envelope": DEPLOY, "bands": BANDS, "domains": {}}
    print(f"\nB5a  tune on what can be characterised, deploy on what the vehicle meets")
    print(f"     (recovery target {100*out['target']:.0f}% of episodes)\n")
    for label, case, dt, hor, rec, dk, socs, widths, kw in specs:
        r = transfer(label, case, dt, hor, rec, socs, widths, dk, n, seed, **kw)
        out["domains"][case] = r
        print(f"  {label}   (recovery target {100*r['target']:.0f}%)"
              if r.get("reachable") else f"  {label}   NO FAMILY REACHES ANY TARGET")
        if not r.get("reachable"):
            print()
            continue
        sd = r["soc_deployed"]
        if sd is None:
            print(f"    no fixed SOC threshold reaches the target even in characterisation")
        else:
            verdict = "HOLDS" if sd["holds"] else "FAILS"
            print(f"    incumbent tuned to {sd['rule']:14} recovers "
                  f"{100*sd['tuned_recovery']:5.1f}% in characterisation and "
                  f"{100*sd['deployed_recovery']:5.1f}% deployed   -> {verdict}")
        rd = r["reserve_deployed"]
        if rd is not None:
            verdict = "HOLDS" if rd["holds"] else "FAILS"
            print(f"    certificate reserve {rd['rule']:14} recovers "
                  f"{100*rd['tuned_recovery']:5.1f}% / {100*rd['deployed_recovery']:5.1f}% "
                  f"  -> {verdict}")
        if r.get("gain_pct") is not None:
            print(f"    knowing the deployment fleet, the best of each family delivers "
                  f"{r['soc_required']['mission_s']/60:.1f} vs "
                  f"{r['reserve_required']['mission_s']/60:.1f} min "
                  f"({r['gain_pct']:+.1f}% to the certificate)")
        print()

    fails = [c for c, r in out["domains"].items()
             if r.get("soc_deployed") is not None and not r["soc_deployed"]["holds"]]
    cert_fails = [c for c, r in out["domains"].items()
                  if r.get("reserve_deployed") is not None
                  and not r["reserve_deployed"]["holds"]]
    out["incumbent_fails_transfer"] = fails
    out["certificate_fails_transfer"] = cert_fails
    print(f"  the tuned fixed threshold fails its transfer test in {len(fails)} of "
          f"{len(out['domains'])} domains: {', '.join(fails) if fails else 'none'}")
    print(f"  the certificate reserve fails in {len(cert_fails)}: "
          f"{', '.join(cert_fails) if cert_fails else 'none'}")

    out["ground"] = ground_time_to_charge()
    path = V.save("b5_domain_transfer.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
