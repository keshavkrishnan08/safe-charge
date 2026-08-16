"""B3 -- a fixed de-rate, tuned honestly, then deployed.

B1 swept the de-rate and found something that should have been obvious earlier: on a single
population, a fixed 1.0C constant-current protocol is safe *and* delivers more charge than the
certificate. The certificate carries a pessimistic estimator and a 12.5 K throttle; a fixed rate
that happens to suit the population carries neither, so it wins. Quoting a gain against 0.5C, a
number we chose, was measuring our own choice of baseline.

But that comparison contains a methodological error, and it is one this study would refuse to
accept in someone else's work: **the fixed rate was tuned and evaluated on the same
distribution.** That is the oldest mistake in empirical work. A manufacturer does not get to
pick a de-rate knowing the population it will meet; it picks one from what it can characterise
in advance, and then ships into weather, ageing and duty cycles that the characterisation did
not contain.

So this experiment does what should have been done first:

  1. **Tune** on a benign, characterisable fleet -- mild ambient, fresh-to-middle-aged cells --
     and take the fastest fixed C-rate that is safe on it. That is the best de-rate an honest
     manufacturer could have chosen with that evidence.
  2. **Deploy** both that rate and the certificate on the fleet the vehicle actually meets:
     cold mornings, hot afternoons, cells at end of life.
  3. Report what each does there.

The certificate is given no advantage: it sees the same widened population, with the same
pessimistic estimator it has carried throughout, and it is not re-tuned between the two stages
because it has nothing to tune.

    python zeroguard/exp/b3_tuned_derate.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V
from zeroguard.exp.b1_baselines import run_ccv, run_filter, DT, HORIZON, TARGET

SEED = 20260815

# What a manufacturer can characterise in advance: a mild climate, cells that are not old yet.
TUNE = dict(env=dict(R=(1.0, 1.4), Q=(0.90, 1.0), plate=(1.0, 1.2)),
            ambients=(25.0,), T0=(18.0, 32.0), label="characterisation fleet")

# What the vehicle actually meets over its life.
DEPLOY = dict(env=dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6)),
              ambients=(-10.0, 0.0, 15.0, 25.0, 35.0, 40.0), T0=(-10.0, 40.0),
              label="deployment fleet")

SWEEP = (0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0)


def population(spec, rng, n):
    for _ in range(n):
        amb = float(rng.choice(spec["ambients"]))
        sc = {k: float(rng.uniform(*v)) for k, v in spec["env"].items()}
        T0 = float(np.clip(rng.uniform(*spec["T0"]), amb - 5.0, 44.0))
        yield amb, sc, float(rng.uniform(0.05, 0.30)), T0


def evaluate(spec, n, seed, rates):
    """Every controller on identical draws from one population."""
    rng = np.random.default_rng(seed)
    acc = {f"ccv_{c:g}C": dict(soc=[], viol=0) for c in rates}
    acc["zeroguard"] = dict(soc=[], viol=0)
    for amb, sc, soc0, T0 in population(spec, rng, n):
        est = V.pessimistic("robotaxi-urban", T_amb=amb)
        plant = P.RobotaxiUrban(T_amb=amb, scale=sc)
        marg = V.margins(est)
        s0 = est.init(soc0, T0)
        r = run_filter(plant, est, s0, amb, marg)
        acc["zeroguard"]["soc"].append(r["soc"]); acc["zeroguard"]["viol"] += int(not r["ok"])
        for c in rates:
            rr = run_ccv(plant, est, s0, amb, c, marg)
            k = f"ccv_{c:g}C"
            acc[k]["soc"].append(rr["soc"]); acc[k]["viol"] += int(not rr["ok"])
    out = {}
    for k, a in acc.items():
        s = np.array(a["soc"])
        out[k] = dict(name=k, trials=len(s), violations=a["viol"],
                      violation_rate=a["viol"] / len(s),
                      mean_soc=float(s.mean()),
                      cp95_upper_pct=100 * stats.cp_upper(a["viol"], len(s)))
    return out


def main(n=400, seed=SEED):
    t0 = time.time()
    print("B3 -- a fixed de-rate, tuned honestly, then deployed\n" + "=" * 78)

    tune = evaluate(TUNE, n, seed, SWEEP)
    safe = [c for c in SWEEP if tune[f"ccv_{c:g}C"]["violations"] == 0]
    chosen = max(safe) if safe else min(SWEEP)
    print(f"\nstage 1 -- tuning on the {TUNE['label']} "
          f"(ambient {TUNE['ambients']}, R<={TUNE['env']['R'][1]}, Q>={TUNE['env']['Q'][0]})")
    print(f"  {'C-rate':>8}{'viol':>7}{'mean SOC':>10}")
    for c in SWEEP:
        r = tune[f"ccv_{c:g}C"]
        print(f"  {c:>7.1f}C{r['violations']:>7}{r['mean_soc']:>10.3f}"
              f"{'   <- fastest safe' if c == chosen else ''}")
    print(f"  the best de-rate this evidence supports: {chosen}C")

    dep = evaluate(DEPLOY, n, seed + 1, SWEEP)
    key = f"ccv_{chosen:g}C"
    d_fix, d_zg = dep[key], dep["zeroguard"]
    print(f"\nstage 2 -- deploying both on the {DEPLOY['label']} "
          f"(ambient {DEPLOY['ambients'][0]:.0f} to {DEPLOY['ambients'][-1]:.0f} C, to end of life)")
    print(f"  {'controller':22}{'viol':>7}{'rate':>9}{'CP95 %':>9}{'mean SOC':>10}")
    for k in (key, "zeroguard"):
        r = dep[k]
        print(f"  {k:22}{r['violations']:>7}{100*r['violation_rate']:>8.1f}%"
              f"{r['cp95_upper_pct']:>9.3f}{r['mean_soc']:>10.3f}")

    # what would a manufacturer have had to ship to stay safe on the deployment fleet?
    safe_dep = [c for c in SWEEP if dep[f"ccv_{c:g}C"]["violations"] == 0]
    needed = max(safe_dep) if safe_dep else None
    out = dict(trials=n, sweep=list(SWEEP),
               tune=dict(label=TUNE["label"], rows=tune, chosen_C=chosen,
                         safe_rates=safe),
               deploy=dict(label=DEPLOY["label"], rows=dep,
                           safe_rates=safe_dep, required_C=needed),
               tuned_rate_violates_on_deployment=bool(d_fix["violations"] > 0),
               tuned_rate_violation_rate=d_fix["violation_rate"],
               tuned_rate_cp95=d_fix["cp95_upper_pct"],
               zeroguard_violations=d_zg["violations"],
               zeroguard_cp95=d_zg["cp95_upper_pct"],
               charge_gap_points=100 * (d_zg["mean_soc"] - d_fix["mean_soc"]))
    if needed is not None:
        rq = dep[f"ccv_{needed:g}C"]
        out["gain_over_safe_fixed_points"] = 100 * (d_zg["mean_soc"] - rq["mean_soc"])
        out["gain_over_safe_fixed_pct"] = 100 * (d_zg["mean_soc"] / rq["mean_soc"] - 1)
        out["safe_fixed_mean_soc"] = rq["mean_soc"]

    print()
    if out["tuned_rate_violates_on_deployment"]:
        print(f"  the de-rate that was safe in characterisation violates in "
              f"{100*out['tuned_rate_violation_rate']:.1f}% of deployment sessions")
        print(f"  the certificate, re-tuned not at all, violates in "
              f"{d_zg['violations']}/{d_zg['trials']}")
    else:
        print(f"  the tuned de-rate remained safe on deployment "
              f"({d_fix['violations']}/{d_fix['trials']})")
    if needed is not None:
        print(f"  to be safe across the deployment fleet a fixed rate must fall to {needed}C, "
              f"and against that the certificate delivers "
              f"{out['gain_over_safe_fixed_points']:+.1f} points "
              f"({out['gain_over_safe_fixed_pct']:+.1f}%)")
    else:
        print("  no fixed rate in the sweep is safe across the deployment fleet")

    path = V.save("b3_tuned_derate.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
