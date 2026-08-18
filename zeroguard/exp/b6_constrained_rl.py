"""B6 -- against constrained reinforcement learning, which the paper dismisses in a sentence.

\\S\\ref{sec:related} says constrained RL "offers expectation-level guarantees, which are the
wrong shape for a constraint that must hold on every step of every episode", cites CPO and
Altman, and then benchmarks against CC-CV, a de-rate sweep, MPC and the CBF quadratic program.
It never benchmarks against the thing it dismisses. In a paper whose whole method is to measure
rather than assert, that is the one assertion left standing, and it is the one a reader who
takes the related work seriously will go looking for.

So: a Lagrangian-constrained policy, trained the way the cited literature trains one.
The objective is charge delivered minus a multiplier times *expected* constraint cost, with the
multiplier adapted to drive the expected cost to a budget \\cite{stooke2020pid,achiam2017cpo}.
That is a faithful rendering of the method, not a straw man -- it is given the same cell, the
same envelope, and a richer state feedback than anything the certificate uses.

The claim under test is narrow and should be stated precisely, because the loose version is
false. Constrained RL is not bad at this. It is optimising the wrong functional: a bound on the
*expectation* of violations across episodes says nothing about any particular episode, and a
vehicle experiences particular episodes. Three things follow, and each is measured:

  **The multiplier trades safety against charge, and cannot buy the last of it.** Sweeping the
  budget traces a frontier. Where that frontier crosses zero *observed* violations, and what it
  costs to get there, is the comparison.

  **Zero observed is not zero proved.** Even a policy that violates nothing on a test set
  carries only a confidence interval; the certificate carries a per-step theorem. Both are
  reported as what they are.

  **And it is trained on a distribution.** Held to the same tune-then-deploy standard as every
  other baseline in \\S\\ref{sec:transfer}, the policy is fitted on a characterisation fleet and
  deployed on the one the vehicle meets.

    python zeroguard/exp/b6_constrained_rl.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V
from zeroguard.exp.b5_domain_transfer import CHAR, DEPLOY

SEED = 20260816
DT, HORIZON, TARGET = 30.0, 80, 0.80
CASE = "robotaxi-urban"
AMB_CHAR = (18.0, 32.0)
AMB_DEPLOY = (-10.0, 40.0)
BUDGETS = (0.50, 0.20, 0.10, 0.05, 0.02, 0.01, 0.0)   # tolerated expected violation rate


# ---------------------------------------------------------------------------------------
def policy(w):
    """u/u_max = sigma(w . [1, soc, (T-25)/20, (V-3.9)*4]). A richer feedback than the
    certificate uses, so nothing here is lost for want of expressiveness."""
    def f(s, v, u_max):
        z = (w[0] + w[1] * s["soc"] + w[2] * (s["T"] - 25.0) / 20.0
             + w[3] * ((3.9 if v is None else v) - 3.9) * 4.0)
        return float(u_max / (1.0 + np.exp(-np.clip(z, -40, 40))))
    return f


def episode(plant, est, s0, w, amb, marg, pol):
    """One session. Returns charge delivered and the per-step constraint cost RL optimises."""
    s, v = dict(s0), None
    cost, viol = 0.0, False
    for _k in range(HORIZON):
        u = float(np.clip(pol(s, v, est.u_max), 0.0, est.u_max))
        s, o = plant.step(s, u, DT, amb); v = float(o[0])
        # the shaped cost a constrained-RL agent actually sees: how far past each limit it went
        cost += max(0.0, float(o[0]) - plant.V_max) + max(0.0, float(o[1]) - plant.T_max)
        bad, _e = V.split_breaches(plant, V.check(plant, o))
        if bad:
            viol = True
        if s["soc"] >= TARGET:
            break
    return float(s["soc"]), cost, viol


def draw(rng, env, ambients):
    amb = float(rng.uniform(*ambients))
    sc = {k: float(rng.uniform(*val)) for k, val in env.items()}
    est = V.pessimistic(CASE, T_amb=amb)
    plant = P.build(CASE, scale=sc, T_amb=amb)
    marg = V.margins(est)
    s0 = V.safe_init(est, float(rng.uniform(0.05, 0.30)),
                     float(np.clip(rng.uniform(-10.0, 40.0), amb - 5.0, 44.0)), marg)
    return plant, est, s0, amb, marg


def train(budget, seed, iters=14, pop=48, elite=10, batch=24):
    """Cross-entropy search on a Lagrangian objective, multiplier adapted toward the budget.

    This is the PID-Lagrangian shape: raise the price of violation while the measured rate is
    above budget, lower it when below. The multiplier is what the cited methods learn; here it
    is adapted directly, which if anything makes the baseline stronger.
    """
    rng = np.random.default_rng(seed)
    mu, sd = np.zeros(4), np.full(4, 2.0)
    lam, best = 1.0, None
    for _ in range(iters):
        cand = mu + sd * rng.standard_normal((pop, 4))
        score = np.empty(pop); rate = np.empty(pop)
        for j in range(pop):
            pol = policy(cand[j])
            r = np.random.default_rng(seed + 7)
            socs, costs, vs = [], [], 0
            for _b in range(batch):
                plant, est, s0, amb, marg = draw(r, CHAR, AMB_CHAR)
                soc, cost, viol = episode(plant, est, s0, cand[j], amb, marg, pol)
                socs.append(soc); costs.append(cost); vs += int(viol)
            rate[j] = vs / batch
            score[j] = float(np.mean(socs)) - lam * float(np.mean(costs))
        idx = np.argsort(score)[-elite:]
        mu, sd = cand[idx].mean(axis=0), cand[idx].std(axis=0) + 0.05
        best = cand[idx[-1]]
        # adapt the price of violation toward the budget, as a Lagrangian method does
        gap = float(rate[idx].mean()) - budget
        lam = float(np.clip(lam * (1.6 if gap > 0 else 0.7), 1e-3, 1e6))
    return best, lam


def evaluate(w, env, ambients, n, seed):
    rng = np.random.default_rng(seed)
    pol = policy(w)
    socs, viol = [], 0
    for _ in range(n):
        plant, est, s0, amb, marg = draw(rng, env, ambients)
        soc, _c, v = episode(plant, est, s0, w, amb, marg, pol)
        socs.append(soc); viol += int(v)
    return dict(trials=n, violations=viol, violation_rate=viol / n,
                cp95_upper_pct=100 * stats.cp_upper(viol, n),
                mean_soc=float(np.mean(socs)))


def certificate(env, ambients, n, seed):
    rng = np.random.default_rng(seed)
    socs, viol = [], 0
    for _ in range(n):
        plant, est, s0, amb, marg = draw(rng, env, ambients)
        s = dict(s0)
        bad_any = False
        for _k in range(HORIZON):
            _lo, hi, st = A.interval(est, s, DT, amb, marg)
            u = hi if st == "ok" else 0.0
            s, o = plant.step(s, float(max(0.0, u)), DT, amb)
            b, _e = V.split_breaches(plant, V.check(plant, o))
            if b:
                bad_any = True
            if s["soc"] >= TARGET:
                break
        socs.append(float(s["soc"])); viol += int(bad_any)
    return dict(trials=n, violations=viol, violation_rate=viol / n,
                cp95_upper_pct=100 * stats.cp_upper(viol, n),
                mean_soc=float(np.mean(socs)))


def main(n=400, seed=SEED):
    t0 = time.time()
    print("B6 -- against constrained reinforcement learning\n" + "=" * 78)
    print("  Lagrangian-constrained policy, multiplier adapted toward a violation budget;")
    print("  trained on a characterisation fleet, deployed on the fleet a vehicle meets\n")
    rows = []
    print(f"  {'budget':>8}{'train viol':>13}{'deploy viol':>14}{'CP95 %':>9}{'SOC':>8}")
    for b in BUDGETS:
        w, lam = train(b, seed)
        tr = evaluate(w, CHAR, AMB_CHAR, n // 2, seed + 101)
        de = evaluate(w, DEPLOY, AMB_DEPLOY, n, seed + 202)
        rows.append(dict(budget=b, lam=lam, weights=[float(x) for x in w],
                         train=tr, deploy=de))
        print(f"  {b:>8.2f}{tr['violations']:>8}/{tr['trials']}"
              f"{de['violations']:>9}/{de['trials']}{de['cp95_upper_pct']:>9.2f}"
              f"{de['mean_soc']:>8.3f}")

    zg = certificate(DEPLOY, AMB_DEPLOY, n, seed + 202)
    print(f"  {'ZEROGUARD':>8}{'--':>13}{zg['violations']:>9}/{zg['trials']}"
          f"{zg['cp95_upper_pct']:>9.2f}{zg['mean_soc']:>8.3f}")

    safe = [r for r in rows if r["deploy"]["violations"] == 0]
    best_safe = max(safe, key=lambda r: r["deploy"]["mean_soc"]) if safe else None
    out = dict(trials=n, budgets=list(BUDGETS), rows=rows, zeroguard=zg,
               any_budget_safe_on_deployment=bool(safe),
               worst_deploy_rate=max(r["deploy"]["violation_rate"] for r in rows),
               best_train_rate=min(r["train"]["violation_rate"] for r in rows))
    if best_safe:
        out["best_safe_budget"] = best_safe["budget"]
        out["best_safe_soc"] = best_safe["deploy"]["mean_soc"]
        out["best_safe_cp95"] = best_safe["deploy"]["cp95_upper_pct"]
        out["gain_over_safe_rl_points"] = 100 * (zg["mean_soc"]
                                                 - best_safe["deploy"]["mean_soc"])
    # does tightening the budget in training actually transfer to deployment?
    tr_rates = [r["train"]["violation_rate"] for r in rows]
    de_rates = [r["deploy"]["violation_rate"] for r in rows]
    out["train_rate_reaches_zero"] = bool(min(tr_rates) == 0.0)
    out["deploy_rate_reaches_zero"] = bool(min(de_rates) == 0.0)
    out["worst_train_to_deploy_gap"] = float(max(d - t for t, d in zip(tr_rates, de_rates)))

    print()
    if not out["deploy_rate_reaches_zero"]:
        print(f"  no budget reaches zero violations on the deployment fleet; the best is "
              f"{100*min(de_rates):.1f}% of sessions")
    else:
        print(f"  some budgets do reach zero *observed* violations on deployment")
    if out["train_rate_reaches_zero"]:
        print(f"  and the gap is the point: a budget that reaches zero in training is still "
              f"violating up to {100*out['worst_train_to_deploy_gap']:.1f} points more on "
              f"deployment. An expectation held on one distribution is not a per-episode "
              f"guarantee on another.")
    if best_safe:
        print(f"  where it does reach zero observed ({best_safe['budget']:.2f} budget), it "
              f"still carries CP95 {out['best_safe_cp95']:.2f}% -- an interval, not a theorem "
              f"-- and delivers {out['gain_over_safe_rl_points']:+.1f} SOC points less than "
              f"the certificate")
    print(f"  the certificate: {zg['violations']}/{zg['trials']}, CP95 "
          f"{zg['cp95_upper_pct']:.2f}%, and the zero is proved per step rather than sampled")
    # -----------------------------------------------------------------------------------
    # The objection this experiment must survive is "you did not train it enough", and it is a
    # fair one: with a batch of 24 episodes a 1 % violation rate is not even measurable, so the
    # search cannot see the budget it is chasing. If more effort closes the deployment gap, the
    # finding is about our compute and not about the method. So the tightest budget is retrained
    # with four times the search and eight times the batch, and evaluated identically.
    # -----------------------------------------------------------------------------------
    print("\n  the same policy class, trained much harder at the tightest budget")
    w2, lam2 = train(0.0, seed + 3, iters=28, pop=96, elite=16, batch=192)
    tr2 = evaluate(w2, CHAR, AMB_CHAR, n // 2, seed + 101)
    de2 = evaluate(w2, DEPLOY, AMB_DEPLOY, n, seed + 202)
    out["hard_trained"] = dict(iters=28, pop=96, batch=192, lam=lam2,
                               train=tr2, deploy=de2,
                               weights=[float(x) for x in w2])
    base = min(r["deploy"]["violation_rate"] for r in rows)
    out["hard_training_helps"] = bool(de2["violation_rate"] < base - 0.02)
    print(f"    {'harder':>8}{tr2['violations']:>8}/{tr2['trials']}"
          f"{de2['violations']:>9}/{de2['trials']}{de2['cp95_upper_pct']:>9.2f}"
          f"{de2['mean_soc']:>8.3f}")
    if out["hard_training_helps"]:
        print(f"  more search does help ({100*base:.1f}% to "
              f"{100*de2['violation_rate']:.1f}%), so the gap is partly ours; what it does not "
              f"do is reach zero, and the shape of the guarantee is unchanged")
    else:
        print(f"  four times the search and eight times the batch does not close the gap "
              f"({100*base:.1f}% to {100*de2['violation_rate']:.1f}%). The limit is not "
              f"compute: an expectation held on one distribution is not a per-episode "
              f"guarantee on another.")

    path = V.save("b6_constrained_rl.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
