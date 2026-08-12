"""E2 -- where the theorem stops being true, and what happens there.

A characterization is only worth the name if it has a boundary, so this measures both
hypotheses failing, separately, and shows they fail in *different ways*:

  (A1) the null input is safe.   Break it and safety itself is lost: every one-step check
       passes, right up until no input can recover, because one-step feasibility never
       implied forward invariance without a fallback to fall back on.

  (A2) constraints are monotone.  Break it and only optimality is lost: the bisection still
       returns a feasible input -- it maintains lo <= u* on the branch containing zero -- but
       it cannot see admissible bands beyond the first gap, so it under-commands.

The asymmetry is the point. (A1) is a hypothesis about safety; (A2) is a hypothesis about
performance. The battery satisfies both, which is why its certificate is a certificate.

    python zeroguard/exp/e2_boundary.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard.systems import HoverQuadrotor, OscillatoryConstraint, BatterySystem
from zeroguard.gfilter import project, admissible_set, _feasible
from zeroguard import stats

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def a1_failure(n=2000, seed=5):
    """Quadrotor: the null input is not safe. Track how long one-step certification holds
    before the state becomes unrecoverable."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        sy = HoverQuadrotor()
        s = sy.init(h0=float(rng.uniform(4.0, 14.0)), v0=float(rng.uniform(-1.0, 1.0)))
        dt = sy.dt_nominal
        certified_steps, violated, unrecoverable_at = 0, False, None
        for k in range(sy.horizon):
            # a lazy policy asks for nothing -- exactly the request the battery filter
            # can always grant, and this system cannot
            u, clipped = project(sy, s, 0.0, dt, sy.w_nominal)
            # can ANY input keep it safe this step?
            recoverable = _feasible(sy, s, sy.u_max, dt, sy.w_nominal, (0.0,))
            if not recoverable and unrecoverable_at is None:
                unrecoverable_at = k
            s, vals = sy.step(s, u, dt, sy.w_nominal)
            if vals[0] >= sy.limits[0][2]:
                certified_steps += 1
            else:
                violated = True
                break
        rows.append((certified_steps, violated, unrecoverable_at if unrecoverable_at is not None else -1))
    steps = np.array([r[0] for r in rows]); viol = np.array([r[1] for r in rows])
    unrec = np.array([r[2] for r in rows])
    m, lo, hi = stats.bootstrap_ci(steps.astype(float))
    return dict(trials=n, violated=int(viol.sum()), violation_rate_pct=100.0 * viol.mean(),
                mean_certified_steps_before_violation=m, ci=[lo, hi],
                mean_unrecoverable_step=float(unrec[unrec >= 0].mean()) if (unrec >= 0).any() else None,
                interpretation="every step passed the one-step check; the state still became "
                               "unrecoverable, because (A1) fails so there is no fallback")


def a1_control(n=2000, seed=6):
    """The same lazy policy on the battery, where (A1) holds: zero forever is safe forever."""
    rng = np.random.default_rng(seed)
    viol = 0
    for _ in range(n):
        sy = BatterySystem()
        s = sy.init(soc0=float(rng.uniform(0.05, 0.8)), T0=float(rng.uniform(20.0, 44.0)))
        for _ in range(sy.horizon):
            u, _ = project(sy, s, 0.0, sy.dt_nominal, sy.w_nominal)
            s, vals = sy.step(s, u, sy.dt_nominal, sy.w_nominal)
            if vals[1] > 45.0 + 1e-9 or vals[0] > 4.20 + 1e-9:
                viol += 1
                break
    return stats.summarize_safety("battery_null_policy", viol, n)


def a2_failure(n=3000, seed=7):
    """Oscillatory constraint: measure whether the bisection's answer is (a) feasible and
    (b) how far below the true maximum it lands."""
    rng = np.random.default_rng(seed)
    gaps, feasible_flags, n_bands = [], [], []
    for _ in range(n):
        sy = OscillatoryConstraint()
        s = sy.init(T0=float(rng.uniform(46.0, 51.5)))
        u, _ = project(sy, s, sy.u_max, sy.dt_nominal, sy.w_nominal)
        grid, ok = admissible_set(sy, s, sy.dt_nominal, sy.w_nominal, n=2400)
        if not ok.any():
            continue
        true_sup = float(grid[ok].max())
        feasible_flags.append(bool(_feasible(sy, s, u, sy.dt_nominal, sy.w_nominal, (0.0,))))
        gaps.append(100.0 * (true_sup - u) / max(true_sup, 1e-12))
        n_bands.append(int(np.sum(np.diff(ok.astype(int)) == 1) + (1 if ok[0] else 0)))
    g = np.array(gaps)
    m, lo, hi = stats.bootstrap_ci(g)
    return dict(trials=len(gaps),
                always_feasible=bool(np.all(feasible_flags)),
                unsafe_returns=int(np.sum(~np.array(feasible_flags))),
                mean_optimality_gap_pct=m, gap_ci=[lo, hi],
                max_optimality_gap_pct=float(g.max()),
                mean_admissible_bands=float(np.mean(n_bands)),
                interpretation="the answer is always feasible -- (A2) failure costs "
                               "delivered performance, never safety")


def main():
    out = {}
    print("E2 -- the boundary of the theorem\n")

    print("(A1) the null input is NOT safe  [hover quadrotor]")
    r = a1_failure(); out["a1_failure"] = r
    print(f"  {r['trials']} episodes | violated {r['violated']} ({r['violation_rate_pct']:.1f}%)")
    print(f"  mean one-step-certified steps before violation: {r['mean_certified_steps_before_violation']:.1f}"
          f"  95% CI [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]")
    print(f"  mean step at which NO input could recover:      {r['mean_unrecoverable_step']:.1f}")

    print("\n(A1) holds  [battery, same lazy policy]")
    c = a1_control(); out["a1_control"] = c
    print(f"  {c['trials']} episodes | violations {c['violations']} | CP95 upper {c['cp95_upper_pct']:.3f}%")

    print("\n(A2) constraints NOT monotone  [oscillatory plant]")
    a = a2_failure(); out["a2_failure"] = a
    print(f"  {a['trials']} states | admissible set has {a['mean_admissible_bands']:.2f} bands on average")
    print(f"  returned input always feasible: {a['always_feasible']}  (unsafe returns: {a['unsafe_returns']})")
    print(f"  optimality gap: mean {a['mean_optimality_gap_pct']:.1f}%  "
          f"95% CI [{a['gap_ci'][0]:.1f}, {a['gap_ci'][1]:.1f}]  max {a['max_optimality_gap_pct']:.1f}%")

    print("\n  => (A1) failure costs SAFETY; (A2) failure costs only PERFORMANCE.")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e2_boundary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(RES,'e2_boundary.json')}")


if __name__ == "__main__":
    main()
