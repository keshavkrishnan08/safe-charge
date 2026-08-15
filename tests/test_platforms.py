"""Anchored Collapse and the vehicle platforms, as executable checks.

`tests/test_safety.py` checks the published filter's guarantees on a cell. This checks the
generalisation and the plants it runs on, in the same spirit: run
`python tests/test_platforms.py` and watch the properties hold rather than taking the module
docstrings on trust.

  1  the vehicle cell reproduces `BatteryROM.probe` bit for bit under the Newtonian law, on
     both discretizations, fresh and aged -- everything in `zeroguard/exp/v*.py` inherits this
  2  with the anchor at zero and no floor constraints, `project_anchored` IS `project_current`
  3  a constraint's *side* is declared, not inferred: an undeclared limit is refused rather
     than guessed at, because guessing it would be silent and unsafe
  4  the admissible set is a single interval on every platform, in both modes
  5  each edge is where feasibility actually flips, checked from both sides
  6  the voltage cap binds before the maximum-power point, which is what keeps the load a
     well-posed floor constraint rather than a two-sided one
  7  the cost is bounded: 20 model evaluations on charge, 40 on discharge, data-independent
  8  a pessimistic estimator shrinks the interval from *both* sides, so one corner dominates
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from safe_charge import BatteryROM, project_current
from zeroguard import anchored as A, platforms as P, vexp as V

W0 = dict(ground=25.0, aerial=15.0, underwater=2.0, space=4.0, reference=25.0)
T0 = dict(ground=(10.0, 42.0), aerial=(-20.0, 45.0), underwater=(-1.0, 18.0),
          space=(-35.0, 35.0), reference=(0.0, 42.0))


def _states(plat, rng, k):
    lo_T, hi_T = T0[plat.domain]
    lo_s = 0.03 if plat.mode == "charge" else max(plat.soc_floor + 0.05, 0.25)
    return [plat.init(float(rng.uniform(lo_s, 0.98)), float(rng.uniform(lo_T, hi_T)))
            for _ in range(k)]


def test_cell_is_bit_exact_with_the_published_rom():
    """1 -- the vehicle cell is the published cell, not a re-implementation of it."""
    p = P.load_params()
    rng = np.random.default_rng(11)
    n = 0
    for rc in ("euler", "exact"):
        for sc in (None, dict(R=1.8, Q=0.85, plate=1.3)):
            rom = BatteryROM(cell_scale=sc, rc=rc)
            cell = P.Cell(cooling=P.Newtonian(p["hA"]), scale=sc, rc=rc)
            for _ in range(4000):
                soc = float(rng.uniform(0.02, 0.98)); T = float(rng.uniform(-15.0, 55.0))
                V1 = float(rng.uniform(-0.2, 0.2)); I = float(rng.uniform(0.0, 15.0))
                dt = 30.0 if rc == "euler" else float(rng.choice([30.0, 120.0]))
                w = float(rng.uniform(-40.0, 50.0))
                s = dict(soc=soc, T=T, V1=V1, aging={"Qloss": 0.0, "Rfac": 1.0})
                a = rom.probe(s, I, dt, w)
                b = cell.probe(soc, T, V1, I, dt, w)
                assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3], \
                    f"cell diverged from the ROM at soc={soc} T={T} I={I} rc={rc}"
                n += 1
    print(f"  1  cell bit-exact with BatteryROM.probe on {n:,} states")


def test_anchored_reduces_to_project_current():
    """2 -- where the original theorem applies, the generalisation is the same function."""
    rng = np.random.default_rng(12)
    plat = P.single_cell()
    rom = BatteryROM()
    marg = (0.03, 0.5, 0.006)
    for _ in range(8000):
        s = dict(soc=float(rng.uniform(0.02, 0.98)), T=float(rng.uniform(-10.0, 44.0)),
                 V1=float(rng.uniform(-0.1, 0.1)), aging={"Qloss": 0.0, "Rfac": 1.0})
        w = float(rng.uniform(-20.0, 45.0))
        a, _ = project_current(rom, s, plat.u_max, 30.0, w, margin=rom.plating_margin(),
                               dV=marg[0], dT=marg[1], dP=marg[2], cool_frac=0.0)
        b, _ = A.project_anchored(plat, s, plat.u_max, 30.0, w, marg)
        assert a == b, f"anchored {b!r} != project_current {a!r} at {s}"
    print("  2  project_anchored == project_current, bit for bit, on 8 000 states")


def test_undeclared_constraint_side_is_refused():
    """3 -- the side of a constraint is data, not a guess.

    The published plating constraint is `phi >= margin`, a lower-bound test on a signal that
    *decreases* in current, so it caps the current from above. Inferring the side from the
    comparison sense would file it as a floor and let the filter command straight through it.
    That is the one modelling error in this method that would be silent and unsafe, so an
    undeclared side has to be an exception rather than a default."""
    plat = P.single_cell()
    plat.limits = ((0, "<=", 4.2), (1, "<=", 45.0), (2, ">=", 0.02))   # 3-tuples: no side
    plat._anchored_split = None
    try:
        A.split(plat.limits)
    except ValueError as e:
        assert "side" in str(e)
        print("  3  a limit with no declared side is refused, not guessed")
        return
    raise AssertionError("an undeclared constraint side was silently accepted")


def test_admissible_set_is_a_single_interval():
    """4 -- one bisection per edge is only valid if feasibility is one connected run."""
    rng = np.random.default_rng(13)
    total = bad = 0
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        marg = V.margins(plat)
        for s in _states(plat, rng, 120):
            _g, ok = A.scan(plat, s, plat.dt_nominal, W0[plat.domain], marg, n=120)
            st = A.structure(ok)
            total += 1
            bad += int(st == "disconnected")
    assert bad == 0, f"{bad} of {total} admissible sets were disconnected"
    print(f"  4  {total:,} dense scans across 16 platforms, 0 disconnected")


def test_the_edges_are_where_feasibility_flips():
    """5 -- both bisections land on real thresholds, not on plausible-looking numbers.

    This is the property that makes the interval an answer rather than an output: the floors
    hold at `u_lo` and fail just below it, the caps hold at `u_hi` and fail just above it. An
    edge that is merely somewhere in the right region would still produce zero violations while
    quietly giving away performance, so the check is two-sided at each edge.

    Note what is *not* asserted here. `Platform.anchor` is a closed-form estimate of the load
    current, computed at the lowest permitted bus voltage; the true lower edge is where the
    floor constraint actually flips, including its margin and the real terminal voltage. The
    two are close and neither is the other, so the anchor is a diagnostic and `u_lo` is the
    answer. `test_the_anchor_estimates_the_lower_edge` states how close they have to be."""
    rng = np.random.default_rng(14)
    checked = 0
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        marg = V.margins(plat)
        hi_f, lo_f = A.split_cached(plat)
        dt, w = plat.dt_nominal, W0[plat.domain]
        for s in _states(plat, rng, 200):
            lo, hi, st = A.interval(plat, s, dt, w, marg)
            if st != "ok":
                continue
            lo_b, hi_b = A._bounds(plat, s)
            eps = max(1e-7, (hi_b - lo_b) * 1e-5)
            assert A._ok(hi_f, plat.probe(s, hi, dt, w), marg), \
                f"{case}: caps fail AT the upper edge {hi}"
            if hi < hi_b - eps:
                assert not A._ok(hi_f, plat.probe(s, hi + eps, dt, w), marg), \
                    f"{case}: caps still hold above the upper edge {hi}"
            assert A._ok(lo_f, plat.probe(s, lo, dt, w), marg), \
                f"{case}: floors fail AT the lower edge {lo}"
            if lo > lo_b + eps:
                assert not A._ok(lo_f, plat.probe(s, lo - eps, dt, w), marg), \
                    f"{case}: floors still hold below the lower edge {lo}"
            checked += 1
    print(f"  5  both edges are true thresholds in all {checked:,} feasible states")


def test_the_anchor_estimates_the_lower_edge():
    """5b -- the diagnostic is honest about what it is.

    `anchor()` divides the load by the lowest permitted bus voltage. That is exact only if the
    pack happens to sit at exactly that voltage and the floor carries no margin, so in general
    it lands near the true lower edge rather than on it -- above it in the warm cases, where
    the real voltage is higher, and below it in the cold ones, where sag eats the difference.
    What has to be true is that it is the right *quantity*: within a factor of two of the edge
    whenever the interval is open, so that a figure drawing it is not misleading."""
    rng = np.random.default_rng(18)
    worst, checked = 1.0, 0
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        if plat.mode != "discharge":
            continue
        marg = V.margins(plat)
        for s in _states(plat, rng, 200):
            lo, hi, st = A.interval(plat, s, plat.dt_nominal, W0[plat.domain], marg)
            if st != "ok" or lo <= 1e-9:
                continue
            a = A.effective_anchor(plat, s)
            worst = max(worst, max(a / lo, lo / max(a, 1e-12)))
            checked += 1
    assert worst < 2.0, f"the anchor estimate is off the lower edge by {worst:.2f}x"
    print(f"  5b the anchor tracks the lower edge within {worst:.2f}x "
          f"({checked:,} states)")


def test_the_voltage_cap_binds_before_the_maximum_power_point():
    """6 -- the floor constraint is a suffix only on the rising branch, and it stays there.

    Delivered bus power is `S V(u) u` with `dV/du < 0`, so it peaks at the maximum-power point
    and falls beyond it. Past that peak the set where the load is met is an interval rather
    than a suffix, and `interval`'s lower bisection would be searching a non-monotone family.

    What prevents it is the voltage cap: a pack reaches its MPP near half its open-circuit
    voltage, far below any usable `V_min`, so the cap binds first and `u_hi < u_MPP`. That is
    a physical ordering rather than a coincidence, but it is also the assumption the method
    rests on, so it is measured on every discharge platform and the margin between the two is
    reported."""
    rng = np.random.default_rng(15)
    worst_ratio = 1e9
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        if plat.mode != "discharge":
            continue
        marg = V.margins(plat)
        dt, w = plat.dt_nominal, W0[plat.domain]
        for s in _states(plat, rng, 60):
            lo, hi, st = A.interval(plat, s, dt, w, marg)
            if st != "ok":
                continue
            g = np.linspace(plat.u_min, plat.u_max, 400)
            Pe = np.array([plat.probe(s, float(u), dt, w)[3] for u in g])
            u_mpp = float(g[int(np.argmax(Pe))])
            assert hi <= u_mpp + 1e-9, \
                f"{case}: upper edge {hi:.3f} is past the MPP {u_mpp:.3f}"
            rising = g <= hi
            assert np.all(np.diff(Pe[rising]) > -1e-9), \
                f"{case}: power is not monotone below the upper edge"
            # only meaningful where a constraint set the edge; where the actuator did, the
            # peak simply lies at or beyond the end of the range and the ratio is 1 by
            # construction rather than by narrowness
            if hi < plat.u_max - 1e-9:
                worst_ratio = min(worst_ratio, u_mpp / max(hi, 1e-12))
    print(f"  6  the voltage cap binds before the MPP on every discharge platform "
          f"(closest approach {worst_ratio:.2f}x)")


def test_cost_is_bounded_and_data_independent():
    """7 -- 20 evaluations on charge, 40 on discharge, whatever the data.

    Two probes to bracket the cap family plus 18 halvings, then two to bracket the floor family
    plus 18 more. The number that matters is not that it is small but that it does not move
    with the state, because a task slot is sized for the worst case and a QP's is unbounded."""
    rng = np.random.default_rng(16)
    worst = {"charge": 0, "discharge": 0}
    for case in P.ALL_CASES:
        plat = V.pessimistic(case)
        marg = V.margins(plat)
        for s in _states(plat, rng, 300):
            c = V.Counted(plat)
            A.project_anchored(c, s, plat.u_max, plat.dt_nominal, W0[plat.domain], marg)
            worst[plat.mode] = max(worst[plat.mode], c.n)
    assert worst["charge"] <= 20, f"charge path used {worst['charge']} evaluations"
    assert worst["discharge"] <= 2 * (18 + 2), \
        f"discharge path used {worst['discharge']} evaluations"
    print(f"  7  bounded cost: {worst['charge']} evaluations on charge, "
          f"{worst['discharge']} on discharge")


def test_pessimism_shrinks_both_edges():
    """8 -- one conservative corner dominates, rather than one corner per edge.

    Scaling the estimator's resistance up predicts more heating and more sag, which lowers the
    upper edge; it also predicts a lower terminal voltage, so more current is needed to serve
    the load, which raises the lower edge. Both edges move inward under the same perturbation.
    If they did not, the two-sided certificate would need a separate worst case per edge and
    `vexp.pessimistic` would be unsound."""
    ITERS = 34          # see below: at the deployed 18 this measures resolution, not physics
    rng = np.random.default_rng(17)
    checked = 0
    worst = 0.0
    for case in P.ALL_CASES:
        base = P.build(case)
        if base.mode != "discharge":
            continue
        marg = V.margins(base)
        # The floor bisection searches [u_min, u_hi], and u_hi itself moves with pessimism, so
        # the two runs being compared do not share a search range. At 18 halvings that alone
        # shifts the reported lower edge by ~1e-4 A on the glider -- larger than the physical
        # effect being tested, and in the opposite direction. This is the third time in this
        # project that bisection resolution has looked like a disagreement, so the tolerance is
        # tied to the resolution rather than to a hopeful constant.
        tol = base.u_max * 2.0 ** -(ITERS - 2)
        for s in _states(base, rng, 60):
            prev = None
            for sR in (1.0, 1.3, 1.6, 1.8):
                est = P.build(case, scale=dict(R=sR, Q=1.0, plate=1.0))
                lo, hi, st = A.interval(est, s, est.dt_nominal, W0[est.domain], marg,
                                        iters=ITERS)
                if st != "ok":
                    break
                if prev is not None:
                    assert hi <= prev[1] + tol, f"{case}: upper edge rose with pessimism"
                    assert lo >= prev[0] - tol, f"{case}: lower edge fell with pessimism"
                    worst = max(worst, max(hi - prev[1], prev[0] - lo, 0.0))
                    checked += 1
                prev = (lo, hi)
    print(f"  8  pessimism shrinks the interval from both sides ({checked:,} comparisons, "
          f"worst wrong-way move {worst:.2e} A)")


def main():
    print("Anchored Collapse and the vehicle platforms\n")
    fns = [test_cell_is_bit_exact_with_the_published_rom,
           test_anchored_reduces_to_project_current,
           test_undeclared_constraint_side_is_refused,
           test_admissible_set_is_a_single_interval,
           test_the_edges_are_where_feasibility_flips,
           test_the_anchor_estimates_the_lower_edge,
           test_the_voltage_cap_binds_before_the_maximum_power_point,
           test_cost_is_bounded_and_data_independent,
           test_pessimism_shrinks_both_edges]
    for f in fns:
        f()
    print(f"\n{len(fns)}/{len(fns)} checks pass")


if __name__ == "__main__":
    main()
