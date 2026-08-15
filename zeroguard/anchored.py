"""The generalisation the vehicles force: an anchor that need not be zero.

Null-Input Collapse needs `u = 0` to be safe. That is true of a cell being charged and false
of almost every vehicle doing its job. Cut the current to a quadrotor's motors and it falls.
Cut it to an AUV and it loses attitude control and its computer. Cut it to a spacecraft in
eclipse and the bus browns out. E2 measured that failure on the quadrotor -- 100 % of episodes
violated after a mean of 22.3 one-step-certified steps -- and concluded the theorem does not
apply. That conclusion is correct and also incomplete, because the *structure* is plainly
still there. It is only the anchor that moved.

Every vehicle has an irreducible load: the power it cannot stop drawing and stay a vehicle.
Hover thrust for a rotorcraft, hotel load for a submersible, bus load for a spacecraft in
eclipse. Charging is the special case where that load is zero, which is exactly why the IECON
filter could anchor at zero and never say so.

    Theorem (Anchored Collapse).  Let a system have state x, scalar input u in [u_min, u_max],
    and constraints c = 1..m, each satisfied on one side of a threshold in u: the *caps* on
    [u_min, u_c], the *floors* on [u_c, u_max].  If
    (A1') u = u_min is safe for every cap, from every x in the safe set, and
    (A2)  each constraint is monotone in u,
    then the admissible set is the interval

        [u_lo, u_hi] = [ max over floors u_c ,  min over caps u_c ]

    and both edges are found by bisection in O(log 1/eps) one-step evaluations -- with no
    optimizer, no convergence test, and no identification of the plant.  The set is non-empty
    exactly when u_lo <= u_hi.  The lower edge is the current the irreducible load demands,
    which is what the *anchor* names.

Null-Input Collapse is the case with no floors, which is why the battery filter needed one
bisection: its lower edge was pinned at zero by construction.

Note what (A1') keeps and what it drops. The original (A1) asks that u = 0 return the state to
the *whole* safe set; (A1') asks only that it satisfy the caps, which is the part that is still
true of a vehicle. Cut a quadrotor's current and nothing heats, nothing sags and nothing
discharges -- every cap is satisfied. What zero cannot do is fly the aircraft, and that is a
floor, not a cap. The failure E2 measured was never that zero became thermally unsafe; it was
that zero stopped serving the load, and the original theorem had no way to say so.

Note that the side of a constraint is *not* its comparison sense. The published plating
constraint is `phi >= margin`, a lower-bound test, but `phi` decreases in current, so it caps
the current from above and belongs to the "hi" family. Sense and side coincide only when the
signal happens to increase in u, so systems declare the side explicitly rather than let it be
guessed. Getting this wrong is silent and unsafe -- it would put a cap in the floor family and
let the filter command straight through it -- so it is declared data, not inference.

Two things follow that the null-input version cannot express:

  * A vehicle gets a *two-sided* certificate. The upper edge is thermal and electrical; the
    lower edge is the load it must keep serving. Safety is the statement that they have not
    crossed.

  * The width u_hi - u_lo is a reserve. It shrinks to zero as the envelope closes, so the
    filter can say how close to unrecoverable the vehicle is, and say it before arrival. The
    null-input filter has no such quantity, because its lower edge never moves.

Cost: two bisections instead of one. Worst case is 40 one-step evaluations against the
original 20 -- two probes to bracket the caps plus 18 halvings, then two to bracket the floors
plus 18 more. Both are fixed, both are branch-bounded, and nothing in the run-time argument
changes: what a safety task slot cannot absorb is an iteration count that depends on the data,
and neither of these does.
"""

HI, LO = "hi", "lo"


def split(limits):
    """Separate declared caps from declared floors, and reject anything undeclared."""
    hi, lo = [], []
    for i, lim in enumerate(limits):
        if len(lim) != 4:
            raise ValueError(
                f"limit {i} = {lim!r} has no declared side; the anchored filter requires "
                f"(index, sense, value, 'hi'|'lo') because the side is not implied by the sense")
        idx, sense, val, side = lim
        if side == HI:
            hi.append((i, idx, sense, val))
        elif side == LO:
            lo.append((i, idx, sense, val))
        else:
            raise ValueError(f"limit {i}: side must be 'hi' or 'lo', got {side!r}")
    return tuple(hi), tuple(lo)


def split_cached(sys_):
    """`split` memoised on the plant, because the bisection asks for it 39 times a call."""
    got = getattr(sys_, "_anchored_split", None)
    if got is not None and got[0] is sys_.limits:
        return got[1], got[2]
    hi, lo = split(sys_.limits)
    sys_._anchored_split = (sys_.limits, hi, lo)
    return hi, lo


def _ok(fam, vals, margins):
    for i, idx, sense, val in fam:
        m = margins[i]
        if sense == "<=":
            if not (vals[idx] <= val - m + 1e-9):
                return False
        else:
            if not (vals[idx] >= val + m - 1e-9):
                return False
    return True


def caps_ok(sys_, s, u, dt, w, margins):
    """Every constraint that bounds u from above."""
    hi, _ = split_cached(sys_)
    return _ok(hi, sys_.probe(s, u, dt, w), margins)


def floors_ok(sys_, s, u, dt, w, margins):
    """Every constraint that bounds u from below."""
    _, lo = split_cached(sys_)
    return _ok(lo, sys_.probe(s, u, dt, w), margins)


def feasible(sys_, s, u, dt, w, margins=None):
    if margins is None:
        margins = (0.0,) * len(sys_.limits)
    vals = sys_.probe(s, u, dt, w)
    hi, lo = split_cached(sys_)
    return _ok(hi, vals, margins) and _ok(lo, vals, margins)


def _bounds(sys_, s):
    lo_b = getattr(sys_, "u_min", 0.0)
    hi_b = sys_.u_max
    c = sys_.cap(s)
    if c is not None:
        hi_b = min(hi_b, c)
    return lo_b, max(lo_b, hi_b)


def effective_anchor(sys_, s):
    """What the load demands, clipped to what the actuator can deliver.

    `Platform.anchor` returns the current the load needs at the *lowest* bus voltage the
    constraints permit, so it is deliberately an over-estimate, and on a tightly sized vehicle
    it can exceed the actuator ceiling. This is the number figures draw and diagnostics report;
    it is **not** what either bisection starts from -- see `interval`."""
    lo_b, hi_b = _bounds(sys_, s)
    return min(max(sys_.anchor(s), lo_b), hi_b)


def interval(sys_, s, dt, w, margins=None, iters=18):
    """The admissible interval [u_lo, u_hi], by two independent bisections.

    Neither bisection needs the anchor, and working out why is what made the final version of
    this function shorter than the first. The caps are satisfied on a prefix `[u_min, u_hi]`
    and `u = 0` is in that prefix from any safe state, exactly as Null-Input Collapse says: cut
    the current and nothing heats, nothing sags, nothing discharges. The floors are satisfied
    on a suffix `[u_lo, u_max]` and `u_max` is in *that* one if anything is. So each family has
    a known-good endpoint of its own and each edge is a plain bisection over the full range.

    The anchor's role is therefore physical rather than algorithmic. It is the current the load
    demands, it is where `u_lo` ends up, and its being nonzero is the entire reason a second
    bisection exists. An earlier version of this function started the upper bisection *at* the
    anchor, which is sound only while the anchor is cap-feasible and produced an interval whose
    reported anchor sat outside it when the two disagreed; `tests/test_platforms.py` caught
    that, and this version cannot express the bug.

    Returns (u_lo, u_hi, status) with status "ok" or "infeasible". Infeasible has three
    distinguishable causes, and the filter should not conflate them:

      * the caps already fail at `u = 0` -- the state is outside the safe set and no input
        recovers it in one step;
      * the floors fail at `u_hi` -- the load costs more than the caps allow, which is the
        interesting case and the one a mission planner wants a warning about;
      * the edges have crossed, which is the same event seen from the other side.
    """
    if margins is None:
        margins = (0.0,) * len(sys_.limits)
    hi_f, lo_f = split_cached(sys_)
    lo_b, hi_b = _bounds(sys_, s)

    # ---- upper edge: caps hold on [u_min, u_hi], and u_min is in it -------------------
    if not _ok(hi_f, sys_.probe(s, lo_b, dt, w), margins):
        return lo_b, lo_b, "infeasible"          # already unsafe; nothing recovers it
    if _ok(hi_f, sys_.probe(s, hi_b, dt, w), margins):
        u_hi = hi_b
    else:
        a, b = lo_b, hi_b
        for _ in range(iters):
            mid = 0.5 * (a + b)
            if _ok(hi_f, sys_.probe(s, mid, dt, w), margins):
                a = mid
            else:
                b = mid
        u_hi = a

    # ---- lower edge: floors hold on [u_lo, u_hi], searched inside the caps -------------
    #
    # The floor search is bounded by u_hi, not by u_max, and that is a soundness requirement
    # rather than an optimisation. Delivered bus power is S V(u) u with V falling in u, so it
    # peaks at the maximum-power point and *falls* beyond it: past the MPP the set where the
    # load is met stops being a suffix and becomes an interval, and a bisection that straddles
    # the peak can report the load unservable when it is not. What keeps the search on the
    # rising branch is the voltage cap. A pack reaches its MPP at roughly half its open-circuit
    # voltage, far below any usable V_min, so u_hi < u_MPP always -- the cap binds first, by a
    # wide margin, and the interval [u_min, u_hi] is monotone by construction.
    # `tests/test_platforms.py` checks that ordering on every discharge platform instead of
    # assuming it; raising the rotorcraft's actuator ceiling to a power cell's 6C is what first
    # pushed u_max past the peak and exposed this.
    if not lo_f:
        u_lo = lo_b
    elif _ok(lo_f, sys_.probe(s, lo_b, dt, w), margins):
        u_lo = lo_b
    elif not _ok(lo_f, sys_.probe(s, u_hi, dt, w), margins):
        return u_hi, u_hi, "infeasible"          # the load cannot be met inside the caps
    else:
        a, b = lo_b, u_hi
        for _ in range(iters):
            mid = 0.5 * (a + b)
            if _ok(lo_f, sys_.probe(s, mid, dt, w), margins):
                b = mid
            else:
                a = mid
        u_lo = b

    if u_lo > u_hi + 1e-12:
        return u_lo, u_hi, "infeasible"          # the envelope has closed
    return u_lo, u_hi, "ok"


def project_anchored(sys_, s, u_req, dt, w, margins=None, iters=18):
    """Project a request onto the admissible interval.

    Returns (u, status) with status in {"unclipped", "clipped-hi", "clipped-lo", "infeasible"}.
    On "infeasible" the returned input is the anchor: the vehicle is told to serve its load and
    that the envelope has closed, which is the only honest answer available.
    """
    u_lo, u_hi, st = interval(sys_, s, dt, w, margins, iters)
    if st == "infeasible":
        return u_lo, "infeasible"
    u = min(max(float(u_req), u_lo), u_hi)
    if u > u_req + 1e-12:
        return u, "clipped-lo"
    if u < u_req - 1e-12:
        return u, "clipped-hi"
    return u, "unclipped"


def reserve(sys_, s, dt, w, margins=None, iters=18):
    """The width of the admissible interval: how much room the vehicle has left."""
    u_lo, u_hi, st = interval(sys_, s, dt, w, margins, iters)
    return dict(u_lo=u_lo, u_hi=u_hi, anchor=effective_anchor(sys_, s),
                demanded_anchor=sys_.anchor(s),
                width=(u_hi - u_lo) if st == "ok" else 0.0, status=st)


def scan(sys_, s, dt, w, margins=None, n=600):
    """Dense feasibility scan, for checking that the interval really is an interval."""
    import numpy as np
    lo_b, hi_b = _bounds(sys_, s)
    g = np.linspace(lo_b, hi_b, n)
    ok = np.array([feasible(sys_, s, float(u), dt, w, margins) for u in g])
    return g, ok


def structure(ok):
    """"empty", "single-interval", or "disconnected" -- the thing (A2) is supposed to buy."""
    if not ok.any():
        return "empty"
    idx = ok.nonzero()[0]
    return "single-interval" if (idx[-1] - idx[0] + 1) == idx.size else "disconnected"
