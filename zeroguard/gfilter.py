"""The null-input projection, written once for any monotone system.

This is `safe_charge.filter.project_current` with the electrochemistry lifted out. The
control flow is identical line for line -- the same cap, the same two short-circuits, the
same fixed iteration count, the same 1e-9 comparison slack -- so that instantiating it on
the battery reproduces the published filter exactly rather than approximately. `exp/e1.py`
checks that bit for bit over 60 000 states.

    Theorem (Null-Input Collapse).  If u = 0 returns every safe state to the safe set, and
    every constrained signal is non-decreasing in u, then {u : one-step transition is safe}
    is a single interval [0, u*], and bisection finds u* in a fixed number of one-step
    evaluations -- with no optimizer, no convergence test, and no identification of the
    plant's uncertain parameters.

The battery is one instance of that theorem. So are a heater, a motor and a power switch.
"""


def _feasible(sys, s, u, dt, w, margins):
    vals = sys.probe(s, u, dt, w)
    for (idx, sense, lim), m in zip(sys.limits, margins):
        v = vals[idx]
        if sense == "<=":
            if not (v <= lim - m + 1e-9):
                return False
        else:
            if not (v >= lim + m - 1e-9):
                return False
    return True


def project(sys, s, u_prop, dt, w, margins=None, iters=18):
    """Largest u in [0, u_prop] whose one-step transition stays inside the safe set.

    Returns (u, clipped). `clipped` is True whenever the request was reduced, including the
    already-unsafe case where even u = 0 fails and the filter commands zero as best effort.
    """
    if margins is None:
        margins = (0.0,) * len(sys.limits)
    u_prop = max(0.0, float(u_prop))
    cap = sys.cap(s)
    if cap is not None:
        u_prop = min(u_prop, cap)
    if _feasible(sys, s, u_prop, dt, w, margins):
        return u_prop, False
    lo, hi = 0.0, u_prop
    if not _feasible(sys, s, 0.0, dt, w, margins):
        return 0.0, True
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _feasible(sys, s, mid, dt, w, margins):
            lo = mid
        else:
            hi = mid
    return lo, True


def admissible_set(sys, s, dt, w, margins=None, n=800):
    """Dense scan of feasibility over [0, u_max]; used to check that the admissible set
    really is one interval anchored at zero, rather than trusting the theorem."""
    import numpy as np
    if margins is None:
        margins = (0.0,) * len(sys.limits)
    grid = np.linspace(0.0, sys.u_max, n)
    ok = np.array([_feasible(sys, s, float(u), dt, w, margins) for u in grid])
    return grid, ok


def interval_diagnosis(ok):
    """Classify a feasibility mask: 'interval-at-zero', 'empty', or 'disconnected'."""
    if not ok.any():
        return "empty"
    if not ok[0]:
        return "not-anchored-at-zero"
    first_false = int(ok.argmin()) if not ok.all() else len(ok)
    return "interval-at-zero" if not ok[first_false:].any() else "disconnected"
