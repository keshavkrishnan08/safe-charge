"""Shared drivers for the vehicle experiments.

Every experiment in `exp/v1_*` .. `exp/v4_*` runs one of three loops, and they are written once
here so that a difference between two domains is a difference in the platform rather than a
difference in the harness.

The pattern throughout is **pessimistic estimator against a drawn plant**. The filter is given
a platform built at the conservative corner of the parameter envelope; the plant it actually
drives is drawn from inside that envelope. This is the only arrangement in which zero
violations means anything, because a filter tested against its own model is testing arithmetic.

One property makes the pessimistic corner well defined for the two-sided certificate, and it is
worth stating because it is not obvious. Scaling the estimator's series resistance *up* is
conservative for **both** edges at once:

  * upper edge -- more predicted heating and more predicted sag, so `u_hi` moves down;
  * lower edge -- lower predicted terminal voltage, so more current is demanded to serve the
    load and `u_lo` moves up.

The interval therefore shrinks from both sides under the same perturbation, so a single corner
dominates rather than requiring a separate corner per edge. `corner_dominates` in each result
file checks this numerically instead of taking it on faith.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from zeroguard import anchored as A
from zeroguard import platforms as P

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# margin policy, carried from the IECON paper
DV, DT0, DP, DSOC = 0.03, 0.5, 0.006, 0.005
K_TH = 15.0                     # thermal throttle per unit of assumed resistance growth
S_R = 1.8                       # datasheet end-of-life resistance bound


def margins(plat, s_R=S_R, dV=DV, dT0=DT0, dP=DP, dsoc=DSOC, load_frac=0.02, K=K_TH):
    """Margins aligned to a platform's declared limits.

    The thermal margin carries the throttle `dT = dT0 + K (s_R - 1)` unchanged from the IECON
    paper; E8 showed K = 15 is the *smallest* value that certifies cells past the rated bound,
    so it is a calibrated constant rather than a tuning knob.
    """
    dT = dT0 + K * (s_R - 1.0)
    if plat.mode == "charge":
        return (dV, dT, dP)
    return (dT, dV, dsoc, load_frac * max(plat.load_W, 1e-9))


# The envelope the estimator must bound. Every draw in every vehicle experiment comes from
# inside this box, and `pessimistic` sits on its worst corner.
ENVELOPE = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))


def pessimistic(case, s_R=S_R, s_Q=None, s_plate=None, **kw):
    """The platform the filter is given: same class, every uncertain channel at its bound.

    An earlier version of this function scaled only resistance, and V1-2 caught it: the plating
    proxy divides by the plate scale, so a plant drawn at 1.6 has a *lower* margin than an
    estimator at 1.0, and 719 of 4 000 sessions crossed the enforced plating margin while the
    certificate reported everything fine. The estimator has to sit on the worst corner of every
    channel it claims to bound, not just the interesting one.
    """
    sc = dict(kw.pop("scale", None) or {})
    sc["R"] = sc.get("R", 1.0) * s_R
    sc["Q"] = sc.get("Q", 1.0) * (ENVELOPE["Q"][0] if s_Q is None else s_Q)
    sc["plate"] = sc.get("plate", 1.0) * (ENVELOPE["plate"][1] if s_plate is None else s_plate)
    return P.build(case, scale=sc, **kw)


def draw_plant(case, rng, env, **kw):
    """A plant drawn from inside the envelope the estimator bounds."""
    sc = {k: float(rng.uniform(*v)) for k, v in env.items() if k in ("R", "Q", "plate")}
    sc.setdefault("R", 1.0); sc.setdefault("Q", 1.0); sc.setdefault("plate", 1.0)
    return P.build(case, scale=sc, **kw), sc


def safe_init(plat, soc, T, marg=None):
    """An initial state the certificate actually makes a claim about.

    Forward invariance says: from inside the safe set, stay inside it. It says nothing at all
    about a state that begins outside, and a harness that starts there is not testing the
    filter, it is testing arithmetic on a vehicle that has already failed.

    V4-4 started a Mars rover at up to `T_air + 60 K`, which at local noon is +50 C against a
    40 C limit, and duly recorded 76 "violations" in states that were unsafe before the filter
    was called. This clamps the initial temperature to `T_max` less the thermal margin, which
    is the boundary of the set the certificate is about.
    """
    if marg is None:
        marg = margins(plat)
    dT = marg[1] if plat.mode == "charge" else marg[0]
    return plat.init(soc, min(float(T), plat.T_max - dT))


def check(plat, vals):
    """Which of a platform's declared limits the *plant* actually breached this step."""
    bad = []
    for i, (idx, sense, val, side) in enumerate(plat.limits):
        if sense == "<=" and vals[idx] > val + 1e-9:
            bad.append(i)
        elif sense == ">=" and vals[idx] < val - 1e-9:
            bad.append(i)
    return bad


def certified_idx(plat):
    """The limits that are *certified* as against merely *enforced*.

    The IECON paper is explicit that the certified set is `S = {T <= T_max, V <= V_max}` and
    that plating is enforced and monitored but never certified, because `u = 0` does not clear
    the plating margin above ~0.82 SOC and so no one-step condition can make it invariant. That
    distinction is load-bearing and it has to survive into the vehicle results, or a plating
    excursion gets counted as a broken theorem when the theorem never claimed it.
    """
    return tuple(i for i, (idx, _s, _v, _side) in enumerate(plat.limits)
                 if not (plat.mode == "charge" and idx == 2))


def split_breaches(plat, bad):
    """(certified breaches, enforced-channel excursions) out of `check`."""
    cert = set(certified_idx(plat))
    return [i for i in bad if i in cert], [i for i in bad if i not in cert]


# =======================================================================================
# Loop 1 -- a charging session (anchor is zero; this is the IECON case at vehicle scale)
# =======================================================================================
def charge_session(plant, est, s0, dt, w, marg, target_soc=0.80, horizon=None,
                   u_request=None, w_plant=None):
    """Charge until the target SOC or the horizon, filtering every step.

    Returns peak temperature and voltage *in the plant*, the SOC delivered, the number of
    steps, and the indices of any limit the plant actually breached.
    """
    horizon = horizon or est.horizon
    s = dict(s0)
    u_req = u_request if u_request is not None else est.u_max
    wp = w if w_plant is None else w_plant
    peak_T, peak_V, min_phi = -1e9, -1e9, 1e9
    viol, enf, steps, clipped = set(), set(), 0, 0
    for k in range(horizon):
        u, st = A.project_anchored(est, s, u_req, dt, w, marg)
        if st == "infeasible":
            u = 0.0
        elif st != "unclipped":
            clipped += 1
        s, o = plant.step(s, float(max(0.0, u)), dt, wp)
        peak_T = max(peak_T, o[1]); peak_V = max(peak_V, o[0]); min_phi = min(min_phi, o[2])
        c, e = split_breaches(plant, check(plant, o))
        viol.update(c); enf.update(e)
        steps = k + 1
        if s["soc"] >= target_soc:
            break
    return dict(peak_T=float(peak_T), peak_V=float(peak_V), min_phi=float(min_phi),
                soc=float(s["soc"]), steps=steps, clipped=clipped,
                violated=sorted(viol), enforced_excursions=sorted(enf),
                ok=not viol, ok_all=not (viol or enf))


# =======================================================================================
# Loop 2 -- a discharge mission (anchor is the load; this is what the generalisation is for)
# =======================================================================================
def discharge_mission(plant, est, s0, dt, w, marg, horizon=None, load_W=None,
                      demand=None, w_plant=None, fly_past_closure=True):
    """Fly, swim or coast until the interval closes, then keep going to measure the lead time.

    The distinction that matters: **closure is not a violation.** When the interval is empty
    the certificate is saying the load can no longer be served inside the envelope, which is
    the true statement and the one a mission planner needs. The failure mode being tested for
    is a plant constraint breached while the certificate still reports feasible.

    `fly_past_closure` continues the episode after the first closure, commanding the anchor,
    so the gap between the warning and the breach can be measured rather than assumed
    positive. Turning it off is the operationally realistic behaviour (land now) and is used
    where the question is endurance rather than lead time.

    Returns the step of first closure, the step of first breach, the lead time between them,
    and whether any breach happened while the certificate still said "ok".
    """
    horizon = horizon or est.horizon
    s = dict(s0)
    wp = w if w_plant is None else w_plant
    if load_W is not None:
        est.load_W = plant.load_W = float(load_W)
        est.limits = est.limits[:3] + ((3, ">=", float(load_W), "lo"),)
        plant.limits = plant.limits[:3] + ((3, ">=", float(load_W), "lo"),)
    closure = breach = -1
    unsafe_while_certified = False
    widths, peak_T, min_V = [], -1e9, 1e9
    steps = 0
    for k in range(horizon):
        u_lo, u_hi, st = A.interval(est, s, dt, w, marg)
        if st == "ok":
            widths.append(u_hi - u_lo)
            req = demand(k) if demand is not None else u_lo
            u = min(max(req, u_lo), u_hi)
        else:
            widths.append(0.0)
            if closure < 0:
                closure = k
            if not fly_past_closure:
                steps = k
                break
            u = max(u_lo, est.anchor(s))
        s, o = plant.step(s, float(u), dt, wp)
        peak_T = max(peak_T, o[1]); min_V = min(min_V, o[0])
        bad, _enf = split_breaches(plant, check(plant, o))
        if bad and breach < 0:
            breach = k
            if closure < 0:
                unsafe_while_certified = True
        steps = k + 1
        if breach >= 0 and closure >= 0:
            break
    lead = (breach - closure) if (breach >= 0 and closure >= 0) else None
    return dict(closure_step=closure, breach_step=breach, lead_steps=lead,
                widths=widths,
                lead_s=(None if lead is None else lead * dt),
                unsafe_while_certified=bool(unsafe_while_certified),
                endurance_s=(closure if closure >= 0 else steps) * dt,
                steps=steps, peak_T=float(peak_T), min_V=float(min_V),
                soc=float(s["soc"]),
                width0=float(widths[0]) if widths else 0.0,
                width_mean=float(np.mean(widths)) if widths else 0.0)


# =======================================================================================
# Loop 2b -- a phased mission (the load moves; the whole point is that the interval follows)
# =======================================================================================
def phased_mission(plant, est, s0, dt, phases, marg, w, w_plant=None, stop_on_closure=True):
    """Run a mission whose load changes by phase.

    `phases` is a sequence of dicts with `name`, `seconds`, `load_W`, and optionally `v_air`
    and `altitude_m`. This is where a real vehicle's certificate lives: the aircraft climbs at
    1.6x hover power, cruises at 0.7x, hovers over the drop, and descends, and the admissible
    interval opens and closes around each of those without the filter being told a mission plan
    exists.

    Returns the per-phase reserve trace and the phase in which the interval closed, if it did.
    """
    s = dict(s0)
    wp = w if w_plant is None else w_plant
    trace, closed_in, t = [], None, 0.0
    breach_in, unsafe_while_certified = None, False
    for ph in phases:
        est.set_load(ph["load_W"]); plant.set_load(ph["load_W"])
        est.set_airspeed(ph.get("v_air"), ph.get("altitude_m"))
        plant.set_airspeed(ph.get("v_air"), ph.get("altitude_m"))
        widths, n = [], int(round(ph["seconds"] / dt))
        pk_T, mn_V, closed_here = -1e9, 1e9, False
        for _ in range(n):
            u_lo, u_hi, st = A.interval(est, s, dt, w, marg)
            if st == "ok":
                widths.append(u_hi - u_lo)
                u = u_lo
            else:
                widths.append(0.0); closed_here = True
                if closed_in is None:
                    closed_in = ph["name"]
                u = max(u_lo, est.anchor(s))
            s, o = plant.step(s, float(u), dt, wp)
            pk_T = max(pk_T, o[1]); mn_V = min(mn_V, o[0])
            bad, _e = split_breaches(plant, check(plant, o))
            if bad and breach_in is None:
                breach_in = ph["name"]
                if closed_in is None:
                    unsafe_while_certified = True
            t += dt
            if closed_here and stop_on_closure:
                break
        trace.append(dict(phase=ph["name"], load_W=ph["load_W"], steps=len(widths),
                          width_min=float(min(widths)) if widths else 0.0,
                          width_mean=float(np.mean(widths)) if widths else 0.0,
                          peak_T=float(pk_T), min_V=float(mn_V), soc=float(s["soc"]),
                          closed=closed_here))
        if closed_here and stop_on_closure:
            break
    return dict(phases=trace, closed_in=closed_in, breach_in=breach_in,
                unsafe_while_certified=bool(unsafe_while_certified),
                completed=closed_in is None, elapsed_s=t, soc=float(s["soc"]))


# =======================================================================================
# Loop 3 -- what the null-input filter does on the same platform
# =======================================================================================
def null_input_mission(plant, est, s0, dt, w, marg, horizon=None, w_plant=None):
    """The pre-generalisation filter, on a platform whose anchor is not zero.

    It has no floor family, so it certifies `u = 0` and commands as little as the caps allow.
    On a discharge platform that means starving the load. This is not a strawman: it is
    literally `project_current`'s structure applied where its hypothesis (A1) is false, and
    measuring what it does is the only way to show the generalisation is necessary rather than
    ornamental.
    """
    horizon = horizon or est.horizon
    s = dict(s0)
    wp = w if w_plant is None else w_plant
    hi_only = tuple(l for l in est.limits if l[3] == "hi")
    marg_hi = tuple(m for m, l in zip(marg, est.limits) if l[3] == "hi")
    saved_lim, saved_marg = est.limits, marg
    est.limits = hi_only
    est._anchored_split = None
    saved_anchor = est.anchor
    est.anchor = lambda _s: 0.0
    try:
        brownout = -1
        served = 0
        for k in range(horizon):
            u, st = A.project_anchored(est, s, 0.0, dt, w, marg_hi)
            u = max(0.0, u)
            s, o = plant.step(s, float(u), dt, wp)
            if o[3] < plant.load_W - 1e-9:          # bus power below the load: brownout
                if brownout < 0:
                    brownout = k
            else:
                served += 1
            if brownout >= 0:
                break
    finally:
        est.limits, est.anchor = saved_lim, saved_anchor
        est._anchored_split = None
    return dict(brownout_step=brownout, steps_served=served,
                brownout_s=(None if brownout < 0 else brownout * dt))


# =======================================================================================
# Instrumentation
# =======================================================================================
class Counted:
    """Wrap a platform to count one-step model evaluations, for the fixed-cost claim."""

    def __init__(self, plat):
        self._p = plat
        self.n = 0

    def __getattr__(self, k):
        return getattr(self._p, k)

    def probe(self, s, u, dt, w):
        self.n += 1
        return self._p.probe(s, u, dt, w)


def scan_structure(plat, states, dt, w, marg, n=256):
    """Dense feasibility scans over many states; returns the structure census."""
    census = {"single-interval": 0, "disconnected": 0, "empty": 0}
    for s in states:
        _, ok = A.scan(plat, s, dt, w, marg, n=n)
        census[A.structure(ok)] += 1
    return census


def spearman(x, y, reps=20000, min_spread=0.0):
    """(rho, p) for a monotonicity claim across a sweep, with an effect-size floor.

    A rank correlation is scale-free, which is its virtue and its trap: it will happily report
    rho = +1.000, p = 0.0002 for a series whose total spread is 2e-8. V3-5 and V3-6 produced
    exactly that. Delivered charge does not move across a decade of dormancy or 6 000 m of
    depth -- the binding constraint in a sealed hull is the plating cap, which neither of those
    channels touches -- and what the sweep varies is the last few bits of a float.

    `min_spread` is the smallest range in `y` worth calling a relationship. Below it the answer
    is nan: there is nothing there, and "nothing there" is the finding.
    """
    from zeroguard.stats import spearman_perm
    y = np.asarray(y, float)
    if min_spread > 0.0 and float(y.max() - y.min()) <= min_spread:
        return float("nan"), float("nan")
    d = spearman_perm(np.asarray(x, float), y, reps=reps)
    return d["rho"], d["p"]


def save(name, obj):
    import json
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path
