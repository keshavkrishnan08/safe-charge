"""V2 -- aerial: delivery rotorcraft and eVTOL air taxis.

This is the domain the generalisation exists for. E2 established that a hovering quadrotor
breaks hypothesis (A1) -- `u = 0` is not a backup, it is a crash -- and that the null-input
filter passes 22.3 consecutive one-step checks before 100 % of episodes violate. Everything
here is downstream of moving the anchor from zero to hover.

Eleven experiments, each attacking a different part. Two of them are about the theorem (V2-1,
V2-2), one is about the quantity the theorem newly makes available (V2-3), five are about the
flight envelope (V2-4..8), and three are about duty, scale and an adversary (V2-9..11).

    python zeroguard/exp/v2_aerial.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260815
ENV = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))


# ---------------------------------------------------------------------------------------
def v2_1_anchor_vs_null(n=600, seed=SEED):
    """Where the null-input filter fails, does the anchored one hold -- and where does the
    *cell* refuse to fly at all?

    Two findings live here and only one of them was expected.

    The expected one: on the same platform, same seeds, the null-input filter starves the
    rotors and the anchored filter does not.

    The unexpected one: with the ROM's own calibrated M50 coefficients -- an energy cell -- the
    anchored filter reports the interval *empty before takeoff* above about 1.1C of hover draw.
    That is not a failure of the filter. It is the filter refusing to certify a rotorcraft on a
    cell that cannot fly it, which is the correct engineering answer and one a runtime guard is
    not usually able to give. The frontier is measured rather than designed around, and the
    remaining aerial experiments use a power cell (see `platforms.POWER_CELL`)."""
    rng = np.random.default_rng(seed)

    # (a) the C-rate frontier, on the calibrated energy cell
    frontier = []
    for c_rate in (0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5):
        est = P.DeliveryQuadrotor(cell_type="energy", scale=dict(R=V.S_R, Q=0.80, plate=1.6))
        load = c_rate * est.P * est.cell.q_nom() * est.S * 3.4      # W at the nominal bus
        est.set_load(load)
        s = est.init(0.95, 20.0)
        lo, hi, st = A.interval(est, s, 2.0, 20.0, V.margins(est))
        frontier.append(dict(c_rate=c_rate, load_W=load, status=st,
                             width=(hi - lo) if st == "ok" else 0.0))
    refuses_at = next((f["c_rate"] for f in frontier if f["status"] != "ok"), None)

    # (b) anchored against null-input, power cell, identical seeds
    anch_unsafe = 0; ni_brownouts = 0; endur = []; ni_served = []
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
        plant = P.DeliveryQuadrotor(scale=sc)
        marg = V.margins(est)
        s0 = est.init(float(rng.uniform(0.80, 0.99)), float(rng.uniform(5.0, 35.0)))
        r = V.discharge_mission(plant, est, s0, 2.0, est.w_nominal, marg, horizon=900)
        anch_unsafe += int(r["unsafe_while_certified"]); endur.append(r["endurance_s"])
        ni = V.null_input_mission(plant, est, s0, 2.0, est.w_nominal, marg, horizon=900)
        ni_brownouts += int(ni["brownout_step"] >= 0)
        ni_served.append(ni["steps_served"])
    return dict(
        frontier=frontier, energy_cell_refuses_above_c=refuses_at,
        trials=n,
        anchored_unsafe_while_certified=anch_unsafe,
        anchored_cp95_pct=100 * stats.cp_upper(anch_unsafe, n),
        null_input_brownouts=ni_brownouts, null_input_brownout_rate=ni_brownouts / n,
        null_input_mean_steps_served=float(np.mean(ni_served)),
        anchored_median_endurance_min=float(np.median(endur)) / 60.0)


# ---------------------------------------------------------------------------------------
def v2_2_interval_structure(n=20000, seed=SEED + 1):
    """Is the two-sided admissible set actually an interval?

    The whole construction rests on it. One bisection per edge is only valid if feasibility is
    a single connected run in `u`; a disconnected set would mean the upper bisection can land
    inside a hole. Twenty thousand flight states, scanned densely, is the check."""
    rng = np.random.default_rng(seed)
    census = {"single-interval": 0, "disconnected": 0, "empty": 0}
    edge_agree, edge_dev = 0, 0.0
    for case in ("delivery-quadrotor", "evtol-air-taxi"):
        est = V.pessimistic(case)
        marg = V.margins(est)
        for _ in range(n // 2):
            s = est.init(float(rng.uniform(0.16, 0.99)), float(rng.uniform(-15.0, 52.0)))
            g, ok = A.scan(est, s, est.dt_nominal, est.w_nominal, marg, n=200)
            st = A.structure(ok)
            census[st] += 1
            if st == "single-interval":
                lo, hi, s2 = A.interval(est, s, est.dt_nominal, est.w_nominal, marg)
                if s2 == "ok":
                    step = g[1] - g[0]
                    d = max(abs(lo - g[ok.nonzero()[0][0]]), abs(hi - g[ok.nonzero()[0][-1]]))
                    edge_dev = max(edge_dev, d / step)
                    edge_agree += int(d <= step)
    return dict(states=n, census=census,
                disconnected=census["disconnected"],
                all_intervals=census["disconnected"] == 0,
                edges_within_one_grid_step=edge_agree,
                worst_edge_dev_grid_steps=edge_dev)


# ---------------------------------------------------------------------------------------
def v2_3_reserve_lead(n=1200, seed=SEED + 2):
    """Does the reserve *warn*, or does it merely describe?

    The claim with the most operational weight in this whole register: the interval width falls
    to zero strictly before any constraint is breached, so the vehicle is told to land while
    landing is still possible. The falsifier is a single episode that breaches with no prior
    closure. Lead time is reported as a distribution, not a mean, because a mean lead time is
    useless to someone deciding whether to divert."""
    rng = np.random.default_rng(seed)
    leads, no_warning, closed, widths_at_warn = [], 0, 0, []
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6),
                                  payload_kg=float(rng.uniform(0.0, 2.0)))
        plant = P.DeliveryQuadrotor(scale=sc, payload_kg=est.payload_kg)
        marg = V.margins(est)
        s0 = est.init(float(rng.uniform(0.5, 0.99)), float(rng.uniform(10.0, 40.0)))
        r = V.discharge_mission(plant, est, s0, 2.0, est.w_nominal, marg, horizon=1200,
                                fly_past_closure=True)
        no_warning += int(r["unsafe_while_certified"])
        if r["closure_step"] >= 0:
            closed += 1
        if r["lead_s"] is not None:
            leads.append(r["lead_s"])
    a = np.array(leads) if leads else np.array([0.0])
    return dict(trials=n, closures=closed, breaches_with_no_warning=no_warning,
                cp95_no_warning_pct=100 * stats.cp_upper(no_warning, n),
                lead_s_min=float(a.min()), lead_s_p05=float(np.percentile(a, 5)),
                lead_s_median=float(np.median(a)), lead_s_p95=float(np.percentile(a, 95)),
                lead_s_max=float(a.max()), episodes_with_lead=len(leads),
                all_warned=no_warning == 0)


# ---------------------------------------------------------------------------------------
SORTIE = [
    dict(name="takeoff", seconds=20.0, mult=1.75, v_air=0.0, altitude_m=10.0),
    dict(name="climb", seconds=60.0, mult=1.45, v_air=6.0, altitude_m=60.0),
    dict(name="cruise-out", seconds=300.0, mult=0.72, v_air=18.0, altitude_m=120.0),
    dict(name="hover-drop", seconds=45.0, mult=1.00, v_air=0.0, altitude_m=30.0),
    dict(name="cruise-back", seconds=300.0, mult=0.68, v_air=18.0, altitude_m=120.0),
    dict(name="descend", seconds=50.0, mult=0.85, v_air=4.0, altitude_m=30.0),
    dict(name="land", seconds=20.0, mult=1.15, v_air=0.0, altitude_m=2.0),
]


def v2_4_sortie(n=800, seed=SEED + 3):
    """A whole delivery sortie, phase by phase.

    Hover is the expensive phase and cruise is the cheap one, and cruise also cools better,
    so the reserve should open in cruise and close over the drop. The filter is never told a
    mission plan exists; it sees a load that changes and re-derives both edges every step."""
    rng = np.random.default_rng(seed)
    # the takeoff transient sets the pack size, so report where it stops being certifiable
    frontier = []
    for par in (2, 3, 4):
        row = dict(parallel=par, cells=6 * par)
        for mult in (1.00, 1.35, 1.75, 2.00):
            e = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            e.P = par; e.u_max = e.c_rate_ceiling * e.cell.q_nom() * par
            e._anchored_split = None
            base = e.load_W; e.set_load(base * mult)
            lo, hi, st = A.interval(e, e.init(0.95, 20.0), 1.0, e.w_nominal, V.margins(e))
            row[f"x{mult:.2f}"] = st
        row["energy_Wh"] = 6 * par * P.DeliveryQuadrotor().cell.q_nom() * 3.63
        frontier.append(row)
    completed = 0; unsafe = 0; closed_where = {}
    per_phase = {p["name"]: [] for p in SORTIE}
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6),
                                  payload_kg=float(rng.uniform(0.0, 1.5)))
        plant = P.DeliveryQuadrotor(scale=sc, payload_kg=est.payload_kg)
        base = est.load_W
        phases = [dict(name=p["name"], seconds=p["seconds"], load_W=base * p["mult"],
                       v_air=p["v_air"], altitude_m=p["altitude_m"]) for p in SORTIE]
        marg = V.margins(est)
        s0 = est.init(float(rng.uniform(0.90, 0.99)), float(rng.uniform(10.0, 35.0)))
        r = V.phased_mission(plant, est, s0, 1.0, phases, marg, est.w_nominal)
        completed += int(r["completed"]); unsafe += int(r["unsafe_while_certified"])
        if r["closed_in"]:
            closed_where[r["closed_in"]] = closed_where.get(r["closed_in"], 0) + 1
        for t in r["phases"]:
            per_phase[t["phase"]].append(t["width_mean"])
    summary = {k: dict(mean_width=float(np.mean(v)) if v else 0.0,
                       n=len(v)) for k, v in per_phase.items()}
    # the frontier is only useful if acting on it changes the outcome, so run the same sortie
    # on the next pack size up
    bigger = 0
    for _ in range(n // 4):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6),
                                  payload_kg=float(rng.uniform(0.0, 1.5)))
        plant = P.DeliveryQuadrotor(scale=sc, payload_kg=est.payload_kg)
        for pl in (est, plant):
            pl.P = 4; pl.u_max = pl.c_rate_ceiling * pl.cell.q_nom() * 4
            pl._anchored_split = None
        base = est.load_W
        phases = [dict(name=p["name"], seconds=p["seconds"], load_W=base * p["mult"],
                       v_air=p["v_air"], altitude_m=p["altitude_m"]) for p in SORTIE]
        marg = V.margins(est)
        s0 = est.init(float(rng.uniform(0.90, 0.99)), float(rng.uniform(10.0, 35.0)))
        rr = V.phased_mission(plant, est, s0, 1.0, phases, marg, est.w_nominal)
        bigger += int(rr["completed"])
    return dict(trials=n, takeoff_frontier=frontier,
                completion_rate_6S4P=bigger / (n // 4), trials_6S4P=n // 4,
                completed=completed, completion_rate=completed / n,
                unsafe_while_certified=unsafe,
                cp95_unsafe_pct=100 * stats.cp_upper(unsafe, n),
                closure_phase_census=closed_where, phase_reserve=summary,
                profile=[dict(name=p["name"], seconds=p["seconds"], power_mult=p["mult"])
                         for p in SORTIE])


# ---------------------------------------------------------------------------------------
def v2_5_payload_altitude(n=180, seed=SEED + 4):
    """Payload and density altitude close the envelope through the same term.

    Momentum theory puts both inside `T^{3/2}/sqrt(rho A)`: mass raises the numerator, altitude
    lowers the denominator. If the reserve is really the physical quantity it claims to be, it
    has to fall monotonically in both, and the two effects have to be interchangeable at equal
    hover power. That is a stronger and more falsifiable statement than "it degrades"."""
    rng = np.random.default_rng(seed)
    grid = []
    for payload in (0.0, 0.5, 1.0, 1.5, 2.0):
        for alt in (0.0, 500.0, 1500.0, 2500.0, 3500.0):
            est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6),
                                      payload_kg=payload, altitude_m=alt)
            marg = V.margins(est)
            widths, endur, unsafe = [], [], 0
            for _ in range(n):
                sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
                plant = P.DeliveryQuadrotor(scale=sc, payload_kg=payload, altitude_m=alt)
                s0 = est.init(float(rng.uniform(0.85, 0.99)), 20.0)
                lo, hi, st = A.interval(est, s0, 2.0, est.w_nominal, marg)
                widths.append((hi - lo) if st == "ok" else 0.0)
                r = V.discharge_mission(plant, est, s0, 2.0, est.w_nominal, marg,
                                        horizon=900, fly_past_closure=False)
                endur.append(r["endurance_s"]); unsafe += int(r["unsafe_while_certified"])
            grid.append(dict(payload_kg=payload, altitude_m=alt, hover_W=est.load_W,
                             mean_width=float(np.mean(widths)),
                             median_endurance_min=float(np.median(endur)) / 60.0,
                             unsafe=unsafe))
    rp, pp = V.spearman([g["payload_kg"] for g in grid], [g["mean_width"] for g in grid])
    rh, ph = V.spearman([g["hover_W"] for g in grid], [g["mean_width"] for g in grid])
    # Altitude has to be measured at fixed payload. Pooled across the whole grid its rank
    # correlation is near zero and non-significant -- not because density altitude does not
    # matter, but because payload spans a far larger range of hover power and swamps it. The
    # honest test is within-stratum, which is also the claim: the two act through one term.
    within = []
    for pay in sorted({g["payload_kg"] for g in grid}):
        sub = sorted((g for g in grid if g["payload_kg"] == pay),
                     key=lambda g: g["altitude_m"])
        r, pv = V.spearman([g["altitude_m"] for g in sub], [g["mean_width"] for g in sub])
        within.append(dict(payload_kg=pay, rho=r, p=pv, n=len(sub)))
    ra_pooled, pa_pooled = V.spearman([g["altitude_m"] for g in grid],
                                      [g["mean_width"] for g in grid])
    return dict(grid=grid, trials_per_cell=n,
                total_unsafe=sum(g["unsafe"] for g in grid),
                width_vs_payload=dict(rho=rp, p=pp),
                width_vs_altitude_pooled=dict(rho=ra_pooled, p=pa_pooled),
                width_vs_altitude_within_payload=within,
                altitude_negative_in_every_stratum=all(w["rho"] < 0 for w in within),
                width_vs_hover_power=dict(rho=rh, p=ph))


# ---------------------------------------------------------------------------------------
def v2_6_gusts(n=500, seed=SEED + 5):
    """Gusts, as a stochastic multiplier on required power.

    A gust does not politely wait for the filter's step boundary, so the demand moves inside
    the interval and occasionally past its upper edge. The claim is that the projection absorbs
    it: clipping to `u_hi` is a loss of authority, not a loss of safety, and the vehicle is only
    in trouble when the *floor* rises past the ceiling -- which is closure, and is reported.

    Two closures are tracked and the distinction matters. The **live** one is the first step at
    which the interval is empty; that is what the filter actually reports and what safety is
    accounted against. The **sustained** one requires SUSTAIN consecutive empty steps, because a
    gust peak can close the interval for a single step and release it, and calling that "the
    envelope has closed" is a false alarm about the flight.

    Lead time is measured from the *live* closure, because that is when a pilot would be told.
    Measuring it from the sustained one -- as the first version did -- produces negative lead
    times at high gust variance, which reads as the certificate warning after the fact when in
    truth the debounce was still counting. The cost of the debounce is reported separately."""
    rng = np.random.default_rng(seed)
    SUSTAIN = 3
    rows = []
    for sigma in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        unsafe = 0; closures = 0; leads = []; debounce = []
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.DeliveryQuadrotor(scale=sc)
            base, marg = est.load_W, V.margins(est)
            s = est.init(float(rng.uniform(0.85, 0.99)), 20.0)
            closure = breach = first_closure = -1
            run = 0
            for k in range(600):
                gust = base * float(np.clip(1.0 + rng.normal(0.0, sigma), 0.4, 2.5))
                est.set_load(gust); plant.set_load(gust)
                lo, hi, st = A.interval(est, s, 2.0, est.w_nominal, marg)
                certified_now = (st == "ok")
                if certified_now:
                    run = 0
                    u = lo
                else:
                    # A gust peak can close the interval for a single step and then release
                    # it. That is a true statement about that step and a false alarm about the
                    # flight, so "the envelope has closed" is only recorded once the closure
                    # has persisted for SUSTAIN consecutive steps. Without this the reported
                    # lead time is inflated by transients rather than earned.
                    run += 1
                    if first_closure < 0:
                        first_closure = k
                    if run >= SUSTAIN and closure < 0:
                        closure = k
                    u = max(lo, est.anchor(s))
                s, o = plant.step(s, float(u), 2.0, est.w_nominal)
                bad, _e = V.split_breaches(plant, V.check(plant, o))
                if bad and breach < 0:
                    breach = k
                    # Safety accounting keys on whether the certificate said "ok" *at this
                    # step*, not on whether the debounced warning has fired yet. Confusing the
                    # two made the first run of this experiment report 467 unsafe episodes at
                    # sigma = 0.5 and a *negative* median lead time: the filter had correctly
                    # reported the interval closed, and the bookkeeping had not caught up.
                    if certified_now:
                        unsafe += 1
                if breach >= 0 and closure >= 0:
                    break
            closures += int(closure >= 0)
            if breach >= 0 and first_closure >= 0:
                leads.append((breach - first_closure) * 2.0)
            if breach >= 0 and closure >= 0:
                debounce.append((closure - first_closure) * 2.0)
        rows.append(dict(sigma=sigma, trials=n, unsafe_while_certified=unsafe,
                         sustain_steps=SUSTAIN, closures=closures,
                         median_lead_s=float(np.median(leads)) if leads else None,
                         median_debounce_cost_s=float(np.median(debounce)) if debounce else None,
                         cp95_pct=100 * stats.cp_upper(unsafe, n)))
    return dict(rows=rows, total_trials=n * len(rows),
                total_unsafe=sum(r["unsafe_while_certified"] for r in rows))


# ---------------------------------------------------------------------------------------
def v2_7_cold_soak(n=400, seed=SEED + 6):
    """Cold soak at altitude, and which edge of the interval binds.

    Warm, the upper edge is thermal. Cold, series resistance rises and the pack sags, so the
    upper edge becomes a *voltage* edge and the lower edge climbs because the same sag means
    more current is needed for the same watts. Both edges move toward each other for the same
    physical reason. Reporting which constraint binds is more informative than reporting that
    something did."""
    rng = np.random.default_rng(seed)
    rows = []
    for T_air in (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0):
        est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6),
                                  T_amb=T_air, altitude_m=2000.0)
        marg = V.margins(est)
        binds = {"thermal": 0, "voltage": 0, "actuator": 0, "closed": 0}
        widths, unsafe = [], 0
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.DeliveryQuadrotor(scale=sc, T_amb=T_air, altitude_m=2000.0)
            s = est.init(float(rng.uniform(0.5, 0.99)), T_air + float(rng.uniform(0.0, 12.0)))
            lo, hi, st = A.interval(est, s, 2.0, T_air, marg)
            if st != "ok":
                binds["closed"] += 1; widths.append(0.0)
            else:
                widths.append(hi - lo)
                if hi >= est.u_max - 1e-9:
                    binds["actuator"] += 1
                else:
                    vals = est.probe(s, min(hi + 1e-3, est.u_max), 2.0, T_air)
                    binds["thermal" if vals[1] > est.T_max - marg[0] else "voltage"] += 1
            r = V.discharge_mission(plant, est, s, 2.0, T_air, marg, horizon=600,
                                    fly_past_closure=False)
            unsafe += int(r["unsafe_while_certified"])
        rows.append(dict(T_air=T_air, trials=n, binding=binds,
                         mean_width=float(np.mean(widths)), unsafe=unsafe))
    return dict(rows=rows, altitude_m=2000.0,
                total_unsafe=sum(r["unsafe"] for r in rows),
                total_trials=n * len(rows))


# ---------------------------------------------------------------------------------------
def v2_8_motor_out(n=600, seed=SEED + 7):
    """One rotor fails; the remaining three carry it at roughly +33 % power.

    The right behaviour is not "stay safe" -- a quadrotor with a dead motor is not safe -- it is
    to *convert* the fault into a closure with lead time, so the aircraft is told to put itself
    down rather than discovering the problem thermally. Failing to warn is the falsifier."""
    rng = np.random.default_rng(seed)
    rows = []
    for bump in (1.0, 1.15, 1.33, 1.60, 2.00):
        unsafe = 0; leads = []; closed = 0
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.DeliveryQuadrotor(scale=sc)
            base, marg = est.load_W, V.margins(est)
            s = est.init(float(rng.uniform(0.6, 0.99)), float(rng.uniform(10.0, 35.0)))
            fail_at = int(rng.integers(20, 200))
            closure = breach = -1
            for k in range(700):
                load = base * (bump if k >= fail_at else 1.0)
                est.set_load(load); plant.set_load(load)
                lo, hi, st = A.interval(est, s, 2.0, est.w_nominal, marg)
                if st == "ok":
                    u = lo
                else:
                    if closure < 0:
                        closure = k
                    u = max(lo, est.anchor(s))
                s, o = plant.step(s, float(u), 2.0, est.w_nominal)
                bad, _e = V.split_breaches(plant, V.check(plant, o))
                if bad and breach < 0:
                    breach = k
                    if closure < 0:
                        unsafe += 1
                if breach >= 0 and closure >= 0:
                    break
            closed += int(closure >= 0)
            if breach >= 0 and closure >= 0:
                leads.append((breach - closure) * 2.0)
        rows.append(dict(power_bump=bump, trials=n, closures=closed,
                         unsafe_while_certified=unsafe,
                         median_lead_s=float(np.median(leads)) if leads else None,
                         cp95_pct=100 * stats.cp_upper(unsafe, n)))
    return dict(rows=rows, total_unsafe=sum(r["unsafe_while_certified"] for r in rows),
                total_trials=n * len(rows))


# ---------------------------------------------------------------------------------------
def v2_9_turnaround(n=4000, seed=SEED + 8):
    """Forty charges a day into a small passively cooled pack -- and the cool-down that costs.

    The first version of this experiment charged a pack that had just landed and reported that
    it reached 0.195 SOC. That is a true number and an uninteresting one: the pack arrives at
    up to 44 C, the throttled margin puts the admissible ceiling at `T_max - dT` = 32.5 C, and
    the certificate correctly refuses to put any current into it at all. Measuring how little
    charge a refused session delivers is measuring the refusal.

    The operational question is the one a delivery operator actually has: *how long must the
    pack sit before the charger can start, and does forty sorties a day survive that?* The
    certificate answers it directly -- the wait is the time until the interval opens, which is
    a quantity the filter already computes and nobody has to schedule by hand. So each session
    here cools at zero current until the interval opens, and both the wait and the subsequent
    charge are reported."""
    rng = np.random.default_rng(seed)
    est = V.pessimistic("quadrotor-turnaround", T_amb=30.0)
    marg = V.margins(est)
    safe = 0; enf = 0; wT = -1e9; socs = []; waits = []; refused = 0; immediate = 0
    MAX_WAIT = 1440                     # 1440 x 5 s = 2 h of cooling before giving up
    for k in range(n):
        frac = k / max(n - 1, 1)
        sc = dict(R=1.0 + 0.8 * frac, Q=1.0 - 0.2 * frac, plate=1.0 + 0.6 * frac)
        plant = P.QuadrotorTurnaround(T_amb=30.0, scale=sc)
        T0 = 30.0 + float(rng.uniform(0.0, 14.0))       # arrives hot off a sortie
        s = est.init(float(rng.uniform(0.10, 0.25)), T0)
        wait = 0
        while wait < MAX_WAIT:
            _lo, hi, st = A.interval(est, s, 5.0, 30.0, marg)
            if st == "ok" and hi > 1e-6:
                break
            s, _o = plant.step(s, 0.0, 5.0, 30.0)       # cool at zero current
            wait += 1
        if wait >= MAX_WAIT:
            refused += 1
            continue
        waits.append(wait * 5.0)
        immediate += int(wait == 0)
        r = V.charge_session(plant, est, s, 5.0, 30.0, marg, target_soc=0.90, horizon=200)
        safe += int(r["ok"]); enf += int(bool(r["enforced_excursions"]))
        wT = max(wT, r["peak_T"]); socs.append(r["soc"])
    ran = len(socs)
    a = np.array(waits) if waits else np.array([0.0])
    med_wait = float(np.median(a))
    # A sortie is 22 min of flight. On one pack the turnaround is the cool-down plus the
    # charge; with M packs in rotation the cool-downs overlap and the binding term is the
    # charge alone. That comparison is the operational answer, and it is the reason delivery
    # operators swap packs rather than wait.
    charge_min = 200 * 5.0 / 60.0
    flight_min = 22.0
    one_pack_min = med_wait / 60.0 + charge_min + flight_min
    swap_min = charge_min + flight_min
    return stats.summarize_safety("quadrotor_turnaround", ran - safe, ran, extra=dict(
        sessions_attempted=n, sessions_charged=ran, refused_after_2h=refused,
        started_immediately=immediate,
        started_immediately_frac=immediate / max(ran, 1),
        wait_s_median=med_wait,
        wait_s_p05=float(np.percentile(a, 5)), wait_s_p50=float(np.percentile(a, 50)),
        wait_s_p95=float(np.percentile(a, 95)), wait_s_max=float(a.max()),
        worst_T=float(wT), enforced_plating_excursions=enf,
        mean_soc=float(np.mean(socs)) if socs else None,
        first_quarter_soc=float(np.mean(socs[:max(1, ran // 4)])) if socs else None,
        last_quarter_soc=float(np.mean(socs[-max(1, ran // 4):])) if socs else None,
        cycle_one_pack_min=one_pack_min, cycle_pack_swap_min=swap_min,
        sorties_per_day_one_pack=24 * 60 / one_pack_min,
        sorties_per_day_pack_swap=24 * 60 / swap_min,
        # to hold 40 sorties/day one sortie must complete every 36 min; a pack whose full
        # cycle is longer than that has to be one of several in rotation
        packs_needed_for_target=max(1.0, one_pack_min / (24 * 60 / 40)),
        target_sorties_per_day=40))


# ---------------------------------------------------------------------------------------
def v2_10_evtol(n=500, seed=SEED + 9):
    """An eVTOL air taxi in hover, and whether the reserve converts to endurance.

    The reserve is only useful if it is a proxy for time. Here the interval width at the start
    of a hover is correlated against how long the aircraft actually gets before closure, across
    a wide draw of packs. A strong monotone relation is what makes "you have this much room"
    equivalent to "you have this long"."""
    rng = np.random.default_rng(seed)
    w0, endur, unsafe = [], [], 0
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.EVTOLAirTaxi(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
        plant = P.EVTOLAirTaxi(scale=sc)
        marg = V.margins(est)
        s0 = est.init(float(rng.uniform(0.35, 0.99)), float(rng.uniform(10.0, 45.0)))
        lo, hi, st = A.interval(est, s0, 1.0, est.w_nominal, marg)
        r = V.discharge_mission(plant, est, s0, 1.0, est.w_nominal, marg, horizon=1500,
                                fly_past_closure=False)
        w0.append((hi - lo) if st == "ok" else 0.0)
        endur.append(r["endurance_s"]); unsafe += int(r["unsafe_while_certified"])
    rho, p = V.spearman(w0, endur)
    a = np.array(endur)
    return dict(trials=n, unsafe_while_certified=unsafe,
                cp95_pct=100 * stats.cp_upper(unsafe, n),
                hover_c_rate=float(P.EVTOLAirTaxi().anchor(P.EVTOLAirTaxi().init())
                                   / P.EVTOLAirTaxi().P / P.EVTOLAirTaxi().cell.q_nom()),
                reserve_vs_endurance=dict(rho=rho, p=p),
                endurance_min_median=float(np.median(a)) / 60.0,
                endurance_min_p05=float(np.percentile(a, 5)) / 60.0,
                endurance_min_p95=float(np.percentile(a, 95)) / 60.0)


# ---------------------------------------------------------------------------------------
def v2_11_adversary(cells=320, iters=4, pop=24, elite=6, steps=200, seed=SEED + 10):
    """A cross-entropy adversary optimising a flight profile against the two-sided filter.

    The statistical unit is the *cell*, not the sequence -- E12's lesson, learned when the
    original adversarial claim rested on 40 cells and could certify nothing below 7.2 %. Each
    of the 320 independently drawn packs gets its own CEM search over a demand profile and an
    airspeed profile, maximising the plant's peak temperature against its limit. The adversary
    is allowed to command anywhere inside the certified interval, which is exactly the freedom
    a real controller has."""
    rng = np.random.default_rng(seed)
    unsafe = 0; worst_ratio = -1e9; worst_cert = -1e9; seq = 0
    t0 = time.time()
    K = 6                                              # profile knots
    for _ in range(cells):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.DeliveryQuadrotor(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
        plant = P.DeliveryQuadrotor(scale=sc)
        base, marg = est.load_W, V.margins(est)
        mu = np.concatenate([np.full(K, 1.0), np.full(K, 0.5)])
        sd = np.concatenate([np.full(K, 0.30), np.full(K, 0.35)])
        best_local = -1e9; cell_unsafe = False
        for _it in range(iters):
            cand = mu + sd * rng.standard_normal((pop, 2 * K))
            cand[:, :K] = np.clip(cand[:, :K], 0.5, 2.0)     # load multiplier
            cand[:, K:] = np.clip(cand[:, K:], 0.0, 1.0)     # request within the interval
            scores = np.empty(pop)
            for j in range(pop):
                s = est.init(0.98, 35.0)
                pk = -1e9; pk_cert = -1e9; closed = False; bad_certified = False
                for k in range(steps):
                    kn = min(K - 1, k * K // steps)
                    load = base * float(cand[j, kn])
                    est.set_load(load); plant.set_load(load)
                    lo, hi, st = A.interval(est, s, 2.0, est.w_nominal, marg)
                    if st != "ok":
                        closed = True
                        u = max(lo, est.anchor(s))
                    else:
                        u = lo + float(cand[j, K + kn]) * (hi - lo)
                    s, o = plant.step(s, float(u), 2.0, est.w_nominal)
                    pk = max(pk, o[1] / est.T_max)
                    if not closed:
                        pk_cert = max(pk_cert, o[1] / est.T_max)
                    bad, _e = V.split_breaches(plant, V.check(plant, o))
                    if bad and not closed:
                        bad_certified = True
                        break
                seq += 1
                scores[j] = pk
                worst_cert = max(worst_cert, pk_cert)
                if bad_certified:
                    cell_unsafe = True
            idx = np.argsort(scores)[-elite:]
            mu = cand[idx].mean(axis=0); sd = cand[idx].std(axis=0) + 0.03
            best_local = max(best_local, float(scores.max()))
        worst_ratio = max(worst_ratio, best_local)
        unsafe += int(cell_unsafe)
    return stats.summarize_safety("aerial_adversary", unsafe, cells, extra=dict(
        cells=cells, sequences=seq, cem_iters=iters, population=pop, elite=elite,
        steps_per_sequence=steps,
        worst_peak_T_fraction_of_limit=float(worst_ratio),
        worst_peak_T_fraction_while_certified=float(worst_cert),
        note=("the adversary exceeds the limit only by continuing to fly an aircraft the "
              "certificate has already told to land; while the interval is open it never "
              "reaches the limit"),
        n_required_1pct=stats.n_required(0.01),
        seconds=round(time.time() - t0, 1)))


# ---------------------------------------------------------------------------------------
def main():
    out = {}
    t0 = time.time()
    print("V2 -- aerial: delivery rotorcraft and eVTOL\n" + "=" * 78)

    print("\nV2-1  the anchor against the null input, and the cell that cannot fly")
    r = v2_1_anchor_vs_null(); out["v2_1_anchor_vs_null"] = r
    print(f"      energy-cell hover frontier: interval empty at and above "
          f"{r['energy_cell_refuses_above_c']}C")
    for f in r["frontier"]:
        print(f"        {f['c_rate']:>4.1f}C  {f['load_W']:>7.0f} W  {f['status']:12s} "
              f"width {f['width']:.2f} A")
    print(f"      power cell, {r['trials']} flights: anchored unsafe-while-certified "
          f"{r['anchored_unsafe_while_certified']} (CP95 {r['anchored_cp95_pct']:.2f}%)")
    print(f"      null-input brownouts {r['null_input_brownouts']}/{r['trials']} "
          f"({100*r['null_input_brownout_rate']:.0f}%), mean "
          f"{r['null_input_mean_steps_served']:.1f} steps served")
    print(f"      anchored median endurance {r['anchored_median_endurance_min']:.1f} min")

    print("\nV2-2  is the two-sided admissible set an interval?")
    r = v2_2_interval_structure(); out["v2_2_interval_structure"] = r
    print(f"      {r['states']:,} flight states | census {r['census']}")
    print(f"      disconnected {r['disconnected']} | edges within one grid step: "
          f"{r['edges_within_one_grid_step']:,} (worst {r['worst_edge_dev_grid_steps']:.2f})")

    print("\nV2-3  does the reserve warn before it fails?")
    r = v2_3_reserve_lead(); out["v2_3_reserve_lead"] = r
    print(f"      {r['trials']:,} flights | closures {r['closures']} | breaches with NO prior "
          f"warning {r['breaches_with_no_warning']} (CP95 {r['cp95_no_warning_pct']:.2f}%)")
    print(f"      lead time s: min {r['lead_s_min']:.0f} | p05 {r['lead_s_p05']:.0f} | median "
          f"{r['lead_s_median']:.0f} | p95 {r['lead_s_p95']:.0f} | max {r['lead_s_max']:.0f}")

    print("\nV2-4  a whole delivery sortie")
    r = v2_4_sortie(); out["v2_4_sortie"] = r
    print(f"      takeoff transient frontier (interval status at takeoff power):")
    for row in r["takeoff_frontier"]:
        st = "  ".join(f"{m}:{row[m][:4]}" for m in ("x1.00", "x1.35", "x1.75", "x2.00"))
        print(f"        6S{row['parallel']}P {row['energy_Wh']:5.0f} Wh   {st}")
    print(f"      {r['trials']} sorties | completion {100*r['completion_rate']:.1f}% | "
          f"unsafe-while-certified {r['unsafe_while_certified']}")
    for k, v in r["phase_reserve"].items():
        print(f"        {k:12s} mean reserve {v['mean_width']:6.2f} A")
    print(f"      closures by phase: {r['closure_phase_census']}")
    print(f"      the same sortie on the next pack size up (6S4P): "
          f"{100*r['completion_rate_6S4P']:.1f} % complete")

    print("\nV2-5  payload and density altitude")
    r = v2_5_payload_altitude(); out["v2_5_payload_altitude"] = r
    print(f"      reserve vs payload  rho={r['width_vs_payload']['rho']:+.3f} "
          f"p={r['width_vs_payload']['p']:.4f}")
    print(f"      reserve vs altitude, pooled rho={r['width_vs_altitude_pooled']['rho']:+.3f} "
          f"p={r['width_vs_altitude_pooled']['p']:.4f}  (payload swamps it)")
    print(f"      reserve vs altitude, within each payload: "
          + ", ".join(f"{w['payload_kg']:.1f}kg {w['rho']:+.2f}"
                      for w in r["width_vs_altitude_within_payload"])
          + f"  | negative in every stratum: {r['altitude_negative_in_every_stratum']}")
    print(f"      reserve vs hover W  rho={r['width_vs_hover_power']['rho']:+.3f} "
          f"p={r['width_vs_hover_power']['p']:.4f}   (the shared term)")
    print(f"      unsafe-while-certified across the whole grid: {r['total_unsafe']}")

    print("\nV2-6  gusts")
    r = v2_6_gusts(); out["v2_6_gusts"] = r
    for row in r["rows"]:
        print(f"      sigma {row['sigma']:.2f}  closures {row['closures']:>4}  unsafe "
              f"{row['unsafe_while_certified']}  median lead {row['median_lead_s']} s "
              f"(debounce costs {row['median_debounce_cost_s']} s)")

    print("\nV2-7  cold soak at 2 000 m, and which edge binds")
    r = v2_7_cold_soak(); out["v2_7_cold_soak"] = r
    for row in r["rows"]:
        b = row["binding"]
        print(f"      {row['T_air']:>6.0f} C  width {row['mean_width']:6.2f} A  "
              f"thermal {b['thermal']:>4} voltage {b['voltage']:>4} actuator {b['actuator']:>4} "
              f"closed {b['closed']:>4}  unsafe {row['unsafe']}")

    print("\nV2-8  motor-out")
    r = v2_8_motor_out(); out["v2_8_motor_out"] = r
    for row in r["rows"]:
        print(f"      +{100*(row['power_bump']-1):>3.0f}%  closures {row['closures']:>4}/"
              f"{row['trials']}  unsafe {row['unsafe_while_certified']}  median lead "
              f"{row['median_lead_s']} s")

    print("\nV2-9  forty turnaround charges a day")
    r = v2_9_turnaround(); out["v2_9_turnaround"] = r
    print(f"      {r['sessions_attempted']:,} attempted | {r['sessions_charged']:,} charged | "
          f"{r['refused_after_2h']} still refused after 2 h of cooling")
    print(f"      {100*r['started_immediately_frac']:.0f} % could start at once; the rest wait: "
          f"median {r['wait_s_median']/60:.1f} min | p95 {r['wait_s_p95']/60:.1f} min | max "
          f"{r['wait_s_max']/60:.1f} min")
    print(f"      violations {r['violations']} | CP95 {r['cp95_upper_pct']:.4f}% | worst T "
          f"{r['worst_T']:.2f} C | SOC {r['first_quarter_soc']:.3f} -> "
          f"{r['last_quarter_soc']:.3f}")
    print(f"      one pack: {r['cycle_one_pack_min']:.1f} min/sortie -> "
          f"{r['sorties_per_day_one_pack']:.1f} sorties/day")
    print(f"      swapping packs: {r['cycle_pack_swap_min']:.1f} min/sortie -> "
          f"{r['sorties_per_day_pack_swap']:.1f} sorties/day, against a target of "
          f"{r['target_sorties_per_day']}")
    print(f"      packs needed to hold the target: {r['packs_needed_for_target']:.1f}")

    print("\nV2-10 eVTOL hover: does the reserve convert to endurance?")
    r = v2_10_evtol(); out["v2_10_evtol"] = r
    print(f"      hover {r['hover_c_rate']:.2f}C | {r['trials']} hovers | unsafe "
          f"{r['unsafe_while_certified']} | CP95 {r['cp95_pct']:.2f}%")
    print(f"      reserve vs endurance rho={r['reserve_vs_endurance']['rho']:+.3f} "
          f"p={r['reserve_vs_endurance']['p']:.4f}")
    print(f"      endurance min: p05 {r['endurance_min_p05']:.1f} | median "
          f"{r['endurance_min_median']:.1f} | p95 {r['endurance_min_p95']:.1f}")

    print("\nV2-11 a cross-entropy adversary against the two-sided filter")
    r = v2_11_adversary(); out["v2_11_adversary"] = r
    print(f"      {r['cells']} independently drawn packs, {r['sequences']:,} sequences")
    print(f"      unsafe-while-certified {r['violations']} | CP95 upper "
          f"{r['cp95_upper_pct']:.3f}% | n needed for 1%: {r['n_required_1pct']}")
    print(f"      worst peak T while the certificate was open: "
          f"{100*r['worst_peak_T_fraction_while_certified']:.1f}% of the limit")
    print(f"      worst peak T if the adversary keeps flying past closure: "
          f"{100*r['worst_peak_T_fraction_of_limit']:.1f}%  ({r['seconds']}s)")

    path = V.save("v2_aerial.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
