"""V3 -- underwater: survey AUVs, buoyancy gliders, under-ice vehicles.

The underwater domain is the one that is hardest for reasons that have nothing to do with the
theorem. A pack inside a pressure hull is thermally isolated from an excellent heat sink -- an
order of magnitude less conductance than a car's loop, into water at 2 C -- so both of the
usual intuitions are wrong at once: it heats slowly *and* it is very cold. Cold is where the
IECON paper's weakest point lives, because plating is enforced and not certified, and the
plating current cap falls with temperature.

It is also the domain that most cleanly separates the anchored generalisation from anything
about flight. A survey AUV sitting perfectly still on the seabed still cannot take `u = 0` for
an answer: the computer, the inertial navigation and the acoustic modem draw their watts
regardless. The anchor is not about staying airborne; it is about staying a vehicle.

    python zeroguard/exp/v3_underwater.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260815
ENV = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))


def binding(plat, s, dt, w, marg, u_hi, status):
    """Which constraint actually set the upper edge."""
    if status != "ok":
        return "closed"
    if u_hi >= plat.cap(s) - 1e-9:
        cap_plate = plat.cell.plate_cap(s["T"]) * plat.P
        return "plating-cap" if cap_plate <= plat.u_max + 1e-9 else "actuator"
    vals = plat.probe(s, min(u_hi * 1.02 + 1e-6, plat.u_max), dt, w)
    if plat.mode == "charge":
        if vals[0] > plat.V_max - marg[0]:
            return "voltage"
        if vals[1] > plat.T_max - marg[1]:
            return "thermal"
        return "plating-margin"
    if vals[1] > plat.T_max - marg[0]:
        return "thermal"
    if vals[0] < plat.V_min + marg[1]:
        return "voltage"
    return "soc"


# ---------------------------------------------------------------------------------------
def v3_1_sealed_hull(n=3000, seed=SEED):
    """A sealed pressure hull: about a twelfth of a car's conductance, into 2 C water.

    The ratio is printed beside the claim on purpose. An earlier version of `platforms.py`
    asserted an order of magnitude and delivered 0.8x, which this experiment caught by
    reporting both numbers on the same line instead of only the one being argued for."""
    rng = np.random.default_rng(seed)
    est = V.pessimistic("survey-auv-dock", T_water=2.0)
    marg = V.margins(est)
    safe = 0; enf = 0; wT = -1e9; socs = []
    car = P.RobotaxiUrban()
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        plant = P.dock_charge_auv(T_water=2.0, scale=sc)
        r = V.charge_session(plant, est, est.init(float(rng.uniform(0.05, 0.3)),
                                                  float(rng.uniform(2.0, 25.0))),
                             60.0, 2.0, marg, target_soc=0.90, horizon=400)
        safe += int(r["ok"]); enf += int(bool(r["enforced_excursions"]))
        wT = max(wT, r["peak_T"]); socs.append(r["soc"])
    return stats.summarize_safety("auv_sealed_hull", n - safe, n, extra=dict(
        hull_UA_W_per_K=est.cell.cooling.hA,
        car_hA_W_per_K=car.cell.cooling.hA,
        conductance_ratio=est.cell.cooling.hA / car.cell.cooling.hA,
        water_C=2.0, worst_T=float(wT), enforced_plating_excursions=enf,
        mean_soc=float(np.mean(socs))))


# ---------------------------------------------------------------------------------------
def v3_2_cold_binding(n=600, seed=SEED + 1):
    """What actually limits a dock recharge in a sealed hull -- and the answer is not what this
    experiment was written to find.

    The hypothesis was that the binding constraint *switches* from temperature to plating as
    the water gets colder. It does not switch, because it was never thermal: plating binds at
    100 % of states at every water temperature from -1.8 C to 25 C. A sealed hull charges so
    slowly -- the plating cap holds it near 0.7C -- that the pack never gets near its thermal
    limit, so the constraint the hull's poor conductance would have threatened is not the one
    doing the work.

    What does change with temperature is *which* plating mechanism binds. Warm, it is the
    instantaneous margin on the potential proxy. Cold, it is increasingly the
    temperature-dependent current cap, which falls to a 0.70C floor below about 5.5 C. That
    shift is the measurable claim and it replaces the one this docstring used to make.

    This matters beyond bookkeeping. The IECON paper certifies temperature and voltage and only
    *enforces* plating -- so on this vehicle, in this medium, the binding constraint is the one
    the theorem does not certify. That is a limitation worth stating plainly rather than a
    result worth dressing up."""
    rng = np.random.default_rng(seed)
    rows = []
    for T_w in (-1.8, 2.0, 4.0, 8.0, 12.0, 18.0, 25.0):
        est = V.pessimistic("survey-auv-dock", T_water=T_w)
        marg = V.margins(est)
        census = {}
        socs = []; viol = 0
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.dock_charge_auv(T_water=T_w, scale=sc)
            s = est.init(float(rng.uniform(0.05, 0.5)), T_w + float(rng.uniform(0.0, 6.0)))
            lo, hi, st = A.interval(est, s, 60.0, T_w, marg)
            b = binding(est, s, 60.0, T_w, marg, hi, st)
            census[b] = census.get(b, 0) + 1
            r = V.charge_session(plant, est, s, 60.0, T_w, marg, target_soc=0.90, horizon=400)
            viol += int(not r["ok"]); socs.append(r["soc"])
        tot = sum(census.values())
        rows.append(dict(T_water=T_w, census=census, trials=n, violations=viol,
                         plating_fraction=(census.get("plating-cap", 0)
                                           + census.get("plating-margin", 0)) / tot,
                         plating_cap_fraction=census.get("plating-cap", 0) / tot,
                         thermal_fraction=census.get("thermal", 0) / tot,
                         mean_soc=float(np.mean(socs))))
    # The share of *plating* is 1.0 at every node, so correlating it with temperature is a
    # question about a constant and returns a tied-rank artefact. The live quantity is the
    # split within plating: the temperature-dependent current cap against the instantaneous
    # margin.
    rho, p = V.spearman([r["T_water"] for r in rows],
                        [r["plating_cap_fraction"] for r in rows])
    return dict(rows=rows, total_violations=sum(r["violations"] for r in rows),
                total_trials=n * len(rows),
                plating_binds_everywhere=all(r["plating_fraction"] > 0.999 for r in rows),
                thermal_never_binds=all(r["thermal_fraction"] < 1e-9 for r in rows),
                cap_share_vs_temperature=dict(rho=rho, p=p),
                cold_cap_fraction=rows[1]["plating_cap_fraction"],
                warm_cap_fraction=rows[-1]["plating_cap_fraction"])


# ---------------------------------------------------------------------------------------
def v3_3_hotel_anchor(n=800, seed=SEED + 2):
    """A nonzero anchor with nothing moving.

    The AUV is stationary on the seabed. Its thrusters are off. It still cannot command zero,
    because the computer and the navigation system are on, and if the pack stops supplying them
    the vehicle stops being a vehicle. This removes the objection that the anchored theorem is
    a story about aircraft."""
    rng = np.random.default_rng(seed)
    out = {}
    for label, hotel, thrust in (("idle-on-seabed", 25.0, 0.0),
                                 ("survey", 25.0, 90.0),
                                 ("transit", 25.0, 240.0)):
        unsafe = 0; brown = 0; endur = []; anchors = []
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.SurveyAUV(hotel_W=hotel, thrust_W=thrust,
                              scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.SurveyAUV(hotel_W=hotel, thrust_W=thrust, scale=sc)
            marg = V.margins(est)
            s0 = est.init(float(rng.uniform(0.4, 0.99)), float(rng.uniform(2.0, 15.0)))
            anchors.append(est.anchor(s0))
            r = V.discharge_mission(plant, est, s0, 60.0, 2.0, marg, horizon=1200,
                                    fly_past_closure=False)
            unsafe += int(r["unsafe_while_certified"]); endur.append(r["endurance_s"])
            ni = V.null_input_mission(plant, est, s0, 60.0, 2.0, marg, horizon=1200)
            brown += int(ni["brownout_step"] >= 0)
        out[label] = dict(hotel_W=hotel, thrust_W=thrust, trials=n,
                          mean_anchor_A=float(np.mean(anchors)),
                          unsafe_while_certified=unsafe,
                          cp95_pct=100 * stats.cp_upper(unsafe, n),
                          null_input_brownouts=brown,
                          median_endurance_h=float(np.median(endur)) / 3600.0)
    return out


# ---------------------------------------------------------------------------------------
def v3_4_deployment(months=6.0, per_day=3.0, seed=SEED + 3):
    """A six-month deployment of dock recharges with nobody aboard to recalibrate anything."""
    rng = np.random.default_rng(seed)
    n = int(months * 30.4 * per_day)
    est = V.pessimistic("survey-auv-dock", T_water=2.0)
    marg = V.margins(est)
    viol = 0; enf = 0; socs = []; wT = -1e9
    for k in range(n):
        frac = k / max(n - 1, 1)
        sc = dict(R=1.0 + 0.8 * frac ** 0.85, Q=1.0 - 0.20 * frac ** 0.8,
                  plate=1.0 + 0.6 * frac)
        plant = P.dock_charge_auv(T_water=2.0, scale=sc)
        r = V.charge_session(plant, est, est.init(float(rng.uniform(0.08, 0.30)),
                                                  2.0 + float(rng.uniform(0.0, 8.0))),
                             60.0, 2.0, marg, target_soc=0.90, horizon=400)
        viol += int(not r["ok"]); enf += int(bool(r["enforced_excursions"]))
        socs.append(r["soc"]); wT = max(wT, r["peak_T"])
    return stats.summarize_safety("auv_deployment", viol, n, extra=dict(
        months=months, recharges_per_day=per_day, worst_T=float(wT),
        enforced_plating_excursions=enf,
        first_quarter_soc=float(np.mean(socs[:n // 4])),
        last_quarter_soc=float(np.mean(socs[-n // 4:]))))


# ---------------------------------------------------------------------------------------
def v3_5_dormancy(n=1200, seed=SEED + 4):
    """Dormancy between missions: calendar ageing is another monotone channel.

    An AUV spends more of its life in a locker than in the water. Calendar ageing raises
    resistance and takes capacity without a single cycle being run, which is exactly the shape
    of every other degradation channel the certificate already bounds -- so the claim is that
    it needs no new machinery, and the test is whether that survives a decade of storage.

    It does, and it does so in a way the experiment was not written to expect: delivered charge
    does not change *at all* across ten years of dormancy, to the last digit. The reason is
    V3-2. In a sealed hull the binding constraint is plating at every temperature, and the
    plating cap depends on temperature rather than on resistance, so the channel that ageing
    moves is not the channel that is limiting. The certificate is not tolerant of dormancy so
    much as blind to it, which is a stronger statement and a different one.

    The first version of this experiment reported `rho = +1.000, p = 0.0002` for that constant
    series -- a tie-handling bug in the rank transform, fixed in `stats._rank`, which turned a
    flat line into a perfect monotone relationship. The degeneracy is now reported as such."""
    rng = np.random.default_rng(seed)
    rows = []
    for months in (0, 6, 12, 24, 48, 96, 120):
        # sqrt-of-time calendar fade, scaled so 120 months reaches the datasheet bound
        f = (months / 120.0) ** 0.5
        est = V.pessimistic("survey-auv-dock", T_water=2.0)
        marg = V.margins(est)
        viol = 0; socs = []
        for _ in range(n // 7):
            sc = dict(R=1.0 + 0.8 * f * float(rng.uniform(0.7, 1.0)),
                      Q=1.0 - 0.20 * f, plate=1.0 + 0.6 * f)
            plant = P.dock_charge_auv(T_water=2.0, scale=sc)
            r = V.charge_session(plant, est, est.init(0.15, 4.0), 60.0, 2.0, marg,
                                 target_soc=0.90, horizon=400)
            viol += int(not r["ok"]); socs.append(r["soc"])
        rows.append(dict(dormant_months=months, trials=n // 7, violations=viol,
                         mean_soc=float(np.mean(socs))))
    socs = [r["mean_soc"] for r in rows]
    rho, p = V.spearman([r["dormant_months"] for r in rows], socs, min_spread=1e-4)
    return dict(rows=rows, total_violations=sum(r["violations"] for r in rows),
                total_trials=sum(r["trials"] for r in rows),
                soc_spread=float(max(socs) - min(socs)),
                insensitive=bool(max(socs) - min(socs) < 1e-6),
                soc_vs_dormancy=dict(rho=rho, p=p))


# ---------------------------------------------------------------------------------------
def v3_6_depth(n=400, seed=SEED + 5):
    """Depth: colder water and a worse-coupled hull, together.

    Both move with depth and both move the same way, so this is a single physical axis rather
    than two. The expected result was a monotone fall in delivered charge; the measured one is
    that delivered charge does not move, for the same reason as V3-5 -- the plating cap binds
    at every depth from the surface to 6 000 m, and it is not the channel depth acts on. What
    depth does change is the *margin* to the thermal constraint, which never becomes binding."""
    rng = np.random.default_rng(seed)
    rows = []
    for depth in (0, 200, 600, 1500, 3000, 6000):
        # thermocline: ~15 C at the surface falling to ~2 C by 1 000 m, then flat
        T_w = 2.0 + 13.0 * np.exp(-depth / 400.0)
        ua = 0.008 * (1.0 - 0.35 * min(depth, 6000) / 6000.0)  # hull squeezed, gap thinner
        est = V.pessimistic("survey-auv-dock", T_water=float(T_w))
        est.cell.cooling = P.HullConduction(ua, 0.050, float(T_w))
        marg = V.margins(est)
        viol = 0; socs = []; census = {}
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.dock_charge_auv(T_water=float(T_w), scale=sc)
            plant.cell.cooling = P.HullConduction(ua, 0.050, float(T_w))
            s = est.init(0.12, float(T_w) + float(rng.uniform(0.0, 5.0)))
            lo, hi, st = A.interval(est, s, 60.0, float(T_w), marg)
            b = binding(est, s, 60.0, float(T_w), marg, hi, st)
            census[b] = census.get(b, 0) + 1
            r = V.charge_session(plant, est, s, 60.0, float(T_w), marg, target_soc=0.90,
                                 horizon=400)
            viol += int(not r["ok"]); socs.append(r["soc"])
        rows.append(dict(depth_m=depth, T_water=float(T_w), hull_UA=float(ua),
                         trials=n, violations=viol, binding=census,
                         mean_soc=float(np.mean(socs))))
    socs = [r["mean_soc"] for r in rows]
    rho, p = V.spearman([r["depth_m"] for r in rows], socs, min_spread=1e-4)
    return dict(rows=rows, total_violations=sum(r["violations"] for r in rows),
                total_trials=n * len(rows), soc_spread=float(max(socs) - min(socs)),
                plating_binds_at_every_depth=all(
                    sum(v for k, v in r["binding"].items() if "plating" in k) == r["trials"]
                    for r in rows),
                soc_vs_depth=dict(rho=rho, p=p))


# ---------------------------------------------------------------------------------------
def v3_7_under_ice(n=1000, transit_min=20.0, seed=SEED + 6):
    """Under an ice shelf there is no abort-to-surface. Does the warning arrive in time?

    **No, and that is the finding.** Waiting for the interval to close gives a median of about
    seven minutes before the vehicle is out of envelope, against a transit of twenty minutes
    back to a known hole in the ice. A certificate that is correct and seven minutes late loses
    the vehicle exactly as surely as one that is wrong.

    The fix is not a better certificate; it is to stop using closure as the trigger. The
    reserve `u_hi - u_lo` is available at every step and falls smoothly toward zero, so an
    operator can abort at a *threshold* on the reserve rather than at its exhaustion. This
    experiment therefore sweeps that threshold and reports the smallest one that buys a
    twenty-minute transit -- which is a design rule the null-input certificate could not have
    produced, because it has no reserve to threshold.

    Reporting the closure-triggered lead time alone would have been a true sentence and a
    misleading paper."""
    rng = np.random.default_rng(seed)
    THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0)
    leads, unwarned, made_it = [], 0, 0
    trig = {th: [] for th in THRESHOLDS}
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        est = P.under_ice_auv(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
        plant = P.under_ice_auv(scale=sc)
        marg = V.margins(est)
        s0 = est.init(float(rng.uniform(0.35, 0.99)), float(rng.uniform(-1.8, 8.0)))
        r = V.discharge_mission(plant, est, s0, 60.0, -1.8, marg, horizon=1500,
                                fly_past_closure=True)
        unwarned += int(r["unsafe_while_certified"])
        if r["lead_s"] is not None:
            leads.append(r["lead_s"])
            made_it += int(r["lead_s"] >= transit_min * 60.0)
        # how much earlier would a reserve threshold have fired?
        w = r["widths"]
        close_at = r["closure_step"] if r["closure_step"] >= 0 else len(w)
        for th in THRESHOLDS:
            k = next((i for i, x in enumerate(w) if x <= th), None)
            if k is not None and k <= close_at:
                trig[th].append((close_at - k) * 60.0)
    a = np.array(leads) if leads else np.array([0.0])
    sweep = []
    for th in THRESHOLDS:
        v = np.array(trig[th]) if trig[th] else np.array([0.0])
        sweep.append(dict(threshold_A=th, episodes=len(trig[th]),
                          median_warning_min=float(np.median(v)) / 60.0,
                          p05_warning_min=float(np.percentile(v, 5)) / 60.0,
                          covers_transit_rate=float((v >= transit_min * 60.0).mean())))
    enough = next((s["threshold_A"] for s in sweep
                   if s["p05_warning_min"] >= transit_min), None)
    return dict(trials=n, transit_min=transit_min,
                breaches_with_no_warning=unwarned,
                cp95_no_warning_pct=100 * stats.cp_upper(unwarned, n),
                episodes_with_lead=len(leads),
                closure_trigger=dict(
                    lead_covers_transit=made_it,
                    lead_covers_transit_rate=made_it / max(len(leads), 1),
                    lead_min_min=float(a.min()) / 60.0,
                    lead_p05_min=float(np.percentile(a, 5)) / 60.0,
                    lead_median_min=float(np.median(a)) / 60.0,
                    lead_p95_min=float(np.percentile(a, 95)) / 60.0),
                reserve_trigger_sweep=sweep,
                threshold_for_transit_A=enough,
                closure_alone_is_insufficient=float(np.median(a)) / 60.0 < transit_min)


# ---------------------------------------------------------------------------------------
def v3_8_transients(n=800, seed=SEED + 7):
    """Obstacle avoidance: thrust goes to full for a few seconds, repeatedly."""
    rng = np.random.default_rng(seed)
    rows = []
    for burst in (2.0, 3.0, 5.0, 8.0):
        unsafe = 0; closures = 0
        for _ in range(n // 4):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.SurveyAUV(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.SurveyAUV(scale=sc)
            base, marg = est.load_W, V.margins(est)
            s = est.init(float(rng.uniform(0.4, 0.99)), float(rng.uniform(2.0, 12.0)))
            closed = False
            for k in range(600):
                avoiding = (k % 40) < 4
                load = base * (burst if avoiding else 1.0)
                est.set_load(load); plant.set_load(load)
                lo, hi, st = A.interval(est, s, 10.0, 2.0, marg)
                if st != "ok":
                    closed = True
                    u = max(lo, est.anchor(s))
                else:
                    u = lo
                s, o = plant.step(s, float(u), 10.0, 2.0)
                bad, _e = V.split_breaches(plant, V.check(plant, o))
                if bad and not closed:
                    unsafe += 1
                    break
                if closed:
                    break
            closures += int(closed)
        rows.append(dict(burst_multiplier=burst, trials=n // 4,
                         closures=closures, unsafe_while_certified=unsafe))
    return dict(rows=rows, total_trials=n,
                total_unsafe=sum(r["unsafe_while_certified"] for r in rows))


# ---------------------------------------------------------------------------------------
def v3_9_no_recalibration(days=180, per_day=3, seed=SEED + 8):
    """180 days with no ground truth: does the estimator buy charge, or safety?

    The oracle is told the plant's true resistance before every single recharge. The bound is
    told nothing after the day it was built. If both record zero violations and the oracle
    delivers more charge, then recalibration is a performance feature and the certificate does
    not depend on it -- which is the property that makes an unattended six-month deployment
    admissible in the first place."""
    rng = np.random.default_rng(seed)
    n = days * per_day
    bound_soc, oracle_soc = [], []
    bound_viol = oracle_viol = 0
    for k in range(n):
        frac = k / max(n - 1, 1)
        sR_true = 1.0 + 0.8 * frac ** 0.85
        sc = dict(R=sR_true, Q=1.0 - 0.2 * frac ** 0.8, plate=1.0 + 0.6 * frac)
        plant = P.dock_charge_auv(T_water=2.0, scale=sc)
        s0_soc = float(rng.uniform(0.08, 0.30)); s0_T = 2.0 + float(rng.uniform(0.0, 8.0))
        for tag, sR_assumed in (("bound", V.S_R), ("oracle", sR_true)):
            est = V.pessimistic("survey-auv-dock", T_water=2.0, s_R=sR_assumed)
            marg = V.margins(est, s_R=sR_assumed)
            r = V.charge_session(plant, est, est.init(s0_soc, s0_T), 60.0, 2.0, marg,
                                 target_soc=0.90, horizon=400)
            if tag == "bound":
                bound_soc.append(r["soc"]); bound_viol += int(not r["ok"])
            else:
                oracle_soc.append(r["soc"]); oracle_viol += int(not r["ok"])
    w = stats.wilcoxon_paired(np.array(oracle_soc), np.array(bound_soc))
    return dict(days=days, cycles=n,
                bound=dict(violations=bound_viol, mean_soc=float(np.mean(bound_soc)),
                           cp95_pct=100 * stats.cp_upper(bound_viol, n)),
                oracle=dict(violations=oracle_viol, mean_soc=float(np.mean(oracle_soc)),
                            cp95_pct=100 * stats.cp_upper(oracle_viol, n)),
                safety_identical=bound_viol == oracle_viol,
                soc_gap_points=100 * (float(np.mean(oracle_soc)) - float(np.mean(bound_soc))),
                wilcoxon=w)


# ---------------------------------------------------------------------------------------
def v3_10_glider_vs_auv(n=600, seed=SEED + 9):
    """A glider at half a watt and an AUV at 175 W: the same certificate, four decades apart.

    If the two-sided construction is really about structure rather than magnitude, then the
    only thing that should differ between these two vehicles is where the anchor sits. The
    reserve ratio and the interval structure are the check."""
    rng = np.random.default_rng(seed)
    out = {}
    for case, dt, T_w in (("buoyancy-glider", 300.0, 4.0), ("under-ice-auv", 60.0, -1.8)):
        est = V.pessimistic(case, T_water=T_w)
        marg = V.margins(est)
        widths, anchors, unsafe, census = [], [], 0, {"single-interval": 0,
                                                      "disconnected": 0, "empty": 0}
        for _ in range(n):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.build(case, T_water=T_w, scale=sc)
            s = est.init(float(rng.uniform(0.2, 0.99)), T_w + float(rng.uniform(0.0, 10.0)))
            lo, hi, st = A.interval(est, s, dt, T_w, marg)
            widths.append((hi - lo) if st == "ok" else 0.0)
            anchors.append(est.anchor(s))
            _g, ok = A.scan(est, s, dt, T_w, marg, n=160)
            census[A.structure(ok)] += 1
            r = V.discharge_mission(plant, est, s, dt, T_w, marg, horizon=600,
                                    fly_past_closure=False)
            unsafe += int(r["unsafe_while_certified"])
        out[case] = dict(load_W=est.load_W, mean_anchor_A=float(np.mean(anchors)),
                         mean_width_A=float(np.mean(widths)), structure=census,
                         unsafe_while_certified=unsafe, trials=n,
                         reserve_ratio=float(np.mean(widths)) / max(np.mean(anchors), 1e-9))
    a, b = out["buoyancy-glider"], out["under-ice-auv"]
    return dict(platforms=out, load_ratio=b["load_W"] / max(a["load_W"], 1e-9),
                anchor_ratio=b["mean_anchor_A"] / max(a["mean_anchor_A"], 1e-9),
                both_single_interval=(a["structure"]["disconnected"] == 0
                                      and b["structure"]["disconnected"] == 0))


# ---------------------------------------------------------------------------------------
def v3_11_buoyancy_pump(n=600, seed=SEED + 10):
    """A glider's buoyancy engine: near-zero draw, punctuated by a 30x burst every dive.

    The extreme of the duty-cycle axis, and the case where an anchor computed at the *mean*
    load would be badly wrong. The anchor here is instantaneous, so the interval simply opens
    and closes around each burst."""
    rng = np.random.default_rng(seed)
    rows = []
    for pump_W in (0.0, 5.0, 15.0, 30.0, 60.0):
        unsafe = 0; closures = 0; dives = []
        for _ in range(n // 5):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.glider_platform(hotel_W=0.5, scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.glider_platform(hotel_W=0.5, scale=sc)
            marg = V.margins(est)
            s = est.init(float(rng.uniform(0.3, 0.99)), 4.0)
            closed = False; k_dive = 0
            for k in range(800):
                pumping = (k % 24) < 2               # two steps of pump per dive cycle
                load = 0.5 + (pump_W if pumping else 0.0)
                est.set_load(load); plant.set_load(load)
                lo, hi, st = A.interval(est, s, 300.0, 4.0, marg)
                if st != "ok":
                    closed = True; u = max(lo, est.anchor(s))
                else:
                    u = lo
                s, o = plant.step(s, float(u), 300.0, 4.0)
                bad, _e = V.split_breaches(plant, V.check(plant, o))
                if bad and not closed:
                    unsafe += 1; break
                if pumping and k % 24 == 0:
                    k_dive += 1
                if closed:
                    break
            closures += int(closed); dives.append(k_dive)
        rows.append(dict(pump_W=pump_W, burst_ratio=pump_W / 0.5, trials=n // 5,
                         closures=closures, unsafe_while_certified=unsafe,
                         median_dives=float(np.median(dives))))
    return dict(rows=rows, total_trials=n,
                total_unsafe=sum(r["unsafe_while_certified"] for r in rows))


# ---------------------------------------------------------------------------------------
def main():
    out = {}
    t0 = time.time()
    print("V3 -- underwater: survey AUVs, gliders, under-ice\n" + "=" * 78)

    print("\nV3-1  a sealed pressure hull in 2 C water")
    r = v3_1_sealed_hull(); out["v3_1_sealed_hull"] = r
    print(f"      hull UA {r['hull_UA_W_per_K']:.4f} W/K vs a car's {r['car_hA_W_per_K']:.4f} "
          f"= {r['conductance_ratio']:.2f}x")
    print(f"      {r['trials']:,} recharges | violations {r['violations']} | CP95 "
          f"{r['cp95_upper_pct']:.3f}% | worst T {r['worst_T']:.2f} C | SOC {r['mean_soc']:.3f}")

    print("\nV3-2  in cold water, what actually binds?")
    r = v3_2_cold_binding(); out["v3_2_cold_binding"] = r
    print(f"      {'T water':>8}{'plating':>10}{'of which cap':>14}{'thermal':>10}{'SOC':>8}")
    for row in r["rows"]:
        print(f"      {row['T_water']:>8.1f}{100*row['plating_fraction']:>9.0f}%"
              f"{100*row['plating_cap_fraction']:>13.0f}%"
              f"{100*row['thermal_fraction']:>9.0f}%{row['mean_soc']:>8.3f}")
    print(f"      plating binds everywhere: {r['plating_binds_everywhere']} | thermal never "
          f"binds: {r['thermal_never_binds']}")
    print(f"      the cap's share of it rises as the water cools: {100*r['warm_cap_fraction']:.0f}%"
          f" at 25 C -> {100*r['cold_cap_fraction']:.0f}% at 2 C  "
          f"(rho={r['cap_share_vs_temperature']['rho']:+.3f} "
          f"p={r['cap_share_vs_temperature']['p']:.4f})")
    print(f"      {r['total_violations']} violations in {r['total_trials']:,}")

    print("\nV3-3  a nonzero anchor with nothing moving")
    r = v3_3_hotel_anchor(); out["v3_3_hotel_anchor"] = r
    for k, v in r.items():
        print(f"      {k:16s} load {v['hotel_W']+v['thrust_W']:>6.0f} W  anchor "
              f"{v['mean_anchor_A']:5.2f} A  unsafe {v['unsafe_while_certified']}  "
              f"null-input brownouts {v['null_input_brownouts']}/{v['trials']}  "
              f"endurance {v['median_endurance_h']:.1f} h")

    print("\nV3-4  a six-month deployment, never recalibrated")
    r = v3_4_deployment(); out["v3_4_deployment"] = r
    print(f"      {r['trials']:,} recharges | violations {r['violations']} | CP95 "
          f"{r['cp95_upper_pct']:.4f}% | SOC {r['first_quarter_soc']:.3f} -> "
          f"{r['last_quarter_soc']:.3f}")

    print("\nV3-5  dormancy: calendar ageing as one more monotone channel")
    r = v3_5_dormancy(); out["v3_5_dormancy"] = r
    for row in r["rows"]:
        print(f"      {row['dormant_months']:>4} months  violations {row['violations']}  "
              f"SOC {row['mean_soc']:.3f}")
    print(f"      delivered charge does not move: spread {r['soc_spread']:.2e} SOC across the "
          f"whole sweep, which is numerical noise, so no rank correlation is reported "
          f"(rho={r['soc_vs_dormancy']['rho']})")

    print("\nV3-6  depth")
    r = v3_6_depth(); out["v3_6_depth"] = r
    for row in r["rows"]:
        print(f"      {row['depth_m']:>5} m  {row['T_water']:>5.1f} C  UA {row['hull_UA']:.4f}  "
              f"viol {row['violations']}  SOC {row['mean_soc']:.3f}  {row['binding']}")
    print(f"      plating binds at every depth: {r['plating_binds_at_every_depth']} | "
          f"delivered charge spread {r['soc_spread']:.2e} SOC (rho={r['soc_vs_depth']['rho']})")

    print("\nV3-7  under ice: does the warning arrive in time to reach a hole?")
    r = v3_7_under_ice(); out["v3_7_under_ice"] = r
    c = r["closure_trigger"]
    print(f"      {r['trials']:,} missions | breaches with no warning "
          f"{r['breaches_with_no_warning']} (CP95 {r['cp95_no_warning_pct']:.2f}%)")
    print(f"      waiting for closure: lead min {c['lead_min_min']:.1f} | p05 "
          f"{c['lead_p05_min']:.1f} | median {c['lead_median_min']:.1f} | p95 "
          f"{c['lead_p95_min']:.1f} min -> covers a {r['transit_min']:.0f}-minute transit in "
          f"{100*c['lead_covers_transit_rate']:.1f}% of episodes")
    print(f"      so closure alone is NOT enough: {r['closure_alone_is_insufficient']}. "
          f"Aborting on a reserve threshold instead:")
    print(f"        {'threshold':>10}{'median warn':>13}{'p05 warn':>10}{'covers transit':>16}")
    for s in r["reserve_trigger_sweep"]:
        print(f"        {s['threshold_A']:>9.1f}A{s['median_warning_min']:>12.1f}m"
              f"{s['p05_warning_min']:>9.1f}m{100*s['covers_transit_rate']:>15.0f}%")
    print(f"      smallest threshold that buys the transit at the 5th percentile: "
          f"{r['threshold_for_transit_A']} A")

    print("\nV3-8  obstacle-avoidance thrust transients")
    r = v3_8_transients(); out["v3_8_transients"] = r
    for row in r["rows"]:
        print(f"      x{row['burst_multiplier']:.0f} burst  closures {row['closures']:>4}/"
              f"{row['trials']}  unsafe {row['unsafe_while_certified']}")

    print("\nV3-9  180 days with no ground truth: charge or safety?")
    r = v3_9_no_recalibration(); out["v3_9_no_recalibration"] = r
    print(f"      {r['cycles']:,} recharges | bound {r['bound']['violations']} violations, "
          f"SOC {r['bound']['mean_soc']:.3f} | oracle {r['oracle']['violations']} violations, "
          f"SOC {r['oracle']['mean_soc']:.3f}")
    print(f"      safety identical: {r['safety_identical']} | oracle buys "
          f"{r['soc_gap_points']:+.2f} SOC points")

    print("\nV3-10 a glider at 0.5 W and an AUV at 175 W")
    r = v3_10_glider_vs_auv(); out["v3_10_glider_vs_auv"] = r
    for k, v in r["platforms"].items():
        print(f"      {k:18s} load {v['load_W']:>7.1f} W  anchor {v['mean_anchor_A']:7.3f} A  "
              f"reserve {v['mean_width_A']:6.2f} A  ratio {v['reserve_ratio']:8.1f}  "
              f"unsafe {v['unsafe_while_certified']}")
    print(f"      load ratio {r['load_ratio']:.0f}x | anchor ratio {r['anchor_ratio']:.0f}x | "
          f"both single-interval: {r['both_single_interval']}")

    print("\nV3-11 the buoyancy engine's burst")
    r = v3_11_buoyancy_pump(); out["v3_11_buoyancy_pump"] = r
    for row in r["rows"]:
        print(f"      pump {row['pump_W']:>5.1f} W ({row['burst_ratio']:>5.0f}x hotel)  "
              f"closures {row['closures']:>4}  unsafe {row['unsafe_while_certified']}  "
              f"dives {row['median_dives']:.0f}")

    path = V.save("v3_underwater.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
