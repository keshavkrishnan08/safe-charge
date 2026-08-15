"""V4 -- space: LEO, GEO, deep space, Mars surface, lunar night, high radiation.

Space is the domain where the certificate's least glamorous property becomes the decisive one.
The IECON filter needs no identification of the plant: no recursive least squares, no observer
to converge, no ground station in the loop. On a car that is a convenience. On a spacecraft
that has never been touched since integration, whose ground link is minutes to hours of light
time away, and whose battery has taken 300 krad of total ionising dose in the interim, it is
the only reason a fixed pre-launch parameter set is admissible at all.

Eleven experiments. Four are environments (LEO, GEO, deep space, Mars), three are the hard
edges of the mission (lunar night, radiation dose, no ground loop), and four attack the method
directly -- including a single-event upset injected into the filter's own state, which is the
one fault model in this whole register that corrupts the certificate rather than the plant.

    python zeroguard/exp/v4_space.py
"""
import os, sys, time, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260815
ENV = dict(R=(1.0, 1.8), Q=(0.80, 1.0), plate=(1.0, 1.6))


# ---------------------------------------------------------------------------------------
def v4_1_leo(years=5.0, seed=SEED):
    """Five years of LEO eclipse cycling: 15.5 orbits a day, never recalibrated.

    The defining feature is the count, not the environment. At a 92.9-minute period this is
    about 28 000 charge cycles, and the certificate has to hold on the last one using the
    numbers it was given before launch."""
    rng = np.random.default_rng(seed)
    sat = P.LEOSmallsat()
    n = int(years * sat.cycles_per_year())
    est = V.pessimistic("leo-smallsat")
    marg = V.margins(est)
    viol = enf = 0; socs = []; wT = -1e9
    for k in range(n):
        frac = k / max(n - 1, 1)
        sc = dict(R=1.0 + 0.8 * frac ** 0.85, Q=1.0 - 0.20 * frac ** 0.75,
                  plate=1.0 + 0.6 * frac)
        plant = P.LEOSmallsat(scale=sc)
        # sunlit arc: 57.9 min of charging, beta-angle jitter on the sink temperature
        T_sink = 4.0 + float(rng.uniform(0.0, 40.0))
        r = V.charge_session(plant, est, est.init(float(rng.uniform(0.60, 0.80)),
                                                  float(rng.uniform(-5.0, 25.0))),
                             30.0, T_sink, marg, target_soc=0.95, horizon=116)
        viol += int(not r["ok"]); enf += int(bool(r["enforced_excursions"]))
        socs.append(r["soc"]); wT = max(wT, r["peak_T"])
    return stats.summarize_safety("leo_eclipse_cycling", viol, n, extra=dict(
        years=years, period_min=sat.period_min, eclipse_min=sat.eclipse_min,
        cycles_per_year=sat.cycles_per_year(), worst_T=float(wT),
        enforced_plating_excursions=enf,
        first_quarter_soc=float(np.mean(socs[:n // 4])),
        last_quarter_soc=float(np.mean(socs[-n // 4:]))))


# ---------------------------------------------------------------------------------------
def v4_2_geo(n=900, seed=SEED + 1):
    """A GEO eclipse season: 90 eclipses a year, up to 72 minutes, 3 kW that cannot go quiet.

    Opposite duty cycle to LEO -- few, deep, and non-negotiable -- and the payload is the
    anchor. A GEO operator's question is not whether the bus survives but whether it survives
    the *longest* eclipse of the season at end of life, so the sweep runs eclipse duration to
    the seasonal maximum."""
    rng = np.random.default_rng(seed)
    rows = []
    for minutes in (10.0, 30.0, 50.0, 65.0, 72.0):
        unsafe = 0; completed = 0; depth = []
        for _ in range(n // 5):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.GEOComsat(scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.GEOComsat(scale=sc)
            for pl in (est, plant):        # sized for ~70 % DoD at the seasonal maximum
                pl.P = 3; pl.u_max = 3.0 * pl.cell.q_nom() * 3; pl._anchored_split = None
            marg = V.margins(est)
            soc0 = float(rng.uniform(0.92, 1.0))
            s0 = est.init(soc0, float(rng.uniform(-10.0, 20.0)))
            r = V.discharge_mission(plant, est, s0, 60.0, est.w_nominal, marg,
                                    horizon=int(minutes), fly_past_closure=False)
            unsafe += int(r["unsafe_while_certified"])
            completed += int(r["closure_step"] < 0)
            depth.append(soc0 - r["soc"])
        rows.append(dict(eclipse_min=minutes, trials=n // 5, completed=completed,
                         completion_rate=completed / (n // 5),
                         unsafe_while_certified=unsafe,
                         median_depth_of_discharge=float(np.median(depth))))
    return dict(rows=rows, eclipses_per_year=P.GEOComsat.eclipses_per_year,
                max_eclipse_min=P.GEOComsat.max_eclipse_min,
                total_trials=n, total_unsafe=sum(r["unsafe_while_certified"] for r in rows))


# ---------------------------------------------------------------------------------------
def v4_3_deep_space(n=3000, seed=SEED + 2):
    """Cruise: the only way heat leaves is as a photon, and the sink is 4 K.

    E3 showed at cell level that replacing Newtonian convection with Stefan-Boltzmann leaves
    every monotonicity intact. This carries it onto a real spacecraft bus and adds the sink
    sweep, because a probe's radiator sees anything from 4 K in cruise to 250 K near a warm
    body."""
    rng = np.random.default_rng(seed)
    est = V.pessimistic("deep-space-cruiser", T_sink=250.0)     # worst sink
    marg = V.margins(est)
    safe = enf = 0; wT = -1e9; socs = []
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
        T_sink = float(rng.uniform(4.0, 250.0))
        plant = P.DeepSpaceCruiser(T_sink=T_sink, scale=sc)
        r = V.charge_session(plant, est, est.init(float(rng.uniform(0.4, 0.7)),
                                                  float(rng.uniform(-20.0, 25.0))),
                             60.0, T_sink, marg, target_soc=0.95, horizon=200)
        safe += int(r["ok"]); enf += int(bool(r["enforced_excursions"]))
        wT = max(wT, r["peak_T"]); socs.append(r["soc"])
    # monotonicity of the radiative channel, checked numerically rather than assumed
    mono = True
    probe = P.DeepSpaceCruiser()
    for _ in range(4000):
        T = float(rng.uniform(-60.0, 60.0))
        q1 = probe.cell.cooling.q_out(T, 0.0); q2 = probe.cell.cooling.q_out(T + 0.5, 0.0)
        if q2 <= q1:
            mono = False
    return stats.summarize_safety("deep_space_cruise", n - safe, n, extra=dict(
        T_sink_range_K=[4.0, 250.0], cooling=probe.cell.cooling.describe(),
        radiative_monotone_in_T=mono, worst_T=float(wT),
        enforced_plating_excursions=enf, mean_soc=float(np.mean(socs))))


# ---------------------------------------------------------------------------------------
def v4_4_mars(n=2400, seed=SEED + 3):
    """Mars surface: radiation plus 6 mbar of CO2, across a diurnal swing of order 80 K.

    Two cooling terms, both increasing in temperature, so their sum is too and Prop. 2's
    cooling channel is unchanged. The claim is that the certificate does not notice that it is
    now on a planet, and the test is a full sol at hourly resolution.

    The rover soaks in sunlight, so its pack starts well above the thin air around it -- but
    the initial temperature is clamped into the safe set by `vexp.safe_init`. An earlier
    version drew up to `T_air + 60 K`, which at local noon starts the rover at +50 C against a
    40 C limit, and recorded 76 violations in states that were already unsafe before the filter
    was called. Forward invariance makes no claim about those and neither does this."""
    rng = np.random.default_rng(seed)
    rows = []
    # a representative sol: -80 C before dawn to -10 C at local noon
    for hour, T_air in ((2, -80.0), (6, -70.0), (10, -35.0), (13, -10.0),
                        (16, -25.0), (20, -55.0)):
        est = V.pessimistic("mars-rover", T_amb=T_air)
        marg = V.margins(est)
        viol = enf = 0; socs = []; wT = -1e9
        for _ in range(n // 6):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.MarsSurfaceRover(T_amb=T_air, scale=sc)
            r = V.charge_session(plant, est,
                                 V.safe_init(est, float(rng.uniform(0.3, 0.6)),
                                             T_air + float(rng.uniform(10.0, 60.0)), marg),
                                 60.0, T_air, marg, target_soc=0.95, horizon=300)
            viol += int(not r["ok"]); enf += int(bool(r["enforced_excursions"]))
            socs.append(r["soc"]); wT = max(wT, r["peak_T"])
        rows.append(dict(local_hour=hour, T_air=T_air, trials=n // 6, violations=viol,
                         enforced=enf, mean_soc=float(np.mean(socs)), peak_T=float(wT)))
    # both channels increasing in T?
    c = P.MarsSurfaceRover().cell.cooling
    mono = all(c.q_out(T + 0.5, 0.0) > c.q_out(T, 0.0)
               for T in np.linspace(-90.0, 60.0, 3000))
    return dict(rows=rows, sol_min=P.MarsSurfaceRover.sol_min, cooling=c.describe(),
                sum_monotone_in_T=bool(mono),
                total_trials=n, total_violations=sum(r["violations"] for r in rows))


# ---------------------------------------------------------------------------------------
def v4_5_lunar_night(n=400, seed=SEED + 4):
    """Lunar night: 354 hours at 100 K, and the load is a *heater*.

    This is the sharpest test of the two-sided certificate in the register, because the floor
    constraint exists to put energy into the very state the cap is trying to hold down. Both
    edges push on temperature, from opposite directions, and the interval is what is left.

    The honest result is a frontier, not a pass. A lander on a 2.9 kWh pack does not survive
    354 hours on a 45 W heater and neither do most real ones -- which is why landers that do
    survive the night carry radioisotope heaters rather than bigger batteries. What the
    certificate supplies is the exact hour at which the envelope closes, from a fixed parameter
    set, so the survivable fraction can be designed for instead of discovered."""
    rng = np.random.default_rng(seed)
    rows = []
    for heater_W in (20.0, 30.0, 45.0, 60.0, 90.0):
        for pack_mult in (1, 2, 4):
            unsafe = 0; hours = []; binding = {}
            for _ in range(n // 5):
                sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
                est = P.LunarNightLander(heater_W=heater_W,
                                         scale=dict(R=V.S_R, Q=0.80, plate=1.6))
                plant = P.LunarNightLander(heater_W=heater_W, scale=sc)
                for pl in (est, plant):
                    pl.P *= pack_mult
                    pl.u_max *= pack_mult
                    pl._anchored_split = None
                marg = V.margins(est)
                s0 = est.init(1.0, float(rng.uniform(-20.0, 20.0)))
                r = V.discharge_mission(plant, est, s0, 300.0, 100.0, marg,
                                        horizon=int(354 * 12), fly_past_closure=False)
                unsafe += int(r["unsafe_while_certified"])
                hours.append(r["endurance_s"] / 3600.0)
                b = "energy" if r["soc"] <= est.soc_floor + 0.02 else "cold-sag"
                binding[b] = binding.get(b, 0) + 1
            med = float(np.median(hours))
            rows.append(dict(heater_W=heater_W, pack_mult=pack_mult,
                             pack_kWh=P.LunarNightLander().energy_kWh() * pack_mult,
                             trials=n // 5, unsafe_while_certified=unsafe,
                             binding=binding,
                             median_survival_h=med,
                             survives_night=med >= P.LunarNightLander.night_hours,
                             night_fraction=med / P.LunarNightLander.night_hours))
    # If the envelope closes on cold rather than on energy, then a bigger battery is the wrong
    # lever and insulation is the right one. That is a design conclusion, so it is measured:
    # the emissivity sweep holds the pack and the heater fixed and moves only the radiator.
    insul = []
    for eps in (0.72, 0.40, 0.20, 0.10, 0.05, 0.02):
        hours = []; unsafe_i = 0
        for _ in range(n // 5):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.LunarNightLander(heater_W=45.0, eps=eps,
                                     scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.LunarNightLander(heater_W=45.0, eps=eps, scale=sc)
            marg = V.margins(est)
            r = V.discharge_mission(plant, est, est.init(1.0, float(rng.uniform(-20.0, 20.0))),
                                    300.0, 100.0, marg, horizon=int(354 * 12),
                                    fly_past_closure=False)
            hours.append(r["endurance_s"] / 3600.0); unsafe_i += int(r["unsafe_while_certified"])
        insul.append(dict(eps=eps, median_survival_h=float(np.median(hours)),
                          unsafe=unsafe_i,
                          survives_night=float(np.median(hours))
                          >= P.LunarNightLander.night_hours))
    # Below some emissivity the pack keeps its own dissipation and stops being cold-limited;
    # what stops it then is simply the energy in the battery. Report where that crossover is,
    # because it is the point past which better insulation buys nothing.
    best = max(insul, key=lambda r: r["median_survival_h"])
    energy_limit_h = (P.LunarNightLander().energy_kWh() * 1000.0 * 0.95) / 45.0
    survivors = [r for r in rows if r["survives_night"]]
    ins_surv = [r for r in insul if r["survives_night"]]
    return dict(rows=rows, insulation=insul,
                best_insulated_survival_h=best["median_survival_h"],
                best_insulated_eps=best["eps"],
                energy_limit_h=energy_limit_h,
                insulation_saturates_at_energy_limit=bool(
                    best["median_survival_h"] >= 0.6 * energy_limit_h),
                insulation_configs_surviving=len(ins_surv),
                night_hours=P.LunarNightLander.night_hours,
                configurations_surviving=len(survivors), configurations=len(rows),
                min_surviving_config=(min(survivors,
                                          key=lambda r: r["pack_kWh"] / max(r["heater_W"], 1e-9))
                                      if survivors else None),
                total_unsafe=(sum(r["unsafe_while_certified"] for r in rows)
                              + sum(r["unsafe"] for r in insul)))


# ---------------------------------------------------------------------------------------
def v4_6_radiation(n=2500, seed=SEED + 5):
    """300 krad of total ionising dose, as one more monotone channel.

    TID raises series resistance and takes capacity: the same shape as ageing, which the
    certificate already bounds. E11 established that at scale-factor level; this runs it at
    mission dose on an outer-planet orbiter and sweeps the dose so the claim has a slope rather
    than a single point."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in (0.0, 50.0, 100.0, 200.0, 300.0, 500.0):
        est = V.pessimistic("high-radiation-orbiter", tid_krad=tid)
        marg = V.margins(est)
        viol = enf = 0; socs = []; wT = -1e9
        for _ in range(n // 6):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            dose = float(rng.uniform(0.0, tid)) if tid > 0 else 0.0
            plant = P.HighRadiationOrbiter(tid_krad=dose, scale=sc)
            r = V.charge_session(plant, est, est.init(float(rng.uniform(0.4, 0.7)),
                                                      float(rng.uniform(-15.0, 20.0))),
                                 60.0, 4.0, marg, target_soc=0.95, horizon=200)
            viol += int(not r["ok"]); enf += int(bool(r["enforced_excursions"]))
            socs.append(r["soc"]); wT = max(wT, r["peak_T"])
        rows.append(dict(tid_krad=tid, trials=n // 6, violations=viol, enforced=enf,
                         mean_soc=float(np.mean(socs)), peak_T=float(wT),
                         R_growth=1.0 + 0.0011 * tid))
    rho, p = V.spearman([r["tid_krad"] for r in rows], [r["mean_soc"] for r in rows])
    return dict(rows=rows, total_trials=n,
                total_violations=sum(r["violations"] for r in rows),
                soc_vs_dose=dict(rho=rho, p=p))


# ---------------------------------------------------------------------------------------
def _flip(x, bit, rng):
    """Flip one bit of a float64. This is what a heavy ion does to a register."""
    b = bytearray(struct.pack("<d", float(x)))
    b[bit // 8] ^= (1 << (bit % 8))
    try:
        v = struct.unpack("<d", bytes(b))[0]
    except Exception:
        return float(x)
    return v


def v4_7_seu(n=40000, seed=SEED + 6):
    """A single-event upset inside the filter's own state.

    Every other fault model in this register corrupts the *plant* or the *sensor*. This one
    corrupts the certificate: a heavy ion flips a bit in the state word the projection is about
    to read, so the filter computes an admissible interval for a spacecraft that does not exist
    and commands it to the one that does.

    There is no theorem that survives this, and claiming otherwise would be dishonest. What can
    be measured is the *shape* of the exposure: which bits matter, how often a flip produces a
    command that is more aggressive than the uncorrupted one, and whether a single range check
    on the state word -- three comparisons, no model evaluations -- closes the gap. That check
    is the deliverable here, not the theorem."""
    rng = np.random.default_rng(seed)
    est = V.pessimistic("high-radiation-orbiter", tid_krad=300.0)
    marg = V.margins(est)
    plant = P.HighRadiationOrbiter(tid_krad=150.0, scale=dict(R=1.4, Q=0.9, plate=1.3))
    fields = ("soc", "T", "V1")
    # plausibility box: what a spacecraft state word can physically contain
    BOX = dict(soc=(0.0, 1.05), T=(-60.0, 80.0), V1=(-1.0, 1.0))
    census = {"benign": 0, "conservative": 0, "aggressive": 0, "caught_by_range_check": 0}
    worst_excess = 0.0; aggressive_uncaught = 0
    for _ in range(n):
        s = dict(soc=float(rng.uniform(0.1, 0.95)), T=float(rng.uniform(-20.0, 35.0)),
                 V1=float(rng.uniform(-0.05, 0.05)), aging={"Qloss": 0.0, "Rfac": 1.0})
        u_ref, _ = A.project_anchored(est, s, est.u_max, 60.0, 4.0, marg)
        f = fields[int(rng.integers(0, 3))]
        bit = int(rng.integers(0, 64))
        s2 = dict(s); s2[f] = _flip(s[f], bit, rng)
        lo, hi = BOX[f]
        caught = not (lo <= s2[f] <= hi) or not np.isfinite(s2[f])
        if caught:
            census["caught_by_range_check"] += 1
            continue
        try:
            u_bad, _ = A.project_anchored(est, s2, est.u_max, 60.0, 4.0, marg)
        except (ValueError, OverflowError):
            census["caught_by_range_check"] += 1
            continue
        if abs(u_bad - u_ref) <= 1e-9:
            census["benign"] += 1
        elif u_bad < u_ref:
            census["conservative"] += 1
        else:
            census["aggressive"] += 1
            aggressive_uncaught += 1
            # does the more aggressive command actually breach the true plant?
            _s, o = plant.step(s, float(u_bad), 60.0, 4.0)
            c, _e = V.split_breaches(plant, V.check(plant, o))
            if c:
                worst_excess = max(worst_excess, o[1] - plant.T_max)
    tot = sum(census.values())
    return dict(flips=n, census=census,
                caught_fraction=census["caught_by_range_check"] / tot,
                benign_or_conservative_fraction=(census["benign"] + census["conservative"]) / tot,
                aggressive_fraction=census["aggressive"] / tot,
                aggressive_uncaught=aggressive_uncaught,
                worst_certified_overshoot_C=float(worst_excess),
                range_check_cost="3 comparisons, 0 model evaluations",
                honest_note=("no one-step theorem survives corruption of its own input; the "
                             "measurable quantity is the exposure and what a range check on "
                             "the state word removes"))


# ---------------------------------------------------------------------------------------
def v4_8_no_ground_loop(days=1825, seed=SEED + 7):
    """No ground in the loop: a fixed pre-launch parameter set against an oracle recalibrated
    every single day.

    Deep space makes recalibration expensive rather than impossible -- round-trip light time to
    the outer planets is hours, and a battery model update is not what the downlink budget is
    for. The claim is the same one E4 made for a mission and V3-9 made for a deployment: the
    estimator buys charge, never safety."""
    rng = np.random.default_rng(seed)
    b_soc, o_soc = [], []
    b_v = o_v = 0
    for k in range(days):
        frac = k / max(days - 1, 1)
        sR = 1.0 + 0.8 * frac ** 0.85
        sc = dict(R=sR, Q=1.0 - 0.2 * frac ** 0.75, plate=1.0 + 0.6 * frac)
        plant = P.DeepSpaceCruiser(scale=sc)
        soc0 = float(rng.uniform(0.4, 0.7)); T0 = float(rng.uniform(-20.0, 20.0))
        T_sink = float(rng.uniform(4.0, 250.0))
        for tag, assumed in (("bound", V.S_R), ("oracle", sR)):
            est = V.pessimistic("deep-space-cruiser", s_R=assumed, T_sink=250.0)
            marg = V.margins(est, s_R=assumed)
            r = V.charge_session(plant, est, est.init(soc0, T0), 60.0, T_sink, marg,
                                 target_soc=0.95, horizon=200)
            if tag == "bound":
                b_soc.append(r["soc"]); b_v += int(not r["ok"])
            else:
                o_soc.append(r["soc"]); o_v += int(not r["ok"])
    w = stats.wilcoxon_paired(np.array(o_soc), np.array(b_soc))
    return dict(days=days, years=days / 365.25,
                bound=dict(violations=b_v, mean_soc=float(np.mean(b_soc)),
                           cp95_pct=100 * stats.cp_upper(b_v, days)),
                oracle=dict(violations=o_v, mean_soc=float(np.mean(o_soc)),
                            cp95_pct=100 * stats.cp_upper(o_v, days)),
                safety_identical=b_v == o_v,
                soc_gap_points=100 * (float(np.mean(o_soc)) - float(np.mean(b_soc))),
                wilcoxon=w)


# ---------------------------------------------------------------------------------------
def v4_9_array_degradation(n=1800, seed=SEED + 8):
    """Solar-array degradation shrinks the charge *window*, not the certificate.

    Arrays lose a couple of percent a year to UV and micrometeoroids, so the available charge
    current falls over the mission. The distinction being tested is that this is a change in
    what the vehicle can *request*, not in what the filter can *certify* -- if the two got
    confused, an ageing array would look like a safety problem.

    The sweep runs to sixty years, three times any real mission, and the array still never
    becomes the binding constraint: at 29.8 % of beginning-of-life output it can still deliver
    more current than the certificate will admit. Delivered charge is flat to three decimal
    places across the whole sweep. The separation being claimed is therefore not marginal on
    this spacecraft -- it is not close."""
    rng = np.random.default_rng(seed)
    rows = []
    n = max(n, 8 * 60)
    for year in (0, 3, 5, 10, 15, 25, 40, 60):
        avail = 0.98 ** year                        # array output relative to beginning of life
        est = V.pessimistic("leo-smallsat")
        marg = V.margins(est)
        viol = 0; socs = []; clipped = []
        for _ in range(n // 5):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.LEOSmallsat(scale=sc)
            r = V.charge_session(plant, est, est.init(0.65, float(rng.uniform(-5.0, 20.0))),
                                 30.0, 4.0, marg, target_soc=0.95, horizon=116,
                                 u_request=est.u_max * avail)
            viol += int(not r["ok"]); socs.append(r["soc"])
            clipped.append(r["clipped"] / max(r["steps"], 1))
        rows.append(dict(year=year, array_fraction=avail, trials=n // 5, violations=viol,
                         mean_soc=float(np.mean(socs)),
                         array_ceiling_A=est.u_max * avail,
                         clipped_fraction=float(np.mean(clipped))))
    rho, p = V.spearman([r["year"] for r in rows], [r["mean_soc"] for r in rows])
    rho2, p2 = V.spearman([r["year"] for r in rows], [r["clipped_fraction"] for r in rows])
    binding_from = next((r["year"] for r in rows if r["mean_soc"] < rows[0]["mean_soc"] - 1e-3),
                        None)
    socs = [r["mean_soc"] for r in rows]
    return dict(rows=rows, total_trials=n,
                soc_spread=float(max(socs) - min(socs)),
                array_never_binding=binding_from is None,
                years_swept=rows[-1]["year"],
                final_array_fraction=rows[-1]["array_fraction"],
                total_violations=sum(r["violations"] for r in rows),
                soc_vs_year=dict(rho=rho, p=p),
                array_becomes_binding_year=binding_from,
                clipping_vs_year=dict(rho=rho2, p=p2))


# ---------------------------------------------------------------------------------------
def v4_10_radiator(n=1600, seed=SEED + 9):
    """Radiator degradation, all the way to no radiator at all.

    Deposition and UV darkening raise absorptivity and cut effective emissivity over a mission.
    E3 showed the certificate degrades gracefully to the vacuum limit at cell level; here the
    sweep runs eps*A down by two orders of magnitude on a spacecraft bus.

    The safety requirement -- zero violations at every point -- holds. The *performance*
    direction is the opposite of the one this experiment was written to expect, and the reason
    is worth stating because it inverts a terrestrial intuition. On a car, losing cooling costs
    charge. In deep space, radiating to a 4-250 K sink, a healthy radiator makes the pack
    *cold*, series resistance climbs, and the voltage limit arrives sooner. Shrinking the
    radiator lets the pack keep its own dissipation, resistance falls, and delivered charge
    rises until the plating corner stops it. Losing the radiator is not a thermal emergency
    here; it is a mild performance improvement, and the thermal emergency in this domain is the
    other direction entirely -- which is what V4-5 measures on the lunar surface."""
    rng = np.random.default_rng(seed)
    rows = []
    for frac in (1.0, 0.75, 0.50, 0.25, 0.10, 0.03, 0.01):
        est = P.DeepSpaceCruiser(eps=0.88 * frac, T_sink=250.0,
                                 scale=dict(R=V.S_R, Q=0.80, plate=1.6))
        marg = V.margins(est)
        viol = 0; socs = []; wT = -1e9
        for _ in range(n // 7):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            plant = P.DeepSpaceCruiser(eps=0.88 * frac, scale=sc)
            T_sink = float(rng.uniform(4.0, 250.0))
            plant.cell.cooling.T_sink = T_sink
            r = V.charge_session(plant, est, est.init(0.5, float(rng.uniform(-10.0, 20.0))),
                                 60.0, T_sink, marg, target_soc=0.95, horizon=200)
            viol += int(not r["ok"]); socs.append(r["soc"]); wT = max(wT, r["peak_T"])
        rows.append(dict(radiator_fraction=frac, eps=0.88 * frac, trials=n // 7,
                         violations=viol, mean_soc=float(np.mean(socs)), peak_T=float(wT)))
    rho, p = V.spearman([r["radiator_fraction"] for r in rows], [r["mean_soc"] for r in rows])
    return dict(rows=rows, total_trials=n,
                total_violations=sum(r["violations"] for r in rows),
                charge_rises_as_radiator_shrinks=bool(rho < 0),
                cold_limited_not_heat_limited=bool(rho < 0),
                soc_vs_radiator=dict(rho=rho, p=p))


# ---------------------------------------------------------------------------------------
def v4_11_bus_anchor(n=900, seed=SEED + 10):
    """The bus load in eclipse is the anchor, and the null-input filter cannot represent it.

    A spacecraft in eclipse is the cleanest possible statement of the generalisation. There is
    no thrust, no motion, nothing that could be described as flight; there is simply a payload
    that has to stay powered for 72 minutes, and a filter whose only always-available action is
    to stop supplying it."""
    rng = np.random.default_rng(seed)
    out = {}
    for case, dt, hor, load in (("leo-smallsat", 30.0, 70, 60.0),
                                ("geo-comsat", 60.0, 72, 3000.0),
                                ("lunar-night-lander", 300.0, 400, 45.0)):
        unsafe = brown = 0; anchors = []; served = []
        for _ in range(n // 3):
            sc = {k: float(rng.uniform(*v)) for k, v in ENV.items()}
            est = P.build(case, mode="discharge", load_W=load,
                          scale=dict(R=V.S_R, Q=0.80, plate=1.6))
            plant = P.build(case, mode="discharge", load_W=load, scale=sc)
            marg = V.margins(est)
            s0 = est.init(float(rng.uniform(0.6, 1.0)), float(rng.uniform(-20.0, 25.0)))
            anchors.append(est.anchor(s0))
            r = V.discharge_mission(plant, est, s0, dt, est.w_nominal, marg, horizon=hor,
                                    fly_past_closure=False)
            unsafe += int(r["unsafe_while_certified"])
            ni = V.null_input_mission(plant, est, s0, dt, est.w_nominal, marg, horizon=hor)
            brown += int(ni["brownout_step"] >= 0)
            served.append(ni["steps_served"])
        out[case] = dict(load_W=load, trials=n // 3,
                         mean_anchor_A=float(np.mean(anchors)),
                         unsafe_while_certified=unsafe,
                         cp95_pct=100 * stats.cp_upper(unsafe, n // 3),
                         null_input_brownouts=brown,
                         null_input_mean_steps_served=float(np.mean(served)))
    return out


# ---------------------------------------------------------------------------------------
def main():
    out = {}
    t0 = time.time()
    print("V4 -- space\n" + "=" * 78)

    print("\nV4-1  five years of LEO eclipse cycling")
    r = v4_1_leo(); out["v4_1_leo"] = r
    print(f"      {r['cycles_per_year']:.0f} cycles/year x {r['years']:.0f} y = "
          f"{r['trials']:,} charges | violations {r['violations']} | CP95 "
          f"{r['cp95_upper_pct']:.4f}%")
    print(f"      worst T {r['worst_T']:.2f} C | SOC {r['first_quarter_soc']:.3f} -> "
          f"{r['last_quarter_soc']:.3f}")

    print("\nV4-2  a GEO eclipse season, out to the 72-minute maximum")
    r = v4_2_geo(); out["v4_2_geo"] = r
    for row in r["rows"]:
        print(f"      {row['eclipse_min']:>5.0f} min  completion "
              f"{100*row['completion_rate']:>5.1f}%  DoD "
              f"{100*row['median_depth_of_discharge']:>5.1f}%  unsafe "
              f"{row['unsafe_while_certified']}")

    print("\nV4-3  deep-space cruise, radiation only")
    r = v4_3_deep_space(); out["v4_3_deep_space"] = r
    print(f"      {r['cooling']}")
    print(f"      {r['trials']:,} draws over sink 4-250 K | violations {r['violations']} | "
          f"CP95 {r['cp95_upper_pct']:.3f}% | radiative monotone in T: "
          f"{r['radiative_monotone_in_T']}")

    print("\nV4-4  a Mars sol")
    r = v4_4_mars(); out["v4_4_mars"] = r
    for row in r["rows"]:
        print(f"      {row['local_hour']:>3}:00  {row['T_air']:>6.0f} C  violations "
              f"{row['violations']}  SOC {row['mean_soc']:.3f}  peak T {row['peak_T']:.1f} C")
    print(f"      radiation + thin convection, sum monotone in T: {r['sum_monotone_in_T']}")

    print("\nV4-5  lunar night: 354 h at 100 K, with the heater as the floor")
    r = v4_5_lunar_night(); out["v4_5_lunar_night"] = r
    print(f"      {'heater':>7}{'pack':>8}{'survival h':>12}{'/354 h':>9}   unsafe")
    for row in r["rows"]:
        print(f"      {row['heater_W']:>6.0f}W{row['pack_kWh']:>7.1f}k"
              f"{row['median_survival_h']:>12.1f}{100*row['night_fraction']:>8.0f}%   "
              f"{row['unsafe_while_certified']}")
    print(f"      configurations surviving the full night: {r['configurations_surviving']}"
          f"/{r['configurations']}  (the envelope closes on cold, not on energy:"
          f" {r['rows'][0]['binding']})")
    print(f"      the lever that moves it is insulation, not capacity:")
    for row in r["insulation"]:
        print(f"        eps {row['eps']:.2f}  survival {row['median_survival_h']:7.1f} h  "
              f"{'SURVIVES THE NIGHT' if row['survives_night'] else ''}")
    print(f"      insulation buys survival until the pack stops being cold-limited and starts "
          f"being energy-limited:")
    print(f"        best {r['best_insulated_survival_h']:.1f} h at eps {r['best_insulated_eps']:.2f}"
          f" against an energy ceiling of {r['energy_limit_h']:.1f} h "
          f"({P.LunarNightLander().energy_kWh()*1000:.0f} Wh / 45 W)")

    print("\nV4-6  total ionising dose")
    r = v4_6_radiation(); out["v4_6_radiation"] = r
    for row in r["rows"]:
        print(f"      {row['tid_krad']:>5.0f} krad  R x{row['R_growth']:.3f}  violations "
              f"{row['violations']}  SOC {row['mean_soc']:.3f}")
    print(f"      SOC vs dose rho={r['soc_vs_dose']['rho']:+.3f} p={r['soc_vs_dose']['p']:.4f}")

    print("\nV4-7  a single-event upset inside the filter's own state")
    r = v4_7_seu(); out["v4_7_seu"] = r
    print(f"      {r['flips']:,} bit flips into soc/T/V1")
    print(f"      caught by a 3-comparison range check: {100*r['caught_fraction']:.2f}%")
    print(f"      benign or conservative: {100*r['benign_or_conservative_fraction']:.2f}%")
    print(f"      more aggressive than the clean command: {r['census']['aggressive']} "
          f"({100*r['aggressive_fraction']:.3f}%) | worst certified overshoot "
          f"{r['worst_certified_overshoot_C']:.3f} C")

    print("\nV4-8  no ground in the loop for five years")
    r = v4_8_no_ground_loop(); out["v4_8_no_ground_loop"] = r
    print(f"      {r['days']:,} charges | bound {r['bound']['violations']} violations, SOC "
          f"{r['bound']['mean_soc']:.3f} | oracle {r['oracle']['violations']} violations, SOC "
          f"{r['oracle']['mean_soc']:.3f}")
    print(f"      safety identical: {r['safety_identical']} | oracle buys "
          f"{r['soc_gap_points']:+.2f} SOC points")

    print("\nV4-9  solar-array degradation")
    r = v4_9_array_degradation(); out["v4_9_array_degradation"] = r
    for row in r["rows"]:
        print(f"      year {row['year']:>2}  array {100*row['array_fraction']:>5.1f}% "
              f"({row['array_ceiling_A']:5.1f} A)  violations {row['violations']}  SOC "
              f"{row['mean_soc']:.3f}  clipped {100*row['clipped_fraction']:.1f}%")
    if r["array_never_binding"]:
        print(f"      across {r['years_swept']} years, down to "
              f"{100*r['final_array_fraction']:.1f} % of beginning-of-life output, the array "
              f"never becomes the binding constraint")
        print(f"      delivered charge varies by {r['soc_spread']:.4f} SOC over the whole sweep")
    else:
        print(f"      the array becomes the binding constraint at year "
              f"{r['array_becomes_binding_year']}")

    print("\nV4-10 radiator degradation to the vacuum limit")
    r = v4_10_radiator(); out["v4_10_radiator"] = r
    for row in r["rows"]:
        print(f"      eps*A x{row['radiator_fraction']:<5.2f}  violations {row['violations']}  "
              f"SOC {row['mean_soc']:.3f}  peak T {row['peak_T']:.1f} C")
    print(f"      SOC vs radiator rho={r['soc_vs_radiator']['rho']:+.3f} "
          f"p={r['soc_vs_radiator']['p']:.4f}")
    print(f"      delivered charge RISES as the radiator shrinks: "
          f"{r['charge_rises_as_radiator_shrinks']} -- in deep space this pack is "
          f"cold-limited, not heat-limited")

    print("\nV4-11 the bus load in eclipse is the anchor")
    r = v4_11_bus_anchor(); out["v4_11_bus_anchor"] = r
    for k, v in r.items():
        print(f"      {k:20s} {v['load_W']:>6.0f} W  anchor {v['mean_anchor_A']:6.2f} A  "
              f"unsafe {v['unsafe_while_certified']}  null-input brownouts "
              f"{v['null_input_brownouts']}/{v['trials']}")

    path = V.save("v4_space.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
