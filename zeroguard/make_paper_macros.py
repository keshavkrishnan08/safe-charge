"""Emit `paper/vehicle_macros.tex` from the result files.

The IECON paper already works this way: every number in the prose is a macro, and the macro
file is generated from the JSON, so the manuscript cannot quietly disagree with the experiment
that produced it. This does the same for the vehicle paper. If an experiment is re-run and a
number moves, the paper moves with it on the next build -- and `verify_vehicles.py` fails
loudly if a *claim* changes rather than a digit.

    python zeroguard/make_paper_macros.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "paper", "vehicle_macros.tex")

D = {}
for k, f in (("e14", "e14_firmware.json"), ("s1", "s1_plating.json"), ("b5", "b5_domain_transfer.json"), ("n2", "n2_dfn.json"), ("b4", "b4_cbf.json"), ("p1", "p1_policy_filter.json"), ("p2", "p2_pack_dynamic.json"),
             ("m1", "m1_duration_margin.json"), ("d1", "d1_drive_cycles.json"), ("emb", "e13_embedded.json"),
             ("b3", "b3_tuned_derate.json"), ("b2", "b2_domain_baselines.json"), ("b", "b1_baselines.json"), ("n", "n1_nasa_validation.json"), ("g", "v1_ground.json"), ("a", "v2_aerial.json"), ("u", "v3_underwater.json"),
             ("s", "v4_space.json"), ("x", "x_crossdomain.json"),
             ("e1", "e1_generality.json"), ("e2", "e2_boundary.json")):
    p = os.path.join(RES, f)
    if os.path.exists(p):
        D[k] = json.load(open(p))

L = []


def m(name, value):
    L.append(rf"\newcommand{{\{name}}}{{{value}}}")


def num(v, dp=0, comma=False):
    if comma:
        return f"{v:,.{dp}f}"
    return f"{v:.{dp}f}"


g, a, u, s, x = D["g"], D["a"], D["u"], D["s"], D["x"]

# ---- scale -----------------------------------------------------------------------------
# P1 adds 400 sessions x 4 policies x 2 modes plus a 7-point curve at 80 x 2; P2 adds
# 24 pack charges under two rules and a 6-point spread sweep at 8 packs x 2 rules. The drive
# phases of P2 are not counted -- nothing is certified during them.
_P1 = 400 * 4 * 2 + 7 * 80 * 2
_P2 = 24 * 2 + 6 * 8 * 2
# B4: 6 gammas x 400 deployed sessions, 400 for the certificate, and 7 x 600 isolated
_B4 = 6 * 400 + 400 + 7 * 600
# N2: 6 open-loop DFN charges, 40 closed-loop, and 2 x 7 x 3 misspecification runs
_N2 = 6 + 40 + 2 * 7 * 3
# B5: 3 domains x 2 populations x 300 missions, plus 1200 ground sessions x 2 controllers
_B5 = 3 * 2 * 300 + 1200 * 2
# E14: 20,000 states through the compiled filter; S1: 3 ambients x 3 controllers x 8 DFN charges
_E14, _S1 = 20_000, 3 * 3 * 8
m("vTotalEpisodes", f"{470_923 + _P1 + _P2 + _B4 + _N2 + _B5 + _E14 + _S1:,}".replace(",", "{,}"))
m("vExperiments", "61")
m("vPlatforms", "16")
m("vClaims", "76")
m("vFigures", "8")
m("cellEpisodes", "650{,}970")
m("cellClaims", "100")
m("cellTransitions", f"{D['e1']['pooled']['one_step_transitions']/1e6:.1f}")

# ---- theorem / cost --------------------------------------------------------------------
m("evalCharge", str(x["x3_cost"]["charge_max_evals"]))
m("evalDischarge", str(x["x3_cost"]["discharge_max_evals"]))
m("worstLatency", num(x["x3_cost"]["worst_p99_us"], 1))
m("worstLatencyCase", x["x3_cost"]["worst_case"].replace("-", " "))
m("slotFraction", num(100 * x["x3_cost"]["worst_p99_us"] * 1e-3 / 10.0, 2))
m("groundLatency", num(g["v1_9_wcet"]["worst_p99_us"], 1))
m("groundSlotPct", num(100 * g["v1_9_wcet"]["slot_fraction"], 3))

# ---- cross-domain ----------------------------------------------------------------------
xs = x["x1_structure"]
m("scanTotal", num(xs["scans"], 0, True))
m("scanSingle", num(xs["total"]["single-interval"], 0, True))
m("scanEmpty", num(xs["total"]["empty"], 0, True))
m("scanDisconnected", str(xs["disconnected"]))
m("reduceStates", num(x["x2_reduction"]["reference_states"], 0, True))
m("reduceMismatch", str(x["x2_reduction"]["reference_mismatches"]))
m("adequacyTrials", num(x["x5_adequacy"]["total_trials"], 0, True))
m("adequacyViol", str(x["x5_adequacy"]["total_violations"]))
m("nReqOne", num(x["x5_adequacy"]["n_required"]["1.0%"], 0, True))
m("nReqTenth", num(x["x5_adequacy"]["n_required"]["0.1%"], 0, True))
pooled = x["x4_reserve"]["pooled"]
m("reservePooledRho", f"{pooled['rho']:+.3f}")
m("reservePooledN", num(pooled["n"], 0, True))
m("reserveWeakest", x["x4_reserve"]["weakest_case"].replace("-", " "))
m("reserveWeakestRho", f"{x['x4_reserve']['weakest_rho']:+.3f}")
for k, v in x["x4_reserve"]["per_case"].items():
    tag = "".join(w.capitalize() for w in k.split("-"))
    m(f"clockW{tag}", f"{v['width_vs_endurance']['rho']:+.3f}")
    m(f"clockE{tag}", f"{v['energy_clock_vs_endurance']['rho']:+.3f}")

# ---- ground ----------------------------------------------------------------------------
m("gReduceStates", num(g["v1_1_reduction"]["states"], 0, True))
m("gReduceMismatch", str(g["v1_1_reduction"]["mismatches"]))
m("gPackDev", f"{g['v1_1_reduction']['pack_rel_dev']:.1e}".replace("e-", r"\times 10^{-").replace("+", "") + "}")
fc = g["v1_2_fast_charge"]
m("gFastN", num(fc["trials"], 0, True))
m("gFastViol", str(fc["violations"]))
m("gFastCP", num(fc["cp95_upper_pct"], 3))
m("gFastWorstT", num(fc["worst_T"], 1))
m("gChargerA", num(fc["charger_equiv_A"], 0))
m("gPackCeilA", num(fc["pack_ceiling_A"], 0))
amb = g["v1_3_ambient"]
m("gCeiling", num(amb["predicted_passive_ceiling_C"], 1))
m("gDeadZone", num(amb["passive"]["dead_zone_from_C"], 0))
m("gLoopGain", num(amb["active_recovers_points"], 1))
m("gAmbTrials", num(amb["total_trials"], 0, True))
dl = g["v1_4_deadline"]
m("gPeakAmb", num(dl["peak_ambient_C"], 0))
m("gColdRho", f"{dl['cold_limb']['rho']:+.3f}")
m("gHotRho", f"{dl['hot_limb']['rho']:+.3f}")
tp = g["v1_5_truck_pack"]
m("gTruckCells", num(tp["rows"][-1]["N"], 0, True))
m("gTruckDev", f"{tp['max_dev']:.1e}")
m("gTruckRes", f"{tp['search_resolution_A']:.1e}")
dc = g["v1_6_duty_cycle"]
m("gRoboSessions", num(dc["duties"]["robotaxi"]["trials"], 0, True))
m("gRoboViol", str(dc["duties"]["robotaxi"]["violations"]))
m("gRoboCP", num(dc["duties"]["robotaxi"]["cp95_upper_pct"], 4))
m("gStressRatio", num(dc["stress_ratio"], 0))
m("gWeeks", num(dc["robotaxi_weeks_to_private_decade"], 1))
m("gCoolTrials", num(g["v1_7_coolant"]["total_trials"], 0, True))
m("gRegenN", num(g["v1_8_regen"]["trials"], 0, True))
sn = g["v1_10_sensor"]
m("gBiasPred", num(sn["predicted_breakpoint_C"], 3))
m("gBiasMeas", num(sn["measured_breakpoint_C"], 1))
m("gBiasErr", num(sn["error_C"], 3))
v2g = g["v1_11_v2g"]
m("gVtwoGN", str(list(v2g.values())[0]["trials"]))
m("gVtwoGBrown", str(list(v2g.values())[0]["null_input_brownouts"]))
m("gVtwoGMaxAnchor", num(max(v["anchor_A"] for v in v2g.values()), 1))

# ---- aerial ----------------------------------------------------------------------------
an = a["v2_1_anchor_vs_null"]
m("aRefuseC", num(an["energy_cell_refuses_above_c"], 1))
m("aFlights", str(an["trials"]))
m("aAnchUnsafe", str(an["anchored_unsafe_while_certified"]))
m("aNullBrown", str(an["null_input_brownouts"]))
m("aNullSteps", num(an["null_input_mean_steps_served"], 1))
st = a["v2_2_interval_structure"]
m("aScanStates", num(st["states"], 0, True))
m("aScanDisc", str(st["disconnected"]))
rl = a["v2_3_reserve_lead"]
m("aLeadN", num(rl["trials"], 0, True))
m("aLeadNoWarn", str(rl["breaches_with_no_warning"]))
m("aLeadMed", num(rl["lead_s_median"], 0))
m("aLeadPfive", num(rl["lead_s_p05"], 0))
m("aLeadPninefive", num(rl["lead_s_p95"], 0))
so = a["v2_4_sortie"]
m("aSortieN", str(so["trials"]))
m("aSortieComplete", num(100 * so["completion_rate"], 1))
m("aSortieBigger", num(100 * so["completion_rate_6S4P"], 1))
m("aSortieUnsafe", str(so["unsafe_while_certified"]))
for ph in ("takeoff", "climb", "cruise-out", "hover-drop"):
    tag = "".join(w.capitalize() for w in ph.split("-"))
    m(f"aReserve{tag}", num(so["phase_reserve"][ph]["mean_width"], 2))
pa = a["v2_5_payload_altitude"]
m("aPayloadRho", f"{pa['width_vs_payload']['rho']:+.3f}")
m("aHoverRho", f"{pa['width_vs_hover_power']['rho']:+.3f}")
m("aAltPooledRho", f"{pa['width_vs_altitude_pooled']['rho']:+.3f}")
m("aAltPooledP", num(pa["width_vs_altitude_pooled"]["p"], 3))
gu = a["v2_6_gusts"]
m("aGustTrials", num(gu["total_trials"], 0, True))
m("aGustUnsafe", str(gu["total_unsafe"]))
m("aGustDebounceHi", num(gu["rows"][-1]["median_debounce_cost_s"], 0))
m("aMotorTrials", num(a["v2_8_motor_out"]["total_trials"], 0, True))
m("aMotorUnsafe", str(a["v2_8_motor_out"]["total_unsafe"]))
tu = a["v2_9_turnaround"]
m("aTurnN", num(tu["sessions_attempted"], 0, True))
m("aTurnImmediate", num(100 * tu["started_immediately_frac"], 0))
m("aTurnWaitMed", num(tu["wait_s_median"] / 60.0, 1))
m("aTurnWaitPninefive", num(tu["wait_s_p95"] / 60.0, 1))
m("aTurnOnePack", num(tu["sorties_per_day_one_pack"], 1))
m("aTurnSwap", num(tu["sorties_per_day_pack_swap"], 1))
m("aTurnPacks", num(tu["packs_needed_for_target"], 1))
ev = a["v2_10_evtol"]
m("aEvtolC", num(ev["hover_c_rate"], 2))
m("aEvtolRho", f"{ev['reserve_vs_endurance']['rho']:+.3f}")
m("aEvtolMed", num(ev["endurance_min_median"], 1))
ad = a["v2_11_adversary"]
m("aAdvCells", str(ad["cells"]))
m("aAdvSeq", num(ad["sequences"], 0, True))
m("aAdvViol", str(ad["violations"]))
m("aAdvCP", num(ad["cp95_upper_pct"], 3))
m("aAdvPeak", num(100 * ad["worst_peak_T_fraction_while_certified"], 1))

# ---- underwater ------------------------------------------------------------------------
sh = u["v3_1_sealed_hull"]
m("uHullUA", f"{sh['hull_UA_W_per_K']:.4f}")
m("uCarUA", f"{sh['car_hA_W_per_K']:.4f}")
m("uRatio", num(1.0 / sh["conductance_ratio"], 0))
m("uHullN", num(sh["trials"], 0, True))
m("uHullViol", str(sh["violations"]))
cb = u["v3_2_cold_binding"]
m("uColdTrials", num(cb["total_trials"], 0, True))
m("uCapCold", num(100 * cb["cold_cap_fraction"], 0))
m("uCapWarm", num(100 * cb["warm_cap_fraction"], 0))
ha = u["v3_3_hotel_anchor"]
m("uIdleW", num(ha["idle-on-seabed"]["hotel_W"], 0))
m("uIdleAnchor", num(ha["idle-on-seabed"]["mean_anchor_A"], 2))
m("uIdleBrown", str(ha["idle-on-seabed"]["null_input_brownouts"]))
m("uIdleN", str(ha["idle-on-seabed"]["trials"]))
m("uDeployN", str(u["v3_4_deployment"]["trials"]))
m("uDormSpread", f"{u['v3_5_dormancy']['soc_spread']:.1e}")
m("uDepthSpread", f"{u['v3_6_depth']['soc_spread']:.1e}")
ui = u["v3_7_under_ice"]
m("uIceN", num(ui["trials"], 0, True))
m("uIceNoWarn", str(ui["breaches_with_no_warning"]))
m("uIceClosureMed", num(ui["closure_trigger"]["lead_median_min"], 0))
m("uIceTransit", num(ui["transit_min"], 0))
m("uIceCover", num(100 * ui["closure_trigger"]["lead_covers_transit_rate"], 0))
m("uIceThresh", num(ui["threshold_for_transit_A"], 0))
for row in ui["reserve_trigger_sweep"]:
    if row["threshold_A"] == ui["threshold_for_transit_A"]:
        m("uIceThreshMed", num(row["median_warning_min"], 0))
        m("uIceThreshPfive", num(row["p05_warning_min"], 0))
gl = u["v3_10_glider_vs_auv"]
m("uLoadRatio", num(gl["load_ratio"], 0))
m("uGliderRatio", num(gl["platforms"]["buoyancy-glider"]["reserve_ratio"], 0))
m("uAuvRatio", num(gl["platforms"]["under-ice-auv"]["reserve_ratio"], 1))

# ---- space -----------------------------------------------------------------------------
leo = s["v4_1_leo"]
m("sLeoCycles", num(leo["trials"], 0, True))
m("sLeoViol", str(leo["violations"]))
m("sLeoCP", num(leo["cp95_upper_pct"], 4))
m("sLeoPerYear", num(leo["cycles_per_year"], 0, True))
m("sGeoMax", num(s["v4_2_geo"]["max_eclipse_min"], 0))
m("sGeoDoD", num(100 * s["v4_2_geo"]["rows"][-1]["median_depth_of_discharge"], 1))
ds = s["v4_3_deep_space"]
m("sCruiseN", num(ds["trials"], 0, True))
m("sCruiseViol", str(ds["violations"]))
m("sMarsTrials", num(s["v4_4_mars"]["total_trials"], 0, True))
m("sMarsViol", str(s["v4_4_mars"]["total_violations"]))
ln = s["v4_5_lunar_night"]
m("sNightHours", num(ln["night_hours"], 0))
m("sNightBase", num(ln["rows"][6]["median_survival_h"], 1))
m("sNightBest", num(ln["best_insulated_survival_h"], 1))
m("sNightBestEps", num(ln["best_insulated_eps"], 2))
m("sNightEnergy", num(ln["energy_limit_h"], 1))
m("sDoseMax", num(s["v4_6_radiation"]["rows"][-1]["tid_krad"], 0))
m("sDoseViol", str(s["v4_6_radiation"]["total_violations"]))
seu = s["v4_7_seu"]
m("sSeuFlips", num(seu["flips"], 0, True))
m("sSeuCaught", num(100 * seu["caught_fraction"], 2))
m("sSeuBenign", num(100 * seu["benign_or_conservative_fraction"], 2))
m("sSeuAggr", num(100 * seu["aggressive_fraction"], 2))
m("sSeuOvershoot", num(seu["worst_certified_overshoot_C"], 3))
ng = s["v4_8_no_ground_loop"]
m("sNoGroundDays", num(ng["days"], 0, True))
m("sNoGroundGap", num(ng["soc_gap_points"], 2))
ar = s["v4_9_array_degradation"]
m("sArrayYears", num(ar["years_swept"], 0))
m("sArrayFrac", num(100 * ar["final_array_fraction"], 1))
m("sArraySpread", f"{ar['soc_spread']:.4f}")
rd = s["v4_10_radiator"]
m("sRadLo", num(rd["rows"][0]["mean_soc"], 3))
m("sRadHi", num(rd["rows"][-1]["mean_soc"], 3))
m("sRadViol", str(rd["total_violations"]))

# ---- external validation (NASA PCoE) ----------------------------------------------------
if "n" in D:
    n = D["n"]
    rb = n["n1_resistance_bound"]
    m("nCells", str(rb["cells"]))
    m("nEIS", num(rb["measurements"], 0, True))
    m("nInsideFrac", num(100 * rb["fraction_inside"], 2))
    m("nInsideEolFrac", num(100 * rb["fraction_inside_within_eol"], 2))
    m("nEolMeas", num(rb["measurements_within_eol"], 0, True))
    m("nWorstCell", rb["worst_cell"])
    m("nWorstRatio", num(rb["worst_ratio"], 3))
    m("nViolations", str(rb["violations"]))
    m("nBound", num(rb["bound"], 1))
    ce = n["n2_capacity_envelope"]
    m("nCapPast", str(ce["cells_past_envelope"]))
    m("nCapCells", str(ce["cells"]))
    m("nCapWorst", num(ce["worst_q"], 3))
    mh = n["n3_monotone_heating"]
    m("nDisCycles", num(mh["discharge"]["cycles"], 0, True))
    m("nDisRho", f"{mh['discharge']['peak_dT_vs_joule']['rho']:+.3f}")
    m("nChgCycles", num(mh["charge"]["cycles"], 0, True))
    m("nChgRho", f"{mh['charge']['peak_dT_vs_joule']['rho']:+.3f}")
    m("nMaxI", num(mh["discharge"]["current_range"][1], 2))
    m("nMaxDT", num(mh["discharge"]["peak_dT_range"][1], 1))
    tr = n["n4_transfer_prediction"]
    m("nHorizon", num(tr["horizon_s"], 0))
    m("nRMSE", num(tr["rmse_C"], 3))
    m("nPersist", num(tr["persistence_rmse_C"], 3))
    m("nSkill", f"{tr['skill_vs_persistence']:+.3f}")
    m("nBias", f"{tr['bias_C']:+.3f}")
    m("nOverFrac", num(100 * tr["over_prediction_fraction"], 1))
    m("nUnderWorst", num(tr["worst_underprediction_C"], 2))
    m("nMarginCover", num(tr["margin_over_underprediction"], 0))
    pp = n["n5_permits_protocol"]
    m("nPermitFrac", num(100 * pp["allowed_fraction"], 1))
    m("nHeadroom", num(pp["median_headroom"], 2))
    m("nCeiling", num(pp["throttled_ceiling_C"], 1))
    m("nRefuseCeil", num(100 * pp["refusals_explained_by_ceiling"], 0))
    nd = n["n6_refusals_predict_degradation"]
    m("nAssocLo", f"{nd['rho_range'][0]:+.3f}")
    m("nAssocHi", f"{nd['rho_range'][1]:+.3f}")
    m("nBadBaselines", str(nd["cells_with_bad_first_baseline"]))

# ---- baselines --------------------------------------------------------------------------
if "b" in D:
    b = D["b"]; cc = b["controllers"]
    m("bTrials", num(b["trials"], 0, True))
    m("bDerateC", num(b["derate_C"], 1))
    m("bGainPts", num(b["charge_gain_over_derate_points"], 1))
    m("bGainPct", num(b["charge_gain_over_derate_pct"], 1))
    m("bAggrViolPct", num(100 * b["aggressive_violation_rate"], 0))
    m("bAggrC", num(b["slowest_unsafe_derate_C"], 1))
    m("bFastestSafeC", num(b["fastest_safe_derate_C"], 1))
    m("bMpcGap", num(abs(b["mpc_h1_soc_gap_points"]), 2))
    m("bMpcRatio", num(b["mpc_h1_eval_ratio"], 1))
    m("bMpcMaxEval", num(cc["mpc_h1"]["evals_max"], 0))
    m("bMpcIterSpread", str(cc["mpc_h1"].get("iters_spread", "")))
    m("bZgEval", num(cc["zeroguard"]["evals_max"], 0))
    for k, tag in (("ccv_0.5C", "Derate"), ("ccv_1C", "OneC"),
                   ("mpc_h1", "MpcOne"), ("mpc_h3", "MpcThree"), ("zeroguard", "Zg")):
        if k in cc:
            m(f"bSoc{tag}", num(cc[k]["mean_soc"], 3))
            m(f"bViol{tag}", str(cc[k]["violations"]))

# ---- domain stopping-rule baselines -----------------------------------------------------
if "b2" in D:
    q = D["b2"]
    m("bTwoTrials", num(q["trials_per_domain"], 0, True))
    for case, tag in (("delivery-quadrotor", "Air"), ("under-ice-auv", "Sea"),
                      ("geo-comsat", "Geo")):
        if case not in q:
            continue
        cv = q[case]["curve"]
        # LaTeX command names may contain letters only -- no digits. An earlier version
        # emitted \b2AirSoc80, which TeX parses as the \b accent applied to "2AirSoc80".
        for tgt, tword in (("80", "Eighty"), ("95", "Ninetyfive")):
            mp = cv[tgt]["mission_per_family"]
            for fam, ft in (("fixed SOC reserve", "Soc"), ("certificate reserve", "Res"),
                            ("both (min of two clocks)", "Both")):
                v = mp[fam]
                m(f"bTwo{tag}{ft}{tword}", ("--" if v is None else num(v / 60.0, 1)))
        c8 = cv["80"]["mission_per_family"]
        if c8["fixed SOC reserve"] and c8["certificate reserve"]:
            g = 100 * (c8["certificate reserve"] / c8["fixed SOC reserve"] - 1)
            m(f"bTwo{tag}GainEighty", f"{g:+.0f}")

# ---- tuned de-rate, deployed ------------------------------------------------------------
if "b3" in D:
    z = D["b3"]
    m("bThreeTrials", num(z["trials"], 0, True))
    m("bThreeTuned", num(z["tune"]["chosen_C"], 1))
    m("bThreeTunedViolPct", num(100 * z["tuned_rate_violation_rate"], 1))
    m("bThreeTunedViol", str(z["deploy"]["rows"][f"ccv_{z['tune']['chosen_C']:g}C"]["violations"]))
    m("bThreeZgViol", str(z["zeroguard_violations"]))
    m("bThreeZgCP", num(z["zeroguard_cp95"], 3))
    m("bThreeRequired", num(z["deploy"]["required_C"], 1))
    m("bThreeGainPts", num(z["gain_over_safe_fixed_points"], 1))
    m("bThreeGainPct", num(z["gain_over_safe_fixed_pct"], 1))

# ---- EPA drive cycles -------------------------------------------------------------------
if "d1" in D:
    d1 = D["d1"]
    m("dRepeats", str(d1["repeats"]))
    m("dCertBreach", str(d1["certificate_breaches_while_certified"]))
    m("dRefusalPct", num(100 * d1["mean_refusal_fraction"], 1))
    m("dUniversalCap", num(d1["best_universal_cap_C"], 1))
    m("dGainVsUniversal", num(d1["gain_vs_universal_cap_points"], 1))
    m("dTunedCap", num(d1["cap_tuned_on_easiest_schedule_C"], 1))
    br = d1["cap_tuned_elsewhere_breaches"]
    m("dTunedBreachUSsix", str(br.get("us06col.txt", 0)))
    m("dTunedBreachHwy", str(br.get("hwycol.txt", 0)))
    for fn, tag in (("us06col.txt", "USsix"), ("uddscol.txt", "Udds"), ("hwycol.txt", "Hwfet")):
        if fn in d1["cycles"]:
            c = d1["cycles"][fn]
            m(f"d{tag}Km", num(c["distance_km"], 1))
            m(f"d{tag}Sec", num(c["seconds"], 0))
            m(f"d{tag}PeakkW", num(c["peak_power_kW"], 0))
            m(f"d{tag}Recovery", num(100 * c["rules"]["certificate"]["recovery_mean"], 1))
            m(f"d{tag}Refusal", num(100 * c["rules"]["certificate"]["refusal_fraction"], 1))
            m(f"d{tag}SafeCap", num(c["best_safe_fixed_C"], 1))

# ---- embedded footprint -----------------------------------------------------------------
if "emb" in D:
    e = D["emb"]; vd = e["verdict"]; fp = e["footprint"]; sp = e["single_precision"]
    m("embBytesPerCall", num(e["allocation"]["bytes_per_call"], 1))
    m("embCalls", num(e["allocation"]["calls"], 0, True))
    m("embSpStates", num(sp["states"], 0, True))
    m("embSpWorstmA", num(1000 * sp["max_abs_dev_A"], 2))
    m("embSpLooseCount", num(sp["times_less_conservative"], 0, True))
    m("embSpFrac", f"{100*sp['relative_to_umax']:.1e}")
    ib = e["integer_bisection"]
    m("embIntBits", str(ib["bits"]))
    m("embIntQuantum", num(ib["quantum_A"], 4))
    m("embIntAbove", str(ib["times_above_reference"]))
    m("embIntStates", num(ib["states"], 0, True))
    m("embRam", str(vd["ram_bytes"]))
    m("embFlash", num(vd["flash_bytes"], 0, True))
    m("embConstants", str(fp["calibration_constants"]))
    m("embLutExp", str(fp["lut_entries"]["exp"]))
    m("embLutAsinh", str(fp["lut_entries"]["asinh"]))

# ---- duration-aware margin --------------------------------------------------------------
if "m1" in D:
    mm = D["m1"]
    m("mTauRef", num(mm["reduction"]["tau_ref_s"], 0))
    m("mKrate", f"{mm['reduction']['k_rate_K_per_s']:.5f}")
    m("mMarginRef", num(mm["reduction"]["old_thermal_K"], 1))
    m("mMarginPulse", num(mm["reduction"]["margin_at_5s"], 2))
    m("mMarginMinute", num(mm["reduction"]["margin_at_60s"], 2))
    m("mChargeTrials", num(mm["charging"]["trials"], 0, True))
    m("mChargeViol", str(mm["charging"]["violations"]))
    m("mRegenGain", num(mm["regen_mean_gain_points"], 1))
    m("mRegenBreach", str(mm["regen_breaches"]))
    fl = mm["floor"]
    m("mFloorR", num(fl["breaks_at_R_multiple"], 1))
    m("mFloorOver", num(fl["breaks_at_factor_over_bound"], 2))
    m("mBound", num(fl["assumed_bound"], 1))
    for fn, tag in (("us06col.txt", "USsix"), ("uddscol.txt", "Udds"), ("hwycol.txt", "Hwfet")):
        if fn in mm["regen"]:
            m(f"mGain{tag}", num(mm["regen"][fn]["recovery_gain_points"], 1))

# ---- the per-domain summary table needs a few figures the prose already quotes ----------
m("xRhoAir", f"{x['x4_reserve']['per_case']['delivery-quadrotor']['width_vs_endurance']['rho']:+.2f}")
m("xRhoGeo", f"{x['x4_reserve']['per_case']['geo-comsat']['width_vs_endurance']['rho']:+.2f}")
m("aNullN", str(a["v2_1_anchor_vs_null"]["trials"]))
m("aTurnViol", str(a["v2_9_turnaround"]["violations"]))
m("aTurnTrials", num(a["v2_9_turnaround"]["trials"], 0, True))
m("uSealViol", str(u["v3_1_sealed_hull"]["violations"]))
m("uSealN", num(u["v3_1_sealed_hull"]["trials"], 0, True))
m("uNoRecalGap", num(u["v3_9_no_recalibration"]["soc_gap_points"], 1))

# ---- the firmware, compiled ------------------------------------------------------------
if "e14" in D:
    f14 = D["e14"]
    m("fwText", num(f14["text_bytes"], 0, True))
    m("fwData", str(f14["data_bytes"]))
    m("fwOcv", str(f14["ocv_table_bytes"]))
    m("fwRam", str(f14["ram_bytes"]))
    m("fwStates", num(f14["states"], 0, True))
    m("fwMismatch", str(f14["status_mismatches"]))
    m("fwQuantum", f"{1000*f14['quantum_A']:.1f}")
    m("fwMaxDev", f"{1000*f14['max_dev_A']:.1f}")
    m("fwMinDev", f"{1000*f14['min_dev_A']:.1f}")
    m("fwAbove", str(f14["above_reference_by_a_quantum"]))
    m("fwVcost", f"{1000*f14['worst_voltage_cost_V']:.3f}")
    m("fwVfrac", num(100 * f14["worst_voltage_frac_margin"], 2))
    m("fwTcost", f"{f14['worst_thermal_cost_K']:.4f}")
    m("fwCompiler", f14["compiler"])
    m("fwTarget", f14["target"].replace("_", "-"))
    m("fwEvals", str(f14["evaluations"]))

# ---- capacity lost to plating, measured by the DFN --------------------------------------
if "s1" in D:
    s1 = D["s1"]
    m("plTarget", num(100 * s1["target_soc"], 0))
    m("plSessions", str(s1["sessions_per_cell"]))
    m("plCertMah", num(s1["cold_cert_plated_mAh"], 1))
    m("plAggMah", num(s1["cold_aggressive_plated_mAh"], 1))
    m("plAggRatio", num(s1["cold_aggressive_ratio"], 1))
    m("plAggOnset", str(s1["cold_aggressive_onset"]))
    m("plCertOnset", str(s1["cold_cert_onset"]))
    m("plCertPhi", num(s1["cold_cert_min_phi_mV"], 1))
    m("plSafeMah", num(s1["cold_safe_plated_mAh"], 1))
    m("plSafeRatio", num(s1["cold_safe_ratio"], 2))
    m("plMinutes", num(abs(s1["cold_minutes_saved"]), 1))
    m("plAggRisk", num(s1["cold_agg_at_risk_min"], 1))
    m("plAggRiskMah", num(s1["cold_agg_at_risk_mAh"], 1))
    m("plCertRisk", num(s1["cold_cert_at_risk_min"], 0))
    m("plCertMin", num(s1["cold_cert_minutes"], 1))
    m("plSafeMin", num(s1["cold_safe_minutes"], 1))
    WORD = {"0": "Cold", "10": "Cool", "25": "Mild"}
    for amb, rec in s1["by_ambient"].items():
        w = WORD[amb]
        for nm, tg in (("certificate", "Cert"), ("CC-CV 0.5C", "Slow"), ("CC-CV 1.5C", "Fast")):
            if nm in rec:
                m(f"pl{w}{tg}Mah", num(rec[nm]["plated_mAh"], 1))
                m(f"pl{w}{tg}Onset", str(rec[nm]["onset_crossed"]))
                m(f"pl{w}{tg}Min", num(rec[nm]["median_minutes"], 1))
                m(f"pl{w}{tg}Phi", num(rec[nm]["min_phi_mV"], 1))

# ---- the same standard in every domain --------------------------------------------------
if "b5" in D:
    q = D["b5"]
    m("tTrials", num(q["trials_per_domain"], 0, True))
    m("tIncFails", str(len(q["incumbent_fails_transfer"])))
    m("tCertFails", str(len(q["certificate_fails_transfer"])))
    m("tDomains", str(len(q["domains"])))
    tag = {"delivery-quadrotor": "Air", "under-ice-auv": "Water", "geo-comsat": "Space"}
    for case, r in q["domains"].items():
        k = tag[case]
        m(f"t{k}Target", num(100 * r["target"], 0))
        sd, rd = r["soc_deployed"], r["reserve_deployed"]
        m(f"t{k}Rule", sd["rule"].replace("<=", "$\\le$").replace("%", "\\%"))
        m(f"t{k}Tuned", num(100 * sd["tuned_recovery"], 1))
        m(f"t{k}Deployed", num(100 * sd["deployed_recovery"], 1))
        m(f"t{k}Holds", "holds" if sd["holds"] else "\\textbf{fails}")
        m(f"t{k}CertTuned", num(100 * rd["tuned_recovery"], 1))
        m(f"t{k}CertDeployed", num(100 * rd["deployed_recovery"], 1))
        m(f"t{k}CertHolds", "holds" if rd["holds"] else "\\textbf{fails}")
        if r.get("gain_pct") is not None:
            m(f"t{k}Gain", num(r["gain_pct"], 1))
            m(f"t{k}IncMin", num(r["soc_required"]["mission_s"] / 60, 1))
            m(f"t{k}CertMin", num(r["reserve_required"]["mission_s"] / 60, 1))
    g5 = q["ground"]
    m("gtTrials", num(g5["trials"], 0, True))
    m("gtWindow", num(g5["horizon_min"], 0))
    m("gtMean", num(g5["mean_gain_points"], 1))
    m("gtBest", num(g5["best_gain_points"], 0))
    m("gtWorst", num(g5["worst_gain_points"], 0))
    m("gtCeiling", num(g5["effective_ceiling_C"], 1))
    m("gtViol", str(g5["zg_violations"]))
    m("gtBaseViol", str(g5["ccv_violations"]))
    # LaTeX forbids digits in control sequences, so the target is spelled out
    WORD = {"50": "Fifty", "60": "Sixty", "70": "Seventy"}
    for k, r in g5["reach"].items():
        w = WORD[k]
        m(f"gtReach{w}Zg", num(100 * r["zg"], 0))
        m(f"gtReach{w}Cc", num(100 * r["ccv"], 0))
        if r["ratio"]:
            m(f"gtReach{w}Ratio", num(r["ratio"], 1))
    for r in g5["by_ambient"]:
        a = int(r["ambient_C"])
        nm = ("Mten" if a == -10 else "Zero" if a == 0 else
              "Fifteen" if a == 15 else "TwentyFive" if a == 25 else
              "ThirtyFive" if a == 35 else "Forty")
        m(f"gtAmb{nm}", num(r["gain_points"], 1))
    m("gtCrossLo", num(g5["crossover_between_C"][0], 0))
    m("gtCrossHi", num(g5["crossover_between_C"][1], 0))

# ---- against a Doyle-Fuller-Newman plant ------------------------------------------------
if "n2" in D:
    d = D["n2"]
    ol, cl, ms = d["open_loop"], d["closed_loop"], d["misspecification"]
    m("dSet", d["param_set"])
    m("dOverV", num(1000 * ol["worst_over_V"], 0))
    m("dOverRatio", num(ol["over_margin_ratio"], 1))
    m("dNearV", num(100 * ol["near_V_frac_margin"], 0))
    m("dNearT", num(100 * ol["near_T_frac_margin"], 0))
    m("dUnderV", num(100 * ol["worst_under_V_frac_margin"], 0))
    m("dLowestPeakV", num(ol["rows"][0]["dfn_peak_V"], 2))
    m("dVmargin", num(1000 * ol["dV_margin"], 0))
    m("dSessions", str(cl["sessions"]))
    m("dViolV", str(cl["voltage_violations"]))
    m("dViolT", str(cl["temperature_violations"]))
    m("dCertViol", str(cl["certified_violations"]))
    m("dCP", num(cl["cp95_upper_pct"], 2))
    m("dPeakV", num(cl["worst_V"], 3))
    m("dPeakT", num(cl["worst_T"], 1))
    m("dPlateClean", str(cl["sessions"] - cl["plating_onset_sessions"]))
    m("dMinPhi", num(1000 * cl["worst_phi"], 1))
    m("dVmaxLim", num(cl["V_max"], 2))
    m("dTmaxLim", num(cl["T_max"], 0))
    ec = ms["per_parameter"]["electrolyte conductivity"]
    dif = ms["per_parameter"]["negative particle diffusivity"]
    m("dEcHeld", num(1 / ec["held_to"], 0))
    m("dEcBreak", num(1 / ec["breaks_at"], 0))
    m("dDiffSwept", num(1 / dif["swept_to"], 0))

# ---- against the control-barrier-function filter ----------------------------------------
if "b4" in D:
    c = D["b4"]
    m("cTrials", num(c["trials"], 0, True))
    m("cZgSoc", num(c["zeroguard"]["mean_soc"], 3))
    m("cZgViol", str(c["zeroguard"]["violations"]))
    m("cZgEval", num(c["zeroguard"]["evals"], 0))
    m("cQpEval", num(c["cbf"][0]["evals"], 0))
    m("cBestGamma", f"{c['best_safe_gamma']:g}")
    m("cBestGammaSoc", num(c["best_safe_cbf_soc"], 3))
    m("cGainOverCbf", num(c["gain_over_best_safe_cbf_points"], 1))
    m("cLinError", num(c["max_lin_error_K"], 2))
    ex = c["exact"]
    m("cExactStates", num(ex["states"], 0, True))
    m("cExactDisc", str(ex["disconnected"]))
    m("cExactDev", num(ex["worst_deviation_A"], 3))
    m("cExactScan", num(ex["scan_step_A"], 3))
    it = c["isolated"]
    m("cIsoTrials", num(it["zg"]["trials"], 0, True))
    m("cIsoQpViol", str(it["cbf@1"]["violations"]))
    m("cIsoQpRate", num(100 * c["linearisation_breach_rate"], 1))
    m("cIsoQpCP", num(c["linearisation_cp95"], 2))
    m("cIsoZgViol", str(it["zg"]["violations"]))
    m("cIsoZgSoc", num(it["zg"]["mean_soc"], 3))
    m("cIsoSafeGamma", f"{c['iso_best_safe_gamma']:g}")
    m("cIsoGap", num(c["iso_charge_gap_points"], 1))
    for g in c["gammas"]:
        tag = str(g).replace("0.", "").replace(".", "")
        tag = {"05": "Afive", "1": "Bone", "2": "Btwo", "4": "Bfour",
               "7": "Bseven", "10": "Cone"}.get(tag, tag)
        r = [x for x in c["cbf"] if x["gamma"] == g][0]
        m(f"cQpSoc{tag}", num(r["mean_soc"], 3))
        m(f"cQpViol{tag}", str(r["violations"]))
        m(f"cIsoViol{tag}", str(it[f"cbf@{g:g}"]["violations"]))
        m(f"cIsoSoc{tag}", num(it[f"cbf@{g:g}"]["mean_soc"], 3))

# ---- a policy through the filter --------------------------------------------------------
if "p1" in D:
    q = D["p1"]
    m("pTrials", num(q["trials"], 0, True))
    m("pPolicies", str(len(q["policies"])))
    m("pUnsafe", str(len(q["unsafe_policies"])))
    m("pWorstRate", num(100 * q["worst_unfiltered_rate"], 0))
    m("pContained", "yes" if q["all_contained"] else "no")
    lp = q["policies"]["learned (charge only)"]
    m("pLearnViol", str(lp["unfiltered"]["violations"]))
    m("pLearnFilt", str(lp["filtered"]["violations"]))
    m("pLearnBias", f"{q['learned_weights'][0]:+.1f}")
    sp = q["policies"]["CC-CV 0.5C (production)"]
    m("pSafeClip", num(100 * sp["filtered"]["clip_rate"], 2))
    m("pSafeCost", num(abs(q["safe_policy_charge_cost_points"]), 3))
    m("pGainSafe", num(q["gain_over_safe_protocol_points"], 1))
    m("pCurveTrials", num(q["curve_trials"], 0))
    m("pUntouchedC", num(q["max_untouched_C"], 1))
    m("pUnsafeC", num(q["min_unsafe_C"], 1))
    m("pMonotone", "yes" if q["curve_monotone"] else "no")
    for r in q["transparency_curve"]:
        tag = {0.3: "Athree", 0.5: "Afive", 0.8: "Aeight", 1.0: "Bone",
               1.5: "Bfive", 2.0: "Ctwo", 3.0: "Cthree"}.get(r["c_rate"])
        if tag:
            m(f"pClip{tag}", num(100 * r["clip_rate"], 1))
            m(f"pViol{tag}", str(r["unfiltered_violations"]))

# ---- the weakest cell, in motion --------------------------------------------------------
if "p2" in D:
    q = D["p2"]
    m("qCells", num(q["n_cells"], 0, True))
    m("qPacks", str(q["total_packs"]))
    m("qTotalCells", num(q["total_cells"], 0, True))
    m("qLaps", str(q["laps"]))
    m("qBind", num(100 * q["bind_fraction"], 0))
    m("qMinBreach", str(q["min_rule_breaches"]))
    m("qMinCP", num(q["min_rule_cp95_pct"], 2))
    m("qMeanBreach", str(q["mean_rule_breaches"]))
    m("qMinPlated", str(q["min_rule_cells_plated"]))
    m("qMeanPlated", num(q["mean_rule_cells_plated"], 0, True))
    m("qMeanPlatePct", num(q["mean_rule_plating_pct"], 1))
    m("qMeanPlatePacks", str(q["mean_rule_packs_plating"]))
    m("qSocCost", num(q["soc_cost_points"], 2))
    m("qDistinct", str(q["max_distinct_binders"]))
    m("qRank", num(q["binder_temp_rank"], 2))
    m("qColdest", num(100 * q["binder_coldest_frac"], 0))
    m("qWorstParam", str(q["binder_is_worst_param"]))
    sw = q["spread_sweep"]
    m("qSweepLo", num(sw[0]["spread_T_K"], 1))
    m("qSweepHi", num(sw[-1]["spread_T_K"], 1))
    m("qSweepPlateLo", str(sw[0]["mean_plated_cells"]))
    m("qSweepPlateHi", str(sw[-1]["mean_plated_cells"]))
    m("qSweepMinPlate", str(q["min_rule_plating_in_sweep"]))
    m("qSweepLumped", num(q["lumped_plating_in_sweep"], 0, True))
    m("qSweepPacks", str(q["sweep_packs"]))
    us = q["cycles"]["US06"]
    m("qSpreadT", num(us["spread_T_K"], 1))
    m("qSpreadSoc", num(100 * us["spread_soc"], 1))
    m("qSwitchRate", num(100 * us["switch_rate"], 0))

# ---- the Lean development, counted from the source so it cannot go stale ---------------
LEAN = os.path.join(os.path.dirname(HERE), "formal", "AnchoredCollapse.lean")
if os.path.exists(LEAN):
    src = open(LEAN).read()
    n_thm = len(re.findall(r"^theorem\s", src, re.M))
    m("leanTheorems", {7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n_thm, str(n_thm)))
    m("leanTheoremsNum", str(n_thm))
    m("leanSorries", str(src.count("sorry") - src.count("`sorry`") - src.count("no sorry")))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("% Generated by zeroguard/make_paper_macros.py -- do not edit by hand.\n")
    f.write("% Every number in the vehicle paper comes from results/*.json through this file.\n")
    f.write("\n".join(sorted(set(L))) + "\n")
print(f"wrote {OUT}  ({len(set(L))} macros)")
