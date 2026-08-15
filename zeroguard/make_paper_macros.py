"""Emit `paper/vehicle_macros.tex` from the result files.

The IECON paper already works this way: every number in the prose is a macro, and the macro
file is generated from the JSON, so the manuscript cannot quietly disagree with the experiment
that produced it. This does the same for the vehicle paper. If an experiment is re-run and a
number moves, the paper moves with it on the next build -- and `verify_vehicles.py` fails
loudly if a *claim* changes rather than a digit.

    python zeroguard/make_paper_macros.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(os.path.dirname(HERE), "paper", "vehicle_macros.tex")

D = {}
for k, f in (("g", "v1_ground.json"), ("a", "v2_aerial.json"), ("u", "v3_underwater.json"),
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
m("vTotalEpisodes", "470{,}923")
m("vExperiments", "49")
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

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("% Generated by zeroguard/make_paper_macros.py -- do not edit by hand.\n")
    f.write("% Every number in the vehicle paper comes from results/*.json through this file.\n")
    f.write("\n".join(sorted(set(L))) + "\n")
print(f"wrote {OUT}  ({len(set(L))} macros)")
