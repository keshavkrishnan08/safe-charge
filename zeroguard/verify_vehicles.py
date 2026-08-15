"""Re-read every vehicle claim out of the result files and fail loudly on drift.

`verify_claims.py` does this for the original ZEROGUARD program. This is the same instrument
pointed at `PAPER_VEHICLES.md`: each claim in the manuscript is restated here as an assertion
against `results/v*.json`, so a number that changes because an experiment was re-run cannot
silently disagree with the prose describing it.

Three kinds of check:

  `eq`    an exact value (counts, booleans, bit-exactness)
  `close` a value within a stated tolerance (means, latencies, correlations)
  `pred`  a predicate over the data (monotone, bounded, all-zero)

    python zeroguard/verify_vehicles.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

_cache = {}


def d(fn):
    if fn not in _cache:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p) as f:
            _cache[fn] = json.load(f)
    return _cache[fn]


def dig(fn, *path):
    x = d(fn)
    for p in path:
        x = x[p]
    return x


CHECKS = []


def eq(label, got, want):
    CHECKS.append((label, lambda: got() == want, lambda: f"{got()!r} != {want!r}"))


def close(label, got, want, tol):
    CHECKS.append((label, lambda: abs(got() - want) <= tol,
                   lambda: f"{got()!r} not within {tol} of {want!r}"))


def pred(label, fn, why):
    CHECKS.append((label, fn, lambda: why))


G, A_, U, S, X = ("v1_ground.json", "v2_aerial.json", "v3_underwater.json",
                  "v4_space.json", "x_crossdomain.json")
N = "n1_nasa_validation.json"
B = "b1_baselines.json"
B2 = "b2_domain_baselines.json"

# ---- domain stopping rules: where the certificate wins, and where it does not ------------
pred("B2    on a rotorcraft no fixed SOC reserve reaches even the 80 % recovery target that "
     "the certificate's reserve does",
     lambda: (dig(B2, "delivery-quadrotor", "curve", "80",
                  "mission_per_family")["certificate reserve"]
              > dig(B2, "delivery-quadrotor", "curve", "80",
                    "mission_per_family")["fixed SOC reserve"]),
     "the fixed SOC rule matched the certificate on the rotorcraft")
pred("B2    and where the mission is charge-limited, a fixed SOC rule is NOT beaten -- reported",
     lambda: (dig(B2, "under-ice-auv", "comparison")["reserve_gain_pct"] <= 0
              or dig(B2, "geo-comsat", "comparison")["reserve_gain_pct"] <= 0),
     "the honest negative result in the charge-limited domains has gone missing")
pred("B2    the combined min-of-two-clocks rule is never far behind the best single rule",
     lambda: all(q is None or q >= -12.0 for q in
                 [dig(B2, c, "comparison").get("combined_gain_pct")
                  for c in ("delivery-quadrotor", "under-ice-auv", "geo-comsat")]),
     "the recommended combined rule lost badly to a single-clock rule somewhere")

# ---- baselines: compared to what? -------------------------------------------------------
eq("B1    the certificate violates nothing where the shipped de-rate also violates nothing",
   lambda: dig(B, "controllers", "zeroguard", "violations"), 0)
pred("B1    and it recovers real charge the de-rate gives away",
     lambda: dig(B, "charge_gain_over_derate_points") > 5.0,
     "the certificate did not materially beat the shipped de-rate")
pred("B1    the de-rate is not paranoia: removing it actually violates",
     lambda: dig(B, "aggressive_violation_rate") > 0.25,
     "the aggressive protocol was safe, which would make the de-rate unjustified")
pred("B1    the filter and MPC solve the same problem, to within a fraction of a point",
     lambda: abs(dig(B, "mpc_h1_soc_gap_points")) < 2.0,
     "the filter and MPC disagreed materially on delivered charge")
pred("B1    but only the filter has a worst case known before the data",
     lambda: dig(B, "zeroguard_bounded_mpc_not"),
     "the bounded-cost distinction between the filter and MPC did not hold")
pred("B1    and MPC's worst observed cost is far above the filter's bound",
     lambda: dig(B, "mpc_h1_eval_ratio") > 5.0,
     "MPC was not materially more expensive in model evaluations")

# ---- external validation: assumptions against measurements from another lab -------------
pred("N1-1  measured resistance growth stays inside the datasheet bound within the claimed life",
     lambda: dig(N, "n1_resistance_bound", "fraction_inside_within_eol") >= 0.999,
     "real cells left the resistance bound inside the life the certificate claims")
# Over the FULL measured life -- including cells NASA ran down to 3 % of initial capacity,
# far past anything the certificate claims -- exactly one measurement of 1,940 exceeds the
# bound, and it does so by 0.1 %. Asserting zero exceedances over the full life would be
# asserting something the data does not support; this asserts what it does.
pred("N1-1  at most one exceedance over the full measured life, and it is marginal",
     lambda: (dig(N, "n1_resistance_bound", "violations") <= 1
              and dig(N, "n1_resistance_bound", "worst_ratio")
              <= 1.01 * dig(N, "n1_resistance_bound", "bound")),
     "more than one cell, or one by more than 1 %, exceeded the resistance bound")
pred("N1-2  the capacity envelope's scope is stated rather than assumed",
     lambda: dig(N, "n2_capacity_envelope", "cells_past_envelope") > 0
     and "note" in d(N)["n2_capacity_envelope"],
     "the capacity scope limitation was not recorded")
pred("N1-3  hypothesis (A2) holds on real cells, model-free",
     lambda: dig(N, "n3_monotone_heating", "monotone_supported"),
     "the energy-balance test did not support monotone heating")
pred("N1-3  and the confounded instantaneous test is reported as confounded",
     lambda: dig(N, "n3_monotone_heating", "instantaneous_test_confounded"),
     "the thermal-lag confound was not recorded")
pred("N1-4  the ROM transfer beats persistence over the horizon",
     lambda: dig(N, "n4_transfer_prediction", "skill_vs_persistence") > 0.0,
     "the transferred model did not beat a persistence baseline")
pred("N1-4  and the thermal margin covers the worst transfer under-prediction",
     lambda: dig(N, "n4_transfer_prediction", "margin_over_underprediction") >= 3.0,
     "the thermal margin does not comfortably cover the transfer error")
pred("N1-5  the filter permits the majority of a real charging protocol",
     lambda: dig(N, "n5_permits_protocol", "allowed_fraction") > 0.5,
     "the filter refused most of a standard measured charging protocol")
pred("N1-6  the refusal-versus-degradation test is reported as inconclusive, not cherry-picked",
     lambda: (not dig(N, "n6_refusals_predict_degradation", "conclusive"))
     and (not dig(N, "n6_refusals_predict_degradation", "signs_agree"))
     and len(dig(N, "n6_refusals_predict_degradation", "variants")) == 4,
     "the inconclusive external association was not reported with all its variants")

# ---- the theorem -----------------------------------------------------------------------
eq("V1-1  vehicle filter on one cell is bit-exact with project_current",
   lambda: dig(G, "v1_1_reduction", "bit_exact"), True)
eq("V1-1  zero mismatches in 100 000 states",
   lambda: dig(G, "v1_1_reduction", "mismatches"), 0)
eq("V1-1  a P-parallel pack's limit is exactly P times one cell's",
   lambda: dig(G, "v1_1_reduction", "pack_exact"), True)
eq("X2   the generalisation reduces to the original theorem, bit for bit",
   lambda: dig(X, "x2_reduction", "bit_exact"), True)
eq("X2   every charge platform's lower edge is exactly zero",
   lambda: dig(X, "x2_reduction", "all_lower_edges_zero"), True)
eq("X1   no admissible set is disconnected, in any domain",
   lambda: dig(X, "x1_structure", "disconnected"), 0)
eq("V2-2 no disconnected admissible set in 20 000 flight states",
   lambda: dig(A_, "v2_2_interval_structure", "disconnected"), 0)
eq("X3   the evaluation count is bounded: 20 on charge, 39 on discharge",
   lambda: dig(X, "x3_cost", "bounded"), True)

# ---- safety: nothing breached while the certificate said feasible -----------------------
eq("V1-2 350 kW robotaxi sessions: zero violations",
   lambda: dig(G, "v1_2_fast_charge", "violations"), 0)
eq("V1-2 the pessimistic corner dominates every draw",
   lambda: dig(G, "v1_2_fast_charge", "corner_dominates"), True)
eq("V1-3 ambient sweep, passive and active: zero violations",
   lambda: dig(G, "v1_3_ambient", "total_violations"), 0)
eq("V1-6 five years of robotaxi duty: zero violations",
   lambda: dig(G, "v1_6_duty_cycle", "duties", "robotaxi", "violations"), 0)
eq("V1-7 the cooling reserve's prediction holds",
   lambda: dig(G, "v1_7_coolant", "prediction_holds"), True)
eq("V1-8 regen pulses at 10 Hz: zero certified violations",
   lambda: dig(G, "v1_8_regen", "violations"), 0)
eq("V1-11 V2G: nothing unsafe while certified, at any export power",
   lambda: sum(v["anchored_unsafe_while_certified"]
               for v in dig(G, "v1_11_v2g").values()), 0)
eq("V2-1 anchored flight: nothing unsafe while certified",
   lambda: dig(A_, "v2_1_anchor_vs_null", "anchored_unsafe_while_certified"), 0)
eq("V2-3 no breach without a prior closure warning",
   lambda: dig(A_, "v2_3_reserve_lead", "breaches_with_no_warning"), 0)
eq("V2-4 a full sortie: nothing unsafe while certified",
   lambda: dig(A_, "v2_4_sortie", "unsafe_while_certified"), 0)
eq("V2-5 payload x altitude grid: nothing unsafe while certified",
   lambda: dig(A_, "v2_5_payload_altitude", "total_unsafe"), 0)
eq("V2-6 gusts to sigma 0.5: nothing unsafe while certified",
   lambda: dig(A_, "v2_6_gusts", "total_unsafe"), 0)
eq("V2-7 cold soak at 2 000 m: nothing unsafe while certified",
   lambda: dig(A_, "v2_7_cold_soak", "total_unsafe"), 0)
eq("V2-8 motor-out to +100 %: nothing unsafe while certified",
   lambda: dig(A_, "v2_8_motor_out", "total_unsafe"), 0)
eq("V2-9 forty turnaround charges a day: zero violations",
   lambda: dig(A_, "v2_9_turnaround", "violations"), 0)
eq("V2-11 the CEM adversary breaks nothing",
   lambda: dig(A_, "v2_11_adversary", "violations"), 0)
eq("V3-1 sealed hull in 2 C water: zero violations",
   lambda: dig(U, "v3_1_sealed_hull", "violations"), 0)
eq("V3-4 six-month deployment: zero violations",
   lambda: dig(U, "v3_4_deployment", "violations"), 0)
eq("V3-5 a decade of dormancy: zero violations",
   lambda: dig(U, "v3_5_dormancy", "total_violations"), 0)
pred("V3-5  and delivered charge does not move at all -- the effect is numerical noise",
     lambda: dig(U, "v3_5_dormancy", "soc_spread") < 1e-4,
     "dormancy moved delivered charge by a measurable amount")
pred("V3-6  plating binds at every depth from the surface to 6 000 m",
     lambda: dig(U, "v3_6_depth", "plating_binds_at_every_depth"),
     "some depth was limited by something other than plating")
eq("V3-6 depth to 6 000 m: zero violations",
   lambda: dig(U, "v3_6_depth", "total_violations"), 0)
eq("V3-7 under ice: no breach without a prior warning",
   lambda: dig(U, "v3_7_under_ice", "breaches_with_no_warning"), 0)
eq("V3-8 obstacle-avoidance bursts: nothing unsafe while certified",
   lambda: dig(U, "v3_8_transients", "total_unsafe"), 0)
eq("V3-11 the buoyancy engine's burst: nothing unsafe while certified",
   lambda: dig(U, "v3_11_buoyancy_pump", "total_unsafe"), 0)
eq("V4-1 five years of LEO cycling: zero violations",
   lambda: dig(S, "v4_1_leo", "violations"), 0)
eq("V4-2 a GEO eclipse season: nothing unsafe while certified",
   lambda: dig(S, "v4_2_geo", "total_unsafe"), 0)
eq("V4-3 deep-space cruise: zero violations",
   lambda: dig(S, "v4_3_deep_space", "violations"), 0)
eq("V4-4 a Mars sol: zero violations",
   lambda: dig(S, "v4_4_mars", "total_violations"), 0)
eq("V4-5 lunar night frontier: nothing unsafe while certified",
   lambda: dig(S, "v4_5_lunar_night", "total_unsafe"), 0)
eq("V4-6 300 krad of dose: zero violations",
   lambda: dig(S, "v4_6_radiation", "total_violations"), 0)
eq("V4-9 solar-array degradation: zero violations",
   lambda: dig(S, "v4_9_array_degradation", "total_violations"), 0)
pred("V4-9  and the array never becomes the binding constraint, even after 60 years",
     lambda: dig(S, "v4_9_array_degradation", "array_never_binding"),
     "array degradation became the binding constraint inside the swept range")
eq("V4-10 radiator to 1 % of nominal: zero violations",
   lambda: dig(S, "v4_10_radiator", "total_violations"), 0)
pred("V4-10 in deep space the pack is cold-limited, so losing the radiator raises charge",
     lambda: dig(S, "v4_10_radiator", "cold_limited_not_heat_limited"),
     "delivered charge fell as the radiator shrank, contradicting the reported mechanism")
pred("V4-5  insulation buys lunar survival until the energy ceiling takes over",
     lambda: dig(S, "v4_5_lunar_night", "insulation_saturates_at_energy_limit"),
     "the insulation sweep did not reach the energy-limited regime")
eq("V4-11 the bus-load anchor: nothing unsafe while certified",
   lambda: sum(v["unsafe_while_certified"] for v in dig(S, "v4_11_bus_anchor").values()), 0)

# ---- the generalisation is necessary, not ornamental -------------------------------------
pred("V1-11 the null-input filter browns out every V2G session",
     lambda: all(v["null_input_brownout_rate"] >= 0.99
                 for v in dig(G, "v1_11_v2g").values()),
     "some V2G export was served by the null-input filter")
pred("V2-1  the null-input filter starves the rotors",
     lambda: dig(A_, "v2_1_anchor_vs_null", "null_input_brownout_rate") >= 0.99,
     "the null-input filter kept a quadrotor flying")
pred("V3-3  a stationary AUV still cannot command zero",
     lambda: all(v["null_input_brownouts"] >= 0.99 * v["trials"]
                 for v in dig(U, "v3_3_hotel_anchor").values()),
     "the null-input filter served an AUV hotel load")
pred("V4-11 the null-input filter cannot hold a bus through eclipse",
     lambda: all(v["null_input_brownouts"] >= 0.99 * v["trials"]
                 for v in dig(S, "v4_11_bus_anchor").values()),
     "the null-input filter held a spacecraft bus")
pred("V2-1  the calibrated energy cell is refused above ~1C of hover",
     lambda: dig(A_, "v2_1_anchor_vs_null", "energy_cell_refuses_above_c") is not None
     and dig(A_, "v2_1_anchor_vs_null", "energy_cell_refuses_above_c") <= 1.5,
     "the energy cell was certified for high-C-rate hover")

# ---- the reserve means something ---------------------------------------------------------
pred("V2-3  no closure ever arrives after its breach",
     lambda: dig(A_, "v2_3_reserve_lead", "lead_s_min") >= 0.0,
     "a breach preceded its own closure warning")
pred("V2-3  the warning is a useful one in the 5th percentile, not just a non-negative one",
     lambda: dig(A_, "v2_3_reserve_lead", "lead_s_p05") > 0.0,
     "the 5th-percentile lead time was zero")
pred("V3-7  waiting for closure does NOT buy an under-ice transit -- reported, not hidden",
     lambda: dig(U, "v3_7_under_ice", "closure_alone_is_insufficient"),
     "closure-triggered warning turned out to be sufficient, so the design rule is moot")
pred("V3-7  and a reserve threshold exists that does buy it",
     lambda: dig(U, "v3_7_under_ice", "threshold_for_transit_A") is not None,
     "no reserve threshold gave a 20-minute warning at the 5th percentile")
pred("X4    every platform is well predicted by one of the certificate's two clocks",
     lambda: dig(X, "x4_reserve", "every_case_has_a_predictive_clock"),
     "some platform had neither a predictive width clock nor a predictive energy clock")
pred("X4    and the width alone is NOT enough everywhere -- reported, not pooled away",
     lambda: "note" in d(X)["x4_reserve"] and not dig(X, "x4_reserve", "raw_all_positive"),
     "the case where interval width fails to predict endurance was not recorded")
pred("X4    the pooled, anchor-normalised relation is strong and significant",
     lambda: dig(X, "x4_reserve", "pooled")["rho"] > 0.5
     and dig(X, "x4_reserve", "pooled")["p"] < 0.01,
     "the pooled reserve-endurance relation did not hold")
pred("V2-10 eVTOL reserve predicts hover endurance",
     lambda: dig(A_, "v2_10_evtol", "reserve_vs_endurance")["rho"] > 0.5,
     "eVTOL reserve did not predict endurance")

# ---- graceful degradation is monotone, not cliffed ---------------------------------------
# The honest continuity claim is monotonicity along each limb, not a bound on the adjacent
# step: with 15 K between grid nodes the largest step is ~half the range, and that number is
# mostly a statement about the grid. What distinguishes the certificate's output from the
# binary built on it is that the binary jumps the *entire* range between adjacent cells while
# the underlying quantity is ordered.
pred("V1-4  the binary hit-rate is more discontinuous than the charge curve underneath it",
     lambda: (dig(G, "v1_4_deadline", "max_adjacent_hit_jump")
              > dig(G, "v1_4_deadline", "max_adjacent_soc_jump_frac_of_range")),
     "the binary service metric was no more discontinuous than the quantity it thresholds")
pred("V1-4  and each limb of the ambient sweep is monotone and significant",
     lambda: (dig(G, "v1_4_deadline", "cold_limb")["rho"] > 0
              and dig(G, "v1_4_deadline", "cold_limb")["p"] < 0.05
              and dig(G, "v1_4_deadline", "hot_limb")["rho"] < 0
              and dig(G, "v1_4_deadline", "hot_limb")["p"] < 0.05),
     "one of the two limbs was not monotone or not significant")
pred("V2-5  reserve falls with payload, and with altitude at every fixed payload",
     lambda: (dig(A_, "v2_5_payload_altitude", "width_vs_payload")["rho"] < 0
              and dig(A_, "v2_5_payload_altitude",
                      "altitude_negative_in_every_stratum")),
     "reserve did not fall with payload, or rose with altitude in some stratum")
pred("V3-2  in a sealed hull the binding constraint is plating at every water temperature",
     lambda: (dig(U, "v3_2_cold_binding", "plating_binds_everywhere")
              and dig(U, "v3_2_cold_binding", "thermal_never_binds")),
     "the binding constraint was not plating everywhere")
pred("V3-2  and the temperature-dependent cap takes a larger share of it as the water cools",
     lambda: dig(U, "v3_2_cold_binding", "cold_cap_fraction")
     > dig(U, "v3_2_cold_binding", "warm_cap_fraction"),
     "the plating current cap did not grow in share in the cold")
pred("V4-3  the radiative cooling channel is monotone in temperature",
     lambda: dig(S, "v4_3_deep_space", "radiative_monotone_in_T"), "radiation was not monotone")
pred("V4-4  radiation plus thin convection is still monotone in temperature",
     lambda: dig(S, "v4_4_mars", "sum_monotone_in_T"), "the Mars cooling sum was not monotone")

# ---- the estimator buys charge, never safety ---------------------------------------------
eq("V3-9  180 days unrecalibrated: safety identical to a daily oracle",
   lambda: dig(U, "v3_9_no_recalibration", "safety_identical"), True)
pred("V3-9  and in a sealed hull the oracle buys nothing either -- plating is what binds",
     lambda: abs(dig(U, "v3_9_no_recalibration", "soc_gap_points")) < 0.05,
     "the sealed-hull oracle bought a material amount of charge")
eq("V4-8  five years without a ground loop: safety identical to a daily oracle",
   lambda: dig(S, "v4_8_no_ground_loop", "safety_identical"), True)
pred("V4-8  and in deep space the oracle buys nothing at all",
     lambda: abs(dig(S, "v4_8_no_ground_loop", "soc_gap_points")) < 0.05,
     "the deep-space oracle bought a material amount of charge")

# ---- closed forms and derived quantities --------------------------------------------------
pred("V1-10 the derived sensor-bias breakpoint predicts the measured one to 0.5 C",
     lambda: dig(G, "v1_10_sensor", "error_C") is not None
     and dig(G, "v1_10_sensor", "error_C") <= 0.5,
     "the breakpoint prediction missed by more than 0.5 C")
pred("V1-3  the passive dead zone begins at the predicted T_max - dT",
     lambda: dig(G, "v1_3_ambient", "passive")["dead_zone_from_C"] is not None
     and abs(dig(G, "v1_3_ambient", "passive")["dead_zone_from_C"]
             - dig(G, "v1_3_ambient", "predicted_passive_ceiling_C")) <= 5.0,
     "the passive ceiling did not appear where the margin predicts")
pred("V1-5  pack-vs-weakest-cell deviation is below the search resolution",
     lambda: dig(G, "v1_5_truck_pack", "dev_below_resolution"),
     "the weakest-cell lemma disagreed by more than the bisection resolution")

# ---- honest limits, stated as checks -------------------------------------------------------
pred("V4-7  the SEU exposure is reported, not certified away",
     lambda: "honest_note" in d(S)["v4_7_seu"]
     and dig(S, "v4_7_seu", "census")["aggressive"] >= 0,
     "the SEU result lost its caveat")
pred("V4-5  the lunar-night result is a frontier, not a blanket pass",
     lambda: dig(S, "v4_5_lunar_night", "configurations_surviving")
     < dig(S, "v4_5_lunar_night", "configurations"),
     "every lunar configuration survived, which would be too good to be true")

# ---- sample adequacy -------------------------------------------------------------------
pred("X5    every headline claim certifies below 1 %",
     lambda: dig(X, "x5_adequacy", "all_certify_below_1pct"),
     "a headline claim was not backed by enough samples to certify below 1 %")
pred("X5    every expected claim was found in the results",
     lambda: dig(X, "x5_adequacy", "claims_found")
     == dig(X, "x5_adequacy", "claims_expected"),
     "a claim listed in the adequacy audit is missing from the results")


def main():
    ok = bad = err = 0
    print(f"verifying {len(CHECKS)} vehicle claims against the result files\n")
    for label, test, why in CHECKS:
        try:
            good = bool(test())
        except FileNotFoundError as e:
            print(f"  ??  {label}\n        missing {os.path.basename(str(e))}")
            err += 1
            continue
        except Exception as e:
            print(f"  ??  {label}\n        {type(e).__name__}: {e}")
            err += 1
            continue
        if good:
            ok += 1
            print(f"  ok  {label}")
        else:
            bad += 1
            print(f"  XX  {label}\n        {why()}")
    print(f"\n{ok} verified, {bad} failed, {err} unresolved, of {len(CHECKS)}")
    if bad or err:
        sys.exit(1)


if __name__ == "__main__":
    main()
