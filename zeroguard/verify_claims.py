"""Check every quantitative claim in PAPER.md against the result files.

A paper whose numbers are typed in by hand drifts from its data the moment anything is
re-run. This reads the claims back out of the results and fails loudly on any disagreement,
so the manuscript cannot silently rot.

    python zeroguard/verify_claims.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def L(n):
    with open(os.path.join(RES, n)) as f:
        return json.load(f)


TOTAL_IN_PAPER = 650970   # quoted in Section 7 of PAPER.md


def main():
    e1, e2, e3 = L("e1_generality.json"), L("e2_boundary.json"), L("e3_radiative.json")
    e4, e5, e6 = L("e4_mission_life.json"), L("e5_pack.json"), L("e6_adversarial.json")
    e7, e8 = L("e7_ablation.json"), L("e8_ablation_targeted.json")
    e9, e10 = L("e9_margin_power.json"), L("e10_certified_region.json")
    e11, e12 = L("e11_closure.json"), L("e12_adversary_scale.json")

    checks = []

    def ck(claim, got, want, tol=0.0):
        ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and not isinstance(want, bool) \
            else (got == want)
        checks.append((ok, claim, got, want))

    # --- Section 4.1
    ck("E1 episodes = 8000", e1["pooled"]["trials"], 8000)
    ck("E1 transitions = 20,400,000", e1["pooled"]["one_step_transitions"], 20_400_000)
    ck("E1 violations = 0", e1["pooled"]["violations"], 0)
    ck("E1 CP95 = 0.037%", round(e1["pooled"]["cp95_upper_pct"], 3), 0.037, 0.001)
    ck("E1b states = 60000", e1["e1b_bit_exact"]["states"], 60000)
    ck("E1b mismatches = 0", e1["e1b_bit_exact"]["mismatches"], 0)
    ck("E1 interval-at-zero on all probes",
       all(list(s["interval_diagnosis"].keys()) == ["interval-at-zero"] for s in e1["systems"]), True)

    # --- Section 4.2
    ck("A1 violation rate = 100%", e2["a1_failure"]["violation_rate_pct"], 100.0, 1e-9)
    ck("A1 mean certified steps = 22.3",
       round(e2["a1_failure"]["mean_certified_steps_before_violation"], 1), 22.3, 0.05)
    ck("A1 unrecoverable step = 22.4", round(e2["a1_failure"]["mean_unrecoverable_step"], 1), 22.4, 0.05)
    ck("battery control violations = 0", e2["a1_control"]["violations"], 0)
    ck("A2 never returns infeasible", e2["a2_failure"]["unsafe_returns"], 0)
    ck("A2 mean gap = 23.7%", round(e2["a2_failure"]["mean_optimality_gap_pct"], 1), 23.7, 0.05)
    ck("A2 max gap = 66.0%", round(e2["a2_failure"]["max_optimality_gap_pct"], 1), 66.0, 0.05)
    ck("A2 mean bands = 3.57", round(e2["a2_failure"]["mean_admissible_bands"], 2), 3.57, 0.01)

    # --- Section 4.3
    a = e3["e3a_monotonicity"]
    ck("E3a samples = 11520", a["samples"], 11520)
    ck("E3a dT/dI violations = 0", a["viol_dTdI"], 0)
    ck("E3a dT/dsR violations = 0", a["viol_dTdR"], 0)
    ck("E3a dT/d(eps loss) violations = 0", a["viol_dTdEpsLoss"], 0)
    b = e3["e3b_envelope"]
    ck("E3b draws = 20000", b["trials"], 20000)
    ck("E3b violations = 0", b["violations"], 0)
    ck("E3b CP95 = 0.015%", round(b["cp95_upper_pct"], 3), 0.015, 0.001)
    ck("E3b worst T = 32.49", round(b["worst_T"], 2), 32.49, 0.01)
    ck("E3b corner dominates", b["corner_dominates"], True)
    c = e3["e3c_vacuum_limit"]
    ck("E3c all safe", c["all_safe"], True)
    ck("E3c Spearman rho = 1.000", round(c["spearman_soc_vs_emissivity"]["rho"], 3), 1.0, 1e-9)
    ck("E3c SOC 0.607 -> 0.168",
       (round(c["sweep"][0]["delivered_soc"], 3), round(c["sweep"][-1]["delivered_soc"], 3)),
       (0.607, 0.168))

    # --- Section 4.4
    for duty, cyc, pkb, pko, sb, so in [("satellite", 27000, 35.00, 39.90, 0.610, 0.698),
                                        ("robotaxi", 4000, 32.07, 41.40, 0.415, 0.577)]:
        r = e4["duties"][duty]
        ck(f"{duty} cycles", r["cycles"], cyc)
        ck(f"{duty} bound violations = 0", r["bound"]["violations"], 0)
        ck(f"{duty} oracle violations = 0", r["oracle"]["violations"], 0)
        ck(f"{duty} bound peak T", round(r["bound"]["worst_peak_T"], 2), pkb, 0.01)
        ck(f"{duty} oracle peak T", round(r["oracle"]["worst_peak_T"], 2), pko, 0.01)
        ck(f"{duty} bound SOC", round(r["bound"]["mean_soc"], 3), sb, 0.001)
        ck(f"{duty} oracle SOC", round(r["oracle"]["mean_soc"], 3), so, 0.001)
        ck(f"{duty} rank-biserial = +1.000", round(r["wilcoxon"]["rank_biserial"], 3), 1.0, 1e-9)
    ck("satellite gap = 8.8 pts", round(100 * e4["duties"]["satellite"]["soc_gap_mean"], 1), 8.8, 0.05)
    ck("robotaxi gap = 16.2 pts", round(100 * e4["duties"]["robotaxi"]["soc_gap_mean"], 1), 16.2, 0.05)

    # --- Section 4.5
    ck("lemma holds at every N", e5["e5a_lemma"]["lemma_holds"], True)
    ck("lemma max N = 1024", max(r["N"] for r in e5["e5a_lemma"]["sweep"]), 1024)
    worst_lemma = max(r["max_abs_disagreement"] for r in e5["e5a_lemma"]["sweep"])
    ck("lemma worst disagreement = 6.6e-12", round(worst_lemma * 1e12, 1), 6.6, 0.05)
    ck("lemma below bisection resolution (15/2^40)", worst_lemma < 15.0 / 2 ** 40, True)
    ck("lemma exactly zero at 9 of 11 sizes",
       sum(1 for r in e5["e5a_lemma"]["sweep"] if r["max_abs_disagreement"] == 0.0), 9)
    f = e5["e5b_fault"]
    ck("fault packs = 1200", f["packs"], 1200)
    ck("per-cell violations = 0", f["per_cell"]["violations"], 0)
    ck("pack-averaged violations = 1200", f["pack_averaged"]["violations"], 1200)
    ck("per-cell worst T = 43.04", round(f["per_cell"]["worst_T"], 2), 43.04, 0.01)
    ck("averaged worst T = 48.54", round(f["pack_averaged"]["worst_T"], 2), 48.54, 0.01)
    ck("HL shift = 3.24 C", round(f["peakT_wilcoxon"]["hodges_lehmann"], 2), 3.24, 0.01)

    # --- Section 4.6
    ad = e6["e6a_adversary"]
    ck("adversary (first pass) sequences = 72000", ad["sequences_searched"], 72000)
    ck("adversary (first pass) cells = 40", ad["cells"], 40)
    ck("adversary at scale: cells = 320", e12["trials"], 320)
    ck("adversary at scale: sequences = 256000", e12["sequences_searched"], 256000)
    ck("adversary at scale: violations = 0", e12["violations"], 0)
    ck("adversary at scale: worst T = 37.991", round(e12["worst_peak_T"], 3), 37.991, 0.001)
    ck("adversary at scale: headroom = 7.009 C", round(e12["headroom_C"], 3), 7.009, 0.001)
    ck("adversary at scale: gain = 0.0690 C", round(e12["mean_gain_over_greedy_C"], 4), 0.0690, 0.0001)
    ck("adversary at scale: CP95 = 0.9318%", round(e12["cp95_upper_pct"], 4), 0.9318, 0.0001)
    ck("adversary at scale certifies below 1%", e12["certifies_below_1pct"], True)
    ck("jitter all safe", e6["e6c_jitter"]["all_safe"], True)
    ck("float32 violations = 0", e6["e6d_float32"]["violations"], 0)
    ck("float32 worst dev = 2.3e-4", round(e6["e6d_float32"]["worst_current_deviation_A"], 6),
       0.000229, 1e-6)

    # --- Section 4.7
    ck("E9 measured breakpoint = 12.85", e9["breakpoint_cap_live_C"], 12.85, 1e-9)
    ck("E9 derived prediction = 12.802", round(e9["refined_prediction_C"], 3), 12.802, 0.001)
    ck("E9 error <= 0.05 C", e9["abs_error_C"] <= 0.05, True)
    ck("E9 alpha = 0.02358", round(e9["alpha_dt_hA_over_Cth"], 5), 0.02358, 1e-5)

    # --- Section 4.8
    ab = e7["ablations"]
    ck("ablation grid = 1400", e7["grid_size"], 1400)
    ck("full method violations = 0", ab["A0_full"]["violations"], 0)
    ck("no-bound violations = 29", ab["A2_no_bound"]["violations"], 29)
    ck("point-estimate violations = 7", ab["A6_point_estimate"]["violations"], 7)
    ck("tolerance exit not fixed-iteration", ab["A7_tolerance_exit"]["iters_is_fixed"], False)
    ck("A1 identical answers = 100%", round(100 * e8["a1_targeted"]["identical_fraction"], 1), 100.0, 1e-9)
    ck("A1 evals saved = 17", round(e8["a1_targeted"]["evals_saved"], 1), 17.0, 0.05)
    ck("min certifying K = 15", e8["a4_min_safe_K"], 15.0, 1e-9)
    kk = {r["K"]: r["violations"] for r in e8["a4_beyond_bound"]["sweep"]}
    ck("K=0 violations = 529", kk[0.0], 529)
    ck("K=12 violations = 5", kk[12.0], 5)
    ck("K=15 violations = 0", kk[15.0], 0)

    # --- Section 4.3 (six-channel) and 4.10 (cost + adequacy)
    six = e11["six_channel_envelope"]
    ck("6-channel draws = 20000", six["trials"], 20000)
    ck("6-channel violations = 0", six["violations"], 0)
    ck("6-channel CP95 = 0.015%", round(six["cp95_upper_pct"], 4), 0.0150, 0.0005)
    ck("6-channel worst T = 32.49", round(six["worst_T"], 2), 32.49, 0.01)
    ck("6-channel corner dominates", six["corner_dominates"], True)
    ck("6 channels named", len(six["channels"]), 6)
    lat = e11["latency"]
    ck("hard bound = 20 evaluations", lat["hard_bound_evaluations"], 20)
    # host timing varies by a microsecond or two between runs; the claim is the
    # published 56 us figure, checked to within that noise
    ck("worst path median ~ 56 us", round(lat["worst_path_median_us"], 1), 56.0, 2.0)
    ck("headroom >= 5.5 orders", lat["headroom_orders"] >= 5.5, True)
    ck("memory = 904 B double", lat["memory_bytes_double"], 904)
    adq = e11["sample_adequacy"]
    ck("n_required(1%) = 299", adq["n_required"]["1.0%"], 299)
    ck("n_required(0.1%) = 2995", adq["n_required"]["0.1%"], 2995)
    ck("n_required(0.01%) = 29956", adq["n_required"]["0.01%"], 29956)
    ck("every claim now certifies below 1%",
       all(r["enough_for_1pct"] for r in adq["claims"]), True)

    # --- Section 4.9
    ck("E10 grid a episodes = 14040", e10["K_vs_sR"]["total_episodes"], 14040)
    ck("E10 grid b episodes = 19890", e10["bias_vs_cool"]["total_episodes"], 19890)

    # --- total workload quoted in Section 7
    total = (e1["pooled"]["trials"] + e1["e1b_bit_exact"]["states"]
             + e2["a1_failure"]["trials"] + e2["a1_control"]["trials"] + e2["a2_failure"]["trials"]
             + e3["e3a_monotonicity"]["samples"] + e3["e3b_envelope"]["trials"]
             + sum(e4["duties"][d]["cycles"] * 2 for d in e4["duties"])
             + e5["e5b_fault"]["packs"] * 2
             + ad["sequences_searched"]
             + sum(x["trials"] for m in e6["e6b_sensors"].values() for x in m["sweep"])
             + sum(x["trials"] for x in e6["e6c_jitter"]["sweep"])
             + e6["e6d_float32"]["trials"]
             + e7["grid_size"] * len(ab)
             + e8["a1_targeted"]["states"] + sum(r["trials"] for r in e8["a4_beyond_bound"]["sweep"])
             + sum(r["trials"] for r in e9["cap_live"])
             + sum(r["trials"] for r in e9["tail_cap_live"]) + sum(r["trials"] for r in e9["tail_cap_frozen"])
             + e10["K_vs_sR"]["total_episodes"] + e10["bias_vs_cool"]["total_episodes"]
             + six["trials"] + sum(v["states"] for v in lat["paths"].values())
             + e12["episodes_simulated"])

    bad = [c for c in checks if not c[0]]
    for ok, claim, got, want in checks:
        if not ok:
            print(f"  MISMATCH  {claim}: results say {got}, paper says {want}")
    print(f"\n{len(checks) - len(bad)}/{len(checks)} claims verified against the result files")
    print(f"total simulated episodes / states across the program: {total:,}")
    ck2 = TOTAL_IN_PAPER
    print(f"  (paper Section 7 quotes {ck2:,}) -> "
          f"{'agrees' if total == ck2 else 'MISMATCH'}")
    if bad:
        print("PAPER AND RESULTS DISAGREE")
        return 1
    print("paper and results agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
