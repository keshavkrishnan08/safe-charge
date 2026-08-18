"""Run the whole ZEROGUARD program: every experiment, then the figures, then the claim check.

Roughly 45 minutes on ten cores. Every script is independently runnable; this only sequences
them and stops if any stage fails.

    python zeroguard/run_all.py
    python zeroguard/run_all.py --figures-only
"""
import os, sys, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

STAGES = [
    ("E1  the principle is not about batteries", "exp/e1_generality.py"),
    ("E2  where the theorem stops being true", "exp/e2_boundary.py"),
    ("E3  cooling-law invariance (vacuum)", "exp/e3_radiative.py"),
    ("E4  31 000 cycles, never recalibrated", "exp/e4_mission_life.py"),
    ("E5  pack structure and per-cell enforcement", "exp/e5_pack.py"),
    ("E6  adversarial and corrupted sensing", "exp/e6_adversarial.py"),
    ("E7  ablations", "exp/e7_ablation.py"),
    ("E8  the two ablations the first grid could not decide", "exp/e8_ablation_targeted.py"),
    ("E9  what a margin is worth", "exp/e9_margin_power.py"),
    ("E10 mapping the certified region", "exp/e10_certified_region.py"),
    ("E12 the adversary, at a sample size that certifies", "exp/e12_adversary_scale.py"),
    ("E11 closing the design-register gaps", "exp/e11_closure.py"),
]

# The vehicle program: Anchored Collapse across four media. `x_crossdomain` reads the four
# domain result files, so it runs last.
VEHICLE_STAGES = [
    ("V1  ground: robotaxis, shuttles, autonomous trucks", "exp/v1_ground.py"),
    ("V2  aerial: delivery rotorcraft and eVTOL", "exp/v2_aerial.py"),
    ("V3  underwater: survey AUVs, gliders, under-ice", "exp/v3_underwater.py"),
    ("V4  space: LEO, GEO, deep space, Mars, lunar night, radiation", "exp/v4_space.py"),
    ("X   cross-domain connectedness", "exp/x_crossdomain.py"),
    ("N1  the assumptions against NASA PCoE measurements", "exp/n1_nasa_validation.py"),
    ("B1  against CC-CV, a shipped de-rate, and MPC", "exp/b1_baselines.py"),
    ("B2  against each domain's own stopping rule", "exp/b2_domain_baselines.py"),
    ("B3  a de-rate tuned honestly, then deployed", "exp/b3_tuned_derate.py"),
    ("D1  the EPA's own driving schedules", "exp/d1_drive_cycles.py"),
    ("E13 deployability on a BMS microcontroller", "exp/e13_embedded.py"),
    ("M1  a margin that knows how long the operation lasts", "exp/m1_duration_margin.py"),
    ("E14 the firmware, compiled and diffed against the reference", "exp/e14_firmware.py"),
    ("S1  capacity lost to plating, measured by the DFN", "exp/s1_plating.py"),
    ("B5  the same standard in every domain", "exp/b5_domain_transfer.py"),
    ("N2  the certificate against a Doyle-Fuller-Newman plant", "exp/n2_dfn.py"),
    ("B4  against the control-barrier-function filter", "exp/b4_cbf.py"),
    ("P1  the policy may be learned; the safety must not be", "exp/p1_policy_filter.py"),
    ("P2  the weakest cell, in motion", "exp/p2_pack_dynamic.py"),
]


def run(path, label):
    t0 = time.time()
    print(f"\n{'='*78}\n {label}\n{'='*78}")
    r = subprocess.run([PY, os.path.join(HERE, path)], cwd=ROOT)
    if r.returncode != 0:
        print(f"\nFAILED: {path}")
        sys.exit(r.returncode)
    print(f"  [{time.time()-t0:.0f}s]")


def main():
    figs_only = "--figures-only" in sys.argv
    cells_only = "--cells-only" in sys.argv
    vehicles_only = "--vehicles-only" in sys.argv
    t0 = time.time()
    if not figs_only:
        if not vehicles_only:
            for label, path in STAGES:
                run(path, label)
        if not cells_only:
            for label, path in VEHICLE_STAGES:
                run(path, label)
    if not vehicles_only:
        run("figures.py", "figures")
        run("verify_claims.py", "checking the manuscript against the results")
    if not cells_only:
        run("figures_vehicles.py", "vehicle figures")
        run("verify_vehicles.py", "checking the vehicle manuscript against the results")
        run("make_paper_macros.py", "regenerating the paper's numbers from the results")
    print(f"\nall stages complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
