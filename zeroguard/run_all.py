"""Run the whole ZEROGUARD program: every experiment, then the figures, then the claim check.

Roughly 25 minutes on ten cores. Every script is independently runnable; this only sequences
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
    t0 = time.time()
    if not figs_only:
        for label, path in STAGES:
            run(path, label)
    run("figures.py", "figures")
    run("verify_claims.py", "checking the manuscript against the results")
    print(f"\nall stages complete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
