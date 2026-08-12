"""E1 -- the principle is not about batteries.

The same projection, unmodified, is asked to certify four systems whose physics have
nothing in common: an electrochemical cell, a resistive heating element, a DC motor's
winding and rotor, and a semiconductor junction. Each is driven by an adversarially greedy
policy that always requests the maximum input, from thousands of randomized initial states,
for a full episode. A single violation anywhere falsifies the theorem.

E1b then checks the sharper claim: instantiated on the battery, the generic filter is not
merely similar to the published `project_current`, it is bit-for-bit identical. That is what
makes "the same filter" a statement of fact rather than a figure of speech.

    python zeroguard/exp/e1_generality.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard.systems import BatterySystem, ResistiveHeater, DCMotorWinding, IGBTJunction
from zeroguard.gfilter import project, admissible_set, interval_diagnosis
from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
N_EPISODES = 2000
N_INTERVAL_PROBES = 400


def samplers(rng):
    """Randomized initial conditions and ambient for each system."""
    return {
        "battery": lambda: (BatterySystem(),
                            dict(soc0=float(rng.uniform(0.05, 0.55)),
                                 T0=float(rng.uniform(15.0, 40.0))),
                            float(rng.uniform(15.0, 35.0))),
        "heater": lambda: (ResistiveHeater(),
                           dict(T0=float(rng.uniform(15.0, 95.0))),
                           float(rng.uniform(10.0, 35.0))),
        "motor": lambda: (DCMotorWinding(),
                          dict(T0=float(rng.uniform(20.0, 120.0)),
                               om0=float(rng.uniform(0.0, 600.0))),
                          float(rng.uniform(15.0, 45.0))),
        "igbt": lambda: (IGBTJunction(),
                         dict(Tj0=float(rng.uniform(30.0, 120.0)),
                              Tc0=float(rng.uniform(30.0, 90.0))),
                         float(rng.uniform(25.0, 60.0))),
    }


def worst_slack(sy, vals):
    """Smallest distance to any limit; negative means violated."""
    out = []
    for idx, sense, lim in sy.limits:
        v = vals[idx]
        out.append(lim - v if sense == "<=" else v - lim)
    return min(out)


def run_system(key, make, n_ep, rng):
    viol = 0
    slacks, works, clip_frac = [], [], []
    t0 = time.time()
    for _ in range(n_ep):
        sy, kw, w = make()
        s = sy.init(**kw)
        dt = sy.dt_nominal
        ep_slack, work, nclip = 1e18, 0.0, 0
        for _ in range(sy.horizon):
            u, c = project(sy, s, sy.u_max, dt, w)
            nclip += int(c)
            s, vals = sy.step(s, u, dt, w)
            ep_slack = min(ep_slack, worst_slack(sy, vals))
            work += u * dt
        slacks.append(ep_slack); works.append(work)
        clip_frac.append(nclip / sy.horizon)
        viol += int(ep_slack < -1e-9)
    dur = time.time() - t0

    # Is the admissible set really one interval anchored at zero?
    diag = {}
    for _ in range(N_INTERVAL_PROBES):
        sy, kw, w = make()
        s = sy.init(**kw)
        _, ok = admissible_set(sy, s, sy.dt_nominal, w, n=600)
        d = interval_diagnosis(ok)
        diag[d] = diag.get(d, 0) + 1

    m, lo, hi = stats.bootstrap_ci(slacks)
    return stats.summarize_safety(key, viol, n_ep, extra=dict(
        physics=make()[0].physics,
        horizon=make()[0].horizon,
        steps_total=int(n_ep * make()[0].horizon),
        worst_slack=float(np.min(slacks)),
        mean_slack=m, slack_ci=[lo, hi],
        mean_clipped_fraction=float(np.mean(clip_frac)),
        mean_work=float(np.mean(works)),
        interval_diagnosis=diag,
        seconds=round(dur, 1)))


def e1b_bit_exact(n_states=60000, seed=11):
    """The generic filter, on the battery, must reproduce project_current exactly."""
    rng = np.random.default_rng(seed)
    scales = [dict(R=1.0, Q=1.0, plate=1.0), dict(R=1.8, Q=0.8, plate=1.6),
              dict(R=1.35, Q=0.9, plate=1.25)]
    mism, n = 0, 0
    worst = 0.0
    per = n_states // (len(scales) * 2)
    for sc in scales:
        for rc in ("euler", "exact"):
            rom = BatteryROM(cell_scale=sc, rc=rc)
            sy = BatterySystem(rom=rom)
            for _ in range(per):
                s = rom.init_state(float(rng.uniform(0.02, 0.95)),
                                   float(rng.uniform(-5.0, 50.0)))
                s["V1"] = float(rng.uniform(0.0, 0.08))
                w = float(rng.uniform(0.0, 40.0))
                cf = float(rng.choice([0.0, 0.25]))
                dT = 0.5 + cf * max(0.0, s["T"] - w)
                a, _ = project_current(rom, s, 3.0 * rom.p["Q_nom"], 30.0, w,
                                       margin=rom.plating_margin(),
                                       dV=0.03, dT=0.5, dP=0.006, cool_frac=cf)
                b, _ = project(sy, s, 3.0 * rom.p["Q_nom"], 30.0, w,
                               margins=(0.03, dT, 0.006))
                n += 1
                if a != b:
                    mism += 1
                    worst = max(worst, abs(a - b))
    return dict(states=n, mismatches=mism, worst_abs_diff=worst)


def main():
    rng = np.random.default_rng(20260811)
    mk = samplers(rng)
    out = {"n_episodes_per_system": N_EPISODES, "systems": []}
    print(f"E1 -- generic filter on four unrelated systems, {N_EPISODES} episodes each\n")
    print(f"{'system':10}{'physics':52}{'episodes':>9}{'steps':>10}{'viol':>6}{'CP95%':>9}{'worst slack':>13}")
    for key, make in mk.items():
        r = run_system(key, make, N_EPISODES, rng)
        out["systems"].append(r)
        print(f"{key:10}{r['physics'][:50]:52}{r['trials']:>9}{r['steps_total']:>10}"
              f"{r['violations']:>6}{r['cp95_upper_pct']:>8.3f}%{r['worst_slack']:>13.6f}")

    print("\ninterval structure (dense scan of feasibility over [0, u_max]):")
    for r in out["systems"]:
        print(f"  {r['name']:10} {r['interval_diagnosis']}")

    print("\nE1b -- generic filter vs published project_current on the battery")
    b = e1b_bit_exact()
    out["e1b_bit_exact"] = b
    print(f"  {b['states']} states, mismatches = {b['mismatches']}, "
          f"worst |diff| = {b['worst_abs_diff']:.3e}")

    tot_steps = sum(r["steps_total"] for r in out["systems"])
    tot_viol = sum(r["violations"] for r in out["systems"])
    out["pooled"] = stats.summarize_safety("pooled", tot_viol,
                                           sum(r["trials"] for r in out["systems"]))
    out["pooled"]["one_step_transitions"] = tot_steps
    print(f"\npooled: {out['pooled']['trials']} episodes, {tot_steps:,} one-step transitions, "
          f"{tot_viol} violations, CP95 upper {out['pooled']['cp95_upper_pct']:.4f}%")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e1_generality.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(RES,'e1_generality.json')}")


if __name__ == "__main__":
    main()
