"""Act II -- the certificate does not depend on how the cell sheds heat.

On Earth a cell loses heat by convection, hA (T - T_amb). In vacuum there is no air, and the
only path out is radiation, eps sigma A (T^4 - T_sink^4). These are different physics: one is
linear in temperature, the other quartic; one needs a medium, the other does not; one has an
ambient, the other has a 4 K sky.

Proposition 2 never asked for either. It asked for monotonicity. T^4 is strictly increasing
in T, so if the argument was really about structure rather than about air, the certificate
should survive the substitution untouched. This measures whether it does.

Three claims:
  E3a  the radiative one-step map is monotone in current, in resistance, and in emissivity
       loss -- the same three directions Prop. 2 needs
  E3b  the certificate holds over a five-channel uncertainty envelope in vacuum
  E3c  safety degrades gracefully as the radiator is taken away entirely (eps -> 0), rather
       than failing at some threshold

    python zeroguard/exp/e3_radiative.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard.systems import RadiativeBatteryROM, SIGMA
from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM = 30.0, 45.0, 4.20
NSTEP, SOC_TGT = 80, 0.80

# five channels, worst-case direction second
ENV = {"R": (1.0, 1.8), "eps": (1.0, 0.60), "Cth": (1.0, 0.80),
       "plate": (1.0, 1.6), "Tsink": (4.0, 250.0)}
CHANS = list(ENV)
CORNER = {k: v[1] for k, v in ENV.items()}


def make(theta):
    r = RadiativeBatteryROM(eps=0.85 * theta["eps"], T_sink=theta["Tsink"])
    r.scale = dict(R=theta["R"], Q=1.0, plate=theta["plate"])
    r.p = dict(r.p); r.p["C_th"] *= theta["Cth"]
    return r


def drive(plant, est, margins=(0.03, 12.5, 0.006), cool_frac=0.0):
    s = plant.init_state(0.10, 25.0)
    Qn = plant.p["Q_nom"]
    mT = mV = -1e9; mP = 1e9
    for _ in range(NSTEP):
        I, _ = project_current(est, s, 3.0 * Qn, DT, 25.0, Vlim=VLIM, Tlim=TLIM,
                               margin=est.plating_margin(), dV=margins[0], dT=margins[1],
                               dP=margins[2], cool_frac=cool_frac)
        s, o = plant.step(s, float(max(0.0, I)), DT, 25.0)
        mT = max(mT, o["T"]); mV = max(mV, o["V"]); mP = min(mP, o["phi_an"])
        if s["soc"] >= SOC_TGT:
            break
    return mT, mV, mP, s["soc"]


def e3a_monotonicity():
    """The three partial derivatives Prop. 2 requires, under the quartic law."""
    bad_I = bad_R = bad_eps = 0; n = 0
    minI = minR = minE = 1e18
    for soc in np.linspace(0.05, 0.85, 24):
        for T in np.linspace(-20.0, 44.0, 20):
            base = RadiativeBatteryROM()
            s = base.init_state(float(soc), float(T)); s["V1"] = 0.02
            for I in np.linspace(0.5, 15.0, 24):
                n += 1
                _, T1, _, _ = base.probe(s, float(I), DT, 25.0)
                _, T2, _, _ = base.probe(s, float(I) + 1e-4, DT, 25.0)
                d = (T2 - T1) / 1e-4; minI = min(minI, d); bad_I += d < -1e-9
                hi = RadiativeBatteryROM(); hi.scale = dict(R=1.4, Q=1.0, plate=1.0)
                _, T3, _, _ = hi.probe(s, float(I), DT, 25.0)
                dR = T3 - T1; minR = min(minR, dR); bad_R += dR < -1e-9
                # emissivity LOSS is the cooling channel: less eps must be hotter
                lo = RadiativeBatteryROM(eps=0.85 * 0.7)
                _, T4, _, _ = lo.probe(s, float(I), DT, 25.0)
                dE = T4 - T1; minE = min(minE, dE); bad_E = dE < -1e-9
                bad_eps += int(bad_E)
    return dict(samples=n, viol_dTdI=int(bad_I), viol_dTdR=int(bad_R),
                viol_dTdEpsLoss=int(bad_eps),
                min_dTdI=float(minI), min_dTdR=float(minR), min_dTdEpsLoss=float(minE))


def e3b_envelope(n_env=20000, seed=20260812):
    rng = np.random.default_rng(seed)
    est = make(CORNER)
    safe = 0; wT = wV = -1e9; wP = 1e9; socs = []
    t0 = time.time()
    for _ in range(n_env):
        th = {k: float(rng.uniform(min(ENV[k]), max(ENV[k]))) for k in CHANS}
        mT, mV, mP, soc = drive(make(th), est)
        wT = max(wT, mT); wV = max(wV, mV); wP = min(wP, mP); socs.append(soc)
        safe += int(mT <= TLIM and mV <= VLIM)
    cT, cV, cP, cSoc = drive(make(CORNER), est)
    m, lo, hi = stats.bootstrap_ci(np.array(socs))
    return stats.summarize_safety("radiative_envelope", n_env - safe, n_env, extra=dict(
        worst_T=float(wT), worst_V=float(wV), worst_plating=float(wP),
        corner_T=float(cT), corner_dominates=bool(cT >= wT - 1e-6),
        mean_soc=m, soc_ci=[lo, hi], seconds=round(time.time() - t0, 1)))


def e3c_vacuum_limit():
    """Take the radiator away, decade by decade, down to nothing."""
    rows = []
    for f in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3, 1e-4, 0.0]:
        th = dict(CORNER); th["eps"] = f if f > 0 else 1e-12
        plant = make(th); est = make(dict(CORNER, eps=max(f, 1e-12)))
        mT, mV, mP, soc = drive(plant, est)
        rows.append(dict(eps_fraction=f, peak_T=float(mT), peak_V=float(mV),
                         delivered_soc=float(soc), safe=bool(mT <= TLIM and mV <= VLIM)))
    socs = [r["delivered_soc"] for r in rows]; eps = [r["eps_fraction"] for r in rows]
    sp = stats.spearman_perm(eps, socs, reps=5000)
    return dict(sweep=rows, all_safe=all(r["safe"] for r in rows),
                spearman_soc_vs_emissivity=sp)


def main():
    out = {}
    print("Act II -- cooling-law invariance (the spacecraft case)\n")

    print("E3a  monotonicity of the quartic map in the three Prop. 2 directions")
    a = e3a_monotonicity(); out["e3a_monotonicity"] = a
    print(f"  {a['samples']:,} samples")
    print(f"  dT'/dI      >= 0 : violations {a['viol_dTdI']}   (min slope {a['min_dTdI']:+.4e})")
    print(f"  dT'/ds_R    >= 0 : violations {a['viol_dTdR']}   (min delta {a['min_dTdR']:+.4e})")
    print(f"  dT'/d(1/eps)>= 0 : violations {a['viol_dTdEpsLoss']}   (min delta {a['min_dTdEpsLoss']:+.4e})")

    print("\nE3b  five-channel envelope, in vacuum")
    b = e3b_envelope(); out["e3b_envelope"] = b
    print(f"  {b['trials']:,} draws | violations {b['violations']} | CP95 upper {b['cp95_upper_pct']:.4f}%")
    print(f"  worst T {b['worst_T']:.2f} C (limit {TLIM}) | worst V {b['worst_V']:.3f} | "
          f"corner dominates interior: {b['corner_dominates']}")
    print(f"  delivered SOC {b['mean_soc']:.3f}  95% CI [{b['soc_ci'][0]:.3f}, {b['soc_ci'][1]:.3f}]")

    print("\nE3c  taking the radiator away entirely")
    c = e3c_vacuum_limit(); out["e3c_vacuum_limit"] = c
    print(f"  {'eps frac':>10}{'peak T':>10}{'SOC':>9}{'safe':>7}")
    for r in c["sweep"]:
        print(f"  {r['eps_fraction']:>10.4g}{r['peak_T']:>10.2f}{r['delivered_soc']:>9.3f}{str(r['safe']):>7}")
    print(f"  all safe: {c['all_safe']} | Spearman(SOC, emissivity) rho="
          f"{c['spearman_soc_vs_emissivity']['rho']:.3f} p={c['spearman_soc_vs_emissivity']['p']:.4g}")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e3_radiative.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(RES,'e3_radiative.json')}")


if __name__ == "__main__":
    main()
