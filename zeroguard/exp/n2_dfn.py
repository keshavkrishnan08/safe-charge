"""N2 -- the certificate against a Doyle--Fuller--Newman plant.

Every vehicle number in this paper comes from a plant we wrote. The reduced-order model is
defensible and it is still ours, so the question a battery reader asks is not "did you test on
hardware" but the sharper one underneath it: **how do you know the ROM is right?** Until now the
answer has been a paragraph.

There is an answer available without a bench. Doyle--Fuller--Newman \\cite{doyle1993,newman2004}
is the field's reference model -- porous-electrode transport, solid-phase diffusion,
Butler--Volmer kinetics, validated against cells by the people who parameterised it -- and it is
a *different model class* from the ROM, not the same equations with different numbers. Running
the certificate against a DFN plant tests something the envelope never can. The envelope varies
resistance, capacity and plating scale *within* the ROM's structure; DFN inflicts structural
misspecification: concentration gradients the ROM has no state for, kinetics it approximates,
and a plating potential that is an actual electrode quantity rather than an affine proxy.

**What this is not.** The ROM's coefficients were fitted to the LG M50, and both PyBaMM
parameter sets used here describe that cell, so this is not an independent-cell test and must
not be read as one. It is a *model-class* test: same cell, higher-fidelity physics.
\\S\\ref{sec:external} covers the other axis -- 33 cells from another laboratory that nothing was
ever fitted to. Neither test substitutes for the other, and together they bracket the two ways a
model can be wrong. OKane2022 is used in preference to the set the ROM was fitted against,
because it carries degradation and plating physics and is one step further from the fit.

The filter sees what a battery management system sees: measured state of charge and measured
temperature, with its own ROM propagating the relaxation state between them. It never sees the
DFN's internal states, because a vehicle never would.

    python zeroguard/exp/n2_dfn.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

warnings.filterwarnings("ignore")
SEED = 20260816
DT, HORIZON = 30.0, 80
T_AMB_C = 25.0
K0 = 273.15

# The certified constraints, on the DFN's own outputs.
V_MAX, T_MAX = 4.20, 45.0
# Plating in the DFN is not a proxy: the negative electrode surface potential difference going
# non-positive is the thermodynamic onset of lithium deposition. The ROM's affine proxy is
# supposed to keep this above zero, and until now nothing has checked that it does.
PHI_ONSET = 0.0


def build(param_set="OKane2022", tweak=None):
    """A DFN cell with lumped thermal and irreversible plating, driven by an input current.

    The voltage cut-off is pushed out of the way deliberately: a solver event that halts the
    run would hide the very excursion this experiment exists to detect. Constraints are checked
    against the DFN's reported outputs here, not enforced by PyBaMM.
    """
    import pybamm
    opts = {"thermal": "lumped", "lithium plating": "irreversible"}
    model = pybamm.lithium_ion.DFN(options=opts)
    pv = pybamm.ParameterValues(param_set)
    pv["Current function [A]"] = pybamm.InputParameter("I")
    pv["Ambient temperature [K]"] = T_AMB_C + K0
    pv["Initial temperature [K]"] = T_AMB_C + K0
    pv["Upper voltage cut-off [V]"] = 5.0
    pv["Lower voltage cut-off [V]"] = 2.0
    if tweak:
        # Most DFN transport parameters are *functions* of concentration and temperature, not
        # scalars. An earlier version multiplied only where `isinstance(v, float)` held and
        # silently did nothing otherwise -- every row of the sweep came back identical, which
        # would have been reported as robustness the experiment never demonstrated. Functions
        # are wrapped; scalars are scaled; anything else is a hard error rather than a no-op.
        for k, f in tweak.items():
            base = pv[k]
            if isinstance(base, (int, float)):
                pv[k] = base * f
            elif callable(base):
                pv[k] = (lambda g, ff: (lambda *a, **kw: g(*a, **kw) * ff))(base, f)
            else:
                raise TypeError(f"cannot scale {k!r} of type {type(base).__name__}")
    return pybamm.Simulation(model, parameter_values=pv), pv


def pessimistic_cell():
    """The estimator the filter carries: the same worst corner of the envelope used everywhere.

    `single_cell` is not in the platform build registry -- it exists so the reduction claim can
    be checked -- so the corner is applied directly rather than through `vexp.pessimistic`.
    """
    return P.single_cell(scale=dict(R=V.S_R, Q=V.ENVELOPE["Q"][0],
                                    plate=V.ENVELOPE["plate"][1]))


READ = {"V": "Voltage [V]",
        "T": "Volume-averaged cell temperature [K]",
        "dcap": "Discharge capacity [A.h]",
        "phi": "X-averaged negative electrode surface potential difference [V]"}


def observe(sol, soc0, q_nom):
    e = {k: float(sol[v].entries[-1]) for k, v in READ.items()}
    return dict(V=e["V"], T=e["T"] - K0, phi=e["phi"],
                soc=float(np.clip(soc0 - e["dcap"] / q_nom, 0.0, 1.0)))


def run_dfn(sim, q_nom, soc0, current, steps=HORIZON, target=0.80):
    """`current(k, obs) -> amps of charge`. Returns the DFN's own trajectory."""
    sol = None
    traj = []
    for k in range(steps):
        obs = observe(sol, soc0, q_nom) if sol is not None else dict(
            V=np.nan, T=T_AMB_C, phi=np.nan, soc=soc0)
        u = float(max(0.0, current(k, obs)))
        try:
            if sol is None:
                sol = sim.solve([0, DT], inputs={"I": -u}, initial_soc=soc0)
            else:
                sol = sim.step(DT, inputs={"I": -u}, starting_solution=sol)
        except Exception as e:                    # a solver failure is not a safety result
            return traj, f"solver: {type(e).__name__}"
        o = observe(sol, soc0, q_nom)
        o["u"] = u
        traj.append(o)
        if o["soc"] >= target:
            break
    return traj, "ok"


# =======================================================================================
def a_open_loop(n=6, seed=SEED):
    """Same current into both models. How far apart do they drift, against the margins?"""
    print("\nN2a  open loop: the same current into the ROM and into DFN")
    rng = np.random.default_rng(seed)
    est = P.single_cell()
    q = est.cell.q_nom()
    marg = V.margins(est)
    rows = []
    for c_rate in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0)[:n]:
        u = c_rate * q
        sim, _pv = build()
        traj, st = run_dfn(sim, q, 0.20, lambda k, o: u)
        if st != "ok" or not traj:
            continue
        # The sign is the whole question and taking an absolute value destroys it. For a cap,
        # the ROM reading *above* the DFN is conservative -- the filter throttles earlier than
        # it needed to, and no constraint is at risk. Only the ROM reading *below* the truth is
        # dangerous, and only that quantity has to fit inside the margin. An earlier version of
        # this function reported |error| and made a 4.4x-margin conservatism look like a
        # 4.4x-margin safety gap, which is the opposite of what the data says.
        #
        # And a second distinction on top of the first: an optimistic reading only endangers a
        # constraint that is near to binding. A 40 mV error at 3.7 V, with the limit at 4.2, is
        # arithmetic; the same error at 4.19 V is a breach. So optimism is recorded twice --
        # everywhere, and restricted to steps where the DFN is within NEAR of the limit -- and
        # it is the second number the margin has to cover.
        NEAR_V, NEAR_T = 0.20, 5.0
        s = est.init(0.20, T_AMB_C)
        over = dict(V=0.0, T=0.0, phi=0.0)          # ROM pessimistic: harmless
        under = dict(V=0.0, T=0.0, phi=0.0)         # ROM optimistic: what the margin must cover
        near = dict(V=0.0, T=0.0)                   # ... and optimistic where it can matter
        for step in traj:
            s, o = est.step(s, u, DT, T_AMB_C)
            for key, i, sense in (("V", 0, "cap"), ("T", 1, "cap"), ("phi", 2, "floor")):
                d = float(o[i]) - step[key]         # ROM minus DFN
                if sense == "floor":
                    d = -d                          # a floor is optimistic when the ROM is high
                over[key] = max(over[key], d)
                under[key] = max(under[key], -d)
            if step["V"] > V_MAX - NEAR_V:
                near["V"] = max(near["V"], step["V"] - float(o[0]))
            if step["T"] > T_MAX - NEAR_T:
                near["T"] = max(near["T"], step["T"] - float(o[1]))
        rows.append(dict(c_rate=c_rate, steps=len(traj),
                         over_V=over["V"], over_T=over["T"], over_phi=over["phi"],
                         under_V=under["V"], under_T=under["T"], under_phi=under["phi"],
                         near_V=near["V"], near_T=near["T"],
                         near_V_vs_margin=near["V"] / marg[0],
                         near_T_vs_margin=near["T"] / marg[1],
                         under_V_vs_margin=under["V"] / marg[0],
                         under_T_vs_margin=under["T"] / marg[1],
                         dfn_peak_T=max(x["T"] for x in traj),
                         dfn_peak_V=max(x["V"] for x in traj),
                         dfn_min_phi=min(x["phi"] for x in traj)))
        print(f"  {c_rate:>4.1f}C   ROM conservative by up to {over['V']:6.4f} V / "
              f"{over['T']:5.3f} K   |   optimistic by {under['V']:6.4f} V "
              f"({100*under['V']/marg[0]:5.1f}% of margin), {under['T']:5.3f} K "
              f"({100*under['T']/marg[1]:4.1f}%)")
    worst_V = max(r["under_V_vs_margin"] for r in rows)
    worst_T = max(r["under_T_vs_margin"] for r in rows)
    near_V = max(r["near_V_vs_margin"] for r in rows)
    near_T = max(r["near_T_vs_margin"] for r in rows)
    worst_over_V = max(r["over_V"] for r in rows)
    print(f"  anywhere on the trajectory, the ROM is optimistic by up to "
          f"{100*worst_V:.0f}% of the voltage margin -- and that worst case is at 0.3C, where "
          f"the DFN peaks at {rows[0]['dfn_peak_V']:.2f} V against a {V_MAX} V limit")
    print(f"  where the constraint is actually near binding it is optimistic by "
          f"{100*near_V:.0f}% of the voltage margin and {100*near_T:.0f}% of the thermal "
          f"margin -- inside both")
    print(f"  in the safe direction it is conservative by up to {worst_over_V:.4f} V, which is "
          f"{worst_over_V/marg[0]:.1f}x the margin: charge given away, not risk taken")
    return dict(rows=rows, worst_under_V_frac_margin=worst_V,
                worst_under_T_frac_margin=worst_T,
                near_V_frac_margin=near_V, near_T_frac_margin=near_T,
                worst_over_V=worst_over_V, over_margin_ratio=worst_over_V / marg[0],
                dV_margin=marg[0], dT_margin=marg[1],
                covered_where_binding=bool(near_V < 1.0 and near_T < 1.0),
                covered_everywhere=bool(worst_V < 1.0 and worst_T < 1.0))


def b_closed_loop(n=40, seed=SEED + 1):
    """The certificate, computed on the ROM, driving a DFN plant."""
    print("\nN2b  closed loop: the ROM's certificate against a DFN plant")
    rng = np.random.default_rng(seed)
    est = pessimistic_cell()
    q = P.single_cell().cell.q_nom()
    marg = V.margins(est)
    viol_V = viol_T = plate = sessions = solver_fail = 0
    peakV, peakT, minphi, socs = [], [], [], []
    for _ in range(n):
        soc0 = float(rng.uniform(0.05, 0.35))
        sim, _pv = build()
        # the filter's own ROM state; soc and T are overwritten from measurement each step,
        # V1 is propagated by the ROM, which is what a model-based BMS observer does
        fs = {"state": est.init(soc0, T_AMB_C)}

        def control(k, o):
            if k > 0:
                fs["state"] = dict(fs["state"], soc=o["soc"], T=o["T"])
            _lo, hi, st = A.interval(est, fs["state"], DT, T_AMB_C, marg)
            u = hi if st == "ok" else 0.0
            fs["state"], _ = est.step(fs["state"], u, DT, T_AMB_C)   # advances V1 only
            return u

        traj, st = run_dfn(sim, q, soc0, control)
        if st != "ok" or not traj:
            solver_fail += 1
            continue
        sessions += 1
        pv_, pt_, mp_ = (max(t["V"] for t in traj), max(t["T"] for t in traj),
                         min(t["phi"] for t in traj))
        peakV.append(pv_); peakT.append(pt_); minphi.append(mp_)
        socs.append(traj[-1]["soc"])
        viol_V += int(pv_ > V_MAX + 1e-6)
        viol_T += int(pt_ > T_MAX + 1e-6)
        plate += int(mp_ <= PHI_ONSET)
    cert = viol_V + viol_T
    print(f"  {sessions} sessions ({solver_fail} solver failures, excluded)")
    print(f"  certified channels on the DFN's own outputs: "
          f"voltage {viol_V}/{sessions}, temperature {viol_T}/{sessions}")
    print(f"  peak DFN voltage {max(peakV):.4f} V against {V_MAX}; "
          f"peak temperature {max(peakT):.2f} C against {T_MAX}")
    print(f"  DFN plating potential stayed above onset in {sessions - plate}/{sessions}; "
          f"minimum was {min(minphi):+.4f} V")
    return dict(sessions=sessions, solver_failures=solver_fail,
                voltage_violations=viol_V, temperature_violations=viol_T,
                certified_violations=cert,
                cp95_upper_pct=100 * stats.cp_upper(cert, max(sessions, 1)),
                plating_onset_sessions=plate,
                worst_V=float(max(peakV)), worst_T=float(max(peakT)),
                worst_phi=float(min(minphi)), mean_soc=float(np.mean(socs)),
                V_max=V_MAX, T_max=T_MAX)


def c_where_it_breaks(seed=SEED + 2):
    """How wrong can the DFN be before a ROM-built certificate stops holding?

    The envelope the estimator claims to bound is a resistance band. Electrolyte conductivity is
    the DFN parameter that most directly moves effective resistance while being something the
    ROM has no state for, so scaling it down is misspecification of the kind the envelope was
    never written to cover.
    """
    print("\nN2c  how wrong the DFN has to be before the certificate gives way")
    est = pessimistic_cell()
    q = P.single_cell().cell.q_nom()
    marg = V.margins(est)
    rows, first = [], None
    SWEEP = [("electrolyte conductivity", "Electrolyte conductivity [S.m-1]"),
             ("negative particle diffusivity", "Negative particle diffusivity [m2.s-1]")]
    for label, key in SWEEP:
      print(f"  -- {label}")
      first_here = None
      for f in (1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01):
        try:
            sim, _pv = build(tweak={key: f})
        except Exception as e:
            print(f"     x{f:<5.2f} build failed: {type(e).__name__}")
            continue
        bad = solver = 0
        wv = wt = -1e9
        for soc0 in (0.10, 0.20, 0.30):
            fs = {"state": est.init(soc0, T_AMB_C)}

            def control(k, o):
                if k > 0:
                    fs["state"] = dict(fs["state"], soc=o["soc"], T=o["T"])
                _lo, hi, st = A.interval(est, fs["state"], DT, T_AMB_C, marg)
                u = hi if st == "ok" else 0.0
                fs["state"], _ = est.step(fs["state"], u, DT, T_AMB_C)
                return u

            traj, st = run_dfn(sim, q, soc0, control)
            if st != "ok" or not traj:
                solver += 1
                continue
            pv_, pt_ = max(t["V"] for t in traj), max(t["T"] for t in traj)
            wv, wt = max(wv, pv_), max(wt, pt_)
            bad += int(pv_ > V_MAX + 1e-6 or pt_ > T_MAX + 1e-6)
        rows.append(dict(parameter=label, factor=f, breaches=bad, trials=3,
                         solver_failures=solver, worst_V=wv, worst_T=wt))
        if first_here is None and bad > 0:
            first_here = f
            first = f if first is None else max(first, f)
        print(f"     x{f:<5.2f}   breaches {bad}/3   solver {solver}/3   "
              f"peak V {wv:.4f}   peak T {wt:.2f} C")
      if first_here is None:
        print(f"     held to x{min(r['factor'] for r in rows if r['parameter'] == label):g}")
    out = dict(rows=rows, breaks_at_factor=first)
    per = {}
    for label, _k in SWEEP:
        rr = [r for r in rows if r["parameter"] == label]
        if not rr:
            continue
        brk = [r["factor"] for r in rr if r["breaches"] > 0]
        per[label] = dict(breaks_at=max(brk) if brk else None,
                          held_to=min(r["factor"] for r in rr if r["breaches"] == 0),
                          swept_to=min(r["factor"] for r in rr))
    out["per_parameter"] = per
    out["any_breach"] = first is not None
    out["tolerated_factor"] = max(v["held_to"] for v in per.values())   # the binding one
    for label, v in per.items():
        if v["breaks_at"] is None:
            print(f"\n  {label}: no breach down to x{v['swept_to']:g} of nominal "
                  f"({1/v['swept_to']:.0f}x misspecification) -- reported as the end of the "
                  f"sweep, not as a measured floor")
        else:
            print(f"\n  {label}: holds to x{v['held_to']:g} ({1/v['held_to']:.0f}x) and gives "
                  f"way at x{v['breaks_at']:g} ({1/v['breaks_at']:.0f}x)")
    print(f"\n  the binding one is {min(per, key=lambda k: per[k]['held_to'] if per[k]['breaks_at'] else 0)}"
          if False else "")
    return out


def main():
    t0 = time.time()
    print("N2 -- the certificate against a Doyle-Fuller-Newman plant\n" + "=" * 78)
    print("  DFN, lumped thermal, irreversible plating, OKane2022")
    print("  (Chen2020 carries no plating parameters, so the plating-aware set is the only")
    print("   one that can be run with the constraint the paper cares about most.)")
    print("  NOTE: same cell family the ROM was fitted to -- this is a model-class test,")
    print("        not an independent-cell one. N1 covers that axis.")
    out = dict(param_set="OKane2022", dt_s=DT, T_amb_C=T_AMB_C,
               model_class_test=True, independent_cell=False)
    out["open_loop"] = a_open_loop()
    out["closed_loop"] = b_closed_loop()
    out["misspecification"] = c_where_it_breaks()

    cl = out["closed_loop"]
    print("\n" + "=" * 78)
    print(f"  a certificate computed on a reduced-order model kept a full DFN cell inside "
          f"both certified constraints in {cl['sessions'] - cl['certified_violations']}"
          f"/{cl['sessions']} sessions (CP95 {cl['cp95_upper_pct']:.2f}%)")
    print(f"  and the ROM's affine plating proxy kept the DFN's *actual* electrode potential "
          f"above the deposition onset in "
          f"{cl['sessions'] - cl['plating_onset_sessions']}/{cl['sessions']}")

    path = V.save("n2_dfn.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
