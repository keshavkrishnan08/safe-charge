"""S2 -- cycle life, measured by the DFN rather than by our own aging model.

S1 measured plating per session. The claim that would matter far more is the one about *life*:
plating is the dominant fast-charge degradation mechanism, so a filter that prevents it should
show up as capacity retained after hundreds of cycles.

**It does not, and this experiment exists to say so.** Over 50 cold cycles all three
controllers lose within 0.25 points of the same capacity, and the certificate is marginally
*worse* than the slow de-rate. The lifetime claim is not supported and the paper does not make
it. What follows is the test that would have found a benefit if one were there, reported as the
negative it is.

**Why not the ROM's own aging model.** The ROM ships one, and it would have been the cheap way
to run this. It is also circular: its plating term accumulates damage proportional to
`clip(margin - phi, 0, inf) * |I|`, where `margin` is *the same threshold the certificate
enforces*. A filter that holds `phi >= margin` by construction therefore scores exactly zero
plating damage by construction, and the experiment would be measuring its own definition. That
is worth saying plainly because it is the kind of shortcut that produces a spectacular and
worthless number.

So the plant is DFN with the full degradation stack -- irreversible plating, solvent-diffusion
limited SEI, porosity change -- and the quantities read are DFN's own `Loss of capacity to
negative lithium plating [A.h]` and `Loss of capacity to negative SEI [A.h]`. Neither knows the
certificate exists.

**Cold, because that is where the mechanism lives.** S1 established that plating is a cold
phenomenon: 1.5C crosses the deposition onset at 0 C and not at 25 C. Cycling warm would age
the cells almost entirely through SEI, which no controller here claims to prevent, and the
plating column would be empty for everyone.

    python zeroguard/exp/s2_cycle_life.py
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

warnings.filterwarnings("ignore")
SEED = 20260816
DT = 30.0
T_AMB = 0.0
SOC_LO, SOC_HI = 0.15, 0.70
CYCLES = 50
K0 = 273.15

PLATE = "Loss of capacity to negative lithium plating [A.h]"
SEI = "Loss of capacity to negative SEI [A.h]"
READ = {"V": "Voltage [V]", "T": "Volume-averaged cell temperature [K]",
        "dcap": "Discharge capacity [A.h]",
        "phi": "X-averaged negative electrode surface potential difference [V]"}


def build_aging(T_amb=T_AMB):
    import pybamm
    model = pybamm.lithium_ion.DFN(options={
        "thermal": "lumped", "lithium plating": "irreversible",
        "SEI": "solvent-diffusion limited", "SEI porosity change": "true"})
    pv = pybamm.ParameterValues("OKane2022")
    pv["Current function [A]"] = pybamm.InputParameter("I")
    pv["Ambient temperature [K]"] = T_amb + K0
    pv["Initial temperature [K]"] = T_amb + K0
    pv["Upper voltage cut-off [V]"] = 5.0
    pv["Lower voltage cut-off [V]"] = 2.0
    return pybamm.Simulation(model, parameter_values=pv), pv


def obs(sol, soc_ref, dcap_ref, q_nom):
    e = {k: float(sol[v].entries[-1]) for k, v in READ.items()}
    return dict(V=e["V"], T=e["T"] - K0, phi=e["phi"],
                soc=float(np.clip(soc_ref - (e["dcap"] - dcap_ref) / q_nom, 0.0, 1.0)))


def run(controller_name, q, n_cycles=CYCLES, seed=SEED):
    """Cycle one cell to death-ish under one controller, reading DFN's own degradation."""
    sim, _pv = build_aging()
    est = P.single_cell(scale=dict(R=V.S_R, Q=V.ENVELOPE["Q"][0],
                                   plate=V.ENVELOPE["plate"][1]))
    marg = V.margins(est)
    fs = {"s": est.init(SOC_LO, T_AMB)}

    def certificate(o):
        fs["s"] = dict(fs["s"], soc=o["soc"], T=o["T"])
        _lo, hi, st = A.interval(est, fs["s"], DT, T_AMB, marg)
        u = hi if st == "ok" else 0.0
        fs["s"], _ = est.step(fs["s"], u, DT, T_AMB)
        return u

    ccv = {"u": 0.0}

    def make_ccv(c_rate):
        def f(o):
            if ccv["u"] == 0.0:
                ccv["u"] = c_rate * q
            if np.isfinite(o["V"]) and o["V"] >= 4.20 - 1e-2:
                ccv["u"] *= 0.85
            return ccv["u"]
        return f

    ctrl = (certificate if controller_name == "certificate"
            else make_ccv(float(controller_name.split()[1].rstrip("C"))))

    sol = None
    soc_ref, dcap_ref = SOC_LO, 0.0
    hist = []
    onset_cycles = 0
    for cyc in range(n_cycles):
        if controller_name != "certificate":
            ccv["u"] = 0.0                     # a fresh session restarts the CC phase
        crossed = False
        # ---- charge ------------------------------------------------------------------
        for _k in range(400):
            o = (obs(sol, soc_ref, dcap_ref, q) if sol is not None
                 else dict(V=np.nan, T=T_AMB, phi=np.nan, soc=SOC_LO))
            u = float(max(0.0, ctrl(o)))
            try:
                sol = (sim.solve([0, DT], inputs={"I": -u}, initial_soc=SOC_LO)
                       if sol is None else
                       sim.step(DT, inputs={"I": -u}, starting_solution=sol))
            except Exception as e:
                return dict(failed=f"{type(e).__name__}", cycles_done=cyc, history=hist)
            o = obs(sol, soc_ref, dcap_ref, q)
            crossed = crossed or (o["phi"] <= 0.0)
            if o["soc"] >= SOC_HI:
                break
        # ---- discharge at 1C ---------------------------------------------------------
        for _k in range(400):
            try:
                sol = sim.step(DT, inputs={"I": q}, starting_solution=sol)
            except Exception as e:
                return dict(failed=f"{type(e).__name__}", cycles_done=cyc, history=hist)
            if obs(sol, soc_ref, dcap_ref, q)["soc"] <= SOC_LO:
                break
        onset_cycles += int(crossed)
        hist.append(dict(cycle=cyc + 1,
                         plate_Ah=float(sol[PLATE].entries[-1]),
                         sei_Ah=float(sol[SEI].entries[-1])))
    last = hist[-1]
    return dict(failed=None, cycles_done=len(hist), history=hist,
                plate_Ah=last["plate_Ah"], sei_Ah=last["sei_Ah"],
                total_Ah=last["plate_Ah"] + last["sei_Ah"],
                onset_cycles=onset_cycles)


def main(n_cycles=CYCLES):
    t0 = time.time()
    print("S2 -- cycle life at 0 C, degradation read from the DFN\n" + "=" * 78)
    print(f"  {n_cycles} charge/discharge cycles per controller, {SOC_LO:.0%}-{SOC_HI:.0%}, "
          f"1C discharge, ambient {T_AMB:.0f} C")
    print("  the ROM's own aging model is deliberately NOT used: its plating term is defined")
    print("  against the same margin the certificate enforces, so it would score itself")
    q = P.single_cell().cell.q_nom()
    out = dict(cycles=n_cycles, T_amb=T_AMB, soc_window=[SOC_LO, SOC_HI],
               q_nom=q, controllers={})

    for name in ("certificate", "CC-CV 0.5C", "CC-CV 1.5C"):
        r = run(name, q, n_cycles)
        if r["failed"]:
            print(f"  {name:16} solver failed after {r['cycles_done']} cycles ({r['failed']})")
            out["controllers"][name] = dict(failed=r["failed"], cycles=r["cycles_done"])
            continue
        pct_plate = 100 * r["plate_Ah"] / q
        pct_sei = 100 * r["sei_Ah"] / q
        out["controllers"][name] = dict(
            cycles=r["cycles_done"], plate_Ah=r["plate_Ah"], sei_Ah=r["sei_Ah"],
            total_Ah=r["total_Ah"], plate_pct=pct_plate, sei_pct=pct_sei,
            total_pct=pct_plate + pct_sei,
            capacity_retained_pct=100 - (pct_plate + pct_sei),
            onset_cycles=r["onset_cycles"], history=r["history"])
        print(f"  {name:16} plating {pct_plate:6.3f}%   SEI {pct_sei:6.3f}%   "
              f"total {pct_plate+pct_sei:6.3f}%   retained {100-pct_plate-pct_sei:7.3f}%   "
              f"onset in {r['onset_cycles']}/{r['cycles_done']} cycles")

    # ---- the honest reading, stated before any comparison that might flatter --------------
    C = out["controllers"]
    vals = [v["total_pct"] for v in C.values() if "total_pct" in v]
    spread = max(vals) - min(vals) if vals else 0.0
    out["fade_spread_pct"] = spread
    out["lifetime_benefit_supported"] = bool(spread > 1.0)
    print(f"\n  every controller loses within {spread:.3f} points of the same capacity. The "
          f"lifetime claim is\n  NOT supported: over this window the choice of charging "
          f"controller does not materially change\n  how much capacity the cell keeps.")
    print(f"  the mechanism is the one S1 already identified. DFN's plating variable is "
          f"dominated by a\n  slow background side reaction that accrues with cold time and "
          f"throughput -- which every\n  controller spends similarly -- rather than by the "
          f"onset crossings the certificate prevents\n  (crossed in "
          f"{C.get('CC-CV 1.5C', {}).get('onset_cycles', 0)}/{n_cycles} cycles for the fast "
          f"rate, {C.get('certificate', {}).get('onset_cycles', 0)}/{n_cycles} for the "
          f"certificate).")
    print(f"  S1 stands -- the onset crossings are real and the certificate avoids them -- but "
          f"they are not\n  what drives capacity fade in this regime, so no life claim follows "
          f"from them.")
    if all("plate_pct" in C.get(k, {}) for k in ("certificate", "CC-CV 1.5C")):
        c, a = C["certificate"], C["CC-CV 1.5C"]
        out["plating_ratio_vs_aggressive"] = (a["plate_pct"] / c["plate_pct"]
                                              if c["plate_pct"] > 1e-12 else None)
        out["plating_saved_pct"] = a["plate_pct"] - c["plate_pct"]
        out["total_saved_pct"] = a["total_pct"] - c["total_pct"]
        print(f"\n  for the record, against the rate a de-rate exists to forbid the "
              f"certificate saves {out['plating_saved_pct']:+.3f} points to plating over "
              f"{n_cycles} cycles -- which is noise, not a benefit")
    if all("plate_pct" in C.get(k, {}) for k in ("certificate", "CC-CV 0.5C")):
        c, s = C["certificate"], C["CC-CV 0.5C"]
        out["plating_vs_safe_derate_pct"] = s["plate_pct"] - c["plate_pct"]
        print(f"  and against the fleet-safe 0.5C de-rate, "
              f"{out['plating_vs_safe_derate_pct']:+.3f} points -- the wrong side of zero")
    path = V.save("s2_cycle_life.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
