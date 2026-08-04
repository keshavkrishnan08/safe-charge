"""Physics-based safety projection (forward-invariant in temperature and voltage).

The CERTIFIED set is S = {V <= Vlim, T <= Tlim}. It is control-invariant because the
always-available action I=0 removes every input-driven heat source (so T only relaxes
toward ambient) and drops V toward the OCV. Guaranteeing that each one-step transition
stays in S therefore renders S forward-invariant: recursive feasibility holds via the
I=0 fallback. (Prop. 1.)

The plating potential phi_an rides the SAME one-dimensional bisection, because it is
affine and decreasing in the applied current, but it is NOT part of the certified set.
Plating is history-dependent, and the I=0 backup does not clear the margin at the cold,
aged, high-SOC corner (above ~0.8 SOC phi_an(I=0) already sits below it), so no one-step
condition can certify it. It is ENFORCED here -- by the one-step margin together with the
temperature-dependent `rom.plate_current_cap` -- and monitored, not certified. See
Sec. IV-B of the paper.

We project the proposed current onto the largest admissible value by binary search on the
ONE-STEP ROM prediction, using measurable states only (V, T) plus the model-predicted
plating potential. Control margins (deltaV, deltaT, deltaP) buy robustness so the
high-fidelity DFN stays feasible despite ROM error.
"""
import numpy as np

def soc_ramp_margin(base, extra, soc0=0.60, soc1=0.85):
    """SOC-dependent plating margin: `base` at low SOC, `base+extra` at high SOC.
    Motivated by the DFN replay: the ROM-DFN plating gap grows at high SOC under
    dynamic profiles (the fixed-C-rate calibration under-estimates it there), so
    the filter must be more conservative exactly where the true risk is highest.
    Pass the returned callable as `project_current(..., margin=...)`."""
    def fn(soc):
        return base + extra * float(np.clip((soc - soc0)/(soc1 - soc0), 0.0, 1.0))
    return fn

def _eff_margin(margin, soc):
    return margin(soc) if callable(margin) else margin

def _feasible(rom, s, I, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP):
    _, o = rom.step(s, I, dt, T_amb)
    m = _eff_margin(margin, o["soc"])
    return (o["V"] <= Vlim - dV + 1e-9) and (o["T"] <= Tlim - dT + 1e-9) \
        and (o["phi_an"] >= m + dP - 1e-9)

def project_current(rom, s, I_prop, dt, T_amb, Vlim=4.20, Tlim=45.0, margin=None,
                    dV=0.03, dT=0.5, dP=0.0, iters=18, cool_frac=None):
    """Return largest I in [0, I_prop] keeping the one-step transition inside S.
    `margin` may be a float or a callable margin(soc).
    `cool_frac` adds a COOLING-ROBUSTNESS budget to the thermal margin: a fault in the
    heat-transfer coefficient is orthogonal to resistance and invisible to the overpotential
    estimator, so we bound it directly by reserving cool_frac*(T-T_amb) of headroom, which
    certifies safety for any cooling loss up to cool_frac/(1+cool_frac). This is the cooling
    channel of the two-channel monotone bound. Defaults to env COOL_FRAC (0.25).
    `dt` must not exceed the ROM's RC time constant (see BatteryROM.step)."""
    if margin is None: margin = rom.plating_margin()
    if cool_frac is None:
        import os as _oscf; cool_frac = float(_oscf.environ.get("COOL_FRAC", 0.25))
    dT = dT + cool_frac * max(0.0, s["T"] - T_amb)   # cooling-channel budget
    I_prop = max(0.0, float(I_prop))
    # temperature-dependent plating-safe current cap (closed-loop DFN-calibrated)
    if hasattr(rom, "plate_current_cap"):
        I_prop = min(I_prop, rom.plate_current_cap(s["T"]))
    if _feasible(rom, s, I_prop, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP):
        return I_prop, False
    lo, hi = 0.0, I_prop
    # if even 0 is infeasible (already-unsafe state), return 0 (best effort)
    if not _feasible(rom, s, 0.0, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP):
        return 0.0, True
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if _feasible(rom, s, mid, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP):
            lo = mid
        else:
            hi = mid
    return lo, True
