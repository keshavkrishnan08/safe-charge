"""Physics-based safety projection (forward-invariant).

For a lumped electro-thermal cell the set S = {V<=Vlim, T<=Tlim, phi_an>=margin}
is CONTROL-INVARIANT: the always-available action I=0 minimizes heat (Q_gen->Q_rev
~0) so T is non-increasing, and lowers V and raises phi_an. Hence guaranteeing
each one-step transition stays in S renders S forward-invariant (recursive
feasibility holds via the I=0 fallback). We therefore project the proposed
current to the largest admissible value by binary search on the ONE-STEP ROM
prediction, using measurable states only (V, T) plus the model-predicted plating
potential. Control margins (deltaV, deltaT, deltaP) buy robustness so the
high-fidelity DFN stays feasible despite ROM error.
"""
import numpy as np

def soc_ramp_margin(base, extra, soc0=0.60, soc1=0.85):
    """SOC-dependent plating margin: `base` at low SOC, `base+extra` at high SOC.
    Motivated by the DFN replay: the ROM-DFN plating gap grows at high SOC under
    dynamic profiles (the fixed-C-rate calibration under-estimates it there), so
    the filter must be more conservative exactly where the true risk is highest."""
    def fn(soc):
        return base + extra * float(np.clip((soc - soc0)/(soc1 - soc0), 0.0, 1.0))
    return fn

def _eff_margin(margin, soc):
    return margin(soc) if callable(margin) else margin

def _feasible(rom, s, I, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP, gamma_th=0.0):
    _, o = rom.step(s, I, dt, T_amb)
    m = _eff_margin(margin, o["soc"])
    return (o["V"] <= Vlim - dV + 1e-9) and (o["T"] <= Tlim - dT + 1e-9) \
        and (o["phi_an"] >= m + dP - 1e-9)

def project_current(rom, s, I_prop, dt, T_amb, Vlim=4.20, Tlim=45.0, margin=None,
                    dV=0.03, dT=0.5, dP=0.0, iters=18, gamma_th=0.0, cool_frac=None):
    """Return largest I in [0, I_prop] keeping the one-step transition inside S.
    `margin` may be a float or a callable margin(soc); `gamma_th`>0 adds a DERIVED thermal
    current cap from the steady-state energy balance (see below).
    `cool_frac` adds a COOLING-ROBUSTNESS budget to the thermal margin: a fault in the
    heat-transfer coefficient is orthogonal to resistance and invisible to the overpotential
    estimator, so we bound it directly by reserving cool_frac*(T-T_amb) of headroom, which
    certifies safety for any cooling loss up to cool_frac/(1+cool_frac). This is the cooling
    channel of the two-channel monotone bound. Defaults to env COOL_FRAC (0.25)."""
    if margin is None: margin = rom.plating_margin()
    if cool_frac is None:
        import os as _oscf; cool_frac = float(_oscf.environ.get("COOL_FRAC", 0.25))
    dT = dT + cool_frac * max(0.0, s["T"] - T_amb)   # cooling-channel budget
    I_prop = max(0.0, float(I_prop))
    # temperature-dependent plating-safe current cap (closed-loop DFN-calibrated)
    if hasattr(rom, "plate_current_cap"):
        I_prop = min(I_prop, rom.plate_current_cap(s["T"]))
    if _feasible(rom, s, I_prop, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP, gamma_th):
        return I_prop, False
    lo, hi = 0.0, I_prop
    # if even 0 is infeasible (already-unsafe state), return 0 (best effort)
    if not _feasible(rom, s, 0.0, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP, gamma_th):
        return 0.0, True
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        if _feasible(rom, s, mid, dt, T_amb, Vlim, Tlim, margin, dV, dT, dP, gamma_th):
            lo = mid
        else:
            hi = mid
    return lo, True
