"""The filter's safety guarantees, as executable checks.

Each test verifies one property on a grid of states, so you can run
`python tests/test_safety.py` and watch the guarantees hold instead of taking them on
faith. Pytest-compatible (functions are named test_*), and it also runs standalone.

  1  one-step invariance in (T, V), and the I=0 backup that makes it hold
  2  safety is monotone in the resistance scale s_R, so an upper bound beats identification
  3  the plating potential is affine and strictly decreasing in the applied current
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from safe_charge import BatteryROM, ocv, project_current

DT, TAMB, VLIM, TLIM = 30.0, 25.0, 4.20, 45.0
TOL = 1e-6


def _grid():
    """States inside the safe set S at ambient (fresh init, so V=OCV<VLIM, T<TLIM)."""
    for soc in np.linspace(0.05, 0.90, 12):
        for T in np.linspace(TAMB - 3, TLIM - 1.0, 8):
            yield float(soc), float(T)


def test_projection_keeps_next_state_safe():
    """The projected current always leaves the NEXT-step T and V inside S (the certified
    constraints). No robustness margins here, so this is the bare invariance claim."""
    rom = BatteryROM()
    m = rom.plating_margin()
    worstT = worstV = 0.0
    for soc, T in _grid():
        s = rom.init_state(soc, T)
        I, _ = project_current(rom, s, 3.0 * rom.p["Q_nom"], DT, TAMB,
                               Vlim=VLIM, Tlim=TLIM, margin=m,
                               dV=0.0, dT=0.0, dP=0.0, cool_frac=0.0)
        assert I >= -TOL
        _, o = rom.step(s, I, DT, TAMB)
        assert o["T"] <= TLIM + 1e-6, f"T overshoot at soc={soc},T={T}: {o['T']}"
        assert o["V"] <= VLIM + 1e-6, f"V overshoot at soc={soc},T={T}: {o['V']}"
        worstT, worstV = max(worstT, o["T"]), max(worstV, o["V"])
    print(f"  invariance: projection safe on all grid states; worst next T={worstT:.3f}C V={worstV:.4f}V")


def test_zero_current_is_a_safe_backup():
    """I=0 adds no self-heating, so temperature only relaxes toward ambient (never above
    max(T, T_amb)) and voltage sits at OCV. With ambient below the limit this keeps every
    safe state safe, which is what makes S forward-invariant for the T/V certificate."""
    rom = BatteryROM()
    for soc, T in _grid():
        s = rom.init_state(soc, T)
        _, o = rom.step(s, 0.0, DT, TAMB)
        assert o["T"] <= max(T, TAMB) + 1e-6, f"I=0 overshot max(T,ambient) at soc={soc},T={T}"
        assert o["T"] <= TLIM + 1e-6, f"I=0 left S at soc={soc},T={T}"
        assert o["V"] <= ocv(soc) + 1e-6, f"I=0 raised V above OCV at soc={soc}"
    print("  backup: I=0 adds no self-heating (T -> ambient) and holds V=OCV, so the safe set is invariant")


def test_safe_current_monotone_in_resistance():
    """Higher resistance scale s_R makes the same current hotter, so the largest safe
    current is non-increasing in s_R. Hence a conservative UPPER bound on s_R certifies
    safety with no identification: a bound removes the need to know the true cell."""
    rom_lo = BatteryROM(cell_scale=dict(R=1.0, Q=1.0, plate=1.0))
    rom_hi = BatteryROM(cell_scale=dict(R=1.8, Q=1.0, plate=1.0))
    m = rom_lo.plating_margin()
    for soc, T in _grid():
        s = rom_lo.init_state(soc, T)
        # same current -> hotter at higher s_R
        I_test = 1.0 * rom_lo.p["Q_nom"]
        _, o_lo = rom_lo.step(s, I_test, DT, TAMB)
        _, o_hi = rom_hi.step(s, I_test, DT, TAMB)
        assert o_hi["T"] >= o_lo["T"] - TOL, f"T not monotone in s_R at soc={soc},T={T}"
        # largest safe current non-increasing in s_R
        I_lo, _ = project_current(rom_lo, s, 3.0 * rom_lo.p["Q_nom"], DT, TAMB,
                                  Vlim=VLIM, Tlim=TLIM, margin=m, dV=0, dT=0, dP=0, cool_frac=0.0)
        I_hi, _ = project_current(rom_hi, s, 3.0 * rom_hi.p["Q_nom"], DT, TAMB,
                                  Vlim=VLIM, Tlim=TLIM, margin=m, dV=0, dT=0, dP=0, cool_frac=0.0)
        assert I_hi <= I_lo + 1e-4, f"safe current not monotone in s_R at soc={soc},T={T}"
    print("  monotonicity: peak T and the safe current are monotone in s_R (an upper bound suffices, no ID)")


def test_plating_potential_affine_decreasing_in_current():
    """Anode plating potential phi_an is affine in current everywhere, and strictly
    decreasing on the fast-charge operating envelope (SOC <= 0.9; the charge target is
    ~0.8). Its slope in I flips sign only above SOC ~0.92, a near-full-charge regime the
    filter never enters. We verify affinity on the full grid and monotonicity on the
    envelope, and report the sign-flip boundary rather than hide it."""
    rom = BatteryROM()
    for soc in np.linspace(0.1, 0.95, 12):        # affinity: everywhere
        for T in (0.0, 15.0, 30.0):
            phi = [rom.phi_an(float(soc), I, float(T)) for I in (0.0, 1.0, 2.0, 3.0)]
            d = np.diff(phi)
            assert np.allclose(d, d[0], atol=1e-9), f"phi_an not affine in I at soc={soc},T={T}"
    for soc in np.linspace(0.1, 0.90, 12):        # strict decrease: operating envelope
        for T in (0.0, 15.0, 30.0):
            phi = [rom.phi_an(float(soc), I, float(T)) for I in (0.0, 1.0, 2.0, 3.0)]
            assert np.all(np.diff(phi) < 0), f"phi_an not decreasing at soc={soc},T={T}: {phi}"
    # report the documented boundary
    socs = np.linspace(0, 1, 1001)
    p1 = rom.p["pl_p1a"] + rom.p["pl_p1b"] * socs + rom.p.get("pl_p1c", 0.0) * socs * socs
    flip = float(socs[np.argmax(p1 <= 0)])
    print(f"  plating: phi_an affine everywhere, strictly decreasing for SOC<=0.9 "
          f"(slope flips only above SOC~{flip:.2f}, outside the charge envelope)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} safety-property checks...")
    for t in tests:
        t()
    print("ALL SAFETY CHECKS PASSED")
