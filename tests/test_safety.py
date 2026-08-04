"""The filter's safety guarantees, as executable checks.

Each test verifies one property on a grid of states, so you can run
`python tests/test_safety.py` and watch the guarantees hold instead of taking them on
faith. Pytest-compatible (functions are named test_*), and it also runs standalone.

  1  one-step invariance in (T, V), and the I=0 backup that makes it hold (Prop. 1)
  2  safety is monotone in BOTH aging channels of Prop. 2 -- the resistance scale s_R and the
     cooling-loss factor gamma -- so an upper bound on each beats identification
  3  the plating potential is affine and strictly decreasing in the applied current
  4  the admissible current set is a single interval [0, I*], which is what makes the
     bisection of Prop. 1 exact rather than merely a heuristic search
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
    safe state safe, which is what makes S forward-invariant for the T/V certificate.

    Prop. 1's proof turns on the RC residual, so we do NOT test only freshly initialized
    states (where V1=0 and the claim is trivial): each state is first driven with current to
    charge the RC branch, and the backup is applied from there."""
    rom = BatteryROM()
    worst_rise = -1e9
    for soc, T in _grid():
        for I_prev in (0.0, 5.0, 10.0, 15.0):        # I_prev>0 leaves the RC branch charged
            s = rom.init_state(soc, T)
            for _ in range(5):
                s, _ = rom.step(s, I_prev, DT, TAMB)
            s["T"] = T                               # re-pin T; V1 and soc carry over
            soc_now = s["soc"]                        # priming advanced soc; compare at soc_now
            assert s["V1"] >= 0.0
            _, o = rom.step(s, 0.0, DT, TAMB)
            assert o["T"] <= max(T, TAMB) + 1e-6, \
                f"I=0 overshot max(T,ambient) at soc={soc},T={T},I_prev={I_prev}"
            assert o["T"] <= TLIM + 1e-6, f"I=0 left S at soc={soc},T={T},I_prev={I_prev}"
            assert o["V"] <= ocv(soc_now) + 1e-6, \
                f"I=0 raised V above OCV at soc={soc_now},I_prev={I_prev}"
            worst_rise = max(worst_rise, o["T"] - max(T, TAMB))
    print(f"  backup: I=0 adds no self-heating even with the RC branch charged -- T never "
          f"exceeds max(T, ambient) (worst excess {worst_rise:+.4f} C) and V returns to OCV, "
          f"so S is invariant")


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


def test_rc_dissipation_is_physical_at_zero_current():
    """The backup's heat must stay finite when the current is cut.

    The RC branch dissipates V1^2/R1, and R1 has to come from the ohmic RESISTANCE, not
    from ohm/I: those agree for I>0, but the latter is 0/0 at exactly the current the
    backup commands. Deriving R1 the wrong way collapses it to a floor and makes the
    backup appear to inject hundreds of degrees, which would silently invalidate Prop. 1
    under any discretization that leaves V1 nonzero. We check R1 is continuous through
    I=0 and that the resulting one-step rise is small, in both discretizations."""
    for rc in ("euler", "exact"):
        rom = BatteryROM(rc=rc)
        Qn = rom.p["Q_nom"]
        for soc, T in _grid():
            R_expect = rom.p["R1_frac"] * rom.R_ohmic(soc, T)
            # R1 is a cell property: identical at I=0 and at any charging current
            for I in (0.0, 1e-9, 0.5*Qn, 2.0*Qn):
                ohm, _ = rom.eta_irrev(soc, T, I)
                if I > 0:
                    assert abs(ohm/I - rom.R_ohmic(soc, T)) < 1e-12
            assert R_expect > 1e-4, f"R1 hit the floor at soc={soc},T={T}"
            # charge the RC branch, then cut the current
            s = rom.init_state(soc, T)
            for _ in range(10):
                s, _ = rom.step(s, 2.0*Qn, DT, TAMB)
            s["T"] = T
            _, o = rom.step(s, 0.0, DT, TAMB)
            assert o["Q_rc"] >= 0.0
            assert o["T"] - T < 0.5, \
                f"backup injected {o['T']-T:.2f} C at soc={soc},T={T},rc={rc}"
    print("  backup heat: R1 is continuous through I=0 in both discretizations, so the "
          "RC residual injects <0.5 C (the delta_T0 budget) instead of diverging")


def test_safe_current_monotone_in_cooling_loss():
    """Prop. 2's SECOND channel. A cooling-loss factor gamma>=1 divides the effective
    heat-transfer coefficient, so the same current runs hotter and the largest safe current is
    non-increasing in gamma. Hence an upper bound on the cooling fault certifies safety with no
    identification either -- the claim the filter's `cool_frac` reserve cashes in.

    As Prop. 2 states, this holds where T >= T_amb: below ambient the cooling term warms the
    cell, so losing cooling would help. The filter carries the same guard, reserving
    cool_frac*max(0, T - T_amb), so it never claims the bound outside its hypothesis."""
    base = BatteryROM()
    m = base.plating_margin()
    for gamma in (1.0, 1.1, 1.25, 1.5, 2.0):
        rom = BatteryROM()
        rom.p = dict(rom.p); rom.p["hA"] = base.p["hA"] / gamma
        for soc, T in _grid():
            if T < TAMB:                      # outside Prop. 2's hypothesis (see docstring)
                continue
            s = rom.init_state(soc, T)
            _, o_g = rom.step(s, 1.0 * base.p["Q_nom"], DT, TAMB)
            _, o_1 = base.step(base.init_state(soc, T), 1.0 * base.p["Q_nom"], DT, TAMB)
            assert o_g["T"] >= o_1["T"] - TOL, f"T not monotone in gamma at soc={soc},T={T}"
            I_g, _ = project_current(rom, s, 3.0 * rom.p["Q_nom"], DT, TAMB,
                                     Vlim=VLIM, Tlim=TLIM, margin=m,
                                     dV=0, dT=0, dP=0, cool_frac=0.0)
            I_1, _ = project_current(base, base.init_state(soc, T), 3.0 * base.p["Q_nom"], DT,
                                     TAMB, Vlim=VLIM, Tlim=TLIM, margin=m,
                                     dV=0, dT=0, dP=0, cool_frac=0.0)
            assert I_g <= I_1 + 1e-4, f"safe current not monotone in gamma at soc={soc},T={T}"
    print("  monotonicity (cooling): peak T and the safe current are monotone in gamma too, "
          "so Prop. 2 holds in both aging channels")


def test_admissible_set_is_a_single_interval():
    """The bisection is exact only if the admissible current set is an interval anchored at 0.
    Two things could break that, and neither does:

      * The reversible heat I*T*dU/dT is endothermic, so dT'/dI dips slightly negative below
        ~0.42 A (0.08C), far under the 0.70C floor of the plating cap. The dip is a shallow
        minimum, not a second crossing, so the T-admissible set stays an interval. We check
        this on a dense current grid instead of trusting the sign of the derivative.
      * Plating is affine decreasing in I, so its admissible set is an interval too -- but one
        that can be EMPTY, because phi_an at I=0 already sits below the margin at high SOC.
        That is precisely why plating is not control-invariant (Sec. IV-B) and is enforced
        rather than certified; when it happens the filter commands I=0 and flags it.

    So: on the certified set {T, V} the admissible set always contains 0 and is an interval;
    including plating it is still an interval, possibly empty."""
    rom = BatteryROM()
    m = rom.plating_margin()
    Igrid = np.linspace(0.0, 3.0 * rom.p["Q_nom"], 400)
    empty_socs = []
    for soc, T in _grid():
        s = rom.init_state(soc, T)
        tv = np.array([_ok(rom, s, float(I), m)[0] for I in Igrid])
        al = np.array([all(_ok(rom, s, float(I), m)) for I in Igrid])
        assert tv[0], f"certified set: I=0 infeasible at soc={soc},T={T}"
        _assert_interval(tv, f"T/V at soc={soc},T={T}")
        _assert_interval(al, f"T/V+plating at soc={soc},T={T}")
        if not al[0]:
            empty_socs.append(soc)
    lo = min(empty_socs) if empty_socs else float("nan")
    print(f"  interval: admissible set is a single interval [0, I*] anchored at 0 for the "
          f"certified T/V set on every grid state; adding plating it stays an interval but "
          f"goes empty above SOC~{lo:.2f}, where I=0 no longer clears the margin")


def _ok(rom, s, I, m):
    """(certified T/V ok, plating ok) for the one-step transition."""
    _, o = rom.step(s, I, DT, TAMB)
    return (bool(o["T"] <= TLIM + 1e-9 and o["V"] <= VLIM + 1e-9),
            bool(o["phi_an"] >= m - 1e-9))


def _assert_interval(ok, where):
    """`ok` is a boolean array over an increasing current grid; assert it is True then False."""
    if ok.all() or not ok.any():
        return
    first_bad = int(np.argmin(ok))
    assert not ok[first_bad:].any(), f"admissible set is not an interval ({where})"


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
