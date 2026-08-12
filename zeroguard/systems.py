"""Monotone control systems with a safe null input.

The IECON filter is written for a battery, but nothing in its argument is
electrochemical. What it actually needs is a system whose constraints are monotone in a
scalar non-negative input, and for which `u = 0` returns the state to the safe set. This
module supplies four such systems from unrelated physics, plus two systems that violate
one hypothesis each, so the boundary of the theorem can be measured rather than asserted.

Every system exposes the same three things:

    u_max                       the largest input the actuator can deliver
    probe(s, u, dt, w)          one-step prediction of each constrained signal
    step(s, u, dt, w)           advance the state
    limits                      ((index, sense, value), ...) with sense in {"<=", ">="}

`probe` returns a tuple of floats; `limits` says how each entry is compared. That is the
whole interface, and it is all `zeroguard.gfilter.project` needs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math import exp, asinh
import numpy as np

from safe_charge.rom import BatteryROM, ocv, TREF, _FRESH

SIGMA = 5.670374419e-8          # Stefan-Boltzmann [W m^-2 K^-4]


# --------------------------------------------------------------------------------------
# 1. Battery -- the published ROM, wrapped so the generic filter can drive it
# --------------------------------------------------------------------------------------
class BatterySystem:
    """The IECON cell. `probe` forwards to the published ROM so that any agreement with
    `safe_charge.filter.project_current` is exact rather than approximate."""
    name = "battery"
    physics = "electrochemical + lumped thermal"
    dt_nominal, w_nominal, horizon = 30.0, 25.0, 80

    def __init__(self, rom=None, Vlim=4.20, Tlim=45.0, margin=None):
        self.rom = rom if rom is not None else BatteryROM()
        self.u_max = 3.0 * self.rom.p["Q_nom"]
        m = self.rom.plating_margin() if margin is None else margin
        self.limits = ((0, "<=", Vlim), (1, "<=", Tlim), (2, ">=", m))

    def probe(self, s, u, dt, w):
        V, T, phi, soc = self.rom.probe(s, u, dt, w)
        return (V, T, phi)

    def step(self, s, u, dt, w):
        s2, o = self.rom.step(s, u, dt, w)
        return s2, (o["V"], o["T"], o["phi_an"])

    def init(self, soc0=0.10, T0=25.0):
        return self.rom.init_state(soc0, T0)

    def cap(self, s):
        return self.rom.plate_current_cap(s["T"])


# --------------------------------------------------------------------------------------
# 2. Resistive heater -- pure Joule heating, no chemistry at all
# --------------------------------------------------------------------------------------
class ResistiveHeater:
    """A heating element under a temperature limit. u is the drive current.

    T' = T + dt (u^2 R(T) - hA (T - T_amb)) / C.  Resistance rises with temperature, which
    is the same positive feedback the battery's Arrhenius term supplies. u = 0 removes all
    generation, so temperature relaxes monotonically to ambient: (A1) holds. Heat is
    u^2 R > 0 and increasing in u: (A2) holds."""
    name = "heater"
    physics = "Joule heating, temperature-dependent resistance"
    dt_nominal, w_nominal, horizon = 5.0, 20.0, 120

    def __init__(self, R0=2.4, alpha=4.1e-3, hA=0.42, C=95.0, Tlim=120.0, u_max=8.0):
        self.R0, self.alpha, self.hA, self.C = R0, alpha, hA, C
        self.u_max = u_max
        self.limits = ((0, "<=", Tlim),)

    def _R(self, T):
        return self.R0 * (1.0 + self.alpha * (T - 20.0))

    def probe(self, s, u, dt, w):
        T = s["T"]
        Tn = T + dt * (u * u * self._R(T) - self.hA * (T - w)) / self.C
        return (Tn,)

    def step(self, s, u, dt, w):
        (Tn,) = self.probe(s, u, dt, w)
        return dict(T=float(Tn)), (Tn,)

    def init(self, T0=20.0):
        return dict(T=float(T0))

    def cap(self, s):
        return self.u_max


# --------------------------------------------------------------------------------------
# 3. DC motor winding -- electromechanical, two coupled states
# --------------------------------------------------------------------------------------
class DCMotorWinding:
    """A DC motor driving a viscous load, under a winding-temperature limit and a speed
    limit. u is armature current.

    Copper loss u^2 R heats the winding; torque k_t u accelerates the rotor against viscous
    drag. u = 0 gives no loss and no torque, so the winding cools and the rotor decelerates:
    (A1) holds on both constraints. Both are increasing in u: (A2) holds."""
    name = "motor"
    physics = "electromechanical, coupled thermal + rotational"
    dt_nominal, w_nominal, horizon = 0.02, 25.0, 4000

    def __init__(self, R=0.31, kt=0.052, J=1.7e-4, b=9.0e-5, hA=0.031, C=42.0,
                 Tlim=155.0, wlim=880.0, u_max=26.0):
        self.R, self.kt, self.J, self.b = R, kt, J, b
        self.hA, self.C = hA, C
        self.u_max = u_max
        self.limits = ((0, "<=", Tlim), (1, "<=", wlim))

    def probe(self, s, u, dt, w_amb):
        T, om = s["T"], s["om"]
        Tn = T + dt * (u * u * self.R - self.hA * (T - w_amb)) / self.C
        omn = om + dt * (self.kt * u - self.b * om) / self.J
        return (Tn, omn)

    def step(self, s, u, dt, w_amb):
        Tn, omn = self.probe(s, u, dt, w_amb)
        return dict(T=float(Tn), om=float(omn)), (Tn, omn)

    def init(self, T0=25.0, om0=0.0):
        return dict(T=float(T0), om=float(om0))

    def cap(self, s):
        return self.u_max


# --------------------------------------------------------------------------------------
# 4. IGBT junction -- power electronics, two-node thermal network
# --------------------------------------------------------------------------------------
class IGBTJunction:
    """A power switch under a junction-temperature limit, with a two-node
    junction-to-case-to-ambient network. u is the switched current.

    Conduction loss is V_ce0 u + R_on u^2 and switching loss is proportional to u; both are
    increasing in u, and u = 0 removes them entirely. (A1) and (A2) hold."""
    name = "igbt"
    physics = "semiconductor conduction + switching loss, 2-node RC network"
    dt_nominal, w_nominal, horizon = 0.005, 40.0, 6000

    def __init__(self, Vce0=0.95, Ron=0.011, Esw=1.9e-4, fsw=8000.0,
                 Rjc=0.28, Rca=0.65, Cj=0.9, Cc=32.0, Tjlim=150.0, u_max=420.0):
        self.Vce0, self.Ron, self.Esw, self.fsw = Vce0, Ron, Esw, fsw
        self.Rjc, self.Rca, self.Cj, self.Cc = Rjc, Rca, Cj, Cc
        self.u_max = u_max
        self.limits = ((0, "<=", Tjlim),)

    def probe(self, s, u, dt, w):
        Tj, Tc = s["Tj"], s["Tc"]
        P = self.Vce0 * u + self.Ron * u * u + self.Esw * self.fsw * u
        q_jc = (Tj - Tc) / self.Rjc
        Tjn = Tj + dt * (P - q_jc) / self.Cj
        return (Tjn,)

    def step(self, s, u, dt, w):
        Tj, Tc = s["Tj"], s["Tc"]
        P = self.Vce0 * u + self.Ron * u * u + self.Esw * self.fsw * u
        q_jc = (Tj - Tc) / self.Rjc
        q_ca = (Tc - w) / self.Rca
        Tjn = Tj + dt * (P - q_jc) / self.Cj
        Tcn = Tc + dt * (q_jc - q_ca) / self.Cc
        return dict(Tj=float(Tjn), Tc=float(Tcn)), (Tjn,)

    def init(self, Tj0=30.0, Tc0=30.0):
        return dict(Tj=float(Tj0), Tc=float(Tc0))

    def cap(self, s):
        return self.u_max


# --------------------------------------------------------------------------------------
# 5. Radiative battery -- the same cell in vacuum (Act II)
# --------------------------------------------------------------------------------------
class RadiativeBatteryROM(BatteryROM):
    """The published cell with Newtonian cooling replaced by Stefan-Boltzmann radiation.

    In vacuum there is no convection, so hA (T - T_amb) becomes eps sigma A (T^4 - T_sink^4).
    Everything else -- overpotential, RC branch, plating proxy, SOC -- is untouched. The
    point of the substitution is that T^4 is still strictly increasing in T, so the cooling
    channel of Prop. 2 is unchanged and the certificate should be indifferent to which law
    applies. This class exists to test that, not to assume it."""

    def __init__(self, *a, eps=0.85, area=0.0042, T_sink=4.0, **k):
        super().__init__(*a, **k)
        self.eps, self.area, self.T_sink = eps, area, T_sink

    def _q_out(self, T_C):
        Tk = T_C + 273.15
        return self.eps * SIGMA * self.area * (Tk ** 4 - self.T_sink ** 4)

    def probe(self, s, I, dt, T_amb):
        p = self.p
        if self.rc == "euler" and dt > p["tau1"]:
            raise ValueError(f"dt={dt}s exceeds tau1={p['tau1']}s")
        soc, T, V1 = s["soc"], s["T"], s["V1"]
        aging = s.get("aging", _FRESH)
        Rfac = aging.get("Rfac", 1.0)
        a0, a3, Ea, A, cT, i0_0, i0_1 = p["eta_params"]
        Rohm = abs(a0 + a3 * exp(6.0 * (soc - 1.0))) \
            * exp(Ea * (1.0 / (T + 273.15) - 1.0 / TREF)) * self.scale["R"]
        i0 = i0_0 * exp(i0_1 * soc)
        i0 = 1e-3 if i0 < 1e-3 else (50.0 if i0 > 50.0 else i0)
        ohm = (I * Rohm) * Rfac
        act = (A * (1.0 + cT * (T - 25.0)) * asinh(I / (2.0 * i0))) * Rfac
        R1 = p.get("R1_frac", 0.3) * max(Rohm * Rfac, 1e-4)
        if self.rc == "exact":
            aa = exp(-dt / p["tau1"]); V1n = aa * V1 + (1.0 - aa) * I * R1
        else:
            V1n = V1 + dt * (-V1 / p["tau1"] + I * R1 / p["tau1"])
        eta = ohm + act + V1n
        V = ocv(soc) + eta
        Q_gen = I * ohm + I * act + V1n * V1n / max(R1, 1e-6) + I * (T + 273.15) * p["dUdT"]
        Tn = T + dt * (Q_gen - self._q_out(T)) / p["C_th"]
        socn = soc + I * dt / (3600.0 * self.Q_eff(aging))
        Tk = T + 273.15
        p0 = (p["pl_p0a"] + p["pl_p0b"] * soc + p.get("pl_p0c", 0.0) * soc * soc
              + p.get("pl_p0d", 0.0) * exp(6.0 * (soc - 1.0)))
        p1 = (p["pl_p1a"] + p["pl_p1b"] * soc + p.get("pl_p1c", 0.0) * soc * soc) \
            * exp(p["pl_ET"] * (1.0 / Tk - 1.0 / TREF))
        return V, Tn, (p0 - p1 * I) / self.scale["plate"], socn

    def step(self, s, I, dt, T_amb):
        V, Tn, phi, socn = self.probe(s, I, dt, T_amb)
        p = self.p
        soc, T, V1 = s["soc"], s["T"], s["V1"]
        aging = s.get("aging", _FRESH); Rfac = aging.get("Rfac", 1.0)
        ohm, act = self.eta_irrev(soc, T, I)
        ohm *= Rfac; act *= Rfac
        R1 = p.get("R1_frac", 0.3) * max(self.R_ohmic(soc, T) * Rfac, 1e-4)
        if self.rc == "exact":
            aa = np.exp(-dt / p["tau1"]); V1n = aa * V1 + (1.0 - aa) * I * R1
        else:
            V1n = V1 + dt * (-V1 / p["tau1"] + I * R1 / p["tau1"])
        out = dict(V=V, T=Tn, phi_an=phi, soc=socn, Q_out=self._q_out(T))
        return dict(soc=float(socn), T=float(Tn), V1=float(V1n), aging=aging), out


# --------------------------------------------------------------------------------------
# 6. Counterexample A -- the null input is NOT safe (breaks A1)
# --------------------------------------------------------------------------------------
class HoverQuadrotor:
    """A quadrotor holding altitude. u is total thrust; the constraint is a floor on
    altitude. Cutting the input does not make this system safe -- it makes it fall.

    This is the honest boundary of the theorem: (A1) fails, so no zero-anchored certificate
    exists, and the filter must say so rather than return a number. It is also why the
    battery result is a *charging* result: on discharge under load, zero current is not a
    backup either."""
    name = "quadrotor"
    physics = "rigid-body vertical dynamics (A1 violated)"
    dt_nominal, w_nominal, horizon = 0.05, 0.0, 200

    def __init__(self, m=1.4, g=9.81, hmin=2.0, u_max=32.0):
        self.m, self.g, self.hmin = m, g, hmin
        self.u_max = u_max
        self.limits = ((0, ">=", hmin),)

    def probe(self, s, u, dt, w):
        h, v = s["h"], s["v"]
        a = u / self.m - self.g
        vn = v + dt * a
        hn = h + dt * vn
        return (hn,)

    def step(self, s, u, dt, w):
        h, v = s["h"], s["v"]
        a = u / self.m - self.g
        vn = v + dt * a
        hn = h + dt * vn
        return dict(h=float(hn), v=float(vn)), (hn,)

    def init(self, h0=10.0, v0=0.0):
        return dict(h=float(h0), v=float(v0))

    def cap(self, s):
        return self.u_max


# --------------------------------------------------------------------------------------
# 7. Counterexample B -- the constraint is NOT monotone in u (breaks A2)
# --------------------------------------------------------------------------------------
class OscillatoryConstraint:
    """A synthetic plant whose constrained signal crosses its limit more than once, so the
    admissible set is a *union* of intervals rather than one.

    (A2) fails by construction. The question this answers is not whether the bisection still
    finds the global optimum -- it cannot -- but whether its failure is dangerous or merely
    wasteful. Because bisection maintains lo <= u* on the branch containing zero, it returns
    the right edge of the *first* admissible interval and never reaches the later ones. It
    therefore under-commands: it gives up performance, not safety. That distinction is worth
    measuring rather than assuming, and it is the reason (A2) is a hypothesis about
    optimality while (A1) is a hypothesis about safety."""
    name = "oscillatory"
    physics = "synthetic, constraint crosses the limit repeatedly (A2 violated)"
    dt_nominal, w_nominal, horizon = 1.0, 25.0, 60

    def __init__(self, Tlim=52.0, u_max=12.0):
        self.u_max = u_max
        self.limits = ((0, "<=", Tlim),)

    def probe(self, s, u, dt, w):
        T = s["T"]
        return (T + 3.9 * np.sin(1.55 * u) + 0.30 * u,)

    def step(self, s, u, dt, w):
        (Tn,) = self.probe(s, u, dt, w)
        return dict(T=float(Tn)), (Tn,)

    def init(self, T0=50.0):
        return dict(T=float(T0))

    def cap(self, s):
        return self.u_max


ALL_MONOTONE = (BatterySystem, ResistiveHeater, DCMotorWinding, IGBTJunction)
COUNTEREXAMPLES = (HoverQuadrotor, OscillatoryConstraint)
