"""Vehicle platforms: ground, air, water, space.

The IECON filter certifies a cell being charged. A vehicle is not a cell being charged. It is
a pack, in a medium that carries heat away in a particular way, serving a load it is not
allowed to stop serving, over a mission during which nobody is going to recalibrate anything.
This module supplies the plants for all four of those differences so the claims about them can
be measured instead of asserted.

Three things vary across the platforms and nothing else does:

  **the medium**, through the cooling law -- liquid loop, free air derated by altitude, a
  sealed hull in cold water, a radiator against 4 K;

  **the pack**, through series/parallel counts, which set how a per-cell certificate becomes a
  vehicle-level current limit;

  **the load**, through `anchor`, which is zero while charging and strictly positive while
  flying, swimming or coasting through eclipse.

The cell underneath is the published ROM's arithmetic in every case -- same overpotential
form, same plating proxy, same coefficients out of `rom_params.json` -- with two changes that
the vehicles require and the original paper never needed. Current may be negative, because
vehicles discharge. And the Newtonian cooling term is replaced by a callable, because a
quadrotor at 400 m, an AUV at 2 C and a spacecraft in vacuum do not share a cooling law.
`tests/test_platforms.py` asserts that with the Newtonian law, zero load and positive current,
this model reproduces `BatteryROM.probe` bit for bit; everything that follows inherits that.

**On the named case studies.** `RobotaxiUrban`, `DeliveryQuadrotor`, `SurveyAUV` and the rest
are *class-representative* parameter sets: pack sizes, hover powers, hotel loads and orbit
geometry chosen to put each platform in the right region of the design space, using published
figures for the vehicle class. They are not any manufacturer's design and no claim here should
be read as one. What the case studies are for is that "it works on autonomous vehicles" is not
a testable sentence and "it holds a 96S45P pack under a 30-minute 350 kW session at 45 C
ambient" is.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import lru_cache
from math import exp, asinh
import numpy as np

from safe_charge.rom import load_params as _load_params_uncached, ocv, TREF, _FRESH


@lru_cache(maxsize=1)
def _params_once():
    return _load_params_uncached()


def load_params():
    """The calibrated coefficients, read from disk exactly once.

    `safe_charge.rom.load_params` opens and parses `rom_params.json` on every call, which is
    invisible when a program builds one cell and ruinous when it builds 33 120 -- V1-5 does
    exactly that, and constructing the truck pack took longer than every other experiment in
    the register combined. The dict is returned by copy so a caller that mutates its own
    parameters (the Mars platform does) cannot poison the cache."""
    return dict(_params_once())

SIGMA = 5.670374419e-8          # Stefan-Boltzmann [W m^-2 K^-4]
G0 = 9.80665


# =======================================================================================
# Cooling laws -- the only place the medium enters
# =======================================================================================
class Newtonian:
    """hA (T - T_amb). The published law; a liquid loop or still air."""
    kind = "newtonian"

    def __init__(self, hA):
        self.hA = float(hA)

    def q_out(self, T_C, w):
        return self.hA * (T_C - w)

    def describe(self):
        return f"hA = {self.hA:.4f} W/K per cell"


class ForcedAir(Newtonian):
    """Free convection augmented by airspeed and derated by air density.

    A rotorcraft cools better in cruise than in hover and worse at altitude. Both effects
    multiply hA by something positive and independent of current, so the cooling channel stays
    strictly increasing in T and the certificate is untouched -- which is the claim, and the
    reason this is a separate class rather than a comment."""
    kind = "forced-air"

    def __init__(self, hA0, v_air=0.0, v_ref=12.0, altitude_m=0.0):
        super().__init__(hA0)
        self.v_air, self.v_ref = float(v_air), float(v_ref)
        self.altitude_m = float(altitude_m)

    def _rho_ratio(self):
        # ISA troposphere, density relative to sea level
        h = self.altitude_m
        return max(0.15, (1.0 - 2.25577e-5 * h) ** 4.2559)

    def q_out(self, T_C, w):
        boost = (1.0 + (self.v_air / self.v_ref) ** 0.8) * self._rho_ratio() ** 0.5
        return self.hA * boost * (T_C - w)

    def describe(self):
        return (f"hA = {self.hA:.4f} W/K, v_air = {self.v_air:.1f} m/s, "
                f"alt = {self.altitude_m:.0f} m (rho/rho0 = {self._rho_ratio():.3f})")


class HullConduction(Newtonian):
    """A sealed pressure hull in cold water.

    The pack cannot vent, so every watt leaves through the hull: a series conductance of the
    internal air gap and the hull wall, into water that is near 2 C below the thermocline and
    is an excellent sink once the heat reaches it. The result is a small UA against a cold
    reservoir, which is a different regime from either the car or the spacecraft -- poorly
    coupled, but to something very cold."""
    kind = "hull"

    def __init__(self, UA_int, UA_hull, T_water=2.0):
        self.UA_int, self.UA_hull = float(UA_int), float(UA_hull)
        super().__init__(1.0 / (1.0 / self.UA_int + 1.0 / self.UA_hull))
        self.T_water = float(T_water)

    def q_out(self, T_C, w):
        return self.hA * (T_C - self.T_water)

    def describe(self):
        return (f"UA_series = {self.hA:.4f} W/K (int {self.UA_int}, hull {self.UA_hull}) "
                f"into {self.T_water:.1f} C water")


class Radiative:
    """eps sigma A (T^4 - T_sink^4). Vacuum: the only way out is a photon.

    T^4 is strictly increasing in T, so Prop. 2's cooling channel is unchanged; E3 established
    that at cell level and this carries it onto the spacecraft platforms."""
    kind = "radiative"

    def __init__(self, eps=0.85, area=0.0042, T_sink=4.0):
        self.eps, self.area, self.T_sink = float(eps), float(area), float(T_sink)

    def q_out(self, T_C, w):
        Tk = T_C + 273.15
        return self.eps * SIGMA * self.area * (Tk ** 4 - self.T_sink ** 4)

    def describe(self):
        return (f"eps = {self.eps:.2f}, A = {self.area*1e4:.1f} cm^2, "
                f"T_sink = {self.T_sink:.0f} K")


class RadiativePlusThin(Radiative):
    """Radiation plus the small convective term a thin atmosphere still provides.

    Mars is ~6 mbar of CO2: not nothing, and not enough. Both terms increase with T, so their
    sum does."""
    kind = "radiative+thin"

    def __init__(self, eps=0.85, area=0.0042, T_sink=4.0, hA=0.006, T_amb=-60.0):
        super().__init__(eps, area, T_sink)
        self.hA, self.T_amb = float(hA), float(T_amb)

    def q_out(self, T_C, w):
        return super().q_out(T_C, w) + self.hA * (T_C - self.T_amb)

    def describe(self):
        return super().describe() + f" + hA = {self.hA:.4f} W/K at {self.T_amb:.0f} C"


# =======================================================================================
# The cell, with a pluggable medium and a sign
# =======================================================================================
class Cell:
    """The published ROM's arithmetic, with the cooling law lifted out and current signed.

    Positive current charges. Negative current discharges: the overpotential changes sign so
    terminal voltage sags instead of rising, SOC falls, and the I^2 R heating stays positive
    because it is a product of two sign-matched terms. The plating proxy is only meaningful on
    charge and is reported regardless so the caller can decide."""

    def __init__(self, cooling=None, scale=None, params=None, rc="exact"):
        self.p = load_params() if params is None else dict(params)
        self.cooling = Newtonian(self.p["hA"]) if cooling is None else cooling
        self.scale = dict(R=1.0, Q=1.0, plate=1.0) if scale is None else dict(scale)
        self.rc = rc

    # -- one-step prediction of every signal any platform constrains ---------------------
    def probe(self, soc, T, V1, I, dt, w, aging=_FRESH):
        p = self.p
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
            a = exp(-dt / p["tau1"]); V1n = a * V1 + (1.0 - a) * I * R1
        else:
            V1n = V1 + dt * (-V1 / p["tau1"] + I * R1 / p["tau1"])
        eta = ohm + act + V1n
        V = float(ocv(soc)) + eta
        Q_gen = I * ohm + I * act + V1n * V1n / max(R1, 1e-6) + I * (T + 273.15) * p["dUdT"]
        Tn = T + dt * (Q_gen - self.cooling.q_out(T, w)) / p["C_th"]
        Qeff = p["Q_nom"] * self.scale["Q"] * (1.0 - aging.get("Qloss", 0.0))
        socn = soc + I * dt / (3600.0 * Qeff)
        Tk = T + 273.15
        p0 = (p["pl_p0a"] + p["pl_p0b"] * soc + p.get("pl_p0c", 0.0) * soc * soc
              + p.get("pl_p0d", 0.0) * exp(6.0 * (soc - 1.0)))
        p1 = (p["pl_p1a"] + p["pl_p1b"] * soc + p.get("pl_p1c", 0.0) * soc * soc) \
            * exp(p["pl_ET"] * (1.0 / Tk - 1.0 / TREF))
        phi = (p0 - p1 * I) / self.scale["plate"]
        return V, Tn, phi, socn, V1n

    def q_nom(self):
        return self.p["Q_nom"] * self.scale["Q"]

    def plating_margin(self):
        return self.p["plating_margin_V"]

    def plate_cap(self, T_C):
        crate = float(np.clip(1.00 + 0.067 * (T_C - 10.0), 0.70, 2.00))
        return crate * self.q_nom()


# =======================================================================================
# Platform base -- the anchored interface
# =======================================================================================
class Platform:
    """A pack in a medium, in one of two modes.

    `mode="charge"`: input is pack charge current, anchor is zero, and Anchored Collapse
    degenerates to Null-Input Collapse. This is the IECON case, at vehicle scale.

    `mode="discharge"`: input is pack discharge current (positive), the anchor is the current
    that serves the irreducible load, and there is a floor constraint that the null-input
    theorem has no way to represent.

    Signal tuple, shared by both modes so one filter drives either:

        0  V_cell   terminal voltage of the representative cell   [V]
        1  T        cell temperature after the step               [C]
        2  phi      plating-potential proxy                       [V]
        3  P_elec   electrical power delivered to the bus         [W]
        4  soc      state of charge after the step                [-]
    """
    domain = "generic"
    case = "generic"
    dt_nominal, horizon = 30.0, 80

    def __init__(self, cell, S, P, mode="charge", T_max=45.0, V_max=4.20, V_min=3.00,
                 soc_floor=0.0, load_W=0.0, w_nominal=25.0, u_max=None, name=None):
        self.cell, self.S, self.P = cell, int(S), int(P)
        self.mode = mode
        self.T_max, self.V_max, self.V_min = T_max, V_max, V_min
        self.soc_floor, self.load_W = soc_floor, float(load_W)
        self.w_nominal = w_nominal
        self.u_min = 0.0
        self.u_max = float(u_max) if u_max is not None else self._default_umax()
        if name:
            self.case = name
        if mode == "charge":
            self.limits = ((0, "<=", V_max, "hi"),
                           (1, "<=", T_max, "hi"),
                           (2, ">=", cell.plating_margin(), "hi"))
        elif mode == "discharge":
            self.limits = ((1, "<=", T_max, "hi"),
                           (0, ">=", V_min, "hi"),
                           (4, ">=", soc_floor, "hi"),
                           (3, ">=", self.load_W, "lo"))
        else:
            raise ValueError(f"mode must be 'charge' or 'discharge', got {mode!r}")

    # -- pack geometry ------------------------------------------------------------------
    def _default_umax(self):
        return 3.0 * self.cell.q_nom() * self.P

    def energy_kWh(self):
        return self.S * self.P * self.cell.q_nom() * 3.63 / 1000.0

    def cell_current(self, u):
        """Pack current to the current the representative cell sees, signed for the mode."""
        i = u / self.P
        return i if self.mode == "charge" else -i

    # -- the anchored interface ---------------------------------------------------------
    def probe(self, s, u, dt, w):
        I = self.cell_current(u)
        V, Tn, phi, socn, _ = self.cell.probe(s["soc"], s["T"], s["V1"], I, dt, w,
                                              s.get("aging", _FRESH))
        P_elec = self.S * V * u
        return (V, Tn, phi, P_elec, socn)

    def step(self, s, u, dt, w):
        I = self.cell_current(u)
        V, Tn, phi, socn, V1n = self.cell.probe(s["soc"], s["T"], s["V1"], I, dt, w,
                                                s.get("aging", _FRESH))
        s2 = dict(soc=float(socn), T=float(Tn), V1=float(V1n), aging=s.get("aging", _FRESH))
        return s2, (V, Tn, phi, self.S * V * u, socn)

    def anchor(self, s):
        """The input the platform is not allowed to go below.

        Zero on charge. On discharge it is the current that serves the load at the *lowest*
        bus voltage the cell constraint permits, so it over-estimates rather than under-, and
        an anchor that clears the caps is a genuine certificate rather than an optimistic one.
        """
        if self.mode == "charge" or self.load_W <= 0.0:
            return 0.0
        return self.load_W / (self.S * self.V_min)

    def set_load(self, W):
        """Change the irreducible load, moving the floor constraint with it.

        Missions have phases. A quadrotor climbs, cruises, hovers over the drop and descends,
        and each phase demands a different power; an AUV alternates transit and survey; a
        spacecraft enters and leaves eclipse. The anchor and the floor both move, and the point
        of the two-sided certificate is that they move *together* and the width between the
        edges is what changes."""
        self.load_W = float(W)
        if self.mode == "discharge":
            self.limits = self.limits[:3] + ((3, ">=", float(W), "lo"),)
            self._anchored_split = None
        return self

    def set_airspeed(self, v_air=None, altitude_m=None):
        """Cruise cools better than hover; altitude cools worse. Only `ForcedAir` cares."""
        c = self.cell.cooling
        if isinstance(c, ForcedAir):
            if v_air is not None:
                c.v_air = float(v_air)
            if altitude_m is not None:
                c.altitude_m = float(altitude_m)
        return self

    def cap(self, s):
        """Actuator/plating ceiling. Plating is a charge phenomenon and does not cap discharge."""
        if self.mode == "charge":
            return min(self.u_max, self.cell.plate_cap(s["T"]) * self.P)
        return self.u_max

    def init(self, soc0=0.5, T0=25.0):
        return dict(soc=float(soc0), T=float(T0), V1=0.0, aging=dict(_FRESH))

    def describe(self):
        return (f"{self.case} [{self.domain}] {self.S}S{self.P}P = {self.S*self.P:,} cells, "
                f"{self.energy_kWh():.2f} kWh, {self.mode}, "
                f"cooling: {self.cell.cooling.describe()}"
                + (f", load {self.load_W:.0f} W" if self.load_W else ""))


# =======================================================================================
# GROUND -- autonomous cars, robotaxis, shuttles, haul trucks
# =======================================================================================
def _ground_cell(cool=1.0, scale=None):
    p = load_params()
    return Cell(cooling=Newtonian(p["hA"] * cool), scale=scale)


def single_cell(scale=None, rc="euler", T_max=45.0, V_max=4.20):
    """A 1S1P Newtonian charge platform: the published cell, wearing the vehicle interface.

    This exists so the reduction claim can be checked rather than argued. Drive this with the
    anchored filter and `safe_charge.project_current` with the same margins and the same
    unclipped request, and the two must agree bit for bit -- same bisection, same bounds, same
    feasibility test, same order of operations."""
    p = load_params()
    plat = Platform(Cell(cooling=Newtonian(p["hA"]), scale=scale, rc=rc), S=1, P=1,
                    mode="charge", T_max=T_max, V_max=V_max, w_nominal=25.0,
                    name="single-cell")
    plat.domain, plat.dt_nominal, plat.horizon = "reference", 30.0, 80
    return plat


class RobotaxiUrban(Platform):
    """Compact urban robotaxi, liquid-cooled, on a 350 kW DC fast charger.

    The hard case for a car is not the pack, it is the duty cycle: a privately owned EV sees
    a fast charger a few times a month and a robotaxi sees one a dozen times a day, so it
    accumulates a private car's decade of fast-charge stress in about six weeks. Everything
    the certificate has to survive -- resistance growth, thermal soak, a coolant loop that is
    no longer new -- arrives roughly fifty times faster."""
    domain, case = "ground", "robotaxi-urban"
    dt_nominal, horizon = 30.0, 80

    def __init__(self, mode="charge", T_amb=25.0, cool=1.0, scale=None, **kw):
        super().__init__(_ground_cell(cool, scale), S=96, P=45, mode=mode,
                         T_max=45.0, V_max=4.20, V_min=3.00, w_nominal=T_amb,
                         load_W=kw.pop("load_W", 0.0), **kw)


class AutonomousShuttle(Platform):
    """Low-speed campus/airport shuttle: small pack, opportunity charging at every stop.

    Many short sessions rather than a few long ones, which puts the certificate in the regime
    where it is invoked constantly and the thermal state never fully relaxes between calls."""
    domain, case = "ground", "autonomous-shuttle"
    dt_nominal, horizon = 10.0, 90

    def __init__(self, mode="charge", T_amb=25.0, cool=0.85, scale=None, **kw):
        super().__init__(_ground_cell(cool, scale), S=72, P=18, mode=mode,
                         T_max=45.0, V_max=4.20, V_min=3.00, w_nominal=T_amb,
                         load_W=kw.pop("load_W", 0.0), **kw)


class HaulTruck(Platform):
    """Autonomous long-haul tractor on megawatt charging.

    Scale is the whole point: 288S115P is 33,120 cells, and a pack-averaged thermal limit on
    that many cells is a statement about a cell that does not exist. E5's weakest-cell lemma
    is what makes a per-cell certificate tractable here, and this is where it earns its
    keep."""
    domain, case = "ground", "haul-truck"
    dt_nominal, horizon = 30.0, 100

    def __init__(self, mode="charge", T_amb=25.0, cool=1.15, scale=None, **kw):
        super().__init__(_ground_cell(cool, scale), S=288, P=115, mode=mode,
                         T_max=45.0, V_max=4.20, V_min=3.00, w_nominal=T_amb,
                         load_W=kw.pop("load_W", 0.0), **kw)


# =======================================================================================
# AERIAL -- delivery rotorcraft, eVTOL air taxis
# =======================================================================================
# The one cell substitution in this file, and the reason for it.
#
# Rotorcraft hover is a 1-3C *continuous* draw. The calibrated M50 is an energy cell, and at
# the datasheet end-of-life resistance bound its terminal voltage under a 1.4C hover load
# falls below any usable pack minimum -- so the anchored filter reports the interval empty
# before the aircraft leaves the ground. That is not a defect in the filter. It is the filter
# working as a *design* tool: the correct engineering answer to "may I fly a rotorcraft on an
# energy cell" is no, and V2-1 measures the C-rate at which the refusal begins rather than
# tuning around it.
#
# The remaining aerial experiments need an aircraft that flies, so they use a power cell:
# roughly 0.4x the series resistance and 0.6x the capacity of the M50, which is about the
# ratio between a 5 Ah energy 21700 and a 3 Ah high-rate one. This is an *extrapolation beyond
# the ROM's calibration* -- the coefficients were fitted to the M50 and no other cell -- and it
# is the largest single modelling liberty taken in this register. It is applied only to the
# aerial platforms, only through the existing scale factors, and every aerial result carries
# it.
POWER_CELL = dict(R=0.40, Q=0.60, plate=1.0)
ENERGY_CELL = dict(R=1.00, Q=1.00, plate=1.00)


def _air_cell(hA_mult=0.35, v_air=0.0, altitude_m=0.0, scale=None, cell_type="power"):
    p = load_params()
    base = dict(POWER_CELL if cell_type == "power" else ENERGY_CELL)
    if scale:
        for k, v in scale.items():
            base[k] = base.get(k, 1.0) * v
    return Cell(cooling=ForcedAir(p["hA"] * hA_mult, v_air=v_air, altitude_m=altitude_m),
                scale=base)


class DeliveryQuadrotor(Platform):
    """A 2.2 kg-class delivery quadrotor, 6S1P, discharging in hover.

    6S3P, 157 Wh, hover at 2.2C, about 22 minutes of endurance. The parallel count is set by
    the *takeoff* transient rather than by endurance: a rotorcraft leaves the ground at 1.5-1.8
    times hover power, and at 6S2P the certificate correctly refuses that transient -- V2-4
    reports the frontier. Sizing a pack so the filter can certify the whole mission, rather
    than only its cheapest phase, is the design use of this method.

    This is the platform on which the null-input theorem is not merely conservative but wrong,
    and E2 measured exactly how wrong. Cutting current does not return this vehicle to the
    safe set; it removes the only thing holding it up. What Anchored Collapse supplies is the
    hover current as a floor, so the certificate becomes two-sided and its width is the
    reserve the aircraft actually has."""
    domain, case = "aerial", "delivery-quadrotor"
    dt_nominal, horizon = 1.0, 1500
    c_rate_ceiling = 6.0        # a power cell's ceiling, matching the cell it is given

    def __init__(self, mode="discharge", mass_kg=2.9, payload_kg=0.0, T_amb=20.0,
                 v_air=0.0, altitude_m=100.0, figure_of_merit=0.62, disk_area=0.342,
                 scale=None, cell_type="power", **kw):
        self.mass_kg, self.payload_kg = mass_kg, payload_kg
        self.fm, self.disk_area = figure_of_merit, disk_area
        self.cell_type = cell_type
        P_hover = self._hover_power(mass_kg + payload_kg, altitude_m)
        cell = _air_cell(0.35, v_air, altitude_m, scale, cell_type)
        if cell_type == "power":
            kw.setdefault("u_max", self.c_rate_ceiling * cell.q_nom() * 3)
        super().__init__(cell, S=6, P=3,
                         mode=mode, T_max=60.0, V_max=4.20, V_min=3.20, soc_floor=0.15,
                         load_W=kw.pop("load_W", P_hover), w_nominal=T_amb, **kw)

    def _hover_power(self, m, alt):
        """Momentum-theory hover power: T^{3/2} / (FM sqrt(2 rho A)), electrical.

        Increasing in mass and in altitude, both through the same term, which is why payload
        and density altitude do the same thing to the reserve and can be swept together."""
        rho = 1.225 * max(0.15, (1.0 - 2.25577e-5 * alt) ** 4.2559)
        T = m * G0
        P_ideal = T ** 1.5 / np.sqrt(2.0 * rho * self.disk_area)
        return float(P_ideal / self.fm / 0.88)      # 0.88 = ESC + motor efficiency


class EVTOLAirTaxi(Platform):
    """A five-seat eVTOL air taxi in the hover phase of a sortie.

    Hover is where an eVTOL's pack is worst-off: a few minutes near 3C into a pack that has to
    still be at flying voltage when it gets to the pad. The floor constraint here is not a
    modelling nicety -- the aircraft cannot trade power for temperature, so the only variable
    left is time, and the width of the interval is what converts to it.

    One parameter here is not the published cell's. A 400 kW hover draw against a 163 kWh pack
    is 2.5C sustained, and the 3C actuator ceiling every other platform inherits leaves so
    little room above the anchor that the interval is empty before any physics happens. eVTOL
    packs use power cells, so the ceiling is set to 6C. The electro-thermal coefficients are
    still the M50's, which is a real limitation and is recorded as one: a power cell has lower
    resistance and would sag and heat less than this model says, so the certificate computed
    here is conservative for the aircraft it represents rather than tuned to it."""
    domain, case = "aerial", "evtol-air-taxi"
    dt_nominal, horizon = 1.0, 1200
    c_rate_ceiling = 6.0

    def __init__(self, mode="discharge", P_hover_W=400_000.0, T_amb=25.0, v_air=0.0,
                 altitude_m=300.0, scale=None, cell_type="power", **kw):
        cell = _air_cell(0.55, v_air, altitude_m, scale, cell_type)
        self.cell_type = cell_type
        kw.setdefault("u_max", self.c_rate_ceiling * cell.q_nom() * 45)
        super().__init__(cell, S=300, P=45, mode=mode,
                         T_max=55.0, V_max=4.20, V_min=3.20, soc_floor=0.20,
                         load_W=kw.pop("load_W", P_hover_W), w_nominal=T_amb, **kw)


class QuadrotorTurnaround(RobotaxiUrban):
    """A delivery quadrotor's *pack*, on the ground, being charged between sorties.

    Same airframe, opposite problem: forty sessions a day into a small passively cooled pack
    that has not finished cooling from the last flight. Kept as its own class because the
    initial temperature distribution is the interesting part and it is not the one a car
    sees."""
    domain, case = "aerial", "quadrotor-turnaround"
    dt_nominal, horizon = 5.0, 200

    def __init__(self, mode="charge", T_amb=25.0, scale=None, cell_type="power", **kw):
        self.cell_type = cell_type
        Platform.__init__(self, _air_cell(0.35, 0.0, 0.0, scale, cell_type), S=6, P=2,
                          mode=mode, T_max=45.0, V_max=4.20, V_min=3.20, w_nominal=T_amb,
                          load_W=kw.pop("load_W", 0.0), **kw)


# =======================================================================================
# UNDERWATER -- survey AUVs, gliders, under-ice vehicles
# =======================================================================================
# A car's per-cell conductance into its liquid loop is about 0.080 W/K. Inside a sealed
# pressure hull the bottleneck is not the hull -- an aluminium tube in moving water is an
# excellent conductor -- it is the still-air gap between the pack and that tube. 0.008 W/K per
# cell across the gap in series with 0.050 W/K through the wall gives 0.0069 W/K, about
# one twelfth of the car's, which is what "thermally isolated from an excellent heat sink"
# actually means numerically. An earlier version of this file used 0.09 and 0.22, giving 0.8x
# a car's -- a claim of an order of magnitude that the numbers did not support, caught when
# V3-1 printed the ratio next to the sentence asserting it.
def _sea_cell(UA_int=0.008, UA_hull=0.050, T_water=2.0, scale=None):
    return Cell(cooling=HullConduction(UA_int, UA_hull, T_water), scale=scale)


class SurveyAUV(Platform):
    """A torpedo-class survey AUV: sealed hull, cold water, hotel load that never stops.

    Two things separate this from every other platform. The pack is inside a pressure hull, so
    its conductance to the outside is roughly an order of magnitude below a car's; and the
    water is 2 C, so plating -- which the IECON paper enforces but does not certify -- is the
    binding constraint on recharge rather than temperature. The vehicle is also the cleanest
    example of a *nonzero anchor with nothing flying*: the computer, the INS and the sonar
    draw their watts whether or not the thrusters do, so even a stationary AUV cannot take
    u = 0 for an answer."""
    domain, case = "underwater", "survey-auv"
    dt_nominal, horizon = 60.0, 600

    def __init__(self, mode="discharge", T_water=2.0, hotel_W=25.0, thrust_W=90.0,
                 UA_int=0.008, UA_hull=0.050, scale=None, **kw):
        self.hotel_W, self.thrust_W = hotel_W, thrust_W
        super().__init__(_sea_cell(UA_int, UA_hull, T_water, scale), S=14, P=6, mode=mode,
                         T_max=50.0, V_max=4.20, V_min=3.00, soc_floor=0.10,
                         load_W=kw.pop("load_W", hotel_W + thrust_W), w_nominal=T_water, **kw)


class BuoyancyGlider:
    """Marker for the glider case study; see `glider_platform`."""


def glider_platform(mode="discharge", T_water=4.0, hotel_W=0.5, pump_W=0.0, scale=None):
    """A buoyancy-driven glider: months of deployment at a fraction of a watt.

    The glider is the extreme of the duty-cycle axis. It has no propeller; it changes its own
    buoyancy in short bursts and sinks and rises for the rest of the time. So its load is
    almost -- but not quite -- zero, and it stays that way for six months without a single
    recalibration. If a never-recalibrated certificate is going to drift, this is where."""
    p = glider = Platform(_sea_cell(0.004, 0.030, T_water, scale), S=10, P=4, mode=mode,
                          T_max=45.0, V_max=4.20, V_min=3.00, soc_floor=0.05,
                          load_W=hotel_W + pump_W, w_nominal=T_water, name="buoyancy-glider")
    p.domain, p.dt_nominal, p.horizon = "underwater", 300.0, 500
    return glider


def under_ice_auv(mode="discharge", T_water=-1.8, hotel_W=35.0, thrust_W=140.0, scale=None):
    """An AUV under an ice shelf, where there is no abort-to-surface.

    Every other platform in this file has a degraded mode that ends with the vehicle stopped
    and intact. This one does not: the surface is a ceiling of ice, so a closed envelope is
    a lost vehicle. It is the strongest argument for a certificate that reports its reserve
    before it runs out, rather than a controller that reports a violation after."""
    a = SurveyAUV(mode=mode, T_water=T_water, hotel_W=hotel_W, thrust_W=thrust_W,
                  UA_int=0.006, UA_hull=0.040, scale=scale)
    a.case = "under-ice-auv"
    return a


def dock_charge_auv(T_water=2.0, scale=None):
    """The same AUV on its docking station. Cold, sealed, and therefore plating-limited."""
    a = SurveyAUV(mode="charge", T_water=T_water, scale=scale)
    a.case = "survey-auv-dock"
    a.dt_nominal, a.horizon = 60.0, 400
    return a


# =======================================================================================
# SPACE -- LEO, GEO, deep space, Mars surface, lunar night, high-radiation
# =======================================================================================
def _space_cell(eps=0.85, area=0.0042, T_sink=4.0, scale=None):
    return Cell(cooling=Radiative(eps, area, T_sink), scale=scale)


class LEOSmallsat(Platform):
    """A smallsat in a ~400 km orbit: 92.9 min period, ~35 min of eclipse, ~5,500 cycles/year.

    The defining feature is not the environment, it is the count. Fifteen and a half charge
    cycles a day for five years is 28,000 cycles with no technician, and the certificate has
    to hold on the last one with the parameters it was given before launch."""
    domain, case = "space", "leo-smallsat"
    dt_nominal, horizon = 30.0, 120
    period_min, eclipse_min = 92.9, 35.0

    def __init__(self, mode="charge", T_sink=4.0, bus_W=60.0, eps=0.85, scale=None, **kw):
        self.bus_W = bus_W
        super().__init__(_space_cell(eps, 0.0042, T_sink, scale), S=8, P=3, mode=mode,
                         T_max=45.0, V_max=4.10, V_min=3.10, soc_floor=0.30,
                         load_W=kw.pop("load_W", bus_W if mode == "discharge" else 0.0),
                         w_nominal=T_sink, **kw)

    def cycles_per_year(self):
        return 365.25 * 24 * 60 / self.period_min


class GEOComsat(Platform):
    """A GEO communications satellite through an eclipse season.

    GEO is the opposite duty cycle from LEO: about ninety eclipses a year clustered into two
    six-week seasons, the longest of them 72 minutes, and a payload that is not allowed to go
    quiet during any of them. Few, deep, and non-negotiable."""
    domain, case = "space", "geo-comsat"
    dt_nominal, horizon = 60.0, 80
    eclipses_per_year, max_eclipse_min = 90, 72.0

    def __init__(self, mode="discharge", T_sink=4.0, bus_W=3000.0, eps=0.85, scale=None, **kw):
        super().__init__(_space_cell(eps, 0.0042, T_sink, scale), S=100, P=8, mode=mode,
                         T_max=40.0, V_max=4.10, V_min=3.20, soc_floor=0.35,
                         load_W=kw.pop("load_W", bus_W if mode == "discharge" else 0.0),
                         w_nominal=T_sink, **kw)


class DeepSpaceCruiser(Platform):
    """A deep-space probe on cruise: radiation to 4 K, and a light-time delay that makes
    ground-in-the-loop recalibration a thing that happens hours later, if at all.

    This is where "never recalibrated" stops being a stress test and becomes the design. The
    IECON certificate's property of needing no identification is not a convenience here; it is
    the only reason a fixed, pre-launch parameter set is admissible."""
    domain, case = "space", "deep-space-cruiser"
    dt_nominal, horizon = 60.0, 200

    def __init__(self, mode="charge", T_sink=4.0, bus_W=200.0, eps=0.88, scale=None, **kw):
        super().__init__(_space_cell(eps, 0.0042, T_sink, scale), S=24, P=16, mode=mode,
                         T_max=40.0, V_max=4.10, V_min=3.10, soc_floor=0.40,
                         load_W=kw.pop("load_W", bus_W if mode == "discharge" else 0.0),
                         w_nominal=T_sink, **kw)


class MarsSurfaceRover(Platform):
    """A Mars surface rover: 6 mbar of CO2, a diurnal swing of order 80 K, and a sol that is
    39 minutes longer than a day.

    The thin atmosphere gives a small convective term on top of radiation. Both increase with
    temperature, so the sum does, and the certificate should not notice -- which is the claim
    this platform exists to test."""
    domain, case = "space", "mars-rover"
    dt_nominal, horizon = 60.0, 300
    sol_min = 24 * 60 + 39.6

    def __init__(self, mode="charge", T_amb=-60.0, bus_W=100.0, scale=None, **kw):
        super().__init__(Cell(cooling=RadiativePlusThin(0.85, 0.0042, 4.0, 0.006, T_amb),
                              scale=scale), S=8, P=12, mode=mode,
                         T_max=40.0, V_max=4.10, V_min=3.10, soc_floor=0.25,
                         load_W=kw.pop("load_W", bus_W if mode == "discharge" else 0.0),
                         w_nominal=T_amb, **kw)


class LunarNightLander(Platform):
    """A lander through lunar night: 354 hours of darkness at 100 K, on heaters.

    Nothing charges for two weeks. The battery's job is to keep itself and the avionics above
    their survival temperature using their own stored energy, which means the load is a
    *heater* -- a floor constraint whose whole purpose is to put energy into the thing the
    upper constraint is trying to keep cool. The two edges of the interval are pushing on the
    same state variable from opposite directions, which is the sharpest test of the two-sided
    certificate in this file."""
    domain, case = "space", "lunar-night-lander"
    dt_nominal, horizon = 300.0, 250
    night_hours = 354.0

    def __init__(self, mode="discharge", T_sink=100.0, heater_W=45.0, eps=0.72, scale=None, **kw):
        super().__init__(_space_cell(eps, 0.0042, T_sink, scale), S=8, P=20, mode=mode,
                         T_max=35.0, V_max=4.10, V_min=3.00, soc_floor=0.05,
                         load_W=kw.pop("load_W", heater_W), w_nominal=T_sink, **kw)


class HighRadiationOrbiter(Platform):
    """An outer-planet orbiter in an intense trapped-radiation belt.

    Total ionising dose raises series resistance and takes capacity. E11 established that this
    enters the certificate exactly like ageing does -- as one more monotone channel -- and this
    platform is where that gets exercised at mission dose rather than at a scale factor."""
    domain, case = "space", "high-radiation-orbiter"
    dt_nominal, horizon = 60.0, 200

    def __init__(self, mode="charge", T_sink=4.0, bus_W=300.0, tid_krad=300.0,
                 eps=0.85, scale=None, **kw):
        self.tid_krad = tid_krad
        f = 1.0 + 0.0011 * tid_krad          # resistance growth with dose
        sc = dict(R=f, Q=1.0 / f ** 0.5, plate=f ** 0.5)
        if scale:
            for k, v in scale.items():
                sc[k] = sc[k] * v
        super().__init__(_space_cell(eps, 0.0042, T_sink, sc), S=16, P=24, mode=mode,
                         T_max=40.0, V_max=4.10, V_min=3.10, soc_floor=0.35,
                         load_W=kw.pop("load_W", bus_W if mode == "discharge" else 0.0),
                         w_nominal=T_sink, **kw)


# =======================================================================================
# Registry
# =======================================================================================
GROUND = ("robotaxi-urban", "autonomous-shuttle", "haul-truck")
AERIAL = ("delivery-quadrotor", "evtol-air-taxi", "quadrotor-turnaround")
UNDERWATER = ("survey-auv", "buoyancy-glider", "under-ice-auv", "survey-auv-dock")
SPACE = ("leo-smallsat", "geo-comsat", "deep-space-cruiser", "mars-rover",
         "lunar-night-lander", "high-radiation-orbiter")

_BUILD = {
    "robotaxi-urban": lambda **k: RobotaxiUrban(**k),
    "autonomous-shuttle": lambda **k: AutonomousShuttle(**k),
    "haul-truck": lambda **k: HaulTruck(**k),
    "delivery-quadrotor": lambda **k: DeliveryQuadrotor(**k),
    "evtol-air-taxi": lambda **k: EVTOLAirTaxi(**k),
    "quadrotor-turnaround": lambda **k: QuadrotorTurnaround(**k),
    "survey-auv": lambda **k: SurveyAUV(**k),
    "buoyancy-glider": lambda **k: glider_platform(**k),
    "under-ice-auv": lambda **k: under_ice_auv(**k),
    "survey-auv-dock": lambda **k: dock_charge_auv(**k),
    "leo-smallsat": lambda **k: LEOSmallsat(**k),
    "geo-comsat": lambda **k: GEOComsat(**k),
    "deep-space-cruiser": lambda **k: DeepSpaceCruiser(**k),
    "mars-rover": lambda **k: MarsSurfaceRover(**k),
    "lunar-night-lander": lambda **k: LunarNightLander(**k),
    "high-radiation-orbiter": lambda **k: HighRadiationOrbiter(**k),
}
ALL_CASES = GROUND + AERIAL + UNDERWATER + SPACE
DOMAINS = dict(ground=GROUND, aerial=AERIAL, underwater=UNDERWATER, space=SPACE)


def build(case, **kw):
    if case not in _BUILD:
        raise KeyError(f"unknown case {case!r}; known: {', '.join(ALL_CASES)}")
    return _BUILD[case](**kw)
