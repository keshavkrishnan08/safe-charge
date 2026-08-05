"""Control-oriented reduced-order electro-thermal-aging model (ROM) of an LG M50
cell, calibrated to PyBaMM (Chen2020 electro-thermal; OKane2022 plating) across
ambients 0/10/25 C. The fitted coefficients ship in _data/rom_params.json; the
calibration harness and its held-out report (voltage RMSE 20.3 mV, temperature RMSE
1.32 C) are described in Sec. V of the paper and are not part of this release.

State: soc, T[C], V1 (RC overpotential), aging {Qloss, Rfac}.
Signals: terminal voltage, heat decomposition (ohmic/activation/reversible),
anode plating potential proxy phi_an (>=0 => no plating vs DFN local signal).
Fast (pure numpy) for large-scale safe-RL training; validated in PyBaMM DFN.
"""
import os, json
import numpy as np
from math import exp as _exp, asinh as _asinh

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
PARAMS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "rom_params.json")
TREF = 298.15
_FRESH = {"Qloss": 0.0, "Rfac": 1.0}   # shared default; never mutated

_ocv = np.loadtxt(os.path.join(CALIB, "ocv.csv"), delimiter=",", skiprows=1)
_OCV_SOC, _OCV_V = _ocv[:, 0], _ocv[:, 1]
def ocv(soc): return np.interp(np.clip(soc, 0.0, 1.0), _OCV_SOC, _OCV_V)

def load_params():
    return json.load(open(PARAMS_JSON))


class BatteryROM:
    """Single-cell ROM. Charge current I>0 charges the cell.

    `rc` selects the RC-branch discretization:
      "euler" (default) explicit Euler, V1 <- V1 + dt*(-V1 + I*R1)/tau1. Valid for
              dt <= tau1; this is what every published number was produced with.
      "exact" zero-order-hold, V1 <- a*V1 + (1-a)*I*R1 with a = exp(-dt/tau1).
              Unconditionally stable, so it is the one to use when studying control
              steps longer than tau1 (see reproduce/step_size.py).
    At dt = tau1 the Euler factor is exactly 0 and the exact factor is e^-1 ~ 0.368,
    so the two agree on the steady state but not on the residual; the certificate's
    thermal margin is what absorbs that difference.
    """
    def __init__(self, params=None, cell_scale=None, rc="euler"):
        self.p = load_params() if params is None else params
        # per-cell multiplicative variation (cell-to-cell / aging)
        self.scale = dict(R=1.0, Q=1.0, plate=1.0) if cell_scale is None else cell_scale
        if rc not in ("euler", "exact"):
            raise ValueError(f"rc must be 'euler' or 'exact', got {rc!r}")
        self.rc = rc

    # ----- overpotential (identical form to calibration) -----
    def R_ohmic(self, soc, T_C):
        """Ohmic resistance [Ohm]. A property of the cell and its temperature, independent
        of the applied current, so it stays well defined at I=0 -- which the RC branch's
        dissipation V1^2/R1 needs, since that term survives after the current is cut."""
        a0, a3, Ea, _A, _cT, _i0_0, _i0_1 = self.p["eta_params"]
        return abs(a0 + a3*np.exp(6.0*(soc-1.0))) \
            * np.exp(Ea*(1.0/(T_C+273.15)-1.0/TREF)) * self.scale["R"]

    def eta_irrev(self, soc, T_C, I):
        _a0, _a3, _Ea, A, cT, i0_0, i0_1 = self.p["eta_params"]
        Rohm = self.R_ohmic(soc, T_C)
        i0 = np.clip(i0_0*np.exp(i0_1*soc), 1e-3, 50.0)
        ohm = I*Rohm
        act = A*(1.0 + cT*(T_C-25.0))*np.arcsinh(I/(2.0*i0))
        return ohm, act

    def phi_an(self, soc, I, T_C):
        """Anode plating-potential proxy [V]; <=0 => plating (per DFN local signal)."""
        p = self.p; T = T_C + 273.15
        p0 = (p["pl_p0a"] + p["pl_p0b"]*soc + p.get("pl_p0c",0.0)*soc*soc
              + p.get("pl_p0d",0.0)*np.exp(6.0*(soc-1.0)))
        p1 = (p["pl_p1a"] + p["pl_p1b"]*soc + p.get("pl_p1c",0.0)*soc*soc) \
             * np.exp(p["pl_ET"]*(1.0/T - 1.0/TREF))
        return (p0 - p1*I) / self.scale["plate"]

    def plating_margin(self):
        return self.p["plating_margin_V"]

    def plate_current_cap(self, T_C):
        """Temperature-dependent plating-safe current limit [A], calibrated from
        CLOSED-LOOP DFN replay (not fixed-C-rate): the instantaneous potential proxy
        under-predicts DFN plating during high-current phases, and the true safe rate
        falls at low temperature. Linear in T between calibrated anchors
        (25 C -> 2C, 10 C -> 1.0C), clipped to [0.70C, 2C]; the 0.70C floor binds only
        below ~5.5 C. This is what makes the realized current profile DFN-plating-safe,
        closing the gap the instantaneous potential proxy misses."""
        Qn = self.p["Q_nom"] * self.scale["Q"]
        crate = float(np.clip(1.00 + 0.067*(T_C - 10.0), 0.70, 2.00))
        return crate * Qn

    def Q_eff(self, aging):
        return self.p["Q_nom"] * self.scale["Q"] * (1.0 - aging.get("Qloss", 0.0))

    def init_state(self, soc0, T0):
        return dict(soc=float(soc0), T=float(T0), V1=0.0,
                    aging={"Qloss": 0.0, "Rfac": 1.0})

    def probe(self, s, I, dt, T_amb):
        """One-step prediction of exactly the four signals the safety filter tests, as a
        tuple (V, T_next, phi_an, soc_next).

        Numerically identical to `step` -- it runs the same arithmetic through the same
        helpers -- but it builds no observation dict and no successor state, so the
        bisection's inner loop allocates nothing. `step` remains the readable reference and
        the one simulations use; `probe` is what a deployment calls. tests/test_safety.py
        asserts the two agree bit for bit."""
        p = self.p
        if self.rc == "euler" and dt > p["tau1"]:
            raise ValueError(f"dt={dt}s exceeds the RC time constant tau1={p['tau1']}s; the "
                             "explicit-Euler RC update is unstable there. Use "
                             "BatteryROM(rc='exact') for longer control steps.")
        soc = s["soc"]; T = s["T"]; V1 = s["V1"]
        aging = s.get("aging", _FRESH)
        Rfac = aging.get("Rfac", 1.0)
        # Inlined R_ohmic / eta_irrev / phi_an on scalars. math.exp and math.asinh are
        # bit-identical to np.exp and np.arcsinh here (verified in tests/test_safety.py),
        # and skipping numpy's scalar-ufunc dispatch is most of this method's speed.
        a0, a3, Ea, A, cT, i0_0, i0_1 = p["eta_params"]
        Rohm = abs(a0 + a3*_exp(6.0*(soc-1.0))) \
            * _exp(Ea*(1.0/(T+273.15)-1.0/TREF)) * self.scale["R"]
        i0 = i0_0*_exp(i0_1*soc)
        i0 = 1e-3 if i0 < 1e-3 else (50.0 if i0 > 50.0 else i0)
        ohm = (I*Rohm) * Rfac
        act = (A*(1.0 + cT*(T-25.0))*_asinh(I/(2.0*i0))) * Rfac
        R1 = p.get("R1_frac", 0.3) * max(Rohm*Rfac, 1e-4)
        if self.rc == "exact":
            a = _exp(-dt/p["tau1"]); V1n = a*V1 + (1.0-a)*I*R1
        else:
            V1n = V1 + dt*(-V1/p["tau1"] + I*R1/p["tau1"])
        eta = ohm + act + V1n                      # same association as step(): rounding matches
        V = ocv(soc) + eta
        Q_gen = I*ohm + I*act + V1n*V1n/max(R1, 1e-6) + I*(T+273.15)*p["dUdT"]
        Tn = T + dt*(Q_gen - p["hA"]*(T - T_amb)) / p["C_th"]
        socn = soc + I*dt/(3600.0*self.Q_eff(aging))
        Tk = T + 273.15
        p0 = (p["pl_p0a"] + p["pl_p0b"]*soc + p.get("pl_p0c",0.0)*soc*soc
              + p.get("pl_p0d",0.0)*_exp(6.0*(soc-1.0)))
        p1 = (p["pl_p1a"] + p["pl_p1b"]*soc + p.get("pl_p1c",0.0)*soc*soc) \
             * _exp(p["pl_ET"]*(1.0/Tk - 1.0/TREF))
        return V, Tn, (p0 - p1*I) / self.scale["plate"], socn

    # ----- one step of dt seconds -----
    def step(self, s, I, dt, T_amb):
        """Advance one control step.

        With rc="euler" (the default, and what every published number used) the RC branch
        decays by (1 - dt/tau1), so dt must satisfy dt <= tau1 = 30 s: beyond that the
        update oscillates, and at dt >= 2*tau1 it diverges. The paper's step dt = tau1
        makes the factor exactly 0, so the discrete update leaves no RC residual and is
        at least as conservative as the continuous-time bound exp(-dt/tau1) ~ 0.368 used
        in the proof of Prop. 1.

        With rc="exact" the branch uses the zero-order-hold factor exp(-dt/tau1), which is
        stable for every dt. Use it to study control steps longer than tau1."""
        p = self.p
        if self.rc == "euler" and dt > p["tau1"]:
            raise ValueError(f"dt={dt}s exceeds the RC time constant tau1={p['tau1']}s; the "
                             "explicit-Euler RC update is unstable there. Use "
                             "BatteryROM(rc='exact') for longer control steps.")
        soc, T, V1 = s["soc"], s["T"], s["V1"]
        aging = s.get("aging", {"Qloss": 0.0, "Rfac": 1.0})
        Rfac = aging.get("Rfac", 1.0)
        ohm, act = self.eta_irrev(soc, T, I)
        ohm *= Rfac; act *= Rfac
        # RC transient on the ohmic part (fast dynamics). R1 is derived from the ohmic
        # RESISTANCE, not from ohm/I: the two agree for I>0, but only the former stays
        # finite at I=0, where the RC branch is still dissipating V1^2/R1 into the cell.
        R1 = p.get("R1_frac", 0.3) * max(self.R_ohmic(soc, T)*Rfac, 1e-4)
        if self.rc == "exact":
            a = np.exp(-dt/p["tau1"]); V1n = a*V1 + (1.0-a)*I*R1
        else:
            V1n = V1 + dt*(-V1/p["tau1"] + I*R1/p["tau1"])
        eta = ohm + act + V1n
        V = ocv(soc) + eta
        Q_ohm = I*ohm; Q_act = I*act; Q_rc = V1n*V1n/max(R1, 1e-6)
        Q_irr = Q_ohm + Q_act + Q_rc
        Q_rev = I*(T+273.15)*p["dUdT"]
        Q_gen = Q_irr + Q_rev
        Tn = T + dt*(Q_gen - p["hA"]*(T - T_amb)) / p["C_th"]
        Qeff = self.Q_eff(aging)
        socn = soc + I*dt/(3600.0*Qeff)
        phi = self.phi_an(soc, I, T)
        out = dict(V=V, T=Tn, eta=eta, ohm=ohm, act=act, Q_ohm=Q_ohm, Q_act=Q_act,
                   Q_rc=Q_rc, Q_rev=Q_rev, Q_irr=Q_irr, Q_gen=Q_gen, phi_an=phi,
                   ocv=ocv(soc), soc=socn, P_out=V*I)
        s2 = dict(soc=float(socn), T=float(Tn), V1=float(V1n), aging=aging)
        return s2, out

    def rollout_current(self, currents, dt, soc0, T0, T_amb):
        s = self.init_state(soc0, T0)
        keys = ["soc","V","T","phi_an","ohm","act","Q_ohm","Q_act","Q_rev","Q_irr","I","P_out","eta"]
        rec = {k: [] for k in keys}
        for I in currents:
            s, o = self.step(s, I, dt, T_amb)
            for k in keys:
                rec[k].append(o[k] if k != "I" else I)
        return {k: np.array(v) for k, v in rec.items()}

    # ----- aging increment over one charge (coarse surrogate; DFN is ground truth) -----
    def age_from_rollout(self, rec, dt):
        p = self.p
        thr_Ah = np.sum(np.abs(rec["I"])) * dt / 3600.0
        Tmean = float(np.mean(rec["T"]))
        dQ_sei = p["sei_A"] * np.exp((Tmean-25.0)/15.0) * (thr_Ah/ (p["Q_nom"]))
        margin = self.plating_margin()
        plate_stress = np.sum(np.clip(margin - rec["phi_an"], 0, None) * np.abs(rec["I"])) * dt/3600.0
        dQ_plate = p["plate_gain"] * plate_stress / p["Q_nom"]
        dQ = dQ_sei + dQ_plate
        return dQ, dQ_sei, dQ_plate


if __name__ == "__main__":
    rom = BatteryROM()
    for (c, T0) in [(2.0,25),(1.0,25),(1.0,0),(0.5,0)]:
        r = rom.rollout_current([c*5.0]*80, 30.0, 0.1, T0, T0)
        stop = int(np.argmax(r["V"] > 4.2)) if (r["V"]>4.2).any() else len(r["V"])-1
        print(f"C={c} T0={T0}C: soc@4.2V={r['soc'][stop]:.3f} Vmax={r['V'][:stop+1].max():.3f} "
              f"Tmax={r['T'][:stop+1].max():.1f} min_phi={r['phi_an'][:stop+1].min():.4f}")
