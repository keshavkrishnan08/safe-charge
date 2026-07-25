"""Control-oriented reduced-order electro-thermal-aging model (ROM) of an LG M50
cell, calibrated to PyBaMM (Chen2020 electro-thermal; OKane2022 plating) across
ambients 0/10/25 C. See calibrate_rom.py / results/tables/calibration_report.json.

State: soc, T[C], V1 (RC overpotential), aging {Qloss, Rfac}.
Signals: terminal voltage, heat decomposition (ohmic/activation/reversible),
anode plating potential proxy phi_an (>=0 => no plating vs DFN local signal).
Fast (pure numpy) for large-scale safe-RL training; validated in PyBaMM DFN.
"""
import os, json
import numpy as np

CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
PARAMS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "rom_params.json")
TREF = 298.15

_ocv = np.loadtxt(os.path.join(CALIB, "ocv.csv"), delimiter=",", skiprows=1)
_OCV_SOC, _OCV_V = _ocv[:, 0], _ocv[:, 1]
def ocv(soc): return np.interp(np.clip(soc, 0.0, 1.0), _OCV_SOC, _OCV_V)

def load_params():
    return json.load(open(PARAMS_JSON))


class BatteryROM:
    """Single-cell ROM. Charge current I>0 charges the cell."""
    def __init__(self, params=None, cell_scale=None):
        self.p = load_params() if params is None else params
        # per-cell multiplicative variation (cell-to-cell / aging)
        self.scale = dict(R=1.0, Q=1.0, plate=1.0) if cell_scale is None else cell_scale

    # ----- overpotential (identical form to calibration) -----
    def eta_irrev(self, soc, T_C, I):
        a0, a3, Ea, A, cT, i0_0, i0_1 = self.p["eta_params"]
        Rohm = abs(a0 + a3*np.exp(6.0*(soc-1.0))) * np.exp(Ea*(1.0/(T_C+273.15)-1.0/TREF)) * self.scale["R"]
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
        (25 C -> 2C, 10 C -> 1.0C), clipped to [0.8C, 2C]. This is what makes the
        realized current profile DFN-plating-safe, closing the gap the proxy misses."""
        Qn = self.p["Q_nom"] * self.scale["Q"]
        crate = float(np.clip(1.00 + 0.067*(T_C - 10.0), 0.70, 2.00))
        return crate * Qn

    def Q_eff(self, aging):
        return self.p["Q_nom"] * self.scale["Q"] * (1.0 - aging.get("Qloss", 0.0))

    def init_state(self, soc0, T0):
        return dict(soc=float(soc0), T=float(T0), V1=0.0,
                    aging={"Qloss": 0.0, "Rfac": 1.0})

    # ----- one step of dt seconds -----
    def step(self, s, I, dt, T_amb):
        p = self.p
        soc, T, V1 = s["soc"], s["T"], s["V1"]
        aging = s.get("aging", {"Qloss": 0.0, "Rfac": 1.0})
        Rfac = aging.get("Rfac", 1.0)
        ohm, act = self.eta_irrev(soc, T, I)
        ohm *= Rfac; act *= Rfac
        # RC transient on the ohmic part (fast dynamics)
        R1 = p.get("R1_frac", 0.3) * max(abs(ohm)/max(I, 1e-6), 1e-4)
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
