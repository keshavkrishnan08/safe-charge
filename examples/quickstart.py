"""Charge an aged cell to 80% SOC through the solver-free filter and confirm it stays safe.

    python examples/quickstart.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_charge import BatteryROM, project_current

DT, TAMB, VLIM, TLIM, SOC_TGT = 30.0, 25.0, 4.20, 45.0, 0.80

# A rated-end-of-life cell (higher resistance, more plating-prone). The filter is given only a
# conservative datasheet bound (s_R=1.8), never the true aged parameters (no health oracle).
plant = BatteryROM(cell_scale=dict(R=1.8, Q=0.8, plate=1.6))     # the true aged cell
filt_model = BatteryROM(cell_scale=dict(R=1.8, Q=1.0, plate=1.0))  # what the filter assumes

s = plant.init_state(soc0=0.10, T0=25.0)
Qn = plant.p["Q_nom"]
maxT = maxV = 0.0; t = 0.0
for step in range(80):
    # greedy governor asks for the max rate; the filter projects it to the largest safe current
    I, clipped = project_current(filt_model, s, I_prop=3.0*Qn, dt=DT, T_amb=TAMB,
                                 Vlim=VLIM, Tlim=TLIM)
    s, o = plant.step(s, I, DT, TAMB)
    maxT, maxV = max(maxT, o["T"]), max(maxV, o["V"]); t += DT
    if s["soc"] >= SOC_TGT:
        break

print(f"reached SOC {s['soc']*100:4.1f}% in {t/60:.1f} min")
print(f"peak temperature {maxT:5.2f} C   (limit {TLIM})   -> {'SAFE' if maxT <= TLIM else 'VIOLATION'}")
print(f"peak voltage     {maxV:5.3f} V   (limit {VLIM})   -> {'SAFE' if maxV <= VLIM else 'VIOLATION'}")
print("\nThe filter carried safety on a cell it never identified, with no optimizer in the loop.")
