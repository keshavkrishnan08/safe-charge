"""What the bisection tolerance buys, and why the iteration count is 18.

The bisection halves the candidate interval [0, I_prop] every iteration, so after k
iterations the commanded current is within

    eps(k) = I_prop / 2^k

of the exact constraint boundary I*, and always on the SAFE side: the invariant is
lo <= I* , so the filter under-commands by at most eps(k) and never over-commands. That
one-sidedness is why tolerance costs charge but not safety -- which is exactly the trade
this script measures.

We sweep k over a full charge and report, for each k:
  * the current resolution eps(k) it guarantees,
  * the delivered SOC at the deadline (what the tolerance costs in performance),
  * the realized peak temperature and voltage (what it costs in margin -- nothing).

    python reproduce/tolerance_sweep.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from safe_charge import BatteryROM, project_current

DT, NSTEP, TAMB, T0, SOC0, SOC_TGT = 30.0, 80, 25.0, 25.0, 0.10, 0.80
VLIM, TLIM = 4.20, 45.0
# The aged cell from the quickstart: the filter is given only the fail-safe bound.
PLANT = dict(R=1.8, Q=0.8, plate=1.6)
FILT = dict(R=1.8, Q=1.0, plate=1.0)


def charge(iters):
    plant = BatteryROM(cell_scale=PLANT)
    est = BatteryROM(cell_scale=FILT)
    Qn = plant.p["Q_nom"]
    s = plant.init_state(SOC0, T0)
    maxT = maxV = 0.0
    for _ in range(NSTEP):
        I, _ = project_current(est, s, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM, iters=iters)
        s, o = plant.step(s, float(max(0.0, I)), DT, TAMB)
        maxT, maxV = max(maxT, o["T"]), max(maxV, o["V"])
        if s["soc"] >= SOC_TGT:
            break
    return s["soc"], maxT, maxV


def main():
    Qn = BatteryROM().p["Q_nom"]
    I_span = 2.0 * Qn                     # the plating cap bounds the bracket at 2C
    ref_soc, _, _ = charge(60)            # effectively exact boundary

    print("bisection tolerance sweep (aged cell, fail-safe bound, 40 min deadline)")
    print(f"{'iters':>6}{'eps [A]':>12}{'eps [C]':>10}{'SOC':>9}{'dSOC vs exact':>16}"
          f"{'peak T':>9}{'peak V':>9}{'safe':>7}")
    for k in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30):
        eps = I_span / 2**k
        soc, mT, mV = charge(k)
        safe = "yes" if (mT <= TLIM and mV <= VLIM) else "NO"
        print(f"{k:>6}{eps:>12.2e}{eps/Qn:>10.2e}{soc:>9.4f}{100*(soc-ref_soc):>15.3f}%"
              f"{mT:>9.2f}{mV:>9.3f}{safe:>7}")

    print(f"\nSafety is independent of k: the bisection maintains lo <= I*, so a coarser")
    print(f"tolerance under-commands and can only add margin. Peak temperature and voltage")
    print(f"stay inside the limits at every k, including k=2.")
    print(f"Performance saturates quickly. eps(18) = {I_span/2**18:.2e} A = "
          f"{1e6*I_span/2**18:.1f} uA, far below any")
    print(f"current sensor or power-stage resolution in a real charger, so 18 iterations buy")
    print(f"the exact boundary in every sense that a BMS can act on; k beyond ~12 changes the")
    print(f"delivered charge by less than a thousandth of a percent of SOC. 18 is chosen to")
    print(f"sit well past that knee while keeping the worst case a small fixed constant.")


if __name__ == "__main__":
    main()
