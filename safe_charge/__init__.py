"""safe_charge: a solver-free, oracle-free safety filter for lithium-ion fast charging.

The entire certified safety path is `filter.project_current`: a bisection over the
reduced-order-model one-step map. It is numpy-only, allocates nothing in its loop, and
runs a statically bounded number of iterations, so its worst-case execution time is a
fixed constant and a functional-safety audit can read the whole safety path on one screen.

Quick start:
    from safe_charge import BatteryROM, project_current
    rom = BatteryROM()
    s = rom.init_state(soc0=0.1, T0=25.0)
    I, _ = project_current(rom, s, I_prop=3.0*rom.p["Q_nom"], dt=30.0, T_amb=25.0)
    # I is the largest current in [0, I_prop] that keeps the next state safe (T, V).
"""
from .rom import BatteryROM, ocv
from .filter import project_current, soc_ramp_margin

__all__ = ["BatteryROM", "ocv", "project_current", "soc_ramp_margin"]
__version__ = "1.0.0"
