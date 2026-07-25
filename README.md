# safe_charge

A solver-free, oracle-free safety filter for lithium-ion fast charging. You hand it any proposed
charging current and it returns the largest current that keeps the cell inside its safe set for the
next step: temperature under 45 &deg;C and terminal voltage under 4.20 V. No optimizer runs in the
loop, and the filter never needs to know the cell's state of health.

The idea is small. Zero current always arrests heating and pulls voltage toward open circuit, so
`I=0` is a backup that is safe from any safe state. That single fact collapses the safety problem to
a one-dimensional search: because each constraint is monotone in current, the set of safe currents is
an interval `[0, I*]`, and a bisection on a reduced-order cell model finds `I*` in a fixed number of
steps. No quadratic program, no matrix solve, no online parameter identification.

The whole safety path is one function, [`safe_charge/filter.py`](safe_charge/filter.py):
**36 source lines, numpy only, a statically bounded 18-iteration loop that allocates nothing inside
it.** Its worst-case run time is a fixed constant you can read off the code. The embedded-QP approach
it replaces puts an iterative solver into the safety-critical path instead.

## Requirements

- Python 3.9 or newer.
- **numpy** &mdash; the only thing the filter and model need.
- **osqp** and **scipy** &mdash; only for the QP baseline in `reproduce/solver_comparison.py`.

```bash
pip install numpy                 # enough to run the filter, the quickstart, and the tests
pip install -r requirements.txt   # adds osqp + scipy for the solver comparison
```

## Quick start

```bash
python examples/quickstart.py
```

Charges a worn cell (higher resistance, more plating-prone) to 80% while the filter is given only a
conservative datasheet resistance bound, never the true aged parameters. Expected output:

```
reached SOC 79.0% in 40.0 min
peak temperature 40.63 C   (limit 45.0)   -> SAFE
peak voltage     4.170 V   (limit 4.2)   -> SAFE
```

Using the filter directly is three lines:

```python
from safe_charge import BatteryROM, project_current

rom = BatteryROM()
s   = rom.init_state(soc0=0.1, T0=25.0)
I, clipped = project_current(rom, s, I_prop=3.0 * rom.p["Q_nom"], dt=30.0, T_amb=25.0)
# I is the largest current in [0, I_prop] that keeps the next state safe.
```

## Run everything

```bash
bash run_all.sh          # override the interpreter with:  PYTHON=python3.11 bash run_all.sh
```

Each piece also runs on its own:

| Command | What it does |
|---|---|
| `python examples/quickstart.py` | Charges an aged cell through the filter and reports peak T and V. |
| `python tests/test_safety.py` | Verifies the three safety guarantees below on a grid of states. |
| `python bench/footprint.py` | Counts the safety path's source lines and dependencies against the OSQP alternative's. |
| `python reproduce/solver_comparison.py [--full]` | Drives the filter and a warm-started OSQP over aged cells and compares cost and safety. |
| `python reproduce/dimension_independence.py` | Shows one fixed 18-iteration certificate covering a 4-channel uncertainty envelope, 500/500 safe. |
| `python reproduce/price_of_oracle_freedom.py` | Measures the charge cost of the fail-safe bound versus a true-cell oracle, per channel. |

Runtimes are seconds each; the numpy-only pieces finish almost instantly. Random seeds are fixed, so
the reproduce scripts print the same numbers every run.

## What the safety checks verify

`tests/test_safety.py` turns the filter's guarantees into properties checked on a grid of states, so
you can watch them hold instead of trusting them:

1. **Invariance.** The projected current always leaves the next-step temperature and voltage inside
   the safe set, and `I=0` adds no self-heating, so a safe state stays safe.
2. **A bound beats identification.** Peak temperature and the largest safe current are monotone in the
   resistance scale, so a conservative upper bound certifies safety with no per-cell identification.
3. **Plating is monotone.** The anode plating potential is affine in current everywhere and strictly
   decreasing across the fast-charge envelope, so it slots into the same one-dimensional search.

## Solver-free, measured

`reproduce/solver_comparison.py --full` drives a real embedded QP (OSQP), warm-started the way a
deployment would, against the filter on cold, worn cells. All of them stay temperature- and
voltage-safe, so this is not a claim that the QP is unsafe. The difference is the cost model. The
full-budget OSQP converges but at a data-dependent iteration count (mean/max climbs at the hard
corner), and capping its iterations to bound its run time makes it stop converging there and apply an
unconverged current. The filter's fixed 18-iteration bisection reaches the exact constraint boundary
every time: bounded run time and exact at once, with no solver in the loop.

`bench/footprint.py` puts numbers on that: the filter's safety path is numpy-only with a static loop
bound, while the QP path adds a compiled ADMM solver (a few dozen bundled C files) to everything a
safety audit has to account for.

## Layout

```
safe_charge/
  filter.py     the safety path (project_current): 36 source lines, numpy only
  rom.py        the reduced-order electro-thermal-plating cell model
  _data/        model parameters (OCV table, model coefficients)
examples/       quickstart
tests/          the safety-property checks
reproduce/      solver comparison, dimension independence, price of oracle-freedom
bench/          software-footprint benchmark
```

## Author

Keshav Krishnan.

## License

MIT. See [LICENSE](LICENSE).
