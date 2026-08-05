# safe_charge

A solver-free, oracle-free safety filter for lithium-ion fast charging.

You hand it any proposed charging current and it returns the largest current that keeps the cell inside
its safe set for the next step: temperature under 45 &deg;C and terminal voltage under 4.20 V. No
optimizer runs in the loop, and the filter never needs to know the cell's state of health. Drop it
between any charging policy and the cell and the output is hard-limited, whatever the policy asks for.

The **certified** set is exactly `{T <= 45 C, V <= 4.20 V}`. The anode plating potential rides the same
one-dimensional search and is enforced by a temperature-dependent current cap, but it is *not* certified:
plating is history-dependent, and the `I=0` backup does not clear its margin at the cold, aged, high-SOC
corner. That scoping is deliberate and matches the paper (Sec. IV-B).

- **Tiny safety path.** The entire safety-critical code is one function, 36 numpy-only source lines.
- **No solver.** A fixed 18-iteration bisection, not a quadratic program. Worst-case run time is a constant.
- **No health oracle.** A conservative datasheet resistance bound replaces per-cell identification.
- **Verifiable.** `python tests/test_safety.py` checks the guarantees on a grid of states in under a second.

---

## Table of contents

- [The idea](#the-idea)
- [The 36-line safety path](#the-36-line-safety-path)
- [Install](#install)
- [Quick start](#quick-start)
- [Using the filter (API)](#using-the-filter-api)
- [Run everything](#run-everything)
- [The experiments, in detail](#the-experiments-in-detail)
- [The safety guarantees](#the-safety-guarantees)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Author and license](#author-and-license)

---

## The idea

Zero current always arrests heating and pulls voltage toward open circuit, so `I=0` is a backup that is
safe from any safe state. That single fact collapses the safety problem to a one-dimensional search:

1. Each hard constraint (temperature, voltage, plating potential) is **monotone** in the applied
   current. Push more current, you get hotter and higher voltage.
2. So the set of currents that keep the *next* state safe is a single interval `[0, I*]`.
3. A **bisection** on a reduced-order cell model finds `I*` in a fixed number of steps.

No quadratic program, no matrix solve, no online parameter identification. For the certified
constraints `I=0` is always inside the interval, so the filter can never paint itself into a corner:
recursive feasibility is free. (For plating the interval can be empty above ~0.82 SOC, where `I=0` no
longer clears the margin; there the filter commands zero current and flags it. `tests/test_safety.py`
checks that interval structure and reports the boundary rather than papering over it.)

Aging is handled by the same monotonicity. Higher internal resistance only makes a given current
hotter, so a **conservative upper bound** on resistance certifies safety without ever measuring the
true value. The filter is initialized with a rated end-of-life bound and stays safe from step one; an
online estimate, if you use one, only buys back charging speed, never safety.

## The 36-line safety path

Everything safety-critical lives in [`safe_charge/filter.py`](safe_charge/filter.py):

```
project_current   29 lines   the bisection and the feasibility short-circuits
_feasible          5 lines   one-step ROM prediction vs the tightened limits
_eff_margin        2 lines   SOC-dependent plating margin lookup
-------------------------
total             36 lines   numpy only, static 18-iteration loop, nothing allocated inside it
```

Its worst-case execution time is a fixed constant you can read off the code. That is the whole point:
a functional-safety audit can bound this path by reading one screen. The embedded-QP approach it
replaces puts a compiled iterative solver (39 bundled C/header files, ~1.1 MB) into the safety-critical
path, whose convergence *is* the failure mode. `python bench/footprint.py` prints both side by side.

## Install

- Python 3.9 or newer.
- **numpy** &mdash; the only thing the filter, the model, the quickstart, and the tests need.
- **osqp** and **scipy** &mdash; only for the QP baseline in `reproduce/solver_comparison.py`.

```bash
pip install numpy                 # enough for the filter, the quickstart, and the safety tests
pip install -r requirements.txt   # adds osqp==1.1.3 + scipy for the solver comparison
```

`osqp` is pinned because its ADMM iteration counts are version-sensitive. The qualitative result holds
across versions; the printed counts are reproducible on 1.1.3 up to your BLAS/platform, which can move
them by a few iterations.

## Quick start

```bash
python examples/quickstart.py
```

Charges a worn cell (higher resistance, more plating-prone) to 80% SOC while the filter is given only a
conservative datasheet resistance bound, never the true aged parameters. You should see:

```
reached SOC 79.0% in 40.0 min
peak temperature 40.63 C   (limit 45.0)   -> SAFE
peak voltage     4.170 V   (limit 4.2)   -> SAFE
```

The peak temperature lands below 45 &deg;C by the cooling-robustness margin, and voltage stays under
4.20 V, on a cell the filter never identified.

## Using the filter (API)

Three lines to project a proposed current onto the safe interval:

```python
from safe_charge import BatteryROM, project_current

rom = BatteryROM()
s   = rom.init_state(soc0=0.1, T0=25.0)
I, clipped = project_current(rom, s, I_prop=3.0 * rom.p["Q_nom"], dt=30.0, T_amb=25.0)
# I is the largest current in [0, I_prop] that keeps the next state safe; clipped=True if I < I_prop.
```

**`project_current(rom, s, I_prop, dt, T_amb, Vlim=4.20, Tlim=45.0, margin=None, dV=0.03, dT=0.5, dP=0.0, iters=18, cool_frac=None)`**

| Argument | Meaning |
|---|---|
| `rom` | a `BatteryROM` (the model the filter predicts with; give it your resistance bound via `cell_scale`) |
| `s` | current state dict from `rom.init_state(...)` or `rom.step(...)` |
| `I_prop` | the current the policy wants (amps); the return value never exceeds it |
| `dt`, `T_amb` | control step (s) and ambient temperature (&deg;C). With the default `rc="euler"` ROM, `dt` must not exceed the RC time constant (30 s); the model raises if it does. Use `BatteryROM(rc="exact")` for longer steps |
| `Vlim`, `Tlim` | the hard voltage and temperature limits |
| `margin` | plating margin, a float or a `margin(soc)` callable; defaults to the ROM's |
| `dV`, `dT`, `dP` | robustness margins that tighten each limit so the high-fidelity plant stays feasible |
| `cool_frac` | cooling-fault reserve; reserves `cool_frac*(T-T_amb)` of thermal headroom (default 0.25) |
| returns | `(I, clipped)`: the safe current and whether it was below `I_prop` |

**`BatteryROM(cell_scale=dict(R=..., Q=..., plate=...), rc="euler")`** &mdash; the reduced-order
electro-thermal-plating cell model. `cell_scale` scales resistance, capacity, and plating relative to
nominal (use it to build a worn cell, or to set the filter's conservative bound). `rc` selects the
RC-branch discretization: `"euler"` is the default and reproduces every published number, and is valid
for `dt <= 30 s`; `"exact"` uses the zero-order-hold factor `exp(-dt/tau)` and is stable at any step.

| Method | Returns |
|---|---|
| `init_state(soc0, T0)` | a fresh state dict at the given SOC and temperature |
| `step(s, I, dt, T_amb)` | `(next_state, obs)`; `obs` has `T`, `V`, `phi_an`, `soc` |
| `probe(s, I, dt, T_amb)` | `(V, T_next, phi_an, soc_next)`; what the filter calls &mdash; same arithmetic as `step`, bit for bit, but allocates nothing |
| `plating_margin()` | the default plating potential margin |
| `plate_current_cap(T_C)` | the temperature-dependent plating-safe current cap |
| `phi_an(soc, I, T_C)` | anode plating potential (negative signals plating) |
| `R_ohmic(soc, T_C)` | ohmic resistance; current-independent, so it stays defined at `I=0` |
| `p` | parameter dict, e.g. `p["Q_nom"]` is the nominal capacity |

`soc_ramp_margin(base, extra, soc0=0.60, soc1=0.85)` builds a `margin(soc)` callable that tightens the
plating margin at high SOC, where the ROM&ndash;DFN plating gap grows; pass it as `margin=`.
`ocv(soc)` is also exported for the open-circuit voltage curve.

## Run everything

```bash
bash run_all.sh          # override the interpreter with:  PYTHON=python3.11 bash run_all.sh
```

That runs all nine pieces in order. Each also runs on its own:

| Command | What it does | Needs |
|---|---|---|
| `python examples/quickstart.py` | Charges an aged cell through the filter, reports peak T and V. | numpy |
| `python tests/test_safety.py` | Verifies the three safety guarantees on a grid of states. | numpy |
| `python bench/footprint.py` | Counts the safety path's source and dependencies vs the OSQP path. | numpy (osqp optional) |
| `python reproduce/solver_comparison.py [--full]` | Filter vs a warm-started OSQP on worn cells: cost and safety. | numpy, scipy, osqp |
| `python reproduce/dimension_independence.py` | One fixed certificate over a 4-channel uncertainty envelope. | numpy |
| `python reproduce/price_of_oracle_freedom.py` | Charge cost of the fail-safe bound vs a true-cell oracle, per channel. | numpy |
| `python bench/timing.py` | Latency by code path, operation count, and memory footprint. | numpy |
| `python reproduce/tolerance_sweep.py` | What the bisection tolerance buys, and why 18 iterations. | numpy |
| `python reproduce/step_size.py` | Safety and charge vs the control step, above and below the RC time constant. | numpy |
| `python reproduce/robustness.py` | Behaviour past the aging bound, anchor comparison, margin-gain sweep. | numpy |

Random seeds are fixed, so the numpy-only reproduce scripts print the same numbers every run. The
solver comparison is deterministic given a fixed OSQP build, but its iteration counts can shift slightly
across platforms.

## The experiments, in detail

### Safety-property checks &mdash; `tests/test_safety.py`

Turns the filter's guarantees into properties checked on a 12&times;8 grid of states, so you can watch
them hold. Runs in under a second. Expected output:

```
Running 8 safety-property checks...
  interval: admissible set is a single interval [0, I*] anchored at 0 for the certified T/V set on every grid state; adding plating it stays an interval but goes empty above SOC~0.82, where I=0 no longer clears the margin
  plating: phi_an affine everywhere, strictly decreasing for SOC<=0.9 (slope flips only above SOC~0.92, outside the charge envelope)
  probe: bit-identical to step on all 6400 states (2 cell scales x 2 discretizations x 2 cooling faults)
  invariance: projection safe on all grid states; worst next T=45.000C V=4.1691V
  backup heat: R1 is continuous through I=0 in both discretizations, so the RC residual injects <0.5 C (the delta_T0 budget) instead of diverging
  monotonicity (cooling): peak T and the safe current are monotone in gamma too, so Prop. 2 holds in both aging channels
  monotonicity: peak T and the safe current are monotone in s_R (an upper bound suffices, no ID)
  backup: I=0 adds no self-heating even with the RC branch charged -- T never exceeds max(T, ambient) (worst excess -0.0034 C) and V returns to OCV, so S is invariant
ALL SAFETY CHECKS PASSED
```

Also runs under pytest (`python -m pytest tests/ -q` &rarr; `8 passed`).

### Software footprint &mdash; `bench/footprint.py`

Counts the safety path's source lines and dependencies, and the trusted-computing-base the QP
alternative adds. With `osqp` installed you get:

```
== Certified safety path (this package) ==
  source lines: 36  (_eff_margin=2, _feasible=5, project_current=29)
  runtime dependencies: numpy only, no third-party solver
  loop bound: static, 18 iterations  |  allocation in the bisection body: none

== Embedded-QP alternative (OSQP linearized MPC) ==
  runtime dependencies: numpy, scipy.sparse, osqp
  adds to the TCB: OSQP solver, 39 bundled C/header files, 1.1 MB
  loop bound: data-dependent (mean/max ~85/775 iterations at the cold corner)  |  heap allocation in loop: yes
```

The bisection body allocates nothing: each iteration calls `BatteryROM.probe()`, which returns a
tuple of the four signals the filter tests instead of building an observation dict, and which the test
suite verifies is bit-identical to `step()`. The claim rests on the *bound*: 18 iterations of
elementary arithmetic, no solver, no convergence test.

### Solver comparison &mdash; `reproduce/solver_comparison.py`

Drives a real embedded QP (OSQP), warm-started the way a deployment would be, against the filter on
cold, worn cells &mdash; handed the *same* fail-safe bound, margins, and plating cap, so any difference is
the solver's. Quick mode (`n=15`, corner only) prints:

```
--- corner population ---
Controller        Iters mean/max  T/V viol %   worst V  mean SOC
QP-full                 85.3/675        0.0%     4.182     0.779
QP-embedded              25.0/25        0.0%     4.184     0.779
filter                     18/18        0.0%     4.181     0.779
```

How to read it: all three stay temperature- and voltage-safe, so this is **not** a "the QP is unsafe"
claim. The point is the cost model. The full-budget OSQP converges but at a data-dependent iteration
count (mean/max climb at the hard corner). Capping its iterations to bound its run time (the
`QP-embedded` row, 25 iterations) makes it stop converging there and apply an unconverged current &mdash;
it still lands under 4.20 V here, but with no convergence guarantee behind it. The filter's fixed 18
iterations reach the exact constraint boundary every time &mdash; bounded run time and exact at once.

The `mean SOC` column prices that simplicity. At this cold corner all three are deadline-limited and
deliver the same charge. Add `--full` for the larger populations (`main n=100 + corner n=50`), where the
warmer main population separates them:

```
--- main population ---
Controller        Iters mean/max  T/V viol %   worst V  mean SOC
QP-full                 36.9/550        0.0%     4.148     0.589
QP-embedded              25.0/25        0.0%     4.172     0.604
filter                     18/18        0.0%     4.069     0.532
```

The filter cedes ~5.7 points of SOC to the foresighted full-budget QP &mdash; the same trade the paper
reports in Sec. VI-E, and the 0.532 figure is the conservative-bound floor quoted in Sec. IV-C.

The exact iteration counts and worst-case voltages depend on your OSQP build (BLAS, platform), so they
may shift by a few counts even on the pinned 1.1.3; the ordering and the qualitative result do not.

### Dimension independence &mdash; `reproduce/dimension_independence.py`

Monotone structure makes the worst case of an entire uncertainty box a single corner, so certifying
against that corner certifies every interior plant. This widens the box from 1 to 5 aging/fault
channels (resistance, cooling, thermal mass, plating, ambient temperature) and shows the certificate
stays a fixed 18-iteration bisection with zero violations. Expected output:

```
dimension sweep (certificate is 18 iterations at every dimension), n=150:
  d=1  ['R']                                     safe=150/150  worstT=44.14C  cert=18 iters
  d=2  ['R', 'cool']                             safe=150/150  worstT=44.27C  cert=18 iters
  d=3  ['R', 'cool', 'Cth']                      safe=150/150  worstT=44.27C  cert=18 iters
  d=4  ['R', 'cool', 'Cth', 'plate']             safe=150/150  worstT=44.27C  cert=18 iters
  d=5  ['R', 'cool', 'Cth', 'plate', 'Tamb']     safe=150/150  worstT=44.49C  cert=18 iters

full 5-channel envelope: 500/500 safe, viol=0.00%, CP95 upper=0.597%, worst 44.49C / 4.141V
corner peakT=44.50C dominates interior worst 44.49C: True
```

`--full` raises the envelope to `n=50000` and each dimension to `n=2000`, tightening the bound a
hundredfold (about ten minutes):

```
full 5-channel envelope: 50000/50000 safe, viol=0.00%, CP95 upper=0.006%, worst 44.50C / 4.141V
corner peakT=44.50C dominates interior worst 44.49C: True
```

The corner's peak temperature dominating the interior worst case is the monotonicity claim, made
empirical. An identification scheme would need one extra state per channel; the certificate needs none.

### Price of oracle-freedom &mdash; `reproduce/price_of_oracle_freedom.py`

The fail-safe bound gives up some charge versus a filter that knows the true cell. This measures how
much, and shows the cost tracks the *binding* constraint, not the number of uncertain channels.
Expected output:

```
price of oracle-freedom over the full charge window (mean oracle SOC - corner SOC), n=300:
  d=1 ['R']                                     penalty= 0.11% SOC  binding=plating (100%)
  d=2 ['R', 'cool']                             penalty= 0.16% SOC  binding=plating (100%)
  d=3 ['R', 'cool', 'Cth']                      penalty= 0.23% SOC  binding=plating (100%)
  d=4 ['R', 'cool', 'Cth', 'plate']             penalty= 1.58% SOC  binding=plating (100%)
  d=5 ['R', 'cool', 'Cth', 'plate', 'Tamb']     penalty= 1.33% SOC  binding=plating (100%)
```

The thermal channels enter safety only through temperature, which is not binding here, so their
combined cost is negligible (~0.23%). Almost the entire price comes from the one channel that binds
(plating). Uncertainty off the binding constraint is nearly free &mdash; adding a fifth channel does not
even increase the price, it slightly lowers it, because a hotter assumed ambient slows the oracle too.
`--full` raises each dimension to `n=1000`.

## The safety guarantees

`tests/test_safety.py` turns the paper's two propositions into executable checks:

1. **Invariance (Prop. 1).** The projected current always leaves the next-step temperature and voltage
   inside the certified set, and `I=0` adds no self-heating &mdash; checked from states whose RC branch has
   been charged by prior current, not just from freshly initialized ones, since the RC residual is what
   the proof turns on.
2. **A bound beats identification (Prop. 2), both channels.** Peak temperature and the largest safe
   current are monotone in the resistance scale `s_R` *and* in the cooling-loss factor `gamma` (where
   `T >= T_amb`, the proposition's hypothesis), so a conservative *upper* bound on each certifies safety
   with no per-cell identification.
3. **Plating is monotone.** The anode plating potential is affine in current everywhere and strictly
   decreasing across the fast-charge envelope, so it enters the same one-dimensional search as
   temperature and voltage.
4. **The search is exact.** The admissible current set is a single interval anchored at 0 for the
   certified constraints, so the bisection converges to the true boundary rather than to some interior
   point. The check also reports where adding plating makes that interval empty (~0.82 SOC), which is
   the concrete reason plating is enforced rather than certified.

## What this repository does and does not contain

It contains the filter, the ROM it predicts with, the executable proposition checks, and the
solver/dimension/price experiments &mdash; everything that runs on numpy in seconds.

It does **not** contain the high-fidelity validation pipeline behind the paper's Tables I&ndash;III and
Figs. 3&ndash;5: the PyBaMM DFN plant, the PPO baseline, the online resistance estimator, and the ROM
calibration harness are outside this release. Numbers here therefore reproduce the propositions and the
cost model, not the DFN-replayed tables.

## Repository layout

```
safe_charge/
  filter.py     the safety path (project_current): 36 source lines, numpy only
  rom.py        the reduced-order electro-thermal-plating cell model
  _data/        model parameters (OCV table, model coefficients)
examples/
  quickstart.py           charge an aged cell through the filter
tests/
  test_safety.py          the three safety guarantees, executable
reproduce/
  solver_comparison.py         filter vs a warm-started embedded QP
  dimension_independence.py    one fixed certificate over a 4-channel envelope
  price_of_oracle_freedom.py   charge cost of the fail-safe bound, per channel
  tolerance_sweep.py           what the bisection tolerance buys
  step_size.py                 safety and charge vs the control step
  robustness.py                past the aging bound, anchors, margin gains
bench/
  footprint.py            source and dependency footprint vs the QP path
  timing.py               latency, operation count, memory
run_all.sh                run all nine in order
requirements.txt          numpy (core) + osqp/scipy (solver baseline)
```

## Reproducibility

- Every reproduce script sets an explicit `numpy` seed, so runs are deterministic to the digit.
- The core (filter, model, tests, quickstart, dimension/price scripts) needs only numpy. Only the
  solver comparison pulls in `osqp` and `scipy`.
- Runtimes are seconds for the numpy-only pieces; the solver comparison is the slow one (many warm-started
  QP solves), a couple of minutes on `--full`.
- `bash run_all.sh` runs the whole set end to end.

## Paper

The write-up this code accompanies lives in [`paper/`](paper/), source and compiled PDF. The numbers in
its text are produced by the experiments here; see [`paper/README.md`](paper/README.md) to build it.

## Citation

This is the reference implementation for:

> K. Krishnan, "Safe Fast Charging Robust to Battery Aging: A Solver-Free Safety Filter Without a Health
> Oracle," in *Proc. IEEE Industrial Electronics Society Annual Conf. (IECON)*, 2026.

```bibtex
@inproceedings{krishnan2026safecharge,
  author    = {Krishnan, Keshav},
  title     = {Safe Fast Charging Robust to Battery Aging: A Solver-Free Safety Filter Without a Health Oracle},
  booktitle = {Proc. IEEE Industrial Electronics Society Annual Conference (IECON)},
  year      = {2026}
}
```

## Author and license

Keshav Krishnan, Park Tudor School, Indianapolis, IN, USA.

Released under the MIT License &mdash; see [LICENSE](LICENSE).
