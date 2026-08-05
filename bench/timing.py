"""Real-time cost of the safety filter: latency, operation count, and memory.

The filter's claim is a BOUNDED worst case, so the worst case is the number that matters --
an average over a mixed workload hides it. The projection has exactly three paths, and we
time each separately after verifying which is which by counting model evaluations:

    1 evaluation   the policy's request is already feasible; return it unchanged
    2 evaluations  even I=0 is infeasible (already-unsafe state); command zero
   20 evaluations  the full 18-iteration bisection -- the real-time case

What this does and does not measure. These are host timings, not a board measurement: there
is no Cortex-M in this loop and we will not invent one. What transfers to a target is the
OPERATION COUNT, a property of the algorithm rather than of this machine. Python per-call
timing is also far noisier than the work being timed, so latency is measured in batches and
reported as the minimum batch mean -- the standard way to read a timing distribution whose
upper tail is the operating system, not the code.

    python bench/timing.py
"""
import os, sys, gc, time, tracemalloc
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np
from safe_charge import BatteryROM, project_current

DT, TAMB, VLIM, TLIM = 30.0, 25.0, 4.20, 45.0
ITERS = 18
BATCH, REPEATS = 500, 60


class _CountingROM(BatteryROM):
    """Counts one-step model evaluations, the unit of work the bisection is bounded in."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.n_step = 0

    def step(self, s, I, dt, T_amb):
        self.n_step += 1
        return super().step(s, I, dt, T_amb)

    def probe(self, s, I, dt, T_amb):
        # the filter's inner loop calls probe(), not step(); count both
        self.n_step += 1
        return super().probe(s, I, dt, T_amb)


def classify_states():
    """Group operating states by how many model evaluations the projection actually uses."""
    rom = _CountingROM()
    Qn = rom.p["Q_nom"]
    by_path = {}
    for soc in np.linspace(0.05, 0.95, 24):
        for T in np.linspace(0.0, 44.0, 24):
            s = rom.init_state(float(soc), float(T))
            rom.n_step = 0
            project_current(rom, s, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM)
            by_path.setdefault(rom.n_step, []).append(s)
    return by_path


def batched_latency(rom, states, Qn):
    """Per-call latency in microseconds, measured in batches with gc disabled.

    Returns (min, median, max) over batch means. The minimum is the cleanest estimate of
    the code's own cost; the spread across batches is host scheduling noise, since every
    call in a given set performs identical work.
    """
    pool = (states * (BATCH // len(states) + 1))[:BATCH]
    for s in pool:                                      # warm interpreter and caches
        project_current(rom, s, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM)
    means = np.empty(REPEATS)
    gc.disable()
    try:
        for r in range(REPEATS):
            t0 = time.perf_counter_ns()
            for s in pool:
                project_current(rom, s, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM)
            means[r] = (time.perf_counter_ns() - t0) / len(pool) / 1e3
    finally:
        gc.enable()
    return means.min(), float(np.median(means)), means.max()


def main():
    rom = BatteryROM()
    Qn = rom.p["Q_nom"]
    by_path = classify_states()
    labels = {1: "1 eval  (request feasible)", 2: "2 evals (I=0 infeasible)",
              2 + ITERS: f"{2+ITERS} evals (full bisection)"}

    print(f"== Latency per projection, by code path "
          f"(host; {REPEATS} batches of {BATCH}, gc off) ==")
    print(f"  {'path':32}{'min':>10}{'median':>10}{'max':>10}   (us/call, over batch means)")
    for n in sorted(by_path):
        lo, mid, hi = batched_latency(rom, by_path[n], Qn)
        print(f"  {labels.get(n, f'{n} evals'):32}{lo:>10.2f}{mid:>10.2f}{hi:>10.2f}"
              f"   n_states={len(by_path[n])}")
    print("  Every call within a path does identical work, so the min-to-max spread is host")
    print("  scheduling noise. The full-bisection row is the real-time number.")

    print("\n== Work per projection (hardware-independent, the number that ports) ==")
    print(f"  hard bound: 2 + iters = {2+ITERS} one-step model evaluations, every input, always")
    print(f"  observed paths: {sorted(by_path)} evaluations")
    print(f"  bisection: fixed {ITERS} iterations, no convergence test, no tolerance exit")
    print(f"  per evaluation: ~40 flops, 2 exp, 1 arcsinh, 1 table interpolation (closed form)")
    print(f"  => worst case ~{(2+ITERS)*40} flops + {(2+ITERS)*2} exp + {2+ITERS} arcsinh")
    print(f"  complexity: O(log(1/eps)) evaluations for current resolution eps; no matrix")
    print(f"  factorization, no rollout, no data-dependent iteration count")

    # --- memory ---
    s0 = by_path[max(by_path)][0]
    project_current(rom, s0, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM)   # warm
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    project_current(rom, s0, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    state_floats = 3 + 2          # soc, T, V1  +  aging{Qloss, Rfac}
    param_floats = sum(1 for v in rom.p.values() if isinstance(v, (int, float))) \
                 + sum(len(v) for v in rom.p.values() if isinstance(v, list))
    ocv_floats = 2 * len(np.loadtxt(os.path.join(ROOT, "safe_charge", "_data", "ocv.csv"),
                                    delimiter=",", skiprows=1))
    total = 8 * (state_floats + param_floats + ocv_floats)

    print("\n== Memory ==")
    print(f"  persistent cell state: {state_floats} floats ({8*state_floats} B)")
    print(f"  model parameters: {param_floats} floats ({8*param_floats} B)")
    print(f"  OCV table: {ocv_floats} floats ({8*ocv_floats} B)")
    print(f"  => {total} B of state and constants in double precision, {total//2} B in single")
    print(f"  no dynamic allocation, no matrix storage, no solver workspace")
    print(f"  (this interpreter allocates {peak-base} B of transient Python objects per call;")
    print(f"   the bisection body itself allocates nothing, since it calls ROM.probe(), which")
    print(f"   returns a tuple instead of building an observation dict)")

    print("\n== Scaling to a target (ESTIMATE, not a board measurement) ==")
    print(f"  The {2+ITERS} evaluations are the invariant. At roughly 1 flop/cycle without FPU")
    print( "  pipelining, ~800 flops plus 40 transcendentals is on the order of 5-10 k cycles,")
    print( "  i.e. of order 100 us on a 100 MHz Cortex-M4F. Against the 30 s control step that")
    print( "  is a duty cycle near 3e-6, five orders of magnitude of headroom. The point is not")
    print( "  the number but that a bound exists at all: no solver, no convergence test, no")
    print( "  data-dependent iteration count.")


if __name__ == "__main__":
    main()
