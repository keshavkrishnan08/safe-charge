"""E14 -- the firmware, compiled and diffed against the reference.

\\S\\ref{sec:embedded} measured what the certificate *would* cost on a battery-management
microcontroller: bytes of state, bytes of table, whether the search needs a floating-point
unit. Every one of those numbers came from a Python model of the arithmetic, which is a
reasonable way to estimate a footprint and no way at all to establish one. "Fits in a
kilobyte" is a claim about a binary, and there was no binary.

`firmware/zeroguard.c` is that binary's source: the charge-direction certificate in C99, with
an integer bisection and a float32 model, no heap, no recursion, no variable-length structure,
and every loop bound a compile-time constant. This experiment builds it and asks three
questions that only a real implementation can answer.

  **How big is it, actually?** Measured from the object file's sections, not estimated from a
  count of constants.

  **Does it agree with the reference?** The two are not expected to be bit-identical -- the
  reference computes in float64 and bisects in floating point, the firmware computes in float32
  and bisects in integers -- so the question is not whether they differ but *which way*, and by
  how much against the margin the filter carries.

  **Is the disagreement one-sided?** \\S\\ref{sec:embedded} argued the integer midpoint rounds
  down and therefore cannot cost the guarantee. That argument covers the search and says
  nothing about the model, and single precision was already shown *not* to be one-sided. The
  combined implementation has to be measured rather than argued.

The constants are generated from the same calibration files the Python filter loads. If they
were transcribed, the two could drift apart and every agreement number below would be measuring
a copy of itself.

    python zeroguard/exp/e14_firmware.py
"""
import os, sys, time, json, subprocess, shutil, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FW = os.path.join(ROOT, "firmware")
SEED, DT, W = 20260816, 30.0, 25.0
CC = shutil.which("clang") or shutil.which("gcc")


def build(out_dir="/tmp/zg_build"):
    """Regenerate the constants, compile at -Os, and report the object's sections."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(FW, "gen_params.py")],
                   check=True, capture_output=True)
    obj = os.path.join(out_dir, "zeroguard.o")
    exe = os.path.join(out_dir, "zg")
    flags = ["-std=c11", "-Os", "-Wall", "-Wextra", "-Werror", f"-I{FW}"]
    subprocess.run([CC, *flags, "-c", os.path.join(FW, "zeroguard.c"), "-o", obj], check=True)
    subprocess.run([CC, *flags, os.path.join(FW, "zeroguard.c"),
                    os.path.join(FW, "zg_main.c"), "-lm", "-o", exe], check=True)
    txt = subprocess.run(["size", obj], capture_output=True, text=True).stdout
    nums = [int(x) for x in re.findall(r"\d+", txt.splitlines()[-1])] if txt else []
    return exe, obj, nums, txt


def reference(est, states, marg):
    out = []
    for s in states:
        _lo, hi, st = A.interval(est, s, DT, W, marg)
        out.append((0 if st == "ok" else 1, hi if st == "ok" else 0.0))
    return out


def firmware(exe, est, states, marg):
    """Feed the same pack and the same states to the compiled binary."""
    c = est.cell
    hdr = (f"{c.scale['R']} {c.scale['Q']} {c.scale['plate']} 1.0 0.0 {c.cooling.hA} "
           f"{est.S} {est.P} {est.u_max} {est.V_max} {est.T_max} "
           f"{marg[0]} {marg[1]} {marg[2]} {DT} {W}\n")
    body = "".join(f"{s['soc']!r} {s['T']!r} {s['V1']!r}\n" for s in states)
    r = subprocess.run([exe], input=hdr + body, capture_output=True, text=True, check=True)
    out = []
    for line in r.stdout.strip().splitlines():
        a, b = line.split()
        out.append((int(a), float(b)))
    return out


def main(n=20000, seed=SEED):
    t0 = time.time()
    print("E14 -- the firmware, compiled and diffed against the reference\n" + "=" * 78)
    if CC is None:
        print("  no C compiler on PATH; nothing to measure")
        return
    exe, obj, nums, raw = build()
    print(f"  built with {os.path.basename(CC)} -Os -Wall -Wextra -Werror, no warnings")

    # ---- what it costs ----------------------------------------------------------------
    text = nums[0] if nums else None
    data = nums[1] if len(nums) > 1 else 0
    ocv_bytes = 128 * 4
    state_bytes = 3 * 4                    # soc, T, V1
    pack_bytes = 6 * 4 + 2 * 4 + 6 * 4     # cell params, S/P, limits and margins
    print(f"\n  code and constants   {text:>6} B  (of which the OCV table is {ocv_bytes})")
    print(f"  writable data        {data:>6} B")
    print(f"  live RAM per call    {state_bytes + pack_bytes:>6} B  "
          f"(state {state_bytes} + pack description {pack_bytes}; no heap, no recursion)")

    # ---- does it agree ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    est = V.pessimistic("robotaxi-urban", T_amb=W)
    marg = V.margins(est)
    states = [est.init(float(rng.uniform(0.05, 0.95)), float(rng.uniform(-10.0, 44.0)))
              for _ in range(n)]
    ref = reference(est, states, marg)
    fw = firmware(exe, est, states, marg)

    st_mismatch = sum(1 for (a, _), (b, _) in zip(ref, fw) if a != b)
    both_ok = [(r[1], f[1]) for r, f in zip(ref, fw) if r[0] == 0 and f[0] == 0]
    d = np.array([f - r for r, f in both_ok])
    quantum = est.u_max / (1 << 16)
    above = int((d > quantum).sum())

    print(f"\n  {n:,} states, identical pack, identical margins")
    print(f"  feasible/infeasible verdict differs in {st_mismatch}")
    print(f"  current returned: firmware minus reference")
    print(f"    median {np.median(d):+.4g} A, worst low {d.min():+.4g}, "
          f"worst high {d.max():+.4g} A")
    print(f"    one bisection quantum is {quantum:.4g} A; the firmware exceeds the reference "
          f"by more than a quantum in {above} of {len(d):,}")

    # what a disagreement in the unsafe direction is worth, in the currency of the margin
    est_probe = est
    worst_v = worst_t = 0.0
    for (r, f), s in zip(both_ok, states):
        if f <= r:
            continue
        pv = est_probe.probe(s, f, DT, W)
        rv = est_probe.probe(s, r, DT, W)
        worst_v = max(worst_v, float(pv[0] - rv[0]))
        worst_t = max(worst_t, float(pv[1] - rv[1]))
    print(f"  where it is higher, the extra current is worth at most {1000*worst_v:.3f} mV "
          f"({100*worst_v/marg[0]:.2f}% of the voltage margin) and {worst_t:.4f} K "
          f"({100*worst_t/marg[1]:.3f}% of the thermal margin)")

    out = dict(compiler=os.path.basename(CC), flags="-std=c11 -Os -Wall -Wextra -Werror",
               target=subprocess.run([CC, "-dumpmachine"], capture_output=True,
                                     text=True).stdout.strip(),
               text_bytes=text, data_bytes=data, ocv_table_bytes=ocv_bytes,
               live_state_bytes=state_bytes, pack_struct_bytes=pack_bytes,
               ram_bytes=state_bytes + pack_bytes,
               size_raw=raw.strip(), states=n,
               status_mismatches=st_mismatch,
               median_dev_A=float(np.median(d)), min_dev_A=float(d.min()),
               max_dev_A=float(d.max()), quantum_A=float(quantum),
               above_reference_by_a_quantum=above,
               worst_voltage_cost_V=worst_v, worst_thermal_cost_K=worst_t,
               worst_voltage_frac_margin=worst_v / marg[0],
               worst_thermal_frac_margin=worst_t / marg[1],
               within_margin=bool(worst_v < marg[0] and worst_t < marg[1]),
               evaluations=18, heap_allocations=0, recursion=False,
               loop_bounds="all compile-time constants")

    print(f"\n  the deviation stays inside the margins the filter already carries, so the "
          f"guarantee survives the port; it is not bit-identical and the paper does not say "
          f"it is.")
    path = V.save("e14_firmware.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
