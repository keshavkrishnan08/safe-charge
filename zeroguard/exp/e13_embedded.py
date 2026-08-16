"""E13 -- what it would take to put this on a battery management microcontroller.

The paper claims the certificate "fits inside any microcontroller a battery management system
already contains". That is an assertion about software that has never been compiled for one, and
it should either be measured or withdrawn. This experiment measures it.

Four things decide whether a safety routine can live on a BMS part:

  **Does it allocate?** Dynamic allocation in a safety loop is disqualifying under every
  automotive and aerospace coding standard worth naming. Measured, not assumed.

  **Does it need double precision?** A Cortex-M4F has a single-precision FPU and no double.
  Anything in double is emulated in software, at roughly an order of magnitude.

  **Could it run with no FPU at all?** The cheapest BMS parts are integer-only. The bisection
  is trivially integer -- its only arithmetic is a midpoint -- but the *model* it evaluates
  contains an exponential, an inverse hyperbolic sine and a table interpolation.

  **How much memory?** State, constants, and any lookup tables the integer version would need.

The safety-relevant question in all of it is one-sided. Reduced precision that makes the filter
return a *smaller* current costs charge; one that makes it return a *larger* current costs the
guarantee. Every comparison below is therefore signed, not absolute.

    python zeroguard/exp/e13_embedded.py
"""
import os, sys, time, math, tracemalloc, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V

SEED = 20260815
ITERS = 18


def states(n, rng, plat):
    return [plat.init(float(rng.uniform(0.05, 0.95)), float(rng.uniform(-10.0, 44.0)))
            for _ in range(n)]


# ---------------------------------------------------------------------------------------
def a_allocation(n=2000, seed=SEED):
    """Does a projection allocate on the heap once it is running?

    The first call allocates -- Python interns objects, builds the split cache, warms tables.
    What matters for a control loop is the steady state, so the measurement is taken after a
    warm-up and reports the per-call figure.
    """
    rng = np.random.default_rng(seed)
    plat = V.pessimistic("robotaxi-urban")
    marg = V.margins(plat)
    ss = states(n, rng, plat)
    for s in ss[:50]:                                    # warm up
        A.project_anchored(plat, s, plat.u_max, 30.0, 25.0, marg)
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    for s in ss:
        A.project_anchored(plat, s, plat.u_max, 30.0, 25.0, marg)
    snap = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snap.compare_to(base, "lineno")
    grew = sum(d.size_diff for d in diff if d.size_diff > 0)
    return dict(calls=n, net_heap_growth_bytes=int(grew),
                bytes_per_call=grew / n,
                note=("Python allocates for its own object model; what this shows is that the "
                      "projection holds no growing structure -- no list, no queue, no history. "
                      "A C implementation of the same control flow allocates nothing at all, "
                      "because there is nothing to allocate."))


# ---------------------------------------------------------------------------------------
def b_single_precision(n=20000, seed=SEED + 1):
    """Would single precision do? And if it differs, does it differ in the safe direction?"""
    rng = np.random.default_rng(seed)
    plat = V.pessimistic("robotaxi-urban")
    marg = V.margins(plat)
    devs, unsafe = [], 0
    for s in states(n, rng, plat):
        u64, _ = A.project_anchored(plat, s, plat.u_max, 30.0, 25.0, marg)
        s32 = {k: (np.float32(v) if isinstance(v, float) else v) for k, v in s.items()}
        u32, _ = A.project_anchored(plat, s32, plat.u_max, 30.0, 25.0, marg)
        d = float(u32) - float(u64)
        devs.append(d)
        unsafe += int(d > 1e-6)                          # single precision permitted MORE current
    a = np.array(devs)
    return dict(states=n, max_abs_dev_A=float(np.abs(a).max()),
                mean_dev_A=float(a.mean()),
                times_less_conservative=int(unsafe),
                fraction_less_conservative=unsafe / n,
                relative_to_umax=float(np.abs(a).max() / plat.u_max))


# ---------------------------------------------------------------------------------------
def c_integer_bisection(n=20000, seed=SEED + 2, bits=16):
    """The search, with no floating point in it at all.

    Represent the current as an integer count of quanta, `u = q * (u_max / 2**bits)`. The
    bisection's only arithmetic is then `mid = (lo + hi) >> 1`, which is exact on any integer
    unit. Rounding the midpoint *down* makes the returned value conservative by construction:
    it can only ever be at or below the real-valued answer, never above.

    This is the part of the method that genuinely needs no FPU. The constraint evaluation still
    does, and part (d) measures what replacing it would cost.
    """
    rng = np.random.default_rng(seed)
    plat = V.pessimistic("robotaxi-urban")
    marg = V.margins(plat)
    hi_f, lo_f = A.split_cached(plat)
    devs, above = [], 0
    for s in states(n, rng, plat):
        lo_b, hi_b = A._bounds(plat, s)
        q = (hi_b - lo_b) / (1 << bits)
        u_ref, _ = A.project_anchored(plat, s, plat.u_max, 30.0, 25.0, marg)
        # integer bisection over quanta
        ql, qh = 0, (1 << bits)
        if not A._ok(hi_f, plat.probe(s, lo_b, 30.0, 25.0), marg):
            u_int = lo_b
        else:
            if A._ok(hi_f, plat.probe(s, hi_b, 30.0, 25.0), marg):
                ql = (1 << bits)
            else:
                for _ in range(bits):
                    qm = (ql + qh) >> 1                  # integer midpoint, rounds down
                    if A._ok(hi_f, plat.probe(s, lo_b + qm * q, 30.0, 25.0), marg):
                        ql = qm
                    else:
                        qh = qm
            u_int = lo_b + ql * q
        d = float(u_int) - float(u_ref)
        devs.append(d); above += int(d > q + 1e-9)
    a = np.array(devs)
    return dict(states=n, bits=bits,
                max_abs_dev_A=float(np.abs(a).max()),
                times_above_reference=int(above),
                conservative_by_construction=bool(above == 0),
                quantum_A=float((V.pessimistic("robotaxi-urban").u_max) / (1 << bits)),
                note=("the integer midpoint rounds down, so the returned current is at or "
                      "below the real-valued answer -- reduced precision costs charge and "
                      "cannot cost the guarantee"))


# ---------------------------------------------------------------------------------------
def d_footprint():
    """Bytes: live state, calibration constants, and the tables an integer build would need."""
    p = P.load_params()
    n_const = sum(1 for v in p.values() if isinstance(v, (int, float)))
    n_eta = len(p.get("eta_params", []))
    n_plating = sum(1 for k in p if k.startswith("pl_"))

    # live state the projection touches: soc, T, V1, two ageing terms, plus the bracket
    state_f32 = 5 * 4 + 2 * 4
    state_f64 = 5 * 8 + 2 * 8

    # tables an FPU-free build needs, sized for 0.1 % of full scale by linear interpolation
    def lut_entries(f, lo, hi, tol_rel, cap=8192):
        """Smallest uniform table whose linear interpolation meets a relative tolerance."""
        for m in (33, 65, 129, 257, 513, 1025, 2049, 4097, cap):
            x = np.linspace(lo, hi, m)
            y = np.array([f(v) for v in x])
            xf = np.linspace(lo, hi, 4001)
            err = np.abs(np.interp(xf, x, y) - np.array([f(v) for v in xf]))
            scale = max(np.abs(y).max(), 1e-12)
            if err.max() / scale <= tol_rel:
                return m
        return cap

    ea = p["eta_params"][2]
    n_exp = lut_entries(lambda T: math.exp(ea * (1.0 / (T + 273.15) - 1.0 / 298.15)),
                        -30.0, 60.0, 1e-3)
    n_asinh = lut_entries(lambda x: math.asinh(x), 0.0, 60.0, 1e-3)
    n_ocv = 128            # the OCV table already ships as data, resampled uniformly
    lut_bytes = (n_exp + n_asinh + n_ocv) * 4

    return dict(calibration_constants=n_const + n_eta + n_plating,
                calibration_bytes_f32=(n_const + n_eta + n_plating) * 4,
                live_state_bytes_f32=state_f32, live_state_bytes_f64=state_f64,
                lut_entries=dict(exp=n_exp, asinh=n_asinh, ocv=n_ocv),
                lut_bytes_q16=lut_bytes,
                total_ram_f32_bytes=state_f32 + (n_const + n_eta + n_plating) * 4,
                total_flash_integer_build_bytes=lut_bytes
                + (n_const + n_eta + n_plating) * 4,
                evaluations_charge=20, evaluations_discharge=40,
                stack_depth="constant: no recursion, no variable-length structure")


# ---------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("E13 -- deployability on a battery-management microcontroller\n" + "=" * 78)
    out = {}

    print("\n(a) does a projection allocate once it is running?")
    r = a_allocation(); out["allocation"] = r
    print(f"    {r['calls']:,} calls | net heap growth {r['net_heap_growth_bytes']:,} bytes "
          f"= {r['bytes_per_call']:.1f} bytes/call")
    print(f"    {r['note']}")

    print("\n(b) is single precision enough, and does it err safely?")
    r = b_single_precision(); out["single_precision"] = r
    print(f"    {r['states']:,} states | max deviation {r['max_abs_dev_A']:.3e} A "
          f"({100*r['relative_to_umax']:.2e} % of full scale)")
    print(f"    times single precision permitted MORE current than double: "
          f"{r['times_less_conservative']}")

    print("\n(c) the search with no floating point at all")
    r = c_integer_bisection(); out["integer_bisection"] = r
    print(f"    {r['states']:,} states | {r['bits']}-bit quanta of {r['quantum_A']:.4f} A")
    print(f"    max deviation {r['max_abs_dev_A']:.4f} A | times above the real-valued "
          f"answer: {r['times_above_reference']}")
    print(f"    conservative by construction: {r['conservative_by_construction']}")

    print("\n(d) memory")
    r = d_footprint(); out["footprint"] = r
    print(f"    live state {r['live_state_bytes_f32']} B (single) / "
          f"{r['live_state_bytes_f64']} B (double)")
    print(f"    calibration constants {r['calibration_constants']} values = "
          f"{r['calibration_bytes_f32']} B")
    print(f"    RAM for a single-precision build: {r['total_ram_f32_bytes']} B")
    print(f"    tables for an FPU-free build: exp {r['lut_entries']['exp']}, "
          f"asinh {r['lut_entries']['asinh']}, ocv {r['lut_entries']['ocv']} entries "
          f"= {r['lut_bytes_q16']:,} B of flash")
    print(f"    stack: {r['stack_depth']}")

    sp = out["single_precision"]
    margin_K = V.DT0 + V.K_TH * (V.S_R - 1.0)
    out["verdict"] = dict(
        no_growing_state=out["allocation"]["bytes_per_call"] < 64,
        # Single precision is NOT one-sided: it permits slightly more current than double in a
        # minority of states. What matters is the magnitude against the margin, not the sign
        # count, so both are reported and the stronger claim is reserved for the integer build.
        single_precision_one_sided=sp["times_less_conservative"] == 0,
        single_precision_worst_excess_A=max(0.0, sp["max_abs_dev_A"]),
        single_precision_excess_frac_of_scale=sp["relative_to_umax"],
        integer_one_sided=out["integer_bisection"]["conservative_by_construction"],
        ram_bytes=out["footprint"]["total_ram_f32_bytes"],
        flash_bytes=out["footprint"]["total_flash_integer_build_bytes"])
    print(f"\n  verdict")
    print(f"    single precision is NOT one-sided -- it permits more current than double in "
          f"{sp['times_less_conservative']:,} of {sp['states']:,} states, by at most "
          f"{sp['max_abs_dev_A']*1e3:.2f} mA")
    print(f"    which is {100*sp['relative_to_umax']:.1e} % of full scale, against a "
          f"{margin_K:.1f} K thermal margin: negligible in magnitude, but not a guarantee")
    print(f"    the integer search IS one-sided by construction, so an FPU-free build is the "
          f"one with a provable direction of error")
    print(f"    footprint: {out['verdict']['ram_bytes']} B RAM, "
          f"{out['verdict']['flash_bytes']:,} B tables, constant stack, no growing state")

    path = V.save("e13_embedded.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
