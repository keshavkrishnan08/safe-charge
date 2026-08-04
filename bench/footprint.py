"""Measure the software footprint of the certified safety path, and the trusted-computing-base
(TCB) it puts a functional-safety audit on the hook for, versus the embedded-QP alternative.

Run it:
    python bench/footprint.py
"""
import os, sys, ast, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def sloc_of(path, names):
    src = open(path).read(); lines = src.splitlines(); tree = ast.parse(src)
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in names:
            seg = lines[n.lineno-1:n.end_lineno]
            out[n.name] = sum(1 for l in seg if l.strip() and not l.strip().startswith("#"))
    return out


def dir_footprint(pkg_dir):
    total = 0; cfiles = 0
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            if f.endswith((".c", ".h", ".cpp", ".pyx")):
                cfiles += 1
    return total, cfiles


filt = sloc_of(os.path.join(ROOT, "safe_charge", "filter.py"),
               {"project_current", "_feasible", "_eff_margin"})
certified = sum(filt.values())

report = {
    "certified_path": {
        "functions": filt,
        "total_sloc": certified,
        "runtime_deps": ["numpy"],
        "third_party_solver": None,
        "loop_bound": "static (18 iterations)",
        # The bisection body allocates nothing; each iteration calls the ROM one-step map,
        # which builds its observation dict. An allocation-free build inlines that map.
        "heap_alloc_in_bisection_body": False,
    },
    "embedded_qp_alternative": {
        "runtime_deps": ["numpy", "scipy.sparse", "osqp"],
        "third_party_solver": "OSQP (compiled ADMM QP solver)",
        "loop_bound": "data-dependent (mean/max ~85/775 iterations at the cold corner; "
                      "reproduce/solver_comparison.py --full)",
        "heap_alloc_in_loop": True,  # sparse matrix assembly + solver setup each step
    },
}

# OSQP TCB size, if installed
try:
    import osqp
    mb, cfiles = dir_footprint(os.path.dirname(osqp.__file__))
    report["embedded_qp_alternative"]["osqp_installed_mb"] = round(mb/1e6, 1)
    report["embedded_qp_alternative"]["osqp_bundled_c_files"] = cfiles
except Exception as e:
    report["embedded_qp_alternative"]["osqp"] = f"not installed ({e})"

print("== Certified safety path (this package) ==")
print(f"  source lines: {certified}  ({', '.join(f'{k}={v}' for k,v in filt.items())})")
print(f"  runtime dependencies: numpy only, no third-party solver")
print(f"  loop bound: static, 18 iterations  |  allocation in the bisection body: none")
print("\n== Embedded-QP alternative (OSQP linearized MPC) ==")
qp = report["embedded_qp_alternative"]
print(f"  runtime dependencies: {', '.join(qp['runtime_deps'])}")
if "osqp_installed_mb" in qp:
    print(f"  adds to the TCB: OSQP solver, {qp['osqp_bundled_c_files']} bundled C/header files, "
          f"{qp['osqp_installed_mb']} MB")
print(f"  loop bound: {qp['loop_bound']}  |  heap allocation in loop: yes (matrix assembly + setup)")
print("\n  (Each bisection iteration calls the ROM one-step map, which builds an observation dict;\n"
      "   an allocation-free deployment inlines it. The bound -- 18 iterations, no convergence\n"
      "   test -- is what the worst-case-time claim rests on.)")
print("\nThe entire safety-critical numerics of the filter are elementary arithmetic and a "
      "statically bounded bisection an auditor can read on one screen; the QP path places an "
      "iterative solver whose convergence is the failure mode into the trusted computing base.")

with open(os.path.join(HERE, "footprint.json"), "w") as fh:
    json.dump(report, fh, indent=2)
