"""L1 -- the Lean audit, recorded from the build rather than transcribed.

The paper says the development is machine-checked and axiom-free. Both halves of that are
claims about a build, and a claim about a build that is typed into a manuscript by hand is a
claim that can rot. This runs `lake build` and records what Lean actually reports, so the
paper's numbers -- theorem count, `sorry` count, and the axiom dependency of each theorem --
are generated from the artifact.

The audit matters most where it is *not* clean. Thirteen of the theorems depend on no axioms
at all, which is unusual and worth saying. The two concrete counterexamples depend on
`propext` -- propositional extensionality -- which enters through core's `Nat` division lemmas.
That is the mildest of Lean's three axioms and is uncontroversial, but it is not nothing, and
the difference between "no axioms" and "no axioms except propext in two evaluation lemmas" is
exactly the sort of thing this project has been careful about everywhere else.

    python zeroguard/exp/l1_lean_audit.py
"""
import os, sys, re, json, subprocess, shutil, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zeroguard import vexp as V

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORMAL = os.path.join(ROOT, "formal")
SRC = os.path.join(FORMAL, "AnchoredCollapse.lean")


def main():
    t0 = time.time()
    print("L1 -- the Lean audit, read from the build\n" + "=" * 78)
    lake = shutil.which("lake") or os.path.expanduser("~/.elan/bin/lake")
    if not os.path.exists(lake):
        print("  no lake on PATH; skipping")
        return
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(lake) + os.pathsep + env["PATH"]
    subprocess.run([lake, "build"], cwd=FORMAL, capture_output=True, text=True, env=env)
    # a clean rebuild is what actually reports the axioms; a replay may not re-emit them
    r = subprocess.run([lake, "build"], cwd=FORMAL, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    if "does not depend on any axioms" not in out:
        subprocess.run([lake, "clean"], cwd=FORMAL, capture_output=True, text=True, env=env)
        r = subprocess.run([lake, "build"], cwd=FORMAL, capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr

    src = open(SRC).read()
    thms = re.findall(r"^theorem\s+(\w+)", src, re.M)
    body = re.sub(r"/-.*?-/", "", src, flags=re.S)          # strip docstrings/comments
    body = re.sub(r"--.*", "", body)
    sorries = len(re.findall(r"\bsorry\b", body))

    clean = set(re.findall(r"'AnchoredCollapse\.(\w+)' does not depend on any axioms", out))
    dep = {m[0]: [a.strip() for a in m[1].split(",")]
           for m in re.findall(r"'AnchoredCollapse\.(\w+)' depends on axioms: \[([^\]]*)\]", out)}

    audit = dict(build_ok=(r.returncode == 0), theorems=len(thms), theorem_names=thms,
                 sorries=sorries, axiom_free=sorted(clean), axiom_using=dep,
                 n_axiom_free=len(clean), n_axiom_using=len(dep),
                 axioms_used=sorted({a for v in dep.values() for a in v}),
                 uses_sorryAx=any("sorryAx" in v for v in dep.values()),
                 uses_choice=any("Classical.choice" in v for v in dep.values()),
                 toolchain=open(os.path.join(FORMAL, "lean-toolchain")).read().strip(),
                 mathlib_free="import Mathlib" not in src)

    print(f"  build {'succeeded' if audit['build_ok'] else 'FAILED'}   toolchain {audit['toolchain']}")
    print(f"  {audit['theorems']} theorems, {audit['sorries']} occurrences of `sorry`, "
          f"mathlib-free: {audit['mathlib_free']}")
    print(f"  {audit['n_axiom_free']} depend on no axioms at all")
    for k, v in sorted(dep.items()):
        print(f"  {k} depends on: {', '.join(v)}")
    print(f"  sorryAx anywhere: {audit['uses_sorryAx']}   Classical.choice anywhere: "
          f"{audit['uses_choice']}")
    path = V.save("l1_lean_audit.json", audit)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
