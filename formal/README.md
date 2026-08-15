# Machine-checked Anchored Collapse

Lean 4 formalisation of the two properties the safety filter rests on. No `sorry`, no mathlib,
and — as Lean's own axiom audit reports — **no dependency on any axiom at all**.

```bash
curl -sSL -o /tmp/elan.tar.gz \
  https://github.com/leanprover/elan/releases/download/v4.2.3/elan-aarch64-apple-darwin.tar.gz
tar xzf /tmp/elan.tar.gz -C /tmp && /tmp/elan-init -y
cd formal && lake build          # prints the axiom audit for all seven theorems
```

| theorem | what it says |
|---|---|
| `admissible_ordConvex` | intersecting caps and floors leaves an **interval** — the collapse theorem |
| `admissible_isCap_of_no_floors` | with no floors it is downward closed: Null-Input Collapse recovered |
| `between_admissible` | everything between two admissible inputs is admissible |
| `bisect_sound` | **every reachable bisection state has a feasible lower endpoint** |
| `bisect_le` | the bracket never inverts |
| `bisect_hi_bad` | the upper endpoint stays infeasible, so it never certifies past the edge |
| `filter_sound` | the two combined: what the filter returns, and everything down to the anchor, is admissible |

`bisect_sound` is the one worth having machine-checked. Convergence is a claim about how *good*
the answer is; this is a claim about whether it is *safe*, and it holds independent of iteration
count, of the midpoint rule, and of floating-point behaviour — the proof never assumes the
midpoint is the arithmetic mean, only that it lies between the endpoints. A filter interrupted
by a deadline still returns an admissible input.

Working through the proofs showed they need less than the paper assumes. Everything is stated
over an arbitrary type carrying a `≤` relation; no order axiom is used anywhere, so the results
hold without transitivity, antisymmetry or totality.
