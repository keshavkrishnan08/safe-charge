/-
  Anchored Collapse, machine-checked.

  This file proves the two properties the safety filter in `zeroguard/anchored.py` relies on,
  with no `sorry` and no dependency on mathlib. Everything is stated over an arbitrary type
  carrying a `≤` relation -- not a linear order, not even a preorder. Working through the
  proofs showed that no order axiom is used anywhere: not transitivity, not antisymmetry, not
  totality. The results hold for any binary relation written `≤`, which is a stronger statement
  than the one the paper needed and costs nothing to make.

  What is proved:

    1. `admissible_ordConvex` -- the admissible input set is order-convex. Since a set of
       reals is an interval exactly when it is order-convex, this *is* the collapse theorem:
       intersecting any number of caps (satisfied on a down-set) with any number of floors
       (satisfied on an up-set) leaves an interval, never a union of pieces. The whole
       two-bisection construction is unsound without it, because a bisection can converge
       into a hole in a disconnected set.

    2. `bisect_sound` -- every state the bisection can reach has a *feasible* lower endpoint
       and an *infeasible* upper endpoint. The filter returns the lower endpoint, so the
       returned input is admissible after any number of iterations, including zero.

    3. `bisect_sound_any_predicate` -- and that guarantee needs **no monotonicity whatsoever**.
       Writing the proof of (2) revealed that it never touches the cap/floor structure, so
       hypothesis (A2) buys optimality rather than safety: a non-monotone constraint set costs
       delivered charge and cannot cost the guarantee.

  Property 2 is the one worth having machine-checked. Convergence is a statement about how
  *good* the answer is; this is a statement about whether the answer is *safe*, and it holds
  independent of iteration count, of the midpoint rule, and of floating-point behaviour --
  the proof never assumes the midpoint is the arithmetic mean, only that it lies between the
  endpoints. A filter that is interrupted, starved of its time budget, or run on a machine
  with a sloppy division still returns a feasible input.

  Correspondence with the implementation. `IsCap`/`IsFloor` are the `"hi"`/`"lo"` families
  that `anchored.split` separates; `Admissible` is `anchored.feasible`; `bisect` is the loop
  in `anchored.interval`. The Python asserts these properties numerically on 64,000 dense
  scans (`exp/x_crossdomain.py`, X1); this proves them.
-/

namespace AnchoredCollapse

/-- A **cap**: a constraint satisfied on a downward-closed set of inputs. Temperature and
    voltage limits are caps -- if a current is cool enough, every smaller current is too. -/
def IsCap {α : Type _} [LE α] (S : α → Prop) : Prop :=
  ∀ ⦃a b : α⦄, S b → a ≤ b → S a

/-- A **floor**: a constraint satisfied on an upward-closed set of inputs. A vehicle's
    irreducible load is a floor -- if a current serves the load, every larger current does. -/
def IsFloor {α : Type _} [LE α] (S : α → Prop) : Prop :=
  ∀ ⦃a b : α⦄, S a → a ≤ b → S b

/-- Order-convexity: anything between two members is a member. Over the reals this is exactly
    "is an interval". -/
def OrdConvex {α : Type _} [LE α] (S : α → Prop) : Prop :=
  ∀ ⦃a b c : α⦄, S a → S c → a ≤ b → b ≤ c → S b

/-- The admissible set: every cap and every floor satisfied at once. `ι` and `κ` index the two
    families and may be empty, finite, or infinite -- the argument never counts them. -/
def Admissible {α : Type _} [LE α] {ι κ : Type _}
    (cap : ι → α → Prop) (flr : κ → α → Prop) (u : α) : Prop :=
  (∀ i, cap i u) ∧ (∀ j, flr j u)

/-! ### 1. The collapse theorem -/

/-- **Anchored Collapse (structure).** An arbitrary intersection of caps and floors is
    order-convex, hence an interval. This is the fact that makes one bisection per edge a
    complete search rather than a heuristic. -/
theorem admissible_ordConvex {α : Type _} [LE α] {ι κ : Type _}
    (cap : ι → α → Prop) (flr : κ → α → Prop)
    (hcap : ∀ i, IsCap (cap i)) (hflr : ∀ j, IsFloor (flr j)) :
    OrdConvex (Admissible cap flr) := by
  rintro a b c ⟨-, hfa⟩ ⟨hcc, -⟩ hab hbc
  refine ⟨fun i => hcap i (hcc i) hbc, fun j => hflr j (hfa j) hab⟩

/-- With no floors, the admissible set is itself a cap: downward closed. This is Null-Input
    Collapse -- the original theorem -- recovered as the special case `κ = Empty`. -/
theorem admissible_isCap_of_no_floors {α : Type _} [LE α] {ι : Type _}
    (cap : ι → α → Prop) (hcap : ∀ i, IsCap (cap i)) :
    IsCap (Admissible cap (fun (_ : Empty) (_ : α) => True)) := by
  rintro a b ⟨hcb, -⟩ hab
  exact ⟨fun i => hcap i (hcb i) hab, fun j => j.elim⟩

/-- A cap family and a floor family that are both satisfied somewhere give a nonempty
    interval containing every point between any two witnesses. -/
theorem between_admissible {α : Type _} [LE α] {ι κ : Type _}
    (cap : ι → α → Prop) (flr : κ → α → Prop)
    (hcap : ∀ i, IsCap (cap i)) (hflr : ∀ j, IsFloor (flr j))
    {lo hi u : α} (hlo : Admissible cap flr lo) (hhi : Admissible cap flr hi)
    (h1 : lo ≤ u) (h2 : u ≤ hi) : Admissible cap flr u :=
  admissible_ordConvex cap flr hcap hflr hlo hhi h1 h2

/-! ### 2. The bisection is sound at every step -/

/-- The bracket the upper bisection maintains: `lo` feasible, `hi` infeasible, `lo ≤ hi`. -/
structure Bracket {α : Type _} [LE α] (P : α → Prop) where
  lo : α
  hi : α
  lo_ok : P lo
  hi_bad : ¬ P hi
  le : lo ≤ hi

/-- One bisection step, with an *arbitrary* midpoint rule. The proof below never uses what
    `mid` computes, only that it lands between the endpoints -- so it covers the arithmetic
    mean, any rounding of it, and any other choice a real implementation might make. -/
def Bracket.step {α : Type _} [LE α] {P : α → Prop} [DecidablePred P]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (B : Bracket P) : Bracket P :=
  let m := mid B.lo B.hi
  have hm := hmid B.lo B.hi B.le
  if h : P m then
    { lo := m, hi := B.hi, lo_ok := h, hi_bad := B.hi_bad, le := hm.2 }
  else
    { lo := B.lo, hi := m, lo_ok := B.lo_ok, hi_bad := h, le := hm.1 }

/-- Iterating the step any number of times. -/
def bisect {α : Type _} [LE α] {P : α → Prop} [DecidablePred P]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b) :
    Nat → Bracket P → Bracket P
  | 0,     B => B
  | n + 1, B => bisect mid hmid n (B.step mid hmid)

/-- **The safety property.** After any number of iterations the returned lower endpoint is
    feasible. Not "converges to"; *is*. The filter may be cut off at any point -- by a
    deadline, an interrupt, a budget -- and what it returns is still admissible. -/
theorem bisect_sound {α : Type _} [LE α] {P : α → Prop} [DecidablePred P]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (n : Nat) (B : Bracket P) : P (bisect mid hmid n B).lo :=
  (bisect mid hmid n B).lo_ok

/-- The bracket never inverts, so the reported interval is never empty by accident. -/
theorem bisect_le {α : Type _} [LE α] {P : α → Prop} [DecidablePred P]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (n : Nat) (B : Bracket P) : (bisect mid hmid n B).lo ≤ (bisect mid hmid n B).hi :=
  (bisect mid hmid n B).le

/-- The upper endpoint stays infeasible, so the bisection never certifies past the true edge. -/
theorem bisect_hi_bad {α : Type _} [LE α] {P : α → Prop} [DecidablePred P]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (n : Nat) (B : Bracket P) : ¬ P (bisect mid hmid n B).hi :=
  (bisect mid hmid n B).hi_bad

/-! ### 2b. Safety needs no monotonicity at all

The theorems above are stated for cap and floor families, but look at what `bisect_sound`
actually used: nothing. `Bracket` carries `P lo` as a field and `Bracket.step` preserves it by
case analysis on a decision, never once appealing to `IsCap`, `IsFloor`, or any property of
`P`. So the safety guarantee holds for an **arbitrary** predicate.

That is a real weakening of the method's hypotheses and it is worth stating separately, because
(A2) -- monotonicity -- is the assumption most likely to fail on a system nobody has checked. If
it fails, the admissible set may be a union of intervals and the bisection will find only the
first one. It will therefore *under-command*: it gives up performance. What it will not do is
return an infeasible input.

Monotonicity buys **optimality**, not safety. The distinction matters for anyone applying this
to a plant whose constraint structure they have not verified: they can lose charge, and they
cannot lose the guarantee. -/

/-- **Safety without monotonicity.** For any predicate whatsoever -- monotone or not, connected
    or not -- every reachable bisection state has a feasible lower endpoint. This is
    `bisect_sound` restated to make explicit that its proof never used the cap/floor structure. -/
theorem bisect_sound_any_predicate {α : Type _} [LE α] (P : α → Prop) [DecidablePred P]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (n : Nat) (B : Bracket P) : P (bisect mid hmid n B).lo :=
  (bisect mid hmid n B).lo_ok

/-- And the returned point is feasible whatever the midpoint rule computes, so the guarantee
    survives a midpoint that is rounded, truncated, or simply wrong -- provided only that it
    lands between the endpoints. -/
theorem bisect_sound_any_midpoint {α : Type _} [LE α] (P : α → Prop) [DecidablePred P]
    (mid mid' : α → α → α)
    (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (hmid' : ∀ a b : α, a ≤ b → a ≤ mid' a b ∧ mid' a b ≤ b)
    (n : Nat) (B : Bracket P) :
    P (bisect mid hmid n B).lo ∧ P (bisect mid' hmid' n B).lo :=
  ⟨(bisect mid hmid n B).lo_ok, (bisect mid' hmid' n B).lo_ok⟩

/-! ### 3. Putting them together -/

/-- **The filter is sound.** If the caps are caps, the floors are floors, and the bisection is
    started from a bracket whose lower endpoint is admissible, then whatever it returns after
    however many iterations satisfies every constraint -- and so does everything between it and
    the anchor. This is the statement `zeroguard/anchored.py` implements. -/
theorem filter_sound {α : Type _} [LE α] {ι κ : Type _}
    (cap : ι → α → Prop) (flr : κ → α → Prop)
    (hcap : ∀ i, IsCap (cap i)) (hflr : ∀ j, IsFloor (flr j))
    [DecidablePred (Admissible cap flr)]
    (mid : α → α → α) (hmid : ∀ a b : α, a ≤ b → a ≤ mid a b ∧ mid a b ≤ b)
    (n : Nat) (B : Bracket (Admissible cap flr))
    (anchor : α) (hanchor : Admissible cap flr anchor) :
    ∀ u, anchor ≤ u → u ≤ (bisect mid hmid n B).lo → Admissible cap flr u := by
  intro u h1 h2
  exact between_admissible cap flr hcap hflr hanchor
    (bisect_sound mid hmid n B) h1 h2

/-! ### 4. Axiom audit

Lean records exactly which axioms each proof depends on. `#print axioms` below is the check
that these are proofs rather than assertions: anything resting on `sorryAx` would say so here.
-/

#print axioms admissible_ordConvex
#print axioms admissible_isCap_of_no_floors
#print axioms between_admissible
#print axioms bisect_sound
#print axioms bisect_le
#print axioms bisect_hi_bad
#print axioms bisect_sound_any_predicate
#print axioms bisect_sound_any_midpoint
#print axioms filter_sound

end AnchoredCollapse

