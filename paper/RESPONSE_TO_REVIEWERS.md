# Response to Reviewers — IECON26-001272

**Safe Fast Charging Robust to Battery Aging: A Solver-Free Safety Filter Without a Health Oracle**

We thank both reviewers for the careful reading and the constructive recommendations. Every point
raised has been addressed. Section and figure numbers below refer to the revised manuscript, which
remains within the page limit. All new quantities are produced by scripts in the public repository
(<https://github.com/keshavkrishnan08/safe-charge>), so each number below can be reproduced.

No previously reported result has changed. One supporting figure in a proof is corrected, and is
declared explicitly in the final section of this letter.

---

## Reviewer 1

> **R1.1 — The authors should add numerical findings in the abstract section.**

Added. The abstract now carries the headline quantities: the peak DFN-replayed temperature
(40.4 °C against the 45 °C limit), the delivered SOC matching CC–CV at 1C (0.773) while removing
that protocol's 22% plating rate, the aged-draw safety record (156/156, Clopper–Pearson 1.9%) across
SOH 1.0–0.8, the cross-chemistry recalibration (100/100, voltage RMSE 26.5 mV), and the per-projection
cost (at most 20 model evaluations, 904 B). The original wording is otherwise unchanged; the numbers
were inserted into the existing sentences rather than replacing them.

> **R1.2 — The authors should add a nomenclature in order to clarify the symbols.**

Added a Nomenclature section before the Introduction, covering all 13 symbol groups used in the
paper: the constrained signals and their limits, the plating overpotential, the current variants,
the certified set and its margin-tightened inner approximation, the one-step constraint peaks, the
three control margins, both aging channels and their estimated and rated values, the two margin
gains, the control step and RC time constant, the heat-generation decomposition, the lumped thermal
parameters, the resistance terms, and the bisection resolution.

> **R1.3 — The quality of figures 1, 4, and 5 is poor.**

All three are rebuilt or re-typeset:

- **Fig. 1** was enlarged from one column to 0.78 of the text width. It was already native vector
  TikZ, so the issue was scale rather than resolution: at column width its node labels were set well
  below body-text size, and they are now legible at the printed size.
- **Figs. 4 and 5** were previously placed in a single column, which scaled them to 0.49× and
  reduced some labels below 2 pt. Both are now full-text-width floats at 1.0×, roughly doubling
  every label. In Fig. 5 all remaining sub-6 pt text was raised so that the figure's minimum type
  size is now 6.6 pt.
- **Fig. 4, panel (a)**: the legend previously occluded the rising "no adaptation" curve and
  overhung the axes frame. It has been repositioned inside the axes and now clears the data.

> **R1.4 — The authors should add future work in the conclusion section.**

The conclusion now orders the next steps and says what each one buys: hardware validation (which
converts a ROM-relative certificate into a rated one), a predictive plating constraint (which would
move plating from enforced to certified), an age-aware current cap, and a pack-level extension.

> **R1.5 — What is the computational complexity of the proposed approach?**

Stated explicitly in Sec. VI-E. The projection costs `O(log(1/ε))` one-step model evaluations for
current resolution `ε`. Because the iteration count is fixed rather than convergence-tested, that
bound is met on *every* input: at most 20 closed-form evaluations, ~800 flops, and 904 B of state and
constants with no dynamic allocation. See also R2.1 for measured timings.

> **R1.6 — The authors should explain in detail the robustness of the proposed approach, because a
> filter that operates without an SoH oracle may pose problems in terms of resilience to the
> uncertainties of battery aging.**

This is the central concern and Sec. VI-C now addresses it directly. The argument is that oracle-freedom
is *safer*, not riskier, because the filter is anchored to a conservative bound rather than to an
estimate:

- Safety is monotone in both aging channels (Prop. 2), so a bound on each certifies the whole
  envelope without identifying anything. A cell aged *past* the rated bound has left its rated range
  and is one a BMS should retire; a higher assumed scale only throttles harder and delivers less
  charge, so the estimator's error costs SOC, never safety.
- We quantify the margin beyond the bound: the first violation appears only when the true resistance
  scale reaches 3.0, i.e. 1.7× the rated end-of-life bound the filter is given.
- The certificate is validated over a 5-channel uncertainty envelope (resistance, cooling, thermal
  mass, plating, ambient) at 50 000 draws with zero violations, a Clopper–Pearson 95% upper bound of
  0.006%, and the monotone corner empirically dominating the interior.
- The gains are shown to be a region, not a tuned point: sweeping `K ∈ [5,25]` and `f ∈ [0.1,0.4]`
  gives zero violations in 27 027 runs against aged, cooling-faulted cells.

> **R1.7 — The authors should mention how the constraints are chosen and what the limitations of the
> proposed approach are.**

Both are now explicit. On provenance (Sec. IV-C): `T_max`, `V_max` and the rated end-of-life scale
`s_R*` are manufacturer limits and the datasheet aging envelope; the cooling reserve `f` is *derived*
from the cooling fault to be covered (reserving `f` of the temperature rise certifies a loss of
`f/(1+f)`, so `f = 0.25` is fixed by the 20% fault claimed, not fitted); only the resistance throttle
`K` is chosen by simulation, and it is set by delivered charge, which it costs monotonically, rather
than by safety. On limitations (Sec. VII): the certificate is ROM-relative and on volume-averaged
temperature, the evidence is simulation over an aged-DFN distribution rather than a hardware rating,
and plating is enforced and monitored rather than certified.

---

## Reviewer 2

We are grateful for the detailed and accurate summary of the contribution, and for the recommendation
to accept.

> **R2.1 — Report the average computation time, worst case computation time, and memory requirements
> of the one-dimensional bisection on a representative embedded processor.**

Added to Sec. VI-E, backed by a new benchmark (`bench/timing.py`). The key point is that average and
worst case *coincide*: the bisection runs a fixed iteration count with no convergence test, so every
call performs identical work and the spread across calls is host scheduling noise rather than
algorithmic variance. We report 56 µs per projection in the reference implementation, 904 B of state
and constants, and no dynamic allocation, against a 30 s control step.

We deliberately report the hardware-independent unit of work as the number that ports — at most 20
one-step model evaluations, ~800 flops plus 40 transcendentals — and give the embedded figure as an
explicit order-of-magnitude estimate (of order 100 µs on a 100 MHz Cortex-M4F) rather than presenting
an unmeasured board number as data. We would rather understate than report a device measurement we
did not take.

> **R2.2 — The influence of the bisection tolerance on the resulting safety margin and charging
> performance should also be briefly discussed.**

Added to Sec. VI-E, with a supporting sweep (`reproduce/tolerance_sweep.py`). The tolerance after `k`
halvings is `I_max / 2^k`, i.e. 38 µA at the deployed `k = 18`. The important structural point is
that **safety is independent of `k`**: the bisection maintains `lo ≤ I*` throughout, so a coarser
tolerance can only under-command the current, never over-command it. Tolerance therefore trades only
delivered charge, and that saturates by `k ≈ 12`; the deployed 18 iterations sit well past the knee.

> **R2.3 — Explain the engineering basis for selecting a 30 s control interval, and discuss whether
> the safety result remains valid when the sampling interval is shorter or longer than the RC time
> constant.**

Added to Sec. IV-A, with a sweep (`reproduce/step_size.py`). The step is pinned by two *opposing*
requirements rather than tuned: the residual argument in Prop. 1 wants `Δt ≳ τ_RC` so the RC residual
has decayed enough for the thermal margin to absorb it, while the explicit discretisation of the RC
branch is stable only for `Δt ≤ τ_RC`. The two meet at `Δt = τ_RC = 30 s`.

Safety is not fragile in this choice. Sweeping `Δt` from 5 s to 120 s leaves zero violations, moves
delivered SOC by under 0.1 points, and never lets the RC residual exceed 10% of the base thermal
margin. For steps longer than `τ_RC` the model must use an unconditionally stable discretisation, so
the released ROM now offers an exact zero-order-hold branch (`BatteryROM(rc="exact")`) alongside the
explicit one used for every published number; the explicit branch now raises rather than silently
oscillating if given `Δt > τ_RC`.

> **R2.4 — Clarify the sources and selection principles of the thermal margin parameters, so that
> readers can distinguish theoretically derived design parameters from parameters tuned through
> simulation.**

Addressed in Sec. IV-C; see R1.7 above for the full breakdown. In short: two constants are
manufacturer data, one is derived in closed form from the fault to be covered, and exactly one (`K`)
is chosen by simulation — and that one is selected on delivered charge, not on safety.

> **R2.5 — Further clarification of the scope of the safety certificate.**

The scope is now stated wherever the certificate is claimed. The certified set is exactly
`S = {T ≤ T_max, V ≤ V_max}`. Plating is *not* in it: plating is history-dependent, and the `I = 0`
backup does not clear the plating margin at the cold, aged, high-SOC corner, so no one-step condition
can certify it. Plating is enforced — by the one-step margin together with the temperature-dependent
current cap — and monitored, and we report it as enforced rather than proven. The released code
carries the same scoping, and the test suite reports the SOC above which the `I = 0` backup stops
clearing the plating margin rather than concealing it.

---

## Declared correction

One number in the proof of Prop. 2 has been corrected. The submitted version stated that the
endothermic reversible heat *"stays below 15% of the ohmic term."* That comparison does not hold in
our model: because `Q_ohm = I²R_Ω` vanishes quadratically in current while `Q_rev` vanishes linearly,
the ratio to the ohmic term reaches ~193% at 1C and grows as current falls. Measured against the
total irreversible heat over the admissible current band, the correct figure is 24%, and the revised
text says so.

The conclusion the sentence supports is unaffected, and is in fact established more directly in the
revision: we verified numerically that `∂_I g_T ≥ 0` holds for every current above 0.08 C, far below
the 0.70 C floor imposed by the plating cap, and that the admissible current set is a single interval
anchored at zero on the whole test grid — which is the property the bisection actually requires.

---

## Reproducibility

All reviewer-requested quantities are generated by scripts in the public repository and are
reproduced by a single command (`bash run_all.sh`): `bench/timing.py` (R2.1),
`reproduce/tolerance_sweep.py` (R2.2), `reproduce/step_size.py` (R2.3), and
`reproduce/robustness.py` (R1.6, R2.4). The safety propositions are executable checks in
`tests/test_safety.py`.
