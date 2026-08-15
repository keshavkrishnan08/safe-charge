# ZEROGUARD
## Monotone Control Theory for Certified Battery Fast Charging in Long-Duration Autonomous Vehicles

---

### Abstract

A robotaxi charges itself several thousand times over its working life and no human inspects
it once. A spacecraft past Mars cannot wait forty minutes for a command, and no technician
will ever recalibrate its battery. Both must hold hard limits on cell temperature and voltage
at every instant, while the cell ages underneath them, and within a time budget that can be
proven in advance rather than hoped for. Existing methods fail one of those three: they run
an optimizer whose iteration count depends on the data, or they lean on a state-of-health
estimate that nobody in the field can obtain.

This work removes both requirements and then shows that neither the removal nor the argument
behind it is about batteries. Zero current is a universally available safe action for a
charging cell, and every constrained signal is monotone in the applied current; those two
facts together collapse a predictive safety filter from a quadratic program to a bisection
over a scalar, resolved in a fixed eighteen iterations, 56 microseconds and 904 bytes, with
no solver and no diagnosis. We state that collapse as a theorem about a class of systems
rather than about a cell, and then test it by running the *same unmodified filter* on a
resistive heater, a DC motor winding and a semiconductor junction: 20.4 million certified
one-step transitions across four unrelated physics, zero violations. We locate the theorem's
boundary by breaking each hypothesis deliberately, and find the two failures are not alike —
losing the safe null input costs safety outright, while losing monotonicity costs only
delivered charge. We show the certificate is indifferent to the law by which the cell sheds
heat, surviving the replacement of Newtonian convection by Stefan–Boltzmann radiation
unchanged, which is what makes the vacuum case tractable. We run 31,000 charge cycles of
satellite and robotaxi duty with the filter never recalibrated, and find it exactly as safe
as an oracle handed the true resistance every cycle, which buys 8.8 and 16.2 points of
delivered charge and nothing else. We prove and verify that a series pack's certificate is
its weakest cell's, exactly, from one cell to 1024, and show that per-cell enforcement
catches a single faulted cell in a hundred that pack-averaged enforcement misses in every
one of 1,200 packs. Finally we predict, from published constants alone, the exact quantity
of sensor error a margin can absorb — 12.802 °C against a measured 12.85 °C — which turns a
safety margin from a tuning parameter into a quantity with a derivation.

---

## 1. The problem

Fast charging a lithium-ion cell is a race against two irreversible failures. Push current
too hard and the cell overheats toward thermal runaway; push it too hard while cold and
lithium plates onto the graphite anode instead of intercalating into it, permanently. Both
are governed by hard limits — temperature below 45 °C, terminal voltage below 4.2 V — and
"hard" is the operative word. A charger that treats them as costs to be traded against
charging speed is not a charger anyone should ship.

That much is standard. What is not standard is the setting this work is aimed at, which we
will call **unattended autonomy**: systems that must charge safely where nobody is watching,
nobody can diagnose, and nobody can intervene in time.

The population is larger than it first appears. A robotaxi sits at the near end — thousands
of unsupervised charge cycles, no driver, no daily inspection. A satellite in low Earth orbit
sits further along: fifteen eclipse cycles a day for five years is roughly 27,000 charges,
with no possibility of servicing. A probe beyond Mars sits at the far end, where the
round-trip light delay makes ground control not merely inconvenient but arithmetically
impossible. These are not three applications. They are three points on one axis, and what
they share is that the charger is alone.

Two properties become non-negotiable in that setting, and current methods supply at most one
of them at a time.

**The run time must be bounded in advance.** Automotive functional safety at the highest
integrity level requires a provable worst-case execution time. An embedded quadratic program
cannot supply one, because its iteration count depends on the data it is given — which is
precisely the failure we demonstrated in the antecedent work by capping a warm-started OSQP
at 25 iterations and watching it stop converging at the cold, aged corner.

**The cell's degraded state must not need to be known.** Every established safe-charging
method anchors to an estimate of state of health. But estimating state of health is exactly
the thing that is impossible on an unattended cell: it requires instrumentation, or a
characterization cycle, or a technician. The requirement is circular, and the circle has to
be broken rather than tightened.

## 2. The idea

The observation the whole of this work rests on is embarrassingly small.

> A battery being charged has one action that is always available and always safe: stop
> charging.

Set the current to zero and every input-driven heat source vanishes at once, so the
temperature can only relax toward ambient; the terminal voltage falls back to the
open-circuit value; nothing gets worse. That is a *backup* in the technical sense — an action
that returns any safe state to the safe set — and its existence is what a predictive safety
filter normally has to search for.

Add the second observation, which is nearly as small: every constrained signal is monotone in
the applied current. More current means more ohmic and activation heating, hence a higher
one-step temperature; more current means more overpotential, hence a higher terminal voltage.

Put the two together and the geometry of the problem collapses. If the constraints only
increase in the current, then the set of currents that keep the next state safe is not some
awkward region requiring optimization — it is a single interval, and because zero is always
in it, that interval is anchored at zero. Finding its right-hand edge is a one-dimensional
root-find, which a bisection resolves in a fixed number of steps. No optimizer. No convergence
test. No parameters to identify.

The aging problem dissolves by the same monotonicity. A cell's internal resistance grows as it
degrades, and higher resistance means more heat for the same current — monotonically. So a
*conservative upper bound* on resistance certifies safety for every cell at or below that
bound, without ever measuring which one you have. The datasheet's rated end-of-life value is
such a bound and is known before deployment. An online estimator, if you have one, only
tightens the bound and buys back charging speed; it is never load-bearing for safety. A cell
that has aged *past* the rated bound has left its rated range and is one a battery management
system should be retiring, not fast-charging.

## 3. The theorem

Stating the collapse carefully makes plain that it has nothing to do with electrochemistry.

> **Theorem (Null-Input Collapse).** Let a system have state *x*, a scalar non-negative input
> *u*, and constraints *g<sub>c</sub>(x, u) ≤ 0*. Suppose
>
> **(A1)** *u = 0* is safe — from every state in the safe set, the one-step map at *u = 0*
> returns to the safe set; and
> **(A2)** each *g<sub>c</sub>* is non-decreasing in *u*.
>
> Then the admissible input set is a single interval *[0, u\*]* anchored at zero, and *u\** is
> found by bisection in *O(log 1/ε)* one-step evaluations — with no optimizer, no convergence
> test, and no identification of the plant's uncertain parameters.

The battery is one instance of that theorem. If the theorem is right, it is not a special one.

## 4. Results

### 4.1 The principle is not about batteries

We wrote the projection once, against the abstract interface the theorem describes, and gave
it four systems with nothing physical in common: the lithium-ion cell; a resistive heating
element under a temperature limit, with resistance rising as it heats; a DC motor's winding
and rotor under coupled thermal and speed limits; and an IGBT junction under a
junction-temperature limit through a two-node thermal network. Each was driven by a greedy
policy demanding maximum input, from thousands of randomized initial states, for a full
episode.

**8,000 episodes. 20,400,000 one-step transitions. Zero violations.** The Clopper–Pearson 95 %
upper bound on the violation rate is 0.037 %. On a dense scan of the input axis at 1,600
sampled states, the admissible set was a single interval anchored at zero every time.

One check makes "the same filter" a statement of fact rather than a figure of speech. Across
60,000 states — three cell scales, both discretizations, with and without the cooling reserve
— the generic projection and the published `project_current` returned **bit-identical**
results. Not close: identical, to the last bit of the mantissa.

Four unrelated physics, one piece of code, no violations. The theorem is not about batteries.

*(Figure F1.)*

### 4.2 Where it stops being true

A characterization is only worth the name if it has a boundary, so we broke each hypothesis
deliberately and separately. The two failures are not the same kind of failure, and the
asymmetry is the most useful thing in this section.

**Break (A1)** with a quadrotor holding altitude, where cutting the input does not make the
system safe but makes it fall. Every single one-step check passed — for a mean of 22.3 steps —
and then **100 % of 2,000 episodes violated**, at almost exactly the step where no input could
recover any longer (22.4). One-step feasibility never implied forward invariance; it only
looked like it did, because in the battery there was always something to fall back on. The
same lazy policy on the battery, where (A1) holds, produced zero violations in 2,000 episodes.

**Break (A2)** with a plant whose constrained signal crosses its limit repeatedly, so the
admissible set is a union of intervals — 3.57 bands on average. The bisection cannot see past
the first gap, so it cannot find the global optimum. But it never returned an infeasible
input, not once in 3,000 states: it maintains `lo ≤ u*` on the branch containing zero, so its
error is always to under-command. The cost is a mean optimality gap of 23.7 % (95 % CI 22.6 to
24.8), reaching 66 % at worst.

So **(A1) is a hypothesis about safety and (A2) is a hypothesis about performance.** Losing
the safe null input loses the certificate; losing monotonicity loses only charge. That is
worth knowing before deploying this anywhere, and it is why the battery result is specifically
a *charging* result — on discharge under load, zero current is not a backup either.

*(Figure F2.)*

### 4.3 The certificate does not depend on how the heat leaves

On Earth a cell sheds heat by convection, proportional to the temperature difference with the
air. In vacuum there is no air, and the only path out is radiation, proportional to the
difference of fourth powers against a 4 K sky. These are different physics: one linear, one
quartic; one needs a medium, one does not; one has an ambient temperature, the other has none.

Proposition 2 never asked for either. It asked for monotonicity, and *T⁴* is strictly
increasing in *T*. So if the argument was really about structure rather than about air, the
certificate should survive the substitution untouched.

It does. Across 11,520 sampled states, all three partial derivatives the proposition requires
remained non-negative under the quartic law, with no violations in any direction. Over a
five-channel uncertainty envelope in vacuum — resistance, emissivity loss, thermal mass,
plating, and sink temperature — **20,000 draws produced zero violations**, a Clopper–Pearson
upper bound of 0.015 %, and a worst observed temperature of 32.49 °C against the 45 °C limit,
with the monotone corner dominating the interior exactly as the theory says it must.

Adding a sixth channel changes nothing. Total ionising dose is the degradation mechanism
peculiar to space — it raises series resistance and takes capacity — and it enters the
argument exactly as terrestrial ageing does, monotonically. Swept jointly with the other five
over 20,000 draws it produced **zero violations**, a Clopper–Pearson upper bound of
0.0150 %, a worst temperature of 32.49 °C, and the corner
still dominating the interior. A sixth uncertain quantity costs the certificate nothing, which
is the practical content of dimension-independence.

Then we took the radiator away entirely, decade by decade, to nothing. Safety held at every
point, and delivered charge fell monotonically from 0.607 to 0.168 (Spearman ρ = 1.000).
The filter does not fail as its cooling disappears; it throttles. That is what graceful
degradation means, and it is the behaviour you want in a system nobody can reach.

*(Figure F3.)*

### 4.4 A whole mission, with nobody to recalibrate

The title of this work promises long duration, so this is the experiment that pays for it. We
simulated two duty cycles end to end with the cell ageing continuously underneath: 27,000
shallow eclipse cycles at satellite cadence, and 4,000 deep charges at robotaxi cadence, each
ending at exactly the rated end of life. The filter was handed the datasheet bound on day one
and told nothing ever again. Against it ran an oracle given the cell's true resistance every
single cycle — the best any diagnostic could possibly do — with a margin that tightens as its
assumption tightens, so the comparison is not rigged.

| | cycles | violations | CP95 upper | peak T | delivered SOC |
|---|---|---|---|---|---|
| satellite, bound only | 27,000 | **0** | 0.011 % | 35.00 °C | 0.610 |
| satellite, oracle | 27,000 | **0** | 0.011 % | 39.90 °C | 0.698 |
| robotaxi, bound only | 4,000 | **0** | 0.075 % | 32.07 °C | 0.415 |
| robotaxi, oracle | 4,000 | **0** | 0.075 % | 41.40 °C | 0.577 |

The oracle buys 8.8 and 16.2 points of delivered charge, with a rank-biserial correlation of
+1.000 and a permutation *p* indistinguishable from zero — it wins on essentially every single
cycle. And it buys nothing else. **The safety columns are identical: zero and zero.**

That is the thesis of this work in one table. The estimator is a performance component, not a
safety component, and can be left out of the trusted path entirely.

*(Figure F5.)*

### 4.5 From one cell to a pack, and why 904 bytes is an architecture

A series pack shares one current across every cell, so its admissible set is the intersection
of the cells' admissible sets. Each of those is an interval anchored at zero, so the
intersection is too, and its right edge is the minimum:

> **Lemma.** *I\*<sub>pack</sub> = min<sub>i</sub> I\*<sub>i</sub>.*

Resolved with 40 halvings so the test measures the lemma rather than the bisection's
coarseness, this held at every N from 1 to 1024 heterogeneous cells, with a worst
disagreement of **6.6 × 10⁻¹² A** — below the 1.4 × 10⁻¹¹ A resolution of the bisection used
to test it, and exactly zero at nine of the eleven sizes. The lemma holds to the limit of what
the measurement can resolve.

The lemma says something practical. The pack's certificate *is* the weakest cell's, so a
filter running independently on every cell loses nothing. And a filter running once on an
averaged cell can be wrong about the only cell that matters.

We tested that directly: 1,200 packs of 100 cells, one cell in each given an anomalous
resistance, both architectures given identical margins and identical faults.

- **A filter on every cell: 0 of 1,200 packs violated.** Worst cell 43.04 °C.
- **One filter on the pack average: 1,200 of 1,200 packs violated.** Worst cell 48.54 °C.

The median peak-temperature difference is 3.24 °C (Hodges–Lehmann, *p* ≈ 0). The averaged
filter is not slightly worse; it fails every time, because the fault it needs to see is
exactly the one averaging destroys.

This is why the footprint is not a performance detail. Battery safety today is enforced once
per pack because certified enforcement is too expensive to replicate. At 904 bytes and 56
microseconds it no longer is — enforcement can live on every cell, where thermal runaway
actually begins.

*(Figure F6.)*

### 4.6 Trying to break it on purpose

A safety argument tested only against well-behaved policies has not been tested.

We replaced the greedy governor with an adversary: a cross-entropy optimizer with full
knowledge of the episode, free to hold current back early so it could push harder later, with
the explicit objective of maximizing peak temperature.

The first version of this experiment searched 72,000 sequences across 40 aged cells and found
nothing — but 40 cells is not enough cells. The statistical unit of a safety claim is the
independent draw, and forty of them bound the violation rate only below 7.2 %, which certifies
nothing anyone should rely on. The adequacy audit in Section 4.10 caught it, so the attack was
re-run at the breadth the claim needs: **320 independently drawn cells,
256,000 candidate sequences.**

**Zero violations.** The worst peak temperature the adversary reached anywhere was
37.991 °C against a 45 °C limit — 7.01 °C of headroom it
could not close — and it beat the naive greedy policy by an average of
0.0690 °C (95 % CI 0.0578 to 0.0805).
The Clopper–Pearson upper bound falls from 7.2 % to 0.9318 %, which does
certify below 1 %. There was nothing to find, because the guarantee is per-step and does not
care what sequence produced the request.

Scheduler jitter of ±30 % on the control interval, using the unconditionally stable
zero-order-hold discretization: zero violations in 1,500 episodes at every level. Single
precision arithmetic: zero violations in 4,000 episodes, with a worst current deviation of
2.3 × 10⁻⁴ A.

Corrupted sensing is the one that should not come out clean, and we designed it so that it
could not. A filter reads sensors; if the sensors lie about temperature in the optimistic
direction, no amount of monotone structure will save it. Zero-mean noise was survivable to a
standard deviation of 4 °C and produced a single violation in 600 at 8 °C. Quantization was
survivable across the whole range tested. Dropout was survivable to a 30 % loss rate. And
optimistic bias broke it, exactly as it must — which turns out to be the most informative
result in this work.

*(Figure F9.)*

### 4.7 What a margin is actually worth

The filter tolerated a sensor reading 12.80 °C low and failed at 12.85 °C. The breakpoint is
sharp: the worst peak temperature rises linearly with the bias and crosses the 45 °C limit
between those two levels, with 500 episodes at every 0.05 °C step.

That number is not arbitrary, and it is not fitted. Write *α = Δt·hA/C<sub>th</sub>* for the
fraction of a step's temperature gap that convection removes. A sensor reading *b* degrees low
corrupts the filter's estimate twice — directly, and through the cooling term, which it also
under-estimates:

*T*<sub>pred</sub> = (*T*−*b*) + (Δt/C<sub>th</sub>)[*Q*<sub>gen</sub> − *hA*((*T*−*b*) − *T*<sub>amb</sub>)]

so that *T*<sub>true</sub> = *T*<sub>pred</sub> + *b*(1 − *α*). At high bias the cooling reserve
is identically zero, because the filter believes the cell is below ambient; the filter
therefore enforces exactly *T*<sub>pred</sub> ≤ *T*<sub>max</sub> − *δ*<sub>T0</sub>, and safety
survives while

> ***b\* = δ<sub>T0</sub> / (1 − α)***

Every term is a published constant. With *δ*<sub>T0</sub> = 12.5 °C and *α* = 0.02358, this
gives **12.802 °C**. The measured bracket is 12.80 to 12.85 °C. The prediction is right to
**0.048 °C**, and nothing in it was tuned.

This is worth more than another zero-violation table. It converts a safety margin from a
number someone chose into a number with a derivation, and it says exactly what that margin
purchases: 12.8 °C of sensor dishonesty, no more and no less. A designer who knows their
sensor's worst-case bias can now compute the margin they need rather than guessing it.

*(Figure F8.)*

### 4.8 What every part of the method is actually for

We removed each mechanism in turn, on a fixed grid of 1,400 aged, cooling-faulted, hot cells
so the comparisons are paired. Two results were the opposite of what we expected, and both
were more useful than confirmation would have been.

| removed | violations (of 1,400) | verdict |
|---|---|---|
| nothing (full method) | 0 | — |
| the conservative bound | **29** | safety |
| bound → point estimate | **7** | safety |
| the *I* = 0 short-circuit | 0 | see below |
| the cooling reserve | 0 | charge only |
| the plating cap | 0 | charge only |
| the resistance throttle *K* | 0 | see below |
| fixed 18 iterations → tolerance exit | 0 | boundedness |

The **conservative bound** is doing the safety work, and replacing it with a realistic noisy
point estimate — unbiased, 12 % standard deviation — reintroduces violations. That is the
central claim of the antecedent work, ablated and confirmed.

The **I = 0 short-circuit** turned out not to be a safety mechanism at all. We tested it on
4,000 states that were already outside the safe set on arrival, expecting the bisection to
return a current it had no grounds to return. It does not: with every input infeasible, every
midpoint tested fails, so the upper bracket collapses onto the lower one and the answer is
zero either way. The answers were **identical in 100 % of states**. What the check actually
buys is 17 model evaluations per call. It is a performance short-circuit, and we had been
describing it wrongly.

The **resistance throttle K** looked, on that grid, like pure conservatism: removing it cost
zero violations and recovered 43 points of delivered charge. But the grid contained only cells
*inside* the rated bound, where the bound alone suffices — so it could not judge a mechanism
whose whole purpose is insurance against cells *past* the bound. Re-run on cells with
resistance scales up to 3.0, the throttle is not conservative at all:

| K | 0 | 2 | 4 | 6 | 8 | 10 | 12 | **15** | 20 |
|---|---|---|---|---|---|---|---|---|---|
| violations / 1,200 | 529 | 400 | 311 | 216 | 115 | 42 | 5 | **0** | 0 |

**K = 15 is the smallest value that certifies this population — exactly the deployed value.**
The original choice was tight, not cautious, and we only learned that by testing it against
the population it was designed for rather than the one that was convenient.

*(Figure F7.)*

### 4.9 The certified region

Every result above reports safety at a point in design space. Mapping it takes two full
two-dimensional grids, each with an independent Monte-Carlo population at every node — 14,040
and 19,890 episodes respectively — reporting the Clopper–Pearson upper bound, which is the
only quantity that still carries information when the observed count is zero.

The throttle-versus-resistance surface shows a clean cliff: certification extends well past
the rated bound at the deployed *K*, and the deployed operating point sits comfortably inside
the certified region rather than on its edge. The bias-versus-cooling-loss surface shows
something we did not expect and would not have guessed: the failure cliff sits at the same
sensor bias regardless of cooling loss. The two faults do not interact. They are separate
channels of the same monotone argument, and the certificate treats them separately because
they are.

*(Figure F4.)*

### 4.10 What it costs, and whether the evidence is enough

Two housekeeping questions that are easy to skip and shouldn't be.

**Cost.** Separating the projection by code path shows what the fixed iteration count buys.
When the request is already admissible the filter answers in a single model evaluation
(4.1 µs); when zero is infeasible it answers in two
(6.8 µs); and the full bisection — the path the worst-case
bound is written against — takes 56.0 µs across
801 states, with a spread of barely three microseconds between its
fastest and slowest run. That is the point of a fixed iteration count: there is no tail to
characterise. Against a 30-second control step it is a duty cycle of
1.9e-06, or 5.7 orders of magnitude of headroom, in
904 bytes.

**Adequacy.** A zero-failure result is only as strong as the number of trials behind it, and
the honest way to report one is beside the number of trials it would have taken. Certifying a
violation rate below 1 % at 95 % confidence needs 299 trials with no
failures; below 0.1 % needs 2,995; below 0.01 % needs
29,956.

Auditing every headline claim against that requirement is what turned up the one weak result
in this work. The cross-domain, vacuum, mission-life, float32 and ablation claims all clear
0.1 %. The pack and full-method claims clear 1 %. And the adversarial claim, on its original
40 cells, cleared **neither** — a Clopper–Pearson bound of 7.2 %, which certifies nothing
worth having. Seventy-two thousand searched sequences is a great deal of searching, but the
statistical unit of that claim is the cell, not the sequence, and forty is not many cells.
Section 4.6 reports the re-run that fixes it.

*(Figure F10.)*

## 5. Limitations

Stated plainly, and stated in advance of the results rather than after them.

**The plant is a model.** The Doyle–Fuller–Newman validation that supports the antecedent
work is not repeated here; every number in this report is reduced-order-model level or
model-versus-model. That is a real limit on what "certified" means, and no amount of Monte
Carlo repairs it. Hardware validation is the next step, and it is the step that would convert
a model-relative certificate into a rated one.

**Plating is enforced, not certified.** The certified set is exactly {*T* ≤ 45 °C, *V* ≤ 4.2 V}.
Lithium plating is history-dependent, and the *I* = 0 backup does not clear its margin above
roughly 0.82 state of charge — we measured that boundary rather than assuming it. Plating is
held by a temperature-dependent current cap and monitored; calling it certified would be
false.

**The certificate is conditional on measurement.** Section 4.7 is not a caveat we discovered;
it is a property we quantified. A filter that reads a lying sensor will act on the lie. What
this work provides is the exact exchange rate.

**(A2) failure is graceful; (A1) failure is not.** Anyone applying the theorem elsewhere must
check the null input first. Section 4.2 exists so that the check is a measurement rather than
an assumption.

## 6. What this enables

Three things follow that did not before, and each traces to a specific measured property
rather than to enthusiasm.

**Safety enforcement can move from the pack to the cell.** Today one battery management system
supervises thousands of cells through averaged measurements, and thermal runaway begins in
one. Section 4.5 shows averaging destroys exactly the signal that matters, and 904 bytes is
small enough to replicate on every cell.

**Cells whose history is unknown become chargeable.** Electric-vehicle packs retire at around
80 % state of health with most of their capacity intact, and stay unused because certifying a
used cell costs more than the cell is worth. A method that needs only the datasheet bound
removes the characterization step rather than accelerating it.

**A learning charger becomes shippable.** The filter carries safety for any input whatsoever —
72,000 adversarially optimized sequences could not find an exception — so a policy can go on
adapting in the field for the life of the vehicle without ever being able to become unsafe.

## 7. Reproducibility

Every number in this report is produced by a script in `zeroguard/exp/`, writes its result to
`zeroguard/results/`, and is redrawn from that file by `zeroguard/figures.py`. Seeds are
fixed. The experimental design, including the falsification criteria quoted in Section 5, was
written to `zeroguard/DESIGN.md` before any of it ran.

The whole program is **650,970** simulated charge episodes and sampled states, and well over
20 million certified one-step transitions. `zeroguard/verify_claims.py` re-reads all **100**
quantitative claims in this report out of the result files and fails if any has drifted; it is
the reason the two late corrections in Sections 4.6 and 4.10 could be made without wondering
what else they broke. The safety-critical path all of it exercises remains 36 lines of numpy.

The design register in `DESIGN.md` records where the delivered experiments diverged from the
plan, including the three gaps the audit found and the fourth that the adequacy check found in
turn. Nothing was quietly dropped.
