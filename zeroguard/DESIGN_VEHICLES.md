# ZEROGUARD — Vehicle Design Register

**Anchored Collapse and the four media: ground, air, water, vacuum**

This is the second design register. Like `DESIGN.md` it is written before the experiments run,
it states what would falsify each claim, and it is audited afterwards against what actually
happened, with divergences recorded rather than quietly corrected.

The first register asked whether the IECON filter is about batteries and answered no. This one
asks a harder question that the answer exposed: **the theorem needs `u = 0` to be safe, and on
a vehicle that is doing its job, it usually is not.** E2 measured that failure precisely — a
hovering quadrotor passed 22.3 consecutive one-step checks and then violated in 100 % of
episodes — and stopped there. Stopping there was correct and it was also the end of the
vehicle story, because a certificate that only holds while the vehicle is parked is not a
vehicle certificate.

---

## 0. Scope and honesty boundaries

Carried unchanged from `DESIGN.md`:

- No Doyle–Fuller–Newman pipeline in this repository. Everything is ROM-level or
  ROM-versus-ROM, and the DFN evidence behind the IECON paper is cited, not re-derived.
- The certified set is `{T, V}` on charge. Plating is enforced and monitored, never certified.

Three new ones this register introduces, stated up front because they bound every number
below:

- **The platforms are class-representative, not vendor designs.** Pack geometry, hover power,
  hotel loads and orbit parameters are chosen to place each platform in the right region of
  the design space using published figures for the vehicle *class*. No claim here is a claim
  about any manufacturer's hardware.
- **One cell, sixteen vehicles.** Every platform uses the M50 coefficients from
  `rom_params.json`, because that is the cell the ROM was calibrated against and substituting
  uncalibrated coefficients would trade a real limitation for a fake result. Where a class
  would really use a different cell — the eVTOL, which needs a power cell — the actuator
  ceiling is raised to match and the mismatch is recorded. The M50's resistance is *higher*
  than a power cell's, so those certificates are conservative for the aircraft rather than
  tuned to it.
- **`u = 0` safety is checked per domain, never assumed.** The whole point is that it fails on
  three of the four media. Any platform whose anchor is nonzero is run through the anchored
  filter and its null-input counterpart side by side, so the difference is measured.

---

## 1. The claim

> **Theorem (Anchored Collapse).** Let a system have state `x`, scalar input `u ∈ [u_min,
> u_max]`, and constraints each satisfied on one side of a threshold in `u`: the *caps* on
> `[u_min, u_c]`, the *floors* on `[u_c, u_max]`. If there is a known anchor `u_a`, feasible
> from every state in the safe set, then the admissible set is the interval `[u_lo, u_hi] ∋
> u_a`, and both edges are found by bisection in `O(log 1/ε)` one-step evaluations — no
> optimizer, no convergence test, no identification.

Null-Input Collapse is the case `u_a = 0` with no floors. **Every vehicle has an irreducible
load** — hover thrust, hotel load, bus load, survival heater — and that load is the anchor.
Charging is the special case where it is zero, which is exactly why the IECON filter could
anchor at zero without ever saying so.

The side of a constraint is **declared, not inferred from its comparison sense.** The
published plating constraint is `phi ≥ margin`, a lower-bound test on a signal that *decreases*
in current, so it is a cap. Guessing from the sense would file it as a floor and let the filter
command straight through it. This is the one place in the method where a silent modelling error
would be unsafe rather than merely conservative, so it is data the plant supplies.

### What the generalisation buys, beyond scope

Two things the null-input certificate cannot express:

1. **A two-sided certificate.** Upper edge thermal and electrical, lower edge the load. Safety
   is the statement that they have not crossed.
2. **A reserve.** `u_hi − u_lo` shrinks to zero as the envelope closes, so the filter reports
   how close to unrecoverable the vehicle is, *before* it arrives. This is the claim with the
   most operational weight in the whole document and it gets the most adversarial treatment
   (V2-3, V3-7, X4).

---

## 2. Platform register

Sixteen platforms across four media. Charge-mode platforms have `u_a = 0` and test the IECON
result at vehicle scale; discharge-mode platforms have `u_a > 0` and test the generalisation.

| domain | case | pack | mode | anchor source |
|---|---|---|---|---|
| ground | `robotaxi-urban` | 96S45P, 78 kWh | charge | — |
| ground | `autonomous-shuttle` | 72S18P, 24 kWh | charge | — |
| ground | `haul-truck` | 288S115P, 601 kWh | charge | — |
| aerial | `delivery-quadrotor` | 6S1P, 0.11 kWh | discharge | hover thrust (momentum theory) |
| aerial | `evtol-air-taxi` | 300S30P, 163 kWh | discharge | 400 kW hover |
| aerial | `quadrotor-turnaround` | 6S1P | charge | — |
| underwater | `survey-auv` | 14S6P, 1.5 kWh | discharge | hotel + thrust, 115 W |
| underwater | `buoyancy-glider` | 10S4P, 0.73 kWh | discharge | 0.5 W hotel |
| underwater | `under-ice-auv` | 14S6P | discharge | 175 W, no abort-to-surface |
| underwater | `survey-auv-dock` | 14S6P | charge | — |
| space | `leo-smallsat` | 8S3P, 0.44 kWh | charge | — |
| space | `geo-comsat` | 100S8P, 14.5 kWh | discharge | 3 kW payload in eclipse |
| space | `deep-space-cruiser` | 24S16P, 7 kWh | charge | — |
| space | `mars-rover` | 8S12P, 1.7 kWh | charge | — |
| space | `lunar-night-lander` | 8S20P, 2.9 kWh | discharge | 45 W survival heater |
| space | `high-radiation-orbiter` | 16S24P, 6 kWh | charge | 300 krad TID |

---

## 3. Experiment register

Eleven experiments per domain, each attacking a different part, plus five cross-domain. Each
row states the claim, what would falsify it, and the decision rule. All seeds fixed.

### V1 — Ground: autonomous cars, robotaxis, shuttles, trucks

| ID | Claim | Falsifier | Decision rule |
|----|-------|-----------|---------------|
| **V1-1** | On charge the vehicle filter reduces exactly to the published `project_current` | any bit-level disagreement | bit-exact on ≥ 100 000 states |
| **V1-2** | A 350 kW session on a 78 kWh pack holds under a full parameter envelope | any violation | 0 / n, CP-95 upper reported |
| **V1-3** | The certificate holds across the ambient range a fleet actually sees (−20…+45 °C) | violation at any ambient | 0 at every point; delivered SOC monotone in ambient |
| **V1-4** | Deadline feasibility degrades continuously, never discontinuously | a cliff in the 30-min-to-80 % rate | Spearman monotone in ambient and SOH |
| **V1-5** | A 33 120-cell truck pack certifies at its weakest cell, exactly | pack `u*` ≠ min over cells | agreement to 1e-9, N up to 33 120 |
| **V1-6** | Robotaxi duty (12 DCFC/day) reaches a private car's decade of stress in ~6 weeks and still holds | violation over 5 simulated years | 0 violations over ≥ 20 000 sessions |
| **V1-7** | Coolant-loop degradation is absorbed by the cooling reserve `f`, and its failure point is predictable | violation before the predicted point | zero violations while `f` covers the loss |
| **V1-8** | Regenerative-braking pulses at `dt = 0.1 s` do not break the certificate | violation under transients | 0 violations, exact-ZOH RC |
| **V1-9** | Worst-case execution time fits an ISO 26262 10 ms task slot with orders to spare | p99 > budget | report p99 and duty cycle |
| **V1-10** | Thermistor bias/dropout is survivable up to the derived breakpoint | violation below the breakpoint | matches the E9 closed form |
| **V1-11** | V2G discharge needs a nonzero anchor even on the ground | anchored and null-input agree | null-input must under-serve the load |

### V2 — Aerial: delivery rotorcraft and eVTOL

| ID | Claim | Falsifier | Decision rule |
|----|-------|-----------|---------------|
| **V2-1** | Where the null-input filter fails (E2), the anchored filter holds | any violation under anchored | anchored 0 violations, null-input > 0, same seeds |
| **V2-2** | The two-sided admissible set is a single interval in flight | any disconnected scan | dense scan, ≥ 20 000 states, 0 disconnected |
| **V2-3** | Interval width is a *predictive* reserve: it reaches zero before any constraint is breached | a violation with no prior closure warning | lead time > 0 in 100 % of episodes; distribution reported |
| **V2-4** | A full delivery sortie (climb/cruise/hover/descend) certifies end to end | violation on any phase | 0 violations across profiles |
| **V2-5** | Payload and density altitude close the envelope through the same term | non-monotone reserve in either | Spearman < 0 for both, permutation p |
| **V2-6** | Gust-driven power transients do not breach the floor | violation under gusts | 0 violations, gust σ swept |
| **V2-7** | Cold soak at altitude is survivable and the binding constraint switches to voltage sag | violation | 0 violations; report which edge binds |
| **V2-8** | A motor-out (+33 % on remaining rotors) is detected as a closure, not a violation | violation without closure warning | closure precedes breach in 100 % |
| **V2-9** | 40 turnaround charges/day into a hot pack holds | violation | 0 / n |
| **V2-10** | eVTOL hover at 2.5C certifies, and the reserve converts to endurance | violation, or non-monotone endurance | 0 violations; endurance monotone in reserve |
| **V2-11** | An adversary optimising the power profile against the two-sided filter cannot break it | any violation | 0 violations at n ≥ 299 (certifies < 1 %) |

### V3 — Underwater: survey AUVs, gliders, under-ice

| ID | Claim | Falsifier | Decision rule |
|----|-------|-----------|---------------|
| **V3-1** | A sealed hull (UA ≈ 1/10 of a car's) still certifies | violation | 0 / n |
| **V3-2** | In 2 °C water the binding constraint on recharge switches from temperature to plating | temperature still binds | report binding fraction; plating > 50 % |
| **V3-3** | A nonzero anchor arises with nothing moving — hotel load alone forces it | null-input serves the load | null-input must brown out |
| **V3-4** | A 6-month deployment of dock recharges holds with no recalibration | violation | 0 violations over ≥ 4 000 cycles |
| **V3-5** | Dormancy between missions (calendar ageing) is another monotone channel | violation | 0 / n |
| **V3-6** | Depth (colder water, worse hull coupling) degrades gracefully | violation, or non-monotone SOC | 0 violations; Spearman monotone |
| **V3-7** | Under ice, closure warning arrives with enough lead time to reach a known hole | lead time < transit time | report lead-time distribution against a 20-min transit |
| **V3-8** | Thrust transients (obstacle avoidance) do not breach the floor | violation | 0 / n |
| **V3-9** | 180 days without recalibration costs charge, never safety | safety differs from an oracle | violations equal (0); SOC gap by Wilcoxon |
| **V3-10** | A glider at 0.5 W and an AUV at 175 W are the same certificate | different structure | both single-interval; reserve ratio reported |
| **V3-11** | The buoyancy pump's burst load is absorbed by the anchor | violation on a burst | 0 / n |

### V4 — Space: LEO, GEO, deep space, Mars, lunar night, high radiation

| ID | Claim | Falsifier | Decision rule |
|----|-------|-----------|---------------|
| **V4-1** | 28 000 LEO eclipse cycles, never recalibrated | violation | 0 end to end |
| **V4-2** | GEO eclipse season (90 eclipses, 72 min max, 3 kW payload) holds | violation | 0 / n |
| **V4-3** | Deep-space cruise, radiation-only cooling to 4 K | violation | 0 / n |
| **V4-4** | Mars diurnal: radiation plus a thin convective term is still monotone | violation, or lost monotonicity | 0 / n; monotonicity verified numerically |
| **V4-5** | Lunar night: the floor is a *heater*, so both edges push on temperature | violation, or empty interval before dawn | 0 violations across 354 h; survival margin reported |
| **V4-6** | 300 krad TID enters as one more monotone channel | violation | 0 / n |
| **V4-7** | A single-event upset in the filter's own state is contained | a flipped bit produces a certified unsafe command | report containment fraction and worst outcome |
| **V4-8** | No ground loop: a fixed pre-launch parameter set is admissible for the whole mission | violation | 0 over the full duration |
| **V4-9** | Solar-array degradation shrinks the charge window, not the certificate | violation | 0 / n; window shrinkage reported |
| **V4-10** | Radiator degradation degrades gracefully to the vacuum limit | violation as `εA → 0` | 0 at every point |
| **V4-11** | The bus-load anchor in eclipse is nonzero and the null-input filter cannot represent it | null-input serves the bus | null-input must brown out |

### X — Cross-domain connectedness

| ID | Claim | Falsifier | Decision rule |
|----|-------|-----------|---------------|
| **X1** | The admissible set is a single interval in every domain, every mode, every state tested | one disconnected scan | 0 disconnected out of ≥ 60 000 scans |
| **X2** | With `u_a = 0` the anchored filter *is* the null-input filter | any disagreement | bit-exact on all charge platforms |
| **X3** | Worst-case cost is fixed across all sixteen platforms | data-dependent iteration count | evaluation count bounded, latency p99 reported |
| **X4** | Reserve width predicts time-to-closure across domains | no monotone relation | Spearman with permutation p, pooled and per domain |
| **X5** | Every zero-failure claim above is backed by enough samples to certify below 1 % | `n` < 299 for any headline claim | report `n_required` beside every claim |

---

## 3a. Divergences from this register

Written after the experiments ran. `DESIGN.md` ends with the rule that a design document
allowed to drift from its experiments is worse than no design document, so every place this
one moved is recorded here rather than edited into agreement above.

**The theorem statement got weaker and truer.** The register above states Anchored Collapse as
requiring "a known anchor `u_a`, feasible from every state in the safe set". Implementing it
showed that neither bisection needs the anchor at all. The caps are satisfied on a prefix and
`u = 0` is in that prefix from any safe state — that is the *original* (A1), unchanged, applied
to the cap family only. The floors are satisfied on a suffix and the actuator ceiling is in
that one. So each edge has its own known-good endpoint and the anchor is a *physical*
quantity — the current the load demands, which is where the lower edge lands — rather than an
algorithmic seed. The shipped statement is (A1′): `u = u_min` is safe **for every cap**. This
is strictly weaker than (A1) and it is what the vehicles satisfy: cutting a quadrotor's current
does not heat it, sag it or discharge it. What zero cannot do is fly the aircraft, and that is
a floor.

The first implementation started the upper bisection *at* the anchor, which is sound only while
the anchor is cap-feasible. `tests/test_platforms.py` caught it: on a tightly sized rotorcraft
the interval came back `ok` with the reported anchor outside it. The current version cannot
express that bug.

**A soundness condition nobody had written down.** Delivered bus power is `S·V(u)·u` with `V`
falling in `u`, so it peaks at the maximum-power point and falls beyond it. Past the peak the
set where the load is met stops being a suffix and becomes an interval, and the lower bisection
would be searching a non-monotone family. What keeps it valid is that a pack reaches its MPP
near half its open-circuit voltage, far below any usable `V_min`, so the voltage cap always
binds first. The floor search is therefore bounded by `u_hi` rather than `u_max`, and the
ordering `u_hi < u_MPP` is now checked on every discharge platform instead of assumed. It
surfaced only when the rotorcraft's actuator ceiling was raised to a power cell's 6C, which
pushed `u_max` past the peak for the first time.

**Cost is 40 evaluations, not 39.** Two probes to bracket each family plus 18 halvings each.
The register's arithmetic was wrong; the property — fixed, data-independent — is unchanged.

**The estimator has to be pessimistic in every channel, not the interesting one.** The first
run of V1-2 returned 719 excursions in 4 000 sessions. The cause was that `vexp.pessimistic`
scaled only resistance, while the plating proxy *divides* by the plate scale, so a plant drawn
at 1.6 had a lower margin than an estimator at 1.0. Fixed by putting the estimator on the worst
corner of every channel it claims to bound. Nothing about the theorem was involved and the bug
would have produced a paper full of unexplained violations.

**Certified against enforced, carried forward.** The IECON paper certifies `S = {T, V}` and
*enforces* plating, because `u = 0` does not clear the plating margin above ~0.82 SOC. The
first vehicle harness counted plating excursions as violations, which made the V1-8 regen
experiment report 183 failures that were all `u = 0` at high SOC — exactly the corner the
original paper says cannot be certified. The distinction is now explicit in
`vexp.certified_idx` and every result reports the two separately.

**Findings that were not designed for.**

| what | where |
|---|---|
| The calibrated M50 — an energy cell — is **refused before takeoff** above ~1.1C of hover draw. The filter is acting as a design tool, not a runtime guard, and the refusal is the correct engineering answer. The remaining aerial experiments therefore use a power cell, which is the register's largest single modelling liberty. | V2-1 |
| With the pack coupled to ambient air, the throttled margin makes fast charging **impossible above 32.5 °C** — `T_max − δ_T` exactly. This is why production EV packs have chillers rather than radiators, and it was found by a sweep written to look for something else. | V1-3 |
| A lander on 2.9 kWh does **not** survive 354 hours of lunar night on a 45 W heater, and neither do most real ones. V4-5's decision rule changed from "zero violations across the night" to "report the survivable frontier", because the pass/fail version would have been a claim about a spacecraft nobody builds. | V4-5 |
| A sealed hull's conductance is **0.086×** a car's, not the 0.8× the first parameter set gave. The experiment printed the ratio next to the sentence asserting an order of magnitude, and they disagreed. | V3-1 |

**Two experiments were re-scoped because the first version could not fail.** V1-4 asked for
80 % SOC in thirty minutes; under the throttled margin this pack delivers 40–50 points in that
time, so every cell of the grid returned zero and the sweep could not distinguish graceful
degradation from a cliff. The target is now 60 %. The same experiment's ambient-monotonicity
test was replaced: delivered charge is **not** monotone in ambient and was never going to be —
it peaks near 20 °C, with series resistance limiting the cold limb and the thermal margin the
hot one — so the two limbs are tested separately rather than pooled into a non-significant
correlation that would have meant nothing.

**A rank-correlation bug that manufactured a result.** `stats.spearman_perm` used
`argsort(argsort(x))` as its rank transform, which breaks ties by array position. V3-5 swept
ten years of dormancy and every node returned the same delivered charge — and the function
reported `ρ = +1.000, p = 0.0002` for a flat line. Mid-ranks (`stats._rank`) fix it, and a
constant series now returns `ρ = nan` with `degenerate: True` rather than a perfect
relationship. Two vehicle results were affected and both changed meaning, not just precision.

**Three claims the data refuted, kept as findings rather than repaired.**

*Plating dominance underwater.* V3-2 was written to show the binding constraint *switching*
from thermal to plating as the water cools. It never switches, because it was never thermal:
plating binds in 100 % of states from −1.8 °C to 25 °C. A sealed hull charges so slowly that
the pack never approaches its thermal limit. What does change with temperature is which
plating mechanism binds. This has a consequence worth stating plainly: on this vehicle, in
this medium, **the binding constraint is the one the IECON paper enforces but does not
certify.** V3-5 and V3-6 inherit it — delivered charge does not move across a decade of
dormancy or 6 000 m of depth, because neither acts on the channel that is limiting. The
certificate is not tolerant of those channels so much as blind to them.

*Under-ice warning time.* V3-7's decision rule was that the closure warning must cover a
20-minute transit. It does not: the median is about seven minutes, and it covers the transit in
essentially no episodes. Rather than soften the rule, the experiment now reports the negative
and then does the thing the negative implies — closure is the wrong trigger, so it sweeps an
abort threshold on the *reserve* and reports the smallest one that buys the transit at the 5th
percentile. That design rule is only available because the two-sided certificate has a reserve
to threshold; the null-input version has nothing to sweep.

*Continuity of the deadline service.* V1-4 asked whether the 30-minute service degrades
continuously. Delivered charge does. The binary hit-rate against a fixed target does not and
cannot — thresholding a smooth quantity produces a threshold. The claim now separates the two
and reports the operational reading: plan on the continuous curve the filter already computes,
not on the binary derived from it.

**An experiment that was measuring its own refusal.** V2-9 charged a drone pack that had just
landed at up to 44 °C and reported that it reached 0.195 SOC. True, and uninteresting: the
throttled margin puts the admissible ceiling at 32.5 °C, so the certificate correctly refuses
to put any current in at all. It now cools at zero current until the interval opens, and
reports the wait, the charge, and the sorties per day that survive it — which is the question
a delivery operator actually has.

**A harness bug that fabricated 76 violations.** V4-4 drew the Mars rover's initial temperature
as `T_air + U(10, 60)`, which at local noon starts the rover at +50 °C against a 40 °C limit.
Forward invariance makes no claim about a state that begins outside the safe set, so those were
not violations of anything. `vexp.safe_init` now clamps the initial state onto the boundary of
the set the certificate is about.

**The reserve claim had to be scoped, and X4 is where.** The register asserts that interval
width predicts time-to-closure across domains. It does in the air (ρ = +0.88 for the eVTOL) and
on the Moon (+0.87) and it does not underwater (+0.02 and −0.12), because an interval can close
for two different reasons: the edges converging, or the state of charge reaching its floor with
the interval still wide open. Only the first is a countdown. The pooled, anchor-normalised
correlation is +0.916, and reporting that alone would have been the most flattering presentation
available. X4 now reports both clocks — the width and `(soc − soc_floor)·Q / u_lo` — per
platform, notes that every platform has one of them significantly positive with the GEO
satellite weakest at +0.367, and recommends acting on the minimum.

**Two instrumentation bugs that manufactured failures rather than hiding them.** Debouncing the
gust closure (three consecutive empty steps before calling it a closure) made the safety
accounting key on the debounced signal rather than the live one, so V2-6 reported 467 unsafe
episodes and *negative* lead times for a filter that had already closed correctly. Safety is
now accounted against `st == "ok"` at the step in question, and the debounce cost is reported
separately (4 s at σ = 0, 96 s at σ = 0.5). Separately, V1-5's resolution bound divided the
*answer* by 2⁴⁵ instead of the *search range*, producing a tolerance tighter than the search
could deliver.

**Bisection resolution, for the third time in this project.** V1-5's pack-versus-weakest-cell
comparison disagreed at 1.4 × 10⁻⁵ A, and the pessimism-monotonicity test showed the lower edge
*falling* under a more conservative estimator. Both were the search's own resolution: the cells
and the pack bisect over slightly different upper bounds, and the floor search's range moves
with `u_hi`. Re-run at 45 and 34 halvings respectively, the deviations are 1.9 × 10⁻¹³ and
exactly zero. Every comparison of two bisected quantities in this repository now carries a
tolerance tied to the resolution rather than to a hopeful constant.

## 4. What would make this fail

Stated in advance:

- If the anchored filter violates on any platform whose anchor it reports as feasible, the
  theorem is wrong as stated and the vehicle framing dies with it.
- If any admissible set comes back disconnected, the two-bisection construction is unfounded
  and the fixed-cost argument goes with it.
- If closure warnings do not precede violations, the reserve is a decoration and should be
  removed from the method rather than defended.
- If the anchored filter and the null-input filter agree on a discharge platform, the
  generalisation is unnecessary and this register is an elaborate way of restating `DESIGN.md`.
- If a platform only certifies because its parameters were chosen to make it certify, that is
  the failure mode of the first register's four "unrelated systems" repeated at larger scale.
  The defence is that every platform's parameters are fixed by its vehicle class before the
  filter is run, and the sweeps (V1-3, V2-5, V3-6, V4-10) are wide enough to contain the
  failure point where one exists.
