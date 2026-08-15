# Anchored Collapse: a solver-free safety certificate for the battery of any autonomous vehicle

*Ground, air, water, vacuum — one projection, forty-nine experiments, sixteen vehicles.*

**Companion to** `PAPER.md`, which established that the IECON filter is not about batteries.
This one asks what the answer to that costs, and what it buys.

---

## 1. What the first answer left broken

The IECON paper proves two things about a lithium-ion cell being fast-charged. `PAPER.md`
showed that neither is electrochemical: the same 36-line projection certified a heater, a motor
winding and an IGBT junction with no modification, across 20.4 million one-step transitions and
zero violations. What the filter actually needs is a system whose constraints are monotone in a
scalar input, and for which the null input is safe.

That second requirement is where the vehicle story stopped. Experiment E2 in the first program
put the filter on a hovering quadrotor and measured what happens when it fails: every one-step
check passed, for a mean of 22.3 consecutive steps, and then 100 % of episodes violated. The
conclusion drawn was that hypothesis (A1) had broken and the theorem did not apply.

That conclusion was right, and it ends the paper's usefulness for anything that moves. Cutting
the current to a quadrotor makes it fall. Cutting it to an AUV loses the computer that is
navigating. Cutting it to a spacecraft in eclipse browns out the payload. A certificate whose
only always-available action is *stop* is a certificate for a vehicle that is parked.

Except that the failure was misdiagnosed, and the misdiagnosis is the whole of this paper.

## 2. The anchor was never at zero; it was zero by accident

Look again at what `u = 0` does to a discharging vehicle. Nothing heats — no current, no
`I²R`. Nothing sags — terminal voltage returns to open-circuit. Nothing discharges. **Every
temperature and voltage constraint is satisfied at zero, on a quadrotor in flight exactly as on
a cell being charged.** What zero cannot do is fly the aircraft.

So the constraint that broke is not one of the ones (A1) was about. It is a new kind: a
*floor*, a requirement that the input be large enough rather than small enough, and the
original theorem has no vocabulary for it. Every vehicle has one, because every vehicle has an
irreducible load — the power it cannot stop drawing and remain a vehicle. Hover thrust for a
rotorcraft. Hotel load for a submersible: the computer and the inertial navigation draw their
watts whether the thrusters do or not, so even an AUV sitting still on the seabed cannot
command zero. Bus load for a spacecraft in eclipse. Contracted export power for a car doing
vehicle-to-grid.

Charging is the special case where that load is zero — which is precisely why the IECON filter
could anchor at zero without ever noticing it was making a choice.

> **Theorem (Anchored Collapse).** Let a system have state `x`, scalar input `u ∈ [u_min,
> u_max]`, and constraints each satisfied on one side of a threshold in `u`: the *caps* on
> `[u_min, u_c]`, the *floors* on `[u_c, u_max]`. If
> **(A1′)** `u = u_min` is safe for every **cap**, from every `x` in the safe set, and
> **(A2)** each constraint is monotone in `u`,
> then the admissible set is the interval
>
> `[u_lo, u_hi] = [ maxᶠˡᵒᵒʳˢ u_c , minᶜᵃᵖˢ u_c ]`
>
> and both edges are found by bisection in `O(log 1/ε)` one-step evaluations — with no
> optimizer, no convergence test, and no identification of the plant. The set is non-empty
> exactly when `u_lo ≤ u_hi`.

Null-Input Collapse is the case with no floors. (A1′) is strictly weaker than (A1): it asks
only that zero satisfy the caps, which is the part that survives contact with a vehicle.

### 2.1 Three things this makes available that the original could not say

**A two-sided certificate.** The upper edge is thermal and electrical; the lower edge is the
load. Safety is now the statement that they have not crossed.

**A reserve.** `u_hi − u_lo` shrinks to zero as the envelope closes, so the filter can report
*how close to unrecoverable* the vehicle is, and report it before arrival. The null-input
filter has no such quantity because its lower edge never moves. This is the claim with the most
operational weight in the paper and it gets the most adversarial treatment.

**A refusal.** When the two edges have crossed before the vehicle has even started, the
certificate is saying the design does not close. §5.1 is an instance: the calibrated cell is
refused for a rotorcraft *before takeoff*, which is a design result a runtime guard is not
normally able to produce.

### 2.2 The side of a constraint is declared, not inferred

The published plating constraint is `φ ≥ margin` — a lower-bound test on a signal that
*decreases* in current, so it caps the current from above. Sense and side coincide only when
the constrained signal happens to increase in `u`. Inferring the side from the comparison
operator would file that cap in the floor family and let the filter command straight through
it: silent, and unsafe. Plants therefore declare the side, and an undeclared limit raises
rather than defaults. `tests/test_platforms.py` check 3 is that refusal.

### 2.3 What it costs

Forty one-step model evaluations, worst case, against the original twenty: two probes to
bracket the cap family plus eighteen halvings, then two to bracket the floors plus eighteen
more. Both are fixed. The property that matters is not that the number is small but that it
does not move with the data, because a safety task slot is sized for the worst case and a QP's
iteration count is unbounded.

### 2.4 One soundness condition, which is not obvious

Delivered bus power is `S·V(u)·u` with `dV/du < 0`, so it peaks at the maximum-power point and
*falls* beyond it. Past that peak the set where the load is met stops being a suffix and
becomes an interval, and the lower bisection would be searching a non-monotone family. What
keeps it valid is the voltage cap: a pack reaches its MPP near half its open-circuit voltage,
far below any usable `V_min`, so `u_hi < u_MPP` always. The floor search is therefore bounded
by `u_hi` rather than `u_max`. This surfaced only when a rotorcraft's actuator ceiling was
raised to a power cell's 6C, which pushed `u_max` past the peak for the first time, and it is
now checked on every discharge platform rather than assumed.

---

## 3. Sixteen vehicles, one cell

Four media, sixteen platforms. Three things vary and nothing else does: the **medium**, through
the cooling law; the **pack**, through series/parallel counts; and the **load**, through the
anchor.

| domain | cooling law | platforms |
|---|---|---|
| ground | Newtonian, liquid loop | robotaxi (96S45P, 78 kWh), shuttle, haul truck (288S115P, 601 kWh) |
| aerial | forced air, derated by density altitude | delivery quadrotor (6S3P), eVTOL air taxi (300S45P, 147 kWh), turnaround charging |
| underwater | sealed-hull conduction into 2 °C water | survey AUV, buoyancy glider, under-ice AUV, dock recharge |
| space | Stefan–Boltzmann to 4 K (plus 6 mbar of CO₂ on Mars) | LEO smallsat, GEO comsat, deep-space cruiser, Mars rover, lunar-night lander, high-radiation orbiter |

The cell underneath is the published ROM's arithmetic in every case — same overpotential form,
same plating proxy, same coefficients — with two changes the vehicles require. Current may be
negative, because vehicles discharge. And the Newtonian cooling term is a callable, because a
quadrotor at 400 m, an AUV at 2 °C and a spacecraft in vacuum do not share a cooling law.
`tests/test_platforms.py` check 1 asserts that under the Newtonian law with positive current
this model reproduces `BatteryROM.probe` **bit for bit**; every result downstream inherits it.

**The platforms are class-representative, not vendor designs.** Pack geometry, hover power,
hotel loads and orbit parameters place each vehicle in the right region of the design space
using published figures for the *class*. No claim here is a claim about anyone's hardware.

---

---

## 4. Ground: robotaxis, shuttles, autonomous trucks

The domain where the original result should hold most directly — the anchor is zero, the medium
is a liquid loop, the physics is what the ROM was calibrated on. A failure here would mean the
vehicle framing is broken rather than stretched. What is new is scale and duty.

**The vehicle filter, on one cell, *is* `project_current`.** Bit-exact on 100,000 states, both
discretizations, fresh and aged, zero mismatches. And a *P*-parallel pack's admissible current
is exactly *P* times one cell's, to a relative deviation of 6.5 × 10⁻¹⁶.

**A 350 kW session on a 78 kWh pack holds.** 4,000 sessions under the full parameter envelope,
zero violations, Clopper–Pearson 95 % upper bound 0.075 %. Worst plant temperature 39.9 °C
against a 45 °C limit, and the pessimistic corner dominates every draw. On this vehicle the
charger is not the binding constraint: 350 kW into a 96S pack is about 985 A, above the pack's
own 3C ceiling of 540 A.

**A 33,120-cell truck pack certifies at its weakest cell, exactly.** Verified from *N* = 1 to
*N* = 33,120, worst deviation 1.9 × 10⁻¹³ A — below the 1.5 × 10⁻¹³ A resolution of a 45-halving
search. At the deployed 18 iterations the same comparison disagrees at 1.4 × 10⁻⁵ A, which is
the search's own resolution and not a fact about packs.

**Five years of robotaxi duty, never recalibrated.** 21,915 fast-charge sessions, zero
violations, CP-95 upper bound 0.0137 %. Delivered charge per session *rises* over the life
(0.372 → 0.403) because early sessions start hotter. At twelve DC fast charges a day against a
private car's one a fortnight — a factor of 171 — a robotaxi accumulates a private car's decade
of fast-charge stress in **three weeks**.

**Worst-case execution time is 71.7 µs at p99**, on the 20-evaluation path, which is 0.72 % of
an ISO 26262 10 ms task slot. Three code paths appear (1, 2 and 20 evaluations) and the count
is bounded, not merely small.

**A margin's purchasing power has a closed form, at vehicle scale.** A thermistor reading low
is survivable up to *b\** = δ_T/(1 − Δt·hA/C_th) = **12.802 °C**, using the throttled margin
δ_T = δ_T0 + K(s_R − 1) = 12.5 K that is actually in force. Measured breakpoint 12.9 °C on a
0.1 °C grid — an error of **0.098 °C**, from published constants with nothing fitted.

**V2G needs a nonzero anchor on the ground.** At 7.4, 11 and 22 kW of contracted export the
anchored filter recorded zero unsafe-while-certified episodes in 600 sessions each, while the
null-input filter browned out the contracted load in **600 of 600** at every power. The
generalisation is not a story about aircraft.

### 4.1 The result this domain was not looking for

Coupled straight to ambient air, the throttled margin puts the admissible ceiling at
`T_max − δ_T` = **32.5 °C**. Above roughly that ambient there is no admissible fast charge at
all, and the certificate delivers nothing — correctly, because at the datasheet end-of-life
resistance bound there is none to deliver. The measured dead zone begins at 35 °C, the first
grid node past the predicted 32.5 °C.

Running the identical sweep with a loop holding coolant at 25 °C recovers the whole range and
buys **+8.7 SOC points** on average. That difference is exactly what the thermal management
system is for, and it is why production EV packs carry chillers rather than radiators. Zero
violations in all 8,000 sessions across both configurations.

A second, smaller finding follows from it. Asked whether the 30-minute service degrades
continuously, the answer depends on which quantity you mean. Delivered charge is ordered and
monotone along each limb of the ambient sweep — rising to a peak at 25 °C (ρ = +0.841,
*p* = 0.0001) and falling past it (ρ = −0.850, *p* = 0.0001), with resistance limiting the cold
side and the thermal margin the hot one. The *binary* hit-rate against a fixed 60 % target
jumps the full range between adjacent cells, because thresholding a smooth quantity produces a
threshold. Dispatch should plan on the curve the filter already computes, not on the binary
derived from it.

---

## 5. Aerial: delivery rotorcraft and eVTOL

The domain the generalisation exists for, and the one where the null-input filter does not
merely under-perform but kills the aircraft. On identical seeds, 600 flights each: the anchored
filter recorded **zero** unsafe-while-certified episodes, and the null-input filter starved the
rotors in **600 of 600**, serving a mean of **0.0** steps before brownout. It does not fly the
aircraft for a while and then fail; it never flies it at all.

**The two-sided admissible set is an interval.** 20,000 flight states scanned on a 200-point
grid: 10,483 single intervals, 9,517 empty, **zero disconnected**, and every bisected edge
within one grid step of the dense scan's.

**Nothing broke anywhere in the flight envelope.** Zero unsafe-while-certified across the
payload × altitude grid (4,500 flights), gusts to σ = 0.5 (3,000), cold soak at 2,000 m
(3,200), motor-out to +100 % power (3,000), and 4,000 turnaround charges. A cross-entropy
adversary given 320 independently drawn packs and 30,720 flight profiles — free to command
anywhere inside the certified interval — reached **79.2 %** of the temperature limit while the
certificate was open and broke nothing (CP-95 upper bound 0.932 %, past the 299 cells needed to
certify below 1 %).

**Payload and density altitude close the envelope through the same term.** Momentum theory puts
both inside `T^{3/2}/√(ρA)`, and the reserve falls with each: ρ = −0.981 against payload,
ρ = −0.998 against hover power — the shared term — and negative against altitude **in every
payload stratum**. Pooled across the grid the altitude correlation is a non-significant −0.192,
because payload spans a far larger range of hover power and swamps it; the within-stratum test
is the one that answers the question.

### 5.1 The certificate refuses to fly the calibrated cell

The most interesting aerial result is a refusal. With the ROM's own M50 coefficients — an
energy cell — the anchored filter reports the interval **empty before takeoff** at and above
**1.2C** of hover draw. The admissible width collapses smoothly to it: 5.06 A at 0.6C, 1.28 A
at 1.0C, 0.16 A at 1.1C, empty at 1.2C.

That is not a filter failure. It is the filter working as a *design tool*: the correct
engineering answer to "may I fly a rotorcraft on an energy cell" is no, and here it arrives
before the aircraft leaves the ground rather than as a thermal event in flight. The remaining
aerial experiments therefore use a power cell — 0.40× resistance, 0.60× capacity — which is the
largest single modelling liberty in this work and is flagged wherever it applies.

### 5.2 The reserve warns, and the takeoff transient sizes the pack

Across 1,200 flights, **every** closure preceded its breach: zero breaches with no prior
warning, minimum lead 0 s, 5th percentile 12 s, median 24 s, 95th percentile 302 s. Under
gusts the picture holds at every variance up to σ = 0.5 with a median lead of 31–38 s, once
the *live* closure is distinguished from a debounced one — requiring three consecutive empty
steps before calling it a closure costs 4 s at σ = 0 and 96 s at σ = 0.5, and measuring lead
from the debounced signal made the first version of that experiment report negative lead times
for a filter that had already closed.

A full delivery sortie — takeoff, climb, cruise, hover over the drop, return, land — shows the
reserve tracking the mission without the filter being told a mission plan exists: 3.07 A in
takeoff, 16.6 A in climb, 31.2 A in cruise, 26.2 A over the drop. **Takeoff is the binding
phase**, and it is what sizes the pack. On 6S2P the certificate refuses a 1.75× takeoff
transient outright; on 6S3P it allows it and 34.2 % of sorties complete; on 6S4P, **78.5 %**.
The frontier is actionable, which is what makes it a design tool rather than a complaint.

An eVTOL air taxi hovering at 3.09C certifies with zero unsafe episodes, a median hover
endurance of 3.0 minutes (p05 0.0, p95 8.7), and interval width predicting that endurance at
ρ = +0.879.

### 5.3 What forty sorties a day actually costs

A delivery pack lands at up to 44 °C, and the throttled margin puts the admissible ceiling at
32.5 °C — so the certificate refuses to charge it. Measuring how little charge a refused
session delivers measures the refusal; the operational question is how long the pack must sit,
and the filter answers it directly, because the wait is the time until the interval opens.

Of 4,000 turnarounds, **18 % could start at once**; the rest waited a median of **60.8 minutes**
(p95 100.9, max 104.2), and none was still refused after two hours. Zero violations throughout.
On a single pack that is a 99.4-minute sortie cycle — **14.5 sorties/day**. Swapping packs so
the cool-downs overlap gives 38.7 minutes and **37.2 sorties/day**, and holding a 40/day target
needs **2.8 packs in rotation**. This is why delivery operators swap packs rather than wait,
and the certificate produces the number rather than the folklore.

---

## 6. Underwater: survey AUVs, gliders, under-ice vehicles

The hardest domain, for reasons that have nothing to do with the theorem. A pack inside a
pressure hull is thermally isolated from an excellent heat sink — **0.0069 W/K per cell against
a car's 0.080, about one twelfth** — and the water is 2 °C, so both intuitions are wrong at
once: it heats slowly *and* it is very cold.

It is also the cleanest separation of the anchored generalisation from anything about flight. A
survey AUV sitting perfectly still on the seabed still cannot command zero: the computer, the
inertial navigation and the modem draw 25 W regardless. Across idle, survey and transit — 25,
115 and 265 W — the anchored filter recorded zero unsafe-while-certified episodes in 800
missions each, and the null-input filter browned out the vehicle in **800 of 800 at every one
of them, including the one where nothing is moving.**

**A sealed hull certifies**: 3,000 dock recharges in 2 °C water, zero violations, CP-95 upper
bound 0.100 %. **A six-month deployment with no recalibration**: 547 recharges, zero
violations. **180 days with no ground truth**: a fixed bound and a daily oracle record
identical safety and identical delivered charge.

### 6.1 The constraint the theorem does not certify is the one that binds

This experiment was written to show the binding constraint *switching* from thermal to plating
as the water cools. It never switches, because it was never thermal. **Plating binds in 100 %
of states at every water temperature from −1.8 °C to 25 °C, and temperature binds in none.** A
sealed hull charges so slowly — the plating cap holds it near 0.7C — that the pack never
approaches its thermal limit. What changes with temperature is which plating mechanism binds:
the current cap takes 35 % of cases at 25 °C and 44 % at 2 °C.

That has a consequence worth stating plainly rather than dressing up. The IECON paper certifies
temperature and voltage and only *enforces* plating. **On this vehicle, in this medium, the
binding constraint is the one the theorem does not certify.**

Two downstream sweeps inherit it and both come back flat. Delivered charge does not move across
ten years of dormancy (spread 2 × 10⁻⁸ SOC) or 6,000 m of depth (7.6 × 10⁻⁸), because neither
channel touches the plating cap. Those spreads are numerical noise, and no rank correlation is
reported over them — an earlier version did, and reported ρ = +1.000, *p* = 0.0002 for a flat
line. The certificate is not *tolerant* of dormancy and depth so much as blind to them.

### 6.2 A negative result, and the design rule it forces

Under an ice shelf there is no abort-to-surface: a closed envelope is a lost vehicle. So the
question is not whether the certificate is correct but whether it is *early*.

**Waiting for the interval to close is too late.** Across 1,000 missions the closure warning
arrives a median of **7 minutes** before the envelope is gone (p05 6 min, p95 8 min), against a
20-minute transit back to a known hole. It covers the transit in **0 % of episodes**. Zero
breaches occurred without a warning — the certificate is correct — and correct seven minutes
late loses the vehicle anyway.

The fix is not a better certificate; it is to stop using closure as the trigger. The reserve is
available at every step and falls smoothly, so the abort can fire on a *threshold* instead of
on exhaustion:

| abort threshold | median warning | 5th percentile | covers the transit |
|---|---|---|---|
| closure (0 A) | 7 min | 6 min | 0 % |
| 2 A | 11 min | 10 min | 0 % |
| 4 A | 20 min | 18 min | 54 % |
| **8 A** | **34 min** | **31 min** | **100 %** |
| 16 A | 58 min | 52 min | 100 % |

Aborting at 8 A of remaining reserve buys 31 minutes at the 5th percentile and covers the
transit in every episode. This is a design rule the null-input certificate could not have
produced, because it has no reserve to threshold — and reporting the closure-triggered lead
time alone would have been a true sentence and a misleading paper.

**A glider at 0.5 W and an AUV at 175 W are the same certificate.** A 350× difference in load
and a 250× difference in anchor; both single-interval in every state scanned, both zero unsafe.
What differs is the reserve *ratio* — 1,575 for the glider against 7.7 for the AUV — which is
the dimensionless statement of how much room each has relative to what it must draw.

---

## 7. Space: LEO, GEO, deep space, Mars, lunar night, radiation

Space is where the certificate's least glamorous property becomes decisive. The filter needs no
identification of the plant — no recursive least squares, no observer to converge, no ground
station in the loop. On a car that is a convenience. On a spacecraft that has not been touched
since integration, whose ground link is hours of light time away, and whose battery has taken
300 krad in the interim, it is the only reason a fixed pre-launch parameter set is admissible.

**28,307 LEO charge cycles — five years at a 92.9-minute period — zero violations**, CP-95
upper bound 0.0106 %, worst temperature 24.6 °C. Delivered charge on the last quarter of the
mission matches the first to three decimal places.

**A GEO eclipse season completes at every duration out to the 72-minute seasonal maximum**,
with a median depth of discharge of 27.4 % on the sized pack and zero unsafe-while-certified
episodes.

**Deep-space cruise, radiating to a 4–250 K sink**: 3,000 draws, zero violations, and the
Stefan–Boltzmann cooling channel verified monotone in temperature numerically rather than
assumed. **A Mars sol** at hourly resolution: zero violations at every local hour across an
80 K diurnal swing, with radiation plus 6 mbar of CO₂ confirmed monotone as a sum.

**300 krad of total ionising dose is one more monotone channel**: zero violations to 500 krad,
with delivered charge falling gently and monotonically in dose (ρ = −0.886, *p* = 0.033).

**Five years with no ground in the loop**: a fixed datasheet bound and an oracle recalibrated
every single day record identical safety (0 violations each) *and identical delivered charge*
— the gap is 0.00 SOC points. In deep space recalibration buys not just no safety but no
performance either, because the binding constraint is the plating corner rather than
resistance.

### 7.1 Two inversions of terrestrial intuition

**Losing the radiator helps.** Sweeping `εA` down by two orders of magnitude produces zero
violations at every point — and delivered charge *rises*, 0.736 → 0.759 (ρ = −1.000). On a car,
losing cooling costs charge. Radiating to a 4–250 K sink, a healthy radiator makes the pack
*cold*, series resistance climbs, and the voltage limit arrives sooner; shrinking it lets the
pack keep its own dissipation. In this domain the pack is cold-limited, not heat-limited.

**Lunar night is a cold problem, so capacity is the wrong lever.** A lander on 2.9 kWh survives
about 5.3 hours of a 354-hour night, and quadrupling the pack buys 7.1 hours — because the
envelope closes on cold-induced voltage sag, not on stored energy. Insulation is the lever that
moves it: dropping emissivity from 0.72 to 0.05 takes survival from 5.3 h to **54.3 h**, at
which point the pack stops being cold-limited and hits its energy ceiling of 61.3 h
(2,904 Wh / 45 W). No configuration survives the full night, which is the honest answer and the
reason real landers that survive lunar night carry radioisotope heaters rather than bigger
batteries.

### 7.2 The one fault the theorem cannot survive, measured anyway

Every other fault model in this work corrupts the plant or the sensor. A single-event upset
corrupts the *certificate*: a heavy ion flips a bit in the state word the projection is about to
read, so the filter computes an admissible interval for a spacecraft that does not exist.

No one-step theorem survives corruption of its own input, and claiming otherwise would be
dishonest. What can be measured is the shape of the exposure. Across **40,000 bit flips** into
`soc`, `T` and `V1`: 5.80 % are caught by a three-comparison range check on the state word,
83.31 % are benign or leave the command *more* conservative, and **10.885 % produce a more
aggressive command than the clean one**. Of those, the worst overshoot actually realised in the
plant was **0.000 °C** — none of them breached anything in one step. The deliverable here is
the range check and the number beside it, not a theorem.

**Solar-array degradation is not a safety question on this spacecraft.** Swept to sixty years —
three times any real mission, down to 29.8 % of beginning-of-life output — the array never
becomes the binding constraint, and delivered charge varies by 0.0009 SOC across the whole
sweep. The separation between what the vehicle can *request* and what the filter can *certify*
is not marginal here; it is not close.

---

## 8. What makes this one method

Forty-four experiments across four media establish that the certificate holds in each. That is
not the same as establishing that it is the *same* certificate — four domains that each worked
for their own reasons would be four papers. Five cross-cutting checks, each of which would fail
if the domains were only superficially related.

**The admissible set is a single interval everywhere.** 64,000 dense scans on a 180-point grid,
across all sixteen platforms and both modes: 45,570 single intervals, 18,430 empty, **zero
disconnected**. The two-bisection construction rests entirely on this and it is measured rather
than argued from monotonicity.

**The generalisation reduces to the original theorem exactly.** Bit-exact against
`project_current` on 30,000 states, and on every one of the nine charge-mode platforms the
lower edge is exactly zero — Null-Input Collapse is recovered, not approximated.

**The cost is fixed, and it is the same fixed cost everywhere.** Twenty model evaluations on
every charge platform, forty on every discharge platform, against theoretical bounds of exactly
20 and 40. Worst p99 latency across all sixteen is **119.4 µs** (the buoyancy glider), which is
1.2 % of a 10 ms task slot. The number that matters is not that it is small but that it does
not move with the data.

**Every zero-failure claim is backed by enough samples to certify something.** Twelve headline
claims, 69,389 pooled trials, zero violations; every one exceeds the 299 trials needed to
certify below 1 % at 95 %, and five exceed the 2,995 needed for 0.1 %. This audit is here
because the first ZEROGUARD program found an adversarial claim resting on 40 cells that bounded
nothing below 7.2 %.

### 8.1 The reserve is one of two clocks, and that is the honest version

The claim that interval width is a *reserve* — a countdown — survives in the air and on the
Moon and does not survive underwater, and pooling would have hidden it.

| platform | width clock | charge clock |
|---|---|---|
| eVTOL air taxi | **+0.879** | +0.115 |
| lunar-night lander | **+0.867** | −0.503 |
| delivery quadrotor | **+0.744** | +0.606 |
| GEO comsat | **+0.367** | +0.084 |
| survey AUV | +0.023 | **+0.990** |
| under-ice AUV | −0.124 | **+0.990** |

An interval can close for two different reasons. **Envelope-driven** closure is the upper edge
falling onto the lower one as the pack heats and sags — there the width *is* the distance still
to travel. **Energy-driven** closure is the state of charge reaching its floor with the
interval still wide open — there the clock is set by how much charge was in the pack, and the
width has nothing to say about it. Both are correctly reported as closures; only the first is a
countdown.

The certificate already carries both quantities:

    t_envelope  ~  u_hi − u_lo            the interval width
    t_energy    =  (soc − soc_floor)·Q / u_lo

Every platform has one of them significantly positive (ρ > 0.3, *p* < 0.001), the GEO satellite
most weakly at +0.367. The operational rule is to act on the **minimum of the two**, which is
conservative whether or not either is individually tight. Reporting the pooled +0.916 alone
would have been the most flattering way to present this and the least honest.

---

## 9. What this does not show

Carried forward, unchanged, and added to.

**No Doyle–Fuller–Newman anywhere in this repository.** Every number here is
reduced-order-model level or model-versus-model. The DFN evidence behind the IECON paper stands
on its own and is cited, not re-derived. A vehicle result that would benefit from DFN replay is
a limitation, not a footnote.

**The certified set is `{T, V}`; plating is enforced and never certified.** §6.1 shows this is
not a technicality: in a sealed hull at any water temperature, **the binding constraint is the
one the theorem does not certify.** The same corner explains why a decade of dormancy, 6,000 m
of depth, six months of deployment and a daily oracle all move delivered charge by nothing.

**One cell, sixteen vehicles.** Every platform uses the M50 coefficients, because that is the
cell the ROM was calibrated against. Where a class would really use a different cell — the
rotorcraft — the substitution is explicit, is 0.40× resistance and 0.60× capacity, and is an
extrapolation beyond the calibration. Every aerial number carries it.

**The platforms are class-representative, not vendor designs.** Pack geometry, hover power,
hotel loads and orbit parameters place each vehicle in the right region of the design space
using published figures for the *class*.

**No theorem survives corruption of its own input.** §7.2 measures the single-event-upset
exposure and offers a three-comparison range check; it does not claim the certificate is
robust to a flipped bit, because it is not.

**Simulation of simulation.** The plants here are models the author wrote, and a filter tested
against models chosen to satisfy its hypotheses is a weaker result than one tested against
hardware. What partially answers it is that three of the four media *break* the original
hypothesis and one platform is refused outright — the register was not constructed so that
everything passes, and §5.1, §6.2, §7.1 and §8.1 are the places where it did not.

---

*Every number in this manuscript is re-read out of `results/v1..v4_*.json` and
`results/x_crossdomain.json` by `verify_vehicles.py` — 76 checks, which fail loudly if the
prose and the data disagree.*

---

## Scale

| domain | experiments | episodes / states |
|---|---|---|
| V1 ground | 11 | 157,142 |
| V2 aerial | 11 | 72,320 |
| V3 underwater | 11 | 20,824 |
| V4 space | 11 | 87,637 |
| X cross-domain | 5 | 133,000 |
| **total** | **49** | **470,923** |

Sixteen platforms, four cooling laws, one unmodified projection. 76 claims re-read out of the
result files by `verify_vehicles.py`; 9 executable property checks in
`tests/test_platforms.py`. The original ZEROGUARD program (650,970 episodes, 100 claims) and
the IECON repository's own test suite both still pass unchanged.
