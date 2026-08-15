# ZEROGUARD

**Monotone Control Theory for Certified Battery Fast Charging in Long-Duration Autonomous
Vehicles**

An extension of the IECON work in `../safe_charge/`, which asks a question the original paper
did not: *is any of this actually about batteries?*

The answer is no, and the consequences are the subject here. The IECON filter needs exactly
two things from a system — a null input that is safe, and constraints that are monotone in
that input — and neither is electrochemical. Stated that way it is a theorem about a class of
control problems, and the battery is one instance.

Then a second question, which the answer to the first made unavoidable: *if it is not about
batteries, is it about vehicles?* The theorem as stated needs `u = 0` to be safe, and on a
vehicle that is doing its job it usually is not — cut the current to a quadrotor and it falls,
to an AUV and it loses its computer, to a spacecraft in eclipse and the bus browns out. That is
the subject of the second half.

```bash
python zeroguard/run_all.py                  # everything: cells, then vehicles, then figures
python zeroguard/run_all.py --cells-only     # just the original program
python zeroguard/run_all.py --vehicles-only  # just the four media
python zeroguard/verify_claims.py            # re-read all 100 cell claims out of the results
python zeroguard/verify_vehicles.py          # re-read every vehicle claim out of the results
```

| file | what it is |
|---|---|
| [`DESIGN.md`](DESIGN.md) | the experimental surface, written **before** anything ran, including what would falsify each claim |
| [`DESIGN_VEHICLES.md`](DESIGN_VEHICLES.md) | the same, for the vehicle program |
| [`PAPER.md`](PAPER.md) | the manuscript |
| [`PAPER_VEHICLES.md`](PAPER_VEHICLES.md) | the vehicle manuscript |
| `systems.py` | four monotone systems from unrelated physics, plus two that break one hypothesis each |
| `gfilter.py` | the projection, written once for any monotone system |
| `anchored.py` | the generalisation: an anchor that need not be zero, and a two-sided certificate |
| `platforms.py` | sixteen vehicles across four media, sharing one cell and one interface |
| `vexp.py` | the three mission loops every vehicle experiment runs |
| `stats.py` | Clopper–Pearson, bootstrap, Wilcoxon with effect sizes, permutation Spearman |
| `exp/` | one script per experiment; each writes JSON to `results/` |
| `figures.py`, `figures_vehicles.py` | eighteen figures, each redrawn from a results file |
| `verify_claims.py`, `verify_vehicles.py` | fail loudly if the manuscript and the data disagree |

## What it found

**The theorem is not about batteries.** The same unmodified projection certified a lithium-ion
cell, a resistive heater, a DC motor winding and an IGBT junction: 8,000 episodes, 20,400,000
one-step transitions, zero violations. Instantiated on the battery it reproduces the published
`project_current` **bit for bit** across 60,000 states.

**Its two hypotheses fail differently.** Break the safe null input and every one-step check
still passes — for a mean of 22.3 steps — and then 100 % of episodes violate. Break
monotonicity and the bisection never once returns an infeasible input; it just gives up 23.7 %
of the available charge. (A1) is about safety, (A2) is about performance.

**The certificate does not depend on how heat leaves the cell.** Replacing Newtonian
convection with Stefan–Boltzmann radiation leaves every monotonicity intact and produces zero
violations in 20,000 vacuum draws. Removing the radiator entirely degrades delivered charge
from 0.607 to 0.168 and never violates.

**A whole mission, never recalibrated, is exactly as safe as an oracle.** Over 31,000 charge
cycles the datasheet bound and a filter told the true resistance every cycle both recorded
zero violations. The oracle bought 8.8 and 16.2 points of delivered charge, and nothing else.

**A pack's certificate is its weakest cell's, exactly.** Verified from N = 1 to N = 1024. With
one faulted cell in a hundred, per-cell enforcement violated in 0 of 1,200 packs and
pack-averaged enforcement in 1,200 of 1,200.

**A margin's purchasing power has a closed form.** *b\** = *δ*<sub>T0</sub>/(1 − Δt·hA/C<sub>th</sub>)
= 12.802 °C predicts the measured sensor-bias breakpoint to 0.048 °C, from published constants
with nothing fitted.

**The design register was audited against what actually ran**, and the three gaps it found
were closed rather than dropped: a six-channel vacuum envelope including radiation drift
(0/20,000), the sample-adequacy check that `stats.n_required` was written for and never
called, and the latency evidence that never reached a figure. The adequacy check then found a
fourth problem of its own — the adversarial claim rested on 40 cells, bounding the violation
rate only below 7.2 % — so the attack was re-run against 320 cells and 256,000 sequences,
bringing it to 0.93 % with zero violations.

**Two things in the original method were described wrongly**, and the ablations found both.
The *I* = 0 short-circuit is not a safety mechanism — the bisection is self-correcting when
every input is infeasible, and returns zero regardless; what the check buys is 17 model
evaluations. And the throttle *K* = 15, which looked over-conservative on cells inside the
rated bound, turns out to be the *smallest* value that certifies cells past it.

## What the vehicle half found

**The failure was misdiagnosed, and that is the whole second paper.** E2 concluded that a
hovering quadrotor breaks (A1) because `u = 0` stops being safe. It does not: cut the current
and nothing heats, nothing sags, nothing discharges — every *cap* is satisfied. What zero
cannot do is fly the aircraft, and that is a **floor**, a constraint the original theorem has
no vocabulary for. Every vehicle has one, because every vehicle has an irreducible load.
Charging is the special case where it is zero.

**Anchored Collapse** adds a second bisection and nothing else: 40 model evaluations instead of
20, both fixed. It reduces to `project_current` bit for bit on 30,000 states, and every one of
the nine charge platforms anchors at exactly zero.

**Zero breaches while certified, across 44 experiments and four media.** 64,000 dense scans, no
disconnected admissible set anywhere. Twelve headline claims, 69,389 pooled trials, zero
violations, every one past the 299 needed to certify below 1 %.

**The certificate refuses designs, not just commands.** With the ROM's own calibrated cell it
reports the interval empty *before takeoff* above 1.2C of hover draw — the right engineering
answer to flying a rotorcraft on an energy cell. A 1.75× takeoff transient is refused on 6S2P,
allowed on 6S3P, and completes 78.5 % of sorties on 6S4P.

**Three results that inverted the hypothesis they were written to test.** In a sealed hull the
binding constraint is plating at *every* water temperature — the one the theorem enforces but
does not certify. In deep space, shrinking the radiator *increases* delivered charge, because
the pack is cold-limited rather than heat-limited. And on the lunar surface a bigger battery
buys almost nothing while insulation takes survival from 5.3 h to 54.3 h, against a 61.3 h
energy ceiling.

**One negative result, kept.** Under an ice shelf the closure warning arrives a median of 7
minutes before the envelope is gone, against a 20-minute transit — it covers the transit in 0 %
of episodes. Closure is the wrong trigger. Aborting instead at 8 A of remaining reserve buys 31
minutes at the 5th percentile and covers it in every episode: a design rule the null-input
certificate could not have produced, because it has no reserve to threshold.

**And the reserve claim is scoped rather than pooled.** Interval width predicts endurance in
the air (ρ = +0.88) and does not underwater (ρ = −0.12), because there the mission ends on
stored charge rather than on the envelope narrowing. The certificate carries both clocks; the
rule is to act on the minimum.

## Scope

Every number here is reduced-order-model level or model-versus-model. The PyBaMM
Doyle–Fuller–Newman pipeline behind the IECON paper is not part of this release, so its
evidence is cited rather than re-derived. Section 5 of `PAPER.md` states the limits in full,
including the two the original paper already carried: the certified set is {*T*, *V*} and
plating is enforced rather than certified, and the certificate is conditional on the sensor —
Section 4.7 quantifies exactly how conditional.
