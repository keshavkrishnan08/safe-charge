# ZEROGUARD

**Monotone Control Theory for Certified Battery Fast Charging in Long-Duration Autonomous
Vehicles**

An extension of the IECON work in `../safe_charge/`, which asks a question the original paper
did not: *is any of this actually about batteries?*

The answer is no, and the consequences are the subject here. The IECON filter needs exactly
two things from a system — a null input that is safe, and constraints that are monotone in
that input — and neither is electrochemical. Stated that way it is a theorem about a class of
control problems, and the battery is one instance.

```bash
python zeroguard/run_all.py          # every experiment, then the figures, then the claim check
python zeroguard/verify_claims.py    # re-read all 80 claims in PAPER.md out of the results
```

| file | what it is |
|---|---|
| [`DESIGN.md`](DESIGN.md) | the experimental surface, written **before** anything ran, including what would falsify each claim |
| [`PAPER.md`](PAPER.md) | the manuscript |
| `systems.py` | four monotone systems from unrelated physics, plus two that break one hypothesis each |
| `gfilter.py` | the projection, written once for any monotone system |
| `stats.py` | Clopper–Pearson, bootstrap, Wilcoxon with effect sizes, permutation Spearman |
| `exp/` | one script per experiment; each writes JSON to `results/` |
| `figures.py` | nine figures, each redrawn from a results file |
| `verify_claims.py` | fails loudly if the manuscript and the data disagree |

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

**Two things in the original method were described wrongly**, and the ablations found both.
The *I* = 0 short-circuit is not a safety mechanism — the bisection is self-correcting when
every input is infeasible, and returns zero regardless; what the check buys is 17 model
evaluations. And the throttle *K* = 15, which looked over-conservative on cells inside the
rated bound, turns out to be the *smallest* value that certifies cells past it.

## Scope

Every number here is reduced-order-model level or model-versus-model. The PyBaMM
Doyle–Fuller–Newman pipeline behind the IECON paper is not part of this release, so its
evidence is cited rather than re-derived. Section 5 of `PAPER.md` states the limits in full,
including the two the original paper already carried: the certified set is {*T*, *V*} and
plating is enforced rather than certified, and the certificate is conditional on the sensor —
Section 4.7 quantifies exactly how conditional.
