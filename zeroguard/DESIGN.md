# ZEROGUARD — Experimental Design

**ZEROGUARD: Monotone Control Theory for Certified Battery Fast Charging in Long-Duration
Autonomous Vehicles**

This document fixes the entire experimental surface before any code runs: what is claimed,
what would falsify it, how it is measured, and which statistical test decides. Nothing is
reported that is not produced by a script in `zeroguard/exp/` and recorded in
`zeroguard/results/`.

---

## 0. Scope and honesty boundaries

The Doyle–Fuller–Newman pipeline (PyBaMM) that validated the IECON paper is **not** in this
repository. Every result below is therefore **ROM-level or ROM-versus-ROM**. The DFN evidence
already published in the IECON paper stands on its own and is cited, not re-derived. Where a
new claim would benefit from DFN replay, that is stated as a limitation rather than papered
over.

Two further boundaries, carried from the IECON paper and not weakened here:

- The certified set is `S = {T <= T_max, V <= V_max}`. Plating is **enforced and monitored**,
  never certified, because `I = 0` does not clear the plating margin above ~0.82 SOC.
- The ROM's explicit-Euler RC branch requires `dt <= tau_RC`. The exact zero-order-hold branch
  (`rc="exact"`) is used wherever a step longer than `tau_RC` is studied.

---

## 1. The central claim

The IECON paper proves two propositions about a battery. The claim of this work is that
**neither proposition is about batteries.**

> **Theorem (Null-Input Collapse).** Let a system have state `x`, scalar input `u >= 0`, and
> constraints `g_c(x, u) <= 0` for `c = 1..m`. If
> **(A1)** `u = 0` is *safe* — from every `x` in the safe set `S`, the one-step map at `u = 0`
> returns to `S`; and
> **(A2)** each `g_c` is *non-decreasing in `u`*,
> then the admissible input set is a single interval `[0, u*]` anchored at zero, and `u*` is
> found exactly by bisection in `O(log 1/eps)` one-step evaluations — with no optimizer, no
> convergence test, and no identification of the plant's uncertain parameters.

The battery is one instance. If the theorem is right, the *same filter code* certifies systems
with no electrochemistry in them at all — and fails, detectably, exactly when (A1) or (A2) is
broken.

That is what Act I tests, and it is the difference between a battery method and a
characterization of a class of control problems.

---

## 2. Experiment register

Each row states the claim, the falsifier, and the decision rule. `n` is the number of
independent runs. All seeds are fixed and recorded.

### Act I — Generality of the principle

| ID | Claim | Falsifier | Design | Decision rule |
|----|-------|-----------|--------|---------------|
| **E1** | The generic filter certifies four physically unrelated systems, unmodified | Any violation on any system | 4 systems × 2000 randomized initial states × full episode | 0 violations required on all four; CP-95 upper bound reported per system |
| **E1b** | The generic filter reduces *exactly* to the published `project_current` on the battery | Any bit-level disagreement | 60 000 (state, current) pairs, 3 cell scales, 2 discretizations | bit-exact equality required |
| **E2** | The principle fails *detectably* when (A1) breaks | Filter silently commands an unsafe input | Quadrotor in hover: `u=0` is fatal | filter must report infeasible, never a false certificate |
| **E3** | The principle fails *detectably* when (A2) breaks | Same | Synthetic non-monotone constraint | bisection must be shown to miss the true `u*`; we detect and flag |

### Act II — Invariance to the physics of cooling (the spacecraft case)

| ID | Claim | Falsifier | Design | Decision rule |
|----|-------|-----------|--------|---------------|
| **E4** | The certificate survives replacing Newtonian cooling with Stefan–Boltzmann radiation | Any violation, or loss of monotonicity | radiative ROM, 5-channel envelope, n = 20 000 | 0 violations; monotonicity in `T` verified analytically and numerically |
| **E5** | Safety degrades gracefully to the vacuum limit | A violation as `hA -> 0` | sweep `hA` over 5 decades to zero | 0 violations at every point; delivered SOC monotone in `hA` |
| **E6** | Radiation-induced parameter drift is just another monotone channel | Violation under the 6th channel | 6-channel envelope, n = 20 000 | 0 violations; CP-95 upper bound reported |

### Act III — Long duration (what the title promises)

| ID | Claim | Falsifier | Design | Decision rule |
|----|-------|-----------|--------|---------------|
| **E7** | Zero violations across a full mission life with **no recalibration ever** | Any violation | 27 000 charge cycles (15/day × 5 y, satellite duty) | 0 violations end to end |
| **E8** | The estimator buys charge, never safety | Safety differs between recalibrated and never-recalibrated | paired runs, same seeds | violations equal (both 0); delivered-SOC gap significant by paired test |

### Act IV — Per-cell architecture

| ID | Claim | Falsifier | Design | Decision rule |
|----|-------|-----------|--------|---------------|
| **E9** | For series packs the admissible set is the intersection, and the weakest cell dominates | `I*` for the pack != min over cells | N = 1..1024, heterogeneous aging | exact agreement to 1e-9 |
| **E10** | Per-cell filters catch a single faulted cell that a pack-averaged filter misses | Pack-averaged filter also catches it | inject one anomalous cell in packs of 100 | per-cell 0 violations, pack-averaged > 0 |

### Act V — Adversarial rigour

| ID | Claim | Falsifier | Design | Decision rule |
|----|-------|-----------|--------|---------------|
| **E11** | An adversary optimizing to break the filter cannot | Any violation | policy maximizing peak `T`, then peak `V`, then both | 0 violations |
| **E12** | Safety survives corrupted sensing | Violation under noise/bias/dropout/quantization | 4 corruption modes × severities, n = 8000 | 0 violations across all |
| **E13** | Safety survives scheduler jitter | Violation under `dt` jitter | `dt` jitter ±30 %, exact-ZOH ROM | 0 violations |
| **E14** | Safety survives reduced arithmetic | Violation in float32 | full re-run in float32 | 0 violations; max deviation reported |

### Act VI — Ablations and micro-causation

Each ablation removes exactly one mechanism and reports the consequence. This is what
establishes that every component is *load-bearing* rather than decorative.

| ID | Mechanism removed | Expected consequence |
|----|-------------------|----------------------|
| **A1** | the `I=0` feasibility short-circuit | filter can no longer detect already-unsafe states |
| **A2** | the conservative bound (use nominal `s_R = 1`) | violations appear as the cell ages |
| **A3** | the cooling reserve `f` | violations appear under cooling faults |
| **A4** | the resistance throttle `K` | thermal margin collapses at end of life |
| **A5** | the plating current cap | plating margin breached in the cold corner |
| **A6** | monotone bound → point estimate | violations under estimator error |
| **A7** | fixed 18 iterations → early exit on tolerance | run-time becomes data-dependent (the QP failure mode) |

### Act VII — Statistical treatment

Every safety claim is a Bernoulli proportion with zero observed failures, so:

- **Clopper–Pearson 95 % one-sided upper bound** on the violation rate, `1 - alpha^(1/n)`.
- **Bootstrap 95 % CI** (10 000 resamples) for every continuous outcome (delivered SOC, peak
  temperature, latency).
- **Paired tests** where the same seed is run under two conditions (E8, ablations): Wilcoxon
  signed-rank, with Hodges–Lehmann median shift and rank-biserial effect size.
- **Rank correlation** (Spearman) for monotonicity claims across sweeps, with permutation p.
- **Sample-size adequacy**: for each headline claim we report the `n` required to certify the
  target rate at 95 % and confirm the run exceeds it.

No p-value is reported without an effect size and a confidence interval.

---

## 3. Figure program

Ten figures, each carrying a claim rather than decorating one. The register below is the
delivered set; where it diverges from the first draft of this document, the divergence is
recorded rather than silently corrected.

1. **F1** Cross-domain certification — four systems, trajectory and the one-step constraint map
2. **F2** The boundary — where (A1) and (A2) break, and how differently
3. **F3** Cooling-law invariance — Newtonian vs radiative, and the vacuum limit
4. **F4** 3-D surface: violation-rate upper bound over (throttle × resistance scale), plus the
   (sensor bias × cooling loss) plane, with the certified boundary drawn on both
5. **F5** Mission-life run — 31 000 cycles, bound versus oracle
6. **F6** Pack scaling — the weakest-cell lemma, and fault localization
7. **F7** Ablation ladder — each mechanism removed, plus the throttle dose-response
8. **F8** Margin purchasing power — the bias sweep against the derived breakpoint
9. **F9** Adversarial and corrupted-sensing robustness
10. **F10** Latency, footprint, and sample adequacy

Two changes from the draft register. Fault localization was promised its own figure and is
instead panel (b) of **F6**, beside the lemma it follows from. And the draft's ninth slot was
latency; the delivered **F9** is robustness and latency moved to **F10**, where it sits with
the sample-adequacy audit that belongs next to it.

### Late additions (E11, E12)

Auditing this register against the delivered experiments found three gaps, all closed:

| ID | What was missing | Where it is now |
|----|------------------|-----------------|
| **E11a** | the six-channel vacuum envelope (radiation drift) promised as E6; the first pass ran five | `exp/e11_closure.py` |
| **E11b** | `stats.n_required` was written and never called, so no claim was checked for adequacy | `exp/e11_closure.py`, figure F10(c) |
| **E11c** | the latency evidence never reached a figure | `exp/e11_closure.py`, figure F10(a,b) |

E11b then found a fourth problem of its own: the adversarial claim rested on 40 cells, a
Clopper–Pearson upper bound of 7.2 %, which certifies nothing. **E12** re-runs the same attack
against 320 independently drawn cells — past the 299 needed to certify below 1 % — because the
statistical unit of that claim is the cell, not the sequence.

## 4. What would make this fail

Stated in advance so the result cannot be rationalized after the fact:

- If the generic filter violates on any non-battery system, the theorem is wrong as stated and
  the paper becomes a battery paper again.
- If the radiative certificate violates, the invariance claim dies and the spacecraft framing
  goes with it.
- If the pack `I*` disagrees with the min over cells, the per-cell architecture argument is
  unfounded.
- If an ablation removes a mechanism with no consequence, that mechanism is decorative and
  should be deleted from the method rather than defended.
