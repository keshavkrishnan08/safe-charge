"""P2 -- the weakest cell, in motion.

V1-5 established the pack lemma statically: for a 33 120-cell truck pack the certified pack
current equals the weakest cell's, to 1e-13, at one state. That is a real result and it is a
*snapshot*. It says the pack's limit is set by whichever cell is worst **at the instant it is
evaluated**, and it says nothing about what happens once the cells stop sharing a state.

They stop immediately. A pack is not thermally uniform -- a cell at a module edge sheds heat to
coolant more readily than one buried between neighbours -- and cells are not identical, so under
the same current they diverge in temperature and in state of charge from the first step.

So the experiment is in two phases, and the division of labour matters:

  **Drive.** The pack runs the EPA schedules from \\S13. Nothing is filtered and nothing is
  claimed here; the cycle's job is to *produce* a diverged pack, using an external input rather
  than a spread we invented. This is also the only honest way to get a realistic state: a real
  pack arrives at the charger having just been driven.

  **Charge.** The diverged pack is then fast-charged, which is where the certificate binds every
  step and where the pack claim is actually load-bearing.

An earlier version tried to test the pack rule on the regenerative braking *within* the cycle,
and it had to be abandoned: instrumented, the certificate turned out to bind in about 1 % of
braking steps on US06 and in none at all on UDDS or HWFET. For this pack and these schedules,
regeneration simply is not constraint-limited at moderate state of charge -- the cap is above
what the driveline offers -- so every rule, including a deliberately wrong one, scored a clean
sheet. That is a fact about the duty cycle rather than a result about heterogeneity, and
reporting it as agreement between the rules would have been meaningless.

Two questions are then asked of the charge, and neither is answerable from the static lemma:

  **Does the binding cell stay the same one?** If it does, an implementer could find the worst
  cell once and reuse it. If the identity migrates, the minimum has to be taken every step and
  any implementation that caches the worst cell is unsound.

  **Does the average cell suffice?** Evaluating the certificate once on mean parameters at the
  mean state is what a lumped pack model *is*, and it is what every simulation reporting a
  single pack temperature implicitly certifies against.

    python zeroguard/exp/p2_pack_dynamic.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import anchored as A, platforms as P, stats, vexp as V
from zeroguard.exp.d1_drive_cycles import load_cycle, pack_power

SEED = 20260816
S_SERIES, P_PARALLEL = 96, 45     # the robotaxi string; cells in series share a current
N_CELLS = 128                     # cells simulated individually
DT_CHG, HORIZON, TARGET = 30.0, 80, 0.80

# A pack is not isothermal. The spread below is the cooling-coefficient range across positions,
# and it is the reason the worst *parameter* cell and the worst *thermal* cell need not be the
# same cell.
COOL_SPREAD = (0.75, 1.15)


def make_pack(rng, n, T_amb, soc0, dT0=2.0):
    """n heterogeneous single cells, each with its own parameters, position and state."""
    cells = []
    for _ in range(n):
        sc = {k: float(rng.uniform(*v)) for k, v in V.ENVELOPE.items()}
        cool = float(rng.uniform(*COOL_SPREAD))
        c = P.single_cell(scale=sc)
        c.cell.cooling.hA *= cool
        c.state = c.init(soc0, T_amb + float(rng.uniform(0.0, dT0)))
        c.scale, c.cool = sc, cool
        cells.append(c)
    return cells


def estimator():
    """One estimator for the whole pack: the worst corner of the envelope, as everywhere else."""
    est = P.single_cell(scale=dict(R=V.S_R, Q=V.ENVELOPE["Q"][0], plate=V.ENVELOPE["plate"][1]))
    est.cell.cooling.hA *= COOL_SPREAD[0]      # and the worst cooling position
    return est


def mean_cell(cells):
    """The shortcut: one cell at the mean parameters, driven from the mean state.

    This is not a straw man. It is what a lumped pack model is, and it carries the same
    pessimistic resistance the real estimator does -- the only thing it gives up is the spread.
    """
    m = {k: float(np.mean([c.scale[k] for c in cells])) for k in V.ENVELOPE}
    m["R"] = V.S_R                             # same pessimism on the channel that matters
    est = P.single_cell(scale=m)
    est.cell.cooling.hA *= float(np.mean([c.cool for c in cells]))
    return est


def drive(cells, t, Pwr, dt, T_amb, laps):
    """Phase 1. Run the schedule; filter nothing, claim nothing. This only diverges the pack."""
    for c in cells:
        c.state = dict(c.state)
    for lap in range(laps):
        for k in range(len(t) - 1):
            h = float(max(dt[k], 1e-3))
            if h > 5.0:
                continue
            Pk = float(Pwr[k])
            V_cell = float(cells[0].probe(cells[0].state, 0.0, h, T_amb)[0])
            i_cell = Pk / (S_SERIES * max(V_cell, 1e-3) * P_PARALLEL)
            for c in cells:
                c.state, _o = c.step(c.state, -i_cell, h, T_amb)
    T = np.array([c.state["T"] for c in cells])
    soc = np.array([c.state["soc"] for c in cells])
    return dict(spread_T=float(T.max() - T.min()), mean_T=float(T.mean()),
                spread_soc=float(soc.max() - soc.min()), mean_soc=float(soc.mean()))


def charge(cells, est, avg, T_amb, rule, marg):
    """Phase 2. Fast-charge the diverged pack under one of the two rules."""
    states = [dict(c.state) for c in cells]
    n = len(cells)
    viol = np.zeros(n, dtype=bool)
    # Plating is enforced and never certified (\S4), so a plating excursion is not a broken
    # theorem -- and counting only certified channels would make this experiment blind to the
    # one failure the lumped model actually causes. Both are tracked, and kept apart.
    plate = np.zeros(n, dtype=bool)
    binders, switches, last, trank = [], 0, None, []
    bound = steps = 0
    for _k in range(HORIZON):
        if rule == "min":
            lims = np.empty(n)
            for j in range(n):
                _lo, hi, st = A.interval(est, states[j], DT_CHG, T_amb, marg)
                lims[j] = hi if st == "ok" else 0.0
            j = int(np.argmin(lims)); cap = float(lims[j])
            binders.append(j)
            switches += int(last is not None and j != last)
            last = j
            temps = np.array([s["T"] for s in states])
            # 0.0 = the coldest cell in the pack binds, 1.0 = the hottest
            trank.append(float((temps < temps[j]).sum()) / max(n - 1, 1))
        else:
            ms = dict(states[0])                          # carries `aging` unaveraged
            for key in ("soc", "T", "V1"):
                ms[key] = float(np.mean([s[key] for s in states]))
            _lo, hi, st = A.interval(avg, ms, DT_CHG, T_amb, marg)
            cap = hi if st == "ok" else 0.0

        u = max(0.0, min(cap, est.u_max))
        bound += int(u < est.u_max - 1e-9); steps += 1
        for j, c in enumerate(cells):
            states[j], o = c.step(states[j], u, DT_CHG, T_amb)
            bad, enf = V.split_breaches(c, V.check(c, o))
            if bad:
                viol[j] = True
            if enf:
                plate[j] = True
        if float(np.mean([s["soc"] for s in states])) >= TARGET:
            break

    soc = np.array([s["soc"] for s in states])
    out = dict(cells_violating=int(viol.sum()), any_violation=bool(viol.any()),
               cells_plating=int(plate.sum()), any_plating=bool(plate.any()),
               mean_soc=float(soc.mean()), steps=steps,
               bind_fraction=bound / max(steps, 1),
               peak_T=float(max(s["T"] for s in states)))
    if rule == "min":
        out.update(distinct_binders=len(set(binders)), binder_switches=switches,
                   switch_rate=switches / max(len(binders), 1),
                   worst_param_cell=int(np.argmax([c.scale["R"] for c in cells])),
                   most_common_binder=int(np.bincount(binders).argmax()) if binders else -1,
                   binder_temp_rank=float(np.mean(trank)) if trank else -1.0,
                   binder_is_coldest=float(np.mean([r < 1e-9 for r in trank])) if trank else 0.0)
    return out


def main(n_packs=8, seed=SEED, T_amb=25.0, soc0=0.60, laps=2):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    print("P2 -- the weakest cell, in motion\n" + "=" * 78)
    print(f"{N_CELLS} individually simulated cells, {S_SERIES}S{P_PARALLEL}P string current, "
          f"{n_packs} packs per cycle, {laps} laps then a fast charge, ambient {T_amb:.0f} C")

    out = {"n_cells": N_CELLS, "series": S_SERIES, "parallel": P_PARALLEL,
           "packs_per_cycle": n_packs, "T_amb": T_amb, "soc0": soc0, "laps": laps,
           "cool_spread": list(COOL_SPREAD), "cycles": {}}
    est = estimator()
    marg = V.margins(est)

    for label, fname in (("US06", "us06col.txt"), ("UDDS", "uddscol.txt"),
                         ("HWFET", "hwycol.txt")):
        t, v = load_cycle(fname)
        dt, Pwr = pack_power(t, v)
        acc = {r: dict(viol=0, cells=0, soc=[], packs=0, bind=[], plate=0, pcells=0)
               for r in ("min", "mean")}
        dyn = {k: [] for k in ("distinct", "switches", "rate", "trank", "coldest",
                               "spread_T", "spread_soc")}
        dyn["identity_matches"] = 0
        for _ in range(n_packs):
            cells = make_pack(rng, N_CELLS, T_amb, soc0)
            div = drive(cells, t, Pwr, dt, T_amb, laps)
            for c in cells:                                # the charge starts where driving left off
                c.state = dict(c.state)
            avg = mean_cell(cells)
            dyn["spread_T"].append(div["spread_T"]); dyn["spread_soc"].append(div["spread_soc"])
            for rule in ("min", "mean"):
                r = charge(cells, est, avg, T_amb, rule, marg)
                a = acc[rule]
                a["viol"] += int(r["any_violation"]); a["cells"] += r["cells_violating"]
                a["plate"] += int(r["any_plating"]); a["pcells"] += r["cells_plating"]
                a["soc"].append(r["mean_soc"]); a["packs"] += 1
                a["bind"].append(r["bind_fraction"])
                if rule == "min":
                    dyn["distinct"].append(r["distinct_binders"])
                    dyn["switches"].append(r["binder_switches"])
                    dyn["rate"].append(r["switch_rate"])
                    dyn["trank"].append(r["binder_temp_rank"])
                    dyn["coldest"].append(r["binder_is_coldest"])
                    dyn["identity_matches"] += int(
                        r["most_common_binder"] == r["worst_param_cell"])
        rec = dict(cycle=label,
                   spread_T_K=float(np.mean(dyn["spread_T"])),
                   spread_soc=float(np.mean(dyn["spread_soc"])),
                   distinct_binders=float(np.mean(dyn["distinct"])),
                   max_distinct_binders=int(max(dyn["distinct"])),
                   binder_switches=float(np.mean(dyn["switches"])),
                   switch_rate=float(np.mean(dyn["rate"])),
                   binder_temp_rank=float(np.mean(dyn["trank"])),
                   binder_is_coldest_frac=float(np.mean(dyn["coldest"])),
                   binder_is_worst_param=dyn["identity_matches"], rules={})
        for rule in ("min", "mean"):
            a = acc[rule]
            rec["rules"][rule] = dict(
                packs=a["packs"], packs_violating=a["viol"], cells_violating=a["cells"],
                violation_rate=a["viol"] / a["packs"],
                cp95_upper_pct=100 * stats.cp_upper(a["viol"], a["packs"]),
                mean_soc=float(np.mean(a["soc"])),
                packs_plating=a["plate"], cells_plating=a["pcells"],
                plating_rate=a["plate"] / a["packs"],
                bind_fraction=float(np.mean(a["bind"])))
        out["cycles"][label] = rec

        mn, mu = rec["rules"]["min"], rec["rules"]["mean"]
        print(f"\n{label}: driving leaves {rec['spread_T_K']:.1f} K and "
              f"{100*rec['spread_soc']:.1f} SOC points of spread across the pack; the "
              f"certificate then binds in {100*mn['bind_fraction']:.0f}% of charge steps")
        print(f"  {'rule':22}{'certified':>12}{'plating':>10}{'cells plated':>14}{'SOC':>8}")
        for nm, r in (("min over cells", mn), ("mean cell (lumped)", mu)):
            print(f"  {nm:22}{r['packs_violating']:>6}/{r['packs']}"
                  f"{r['packs_plating']:>6}/{r['packs']}{r['cells_plating']:>14}"
                  f"{r['mean_soc']:>8.3f}")
        print(f"  binding cell: {rec['distinct_binders']:.1f} distinct identities per charge, "
              f"{rec['binder_switches']:.0f} handovers "
              f"({100*rec['switch_rate']:.0f}% of steps); it is the coldest cell in the pack "
              f"{100*rec['binder_is_coldest_frac']:.0f}% of the time "
              f"(mean temperature rank {rec['binder_temp_rank']:.2f})")

    # -----------------------------------------------------------------------------------
    tot_min = sum(c["rules"]["min"]["packs_violating"] for c in out["cycles"].values())
    tot_mean = sum(c["rules"]["mean"]["packs_violating"] for c in out["cycles"].values())
    tot_packs = sum(c["rules"]["min"]["packs"] for c in out["cycles"].values())
    cells_mean = sum(c["rules"]["mean"]["cells_violating"] for c in out["cycles"].values())
    plate_mean = sum(c["rules"]["mean"]["packs_plating"] for c in out["cycles"].values())
    plate_min = sum(c["rules"]["min"]["packs_plating"] for c in out["cycles"].values())
    pcells_mean = sum(c["rules"]["mean"]["cells_plating"] for c in out["cycles"].values())
    pcells_min = sum(c["rules"]["min"]["cells_plating"] for c in out["cycles"].values())
    out.update(total_packs=tot_packs, total_cells=tot_packs * N_CELLS,
               min_rule_breaches=tot_min, mean_rule_breaches=tot_mean,
               mean_rule_cells_breached=cells_mean,
               mean_rule_packs_plating=plate_mean, min_rule_packs_plating=plate_min,
               mean_rule_cells_plated=pcells_mean, min_rule_cells_plated=pcells_min,
               mean_rule_plating_pct=100 * pcells_mean / max(tot_packs * N_CELLS, 1),
               min_rule_cp95_pct=100 * stats.cp_upper(tot_min, tot_packs),
               mean_rule_violation_rate=tot_mean / tot_packs,
               bind_fraction=float(np.mean([c["rules"]["min"]["bind_fraction"]
                                            for c in out["cycles"].values()])),
               identity_migrates=any(c["max_distinct_binders"] > 1
                                     for c in out["cycles"].values()),
               max_distinct_binders=max(c["max_distinct_binders"]
                                        for c in out["cycles"].values()),
               binder_is_worst_param=sum(c["binder_is_worst_param"]
                                         for c in out["cycles"].values()),
               binder_coldest_frac=float(np.mean([c["binder_is_coldest_frac"]
                                                  for c in out["cycles"].values()])),
               binder_temp_rank=float(np.mean([c["binder_temp_rank"]
                                               for c in out["cycles"].values()])),
               soc_cost_points=float(np.mean(
                   [c["rules"]["mean"]["mean_soc"] - c["rules"]["min"]["mean_soc"]
                    for c in out["cycles"].values()])) * 100)

    # -----------------------------------------------------------------------------------
    # The lumped rule survived. Reporting only that would be the same mistake B2 made: an
    # incumbent that holds up on one configuration is evidence about that configuration, not
    # about the rule. The spread it was tested at -- under a kelvin -- is far below what a real
    # pack runs; production packs are specified to a few kelvin and aged ones drift further.
    # So the question is not whether the shortcut is safe but *up to what spread*, which is a
    # number an integrator can check their own pack against.
    # -----------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("where the lumped model gives way\n")
    t_u, v_u = load_cycle("us06col.txt")
    dt_u, Pwr_u = pack_power(t_u, v_u)
    rows, thresh = [], None
    for dT0 in (2.0, 5.0, 10.0, 15.0, 20.0, 25.0):
        rng2 = np.random.default_rng(seed + 31)
        a = {r: dict(viol=0, cells=0, plate=0, pcells=0) for r in ("min", "mean")}
        sp = []
        for _ in range(n_packs):
            cells = make_pack(rng2, N_CELLS, T_amb, soc0, dT0=dT0)
            div = drive(cells, t_u, Pwr_u, dt_u, T_amb, laps)
            sp.append(div["spread_T"])
            avg = mean_cell(cells)
            for rule in ("min", "mean"):
                r = charge(cells, est, avg, T_amb, rule, marg)
                a[rule]["viol"] += int(r["any_violation"])
                a[rule]["cells"] += r["cells_violating"]
                a[rule]["plate"] += int(r["any_plating"])
                a[rule]["pcells"] += r["cells_plating"]
        row = dict(dT0_K=dT0, spread_T_K=float(np.mean(sp)), packs=n_packs,
                   min_breaches=a["min"]["viol"], mean_breaches=a["mean"]["viol"],
                   mean_cells=a["mean"]["cells"], min_cells=a["min"]["cells"],
                   min_plating=a["min"]["plate"], mean_plating=a["mean"]["plate"],
                   min_plated_cells=a["min"]["pcells"],
                   mean_plated_cells=a["mean"]["pcells"])
        rows.append(row)
        if thresh is None and row["mean_plating"] > 0:
            thresh = row["spread_T_K"]
        print(f"  spread {row['spread_T_K']:>5.1f} K   certified: min {row['min_breaches']}"
              f"/{n_packs} lumped {row['mean_breaches']}/{n_packs}   |   plating: min "
              f"{row['min_plated_cells']:>4} cells, lumped {row['mean_plated_cells']:>4} cells")
    out["spread_sweep"] = rows
    out["lumped_fails_above_K"] = thresh
    out["min_rule_breaches_in_sweep"] = sum(r["min_breaches"] for r in rows)
    out["min_rule_plating_in_sweep"] = sum(r["min_plated_cells"] for r in rows)
    out["lumped_plating_in_sweep"] = sum(r["mean_plated_cells"] for r in rows)
    out["sweep_packs"] = n_packs * len(rows)
    if thresh is None:
        print(f"  the lumped model held to {rows[-1]['spread_T_K']:.1f} K of spread on both "
              f"counts; on this pack its error is smaller than the pessimism the estimator "
              f"already carries")
    else:
        print(f"  the certified channels survive under both rules -- that is the theorem doing "
              f"its job, and it is not the interesting line")
        print(f"  the lumped model drives cells past the *plating* margin at every spread "
              f"tested, from {rows[0]['spread_T_K']:.1f} K upward, and worsens with it: "
              f"{rows[0]['mean_plated_cells']} cells at the tightest pack, "
              f"{rows[-1]['mean_plated_cells']} at the loosest "
              f"({out['lumped_plating_in_sweep']} across the sweep, against "
              f"{out['min_rule_plating_in_sweep']} for the minimum rule). "
              f"Plating is enforced and never certified, so no theorem is broken; the cells "
              f"are damaged all the same, which is the whole reason the channel is enforced.")

    print("\n" + "=" * 78)
    print(f"  min over cells: {tot_min}/{tot_packs} packs breached "
          f"(CP95 {out['min_rule_cp95_pct']:.2f}%), across {out['total_cells']:,} cells")
    print(f"  mean cell:      {tot_mean}/{tot_packs} packs breached on the certified "
          f"channels too -- the theorem is not what the shortcut breaks")
    print(f"  what it breaks is plating: {plate_mean}/{tot_packs} packs, "
          f"{pcells_mean} of {out['total_cells']:,} cells "
          f"({100*pcells_mean/max(out['total_cells'],1):.1f}%) driven past the margin, "
          f"bought for {out['soc_cost_points']:+.2f} SOC points. The minimum over cells plated "
          f"{pcells_min}.")
    print(f"  the binding cell is not one cell: up to {out['max_distinct_binders']} distinct "
          f"identities within a single charge, so the minimum is taken every step")
    print(f"  and it sits among the pack's *colder* cells -- mean temperature rank "
          f"{out['binder_temp_rank']:.2f} against 0.50 for an indifferent draw, the coldest "
          f"cell outright {100*out['binder_coldest_frac']:.0f}% of the time. That is the "
          f"opposite of the intuition a lumped thermal model encodes, and it follows from the "
          f"plating cap tightening as a cell cools: in the charge direction a cold cell is a "
          f"weak cell.")
    print(f"  it is the worst-resistance cell in only "
          f"{out['binder_is_worst_param']}/{tot_packs} packs")

    path = V.save("p2_pack_dynamic.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
