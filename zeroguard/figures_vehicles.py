"""Vehicle figures for ZEROGUARD: the four media, and what connects them.

Eight figures, drawn from `results/v1..v4_*.json` and `results/x_crossdomain.json`. Same rules
as `figures.py`: the validated Okabe-Ito order, a distinct marker per series so identity
survives greyscale, one y-scale per panel, nothing smoothed, nothing drawn by hand.

Two conventions specific to these figures, because the vehicle results have a distinction the
cell-level ones did not:

  **closure is not violation.** An empty interval means the certificate is correctly reporting
  that the load can no longer be served inside the envelope. It is drawn in amber. A *breach
  while the certificate still says feasible* is the actual failure mode and is drawn in red --
  and every red count in this file is zero, which is the point.

  **the anchor is always marked.** On any panel showing an admissible interval, the anchor is
  drawn as a horizontal line, because the entire difference between this work and the original
  is that the line is not at zero.

    python zeroguard/figures_vehicles.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from zeroguard.figures import C, M, INK, MUTED, GRID, SEQ, COL, DBL, style, save

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

DOMAIN_C = dict(ground=C[0], aerial=C[1], underwater=C[2], space=C[3], reference=C[4])
DOMAIN_M = dict(ground="o", aerial="^", underwater="s", space="D", reference="v")
AMBER, RED = "#E69F00", "#C1272D"


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def tag(ax, txt, y=1.02):
    """Panel title, wrapped to the panel.

    `figures._tag` sets a left-aligned axes title, which is right for the short labels in the
    cell-level figures and wrong here: these panels carry sentences, and at three-to-a-row on a
    7.16 in page the titles ran straight into their neighbours. This one clips the title to the
    axes and wraps it instead."""
    ax.set_title(txt, loc="left", pad=3, fontsize=7.4, wrap=True)
    ax.title.set_position((0.0, y))


# =======================================================================================
def fv1_domains():
    """The anchor moves: what the null-input filter does on each of the four media."""
    v1, v2 = load("v1_ground.json"), load("v2_aerial.json")
    v3, v4 = load("v3_underwater.json"), load("v4_space.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.9), constrained_layout=True)

    # (a) anchor magnitude per platform, log scale
    ax = axes[0]
    rows = []
    for k, v in v1["v1_11_v2g"].items():
        rows.append(("ground", f"V2G {v['export_kW']:.0f} kW", v["anchor_A"]))
    for k, v in v3["v3_3_hotel_anchor"].items():
        rows.append(("underwater", k, v["mean_anchor_A"]))
    for k, v in v4["v4_11_bus_anchor"].items():
        rows.append(("space", k.replace("-", " "), v["mean_anchor_A"]))
    q = v2["v2_1_anchor_vs_null"]
    # read the hover anchors off the platforms rather than hard-coding them, so a change to
    # the airframe cannot leave the figure quietly describing an aircraft that no longer exists
    from zeroguard import platforms as _P
    for cls, lab in ((_P.DeliveryQuadrotor, "quadrotor hover"),
                     (_P.EVTOLAirTaxi, "eVTOL hover")):
        pl = cls()
        rows.append(("aerial", lab, pl.anchor(pl.init())))
    rows.sort(key=lambda r: r[2])
    y = np.arange(len(rows))
    for i, (dom, lab, val) in enumerate(rows):
        ax.barh(i, max(val, 1e-3), color=DOMAIN_C[dom], height=0.68, alpha=0.9)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], fontsize=5.6)
    ax.set_xscale("log"); ax.set_xlabel("anchor current $u_a$  [A]")
    ax.axvline(1e-3, color=INK, lw=1.0, ls="--")
    ax.text(0.02, 0.02, "null-input theorem lives here", transform=ax.transAxes,
            fontsize=5.4, color=INK, va="bottom", ha="left")
    tag(ax, "(a)  every vehicle's irreducible load")

    # (b) null-input brownout rate vs anchored
    ax = axes[1]
    labels, ni, an = [], [], []
    for k, v in v1["v1_11_v2g"].items():
        labels.append(f"V2G\n{v['export_kW']:.0f} kW")
        ni.append(100 * v["null_input_brownout_rate"])
        an.append(100 * v["anchored_unsafe_while_certified"] / v["trials"])
    labels.append("quadrotor\nhover")
    ni.append(100 * q["null_input_brownout_rate"])
    an.append(100 * q["anchored_unsafe_while_certified"] / q["trials"])
    for k, v in list(v3["v3_3_hotel_anchor"].items())[:1]:
        labels.append("AUV\nseabed")
        ni.append(100 * v["null_input_brownouts"] / v["trials"])
        an.append(100 * v["unsafe_while_certified"] / v["trials"])
    for k, v in list(v4["v4_11_bus_anchor"].items())[:2]:
        labels.append(k.replace("-", "\n"))
        ni.append(100 * v["null_input_brownouts"] / v["trials"])
        an.append(100 * v["unsafe_while_certified"] / v["trials"])
    x = np.arange(len(labels))
    ax.bar(x - 0.19, ni, 0.36, color=RED, label="null-input: load starved", alpha=0.9)
    ax.bar(x + 0.19, an, 0.36, color=C[2], label="anchored: unsafe while certified")
    for i, v in enumerate(an):
        if v == 0:
            ax.text(x[i] + 0.19, 3, "0", ha="center", fontsize=6, color=C[2], weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=5.4,
                       rotation=38, ha="right")
    ax.set_ylabel("episodes  [%]"); ax.set_ylim(0, 128)
    ax.legend(loc="upper left", fontsize=5.4, ncol=1)
    tag(ax, "(b)  the generalisation is not optional")

    # (c) the C-rate frontier: where the calibrated cell refuses to fly
    ax = axes[2]
    fr = q["frontier"]
    cr = [f["c_rate"] for f in fr]
    wd = [f["width"] for f in fr]
    ok = [f["status"] == "ok" for f in fr]
    ax.plot(cr, wd, "-", color=C[0], lw=1.8, zorder=2)
    ax.scatter([c for c, o in zip(cr, ok) if o], [w for w, o in zip(wd, ok) if o],
               s=26, color=C[0], marker="o", zorder=3, label="interval non-empty")
    ax.scatter([c for c, o in zip(cr, ok) if not o], [w for w, o in zip(wd, ok) if not o],
               s=34, color=RED, marker="X", zorder=3, label="refused: interval empty")
    b = q["energy_cell_refuses_above_c"]
    if b:
        ax.axvline(b, color=RED, lw=1.0, ls="--")
        ax.text(b + 0.04, ax.get_ylim()[1] * 0.62, f"refusal at {b:.1f}C",
                fontsize=6, color=RED)
    ax.set_xlabel("hover draw  [C]"); ax.set_ylabel("admissible width  [A]")
    ax.legend(loc="upper right", fontsize=5.8)
    tag(ax, "(c)  a design tool, not a guard")
    fig.suptitle("The anchor moves off zero, and three of the four media require it",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV1_domains")


# =======================================================================================
def fv2_reserve():
    """The reserve: does interval width warn, and does it warn in time?"""
    v2, v3 = load("v2_aerial.json"), load("v3_underwater.json")
    x = load("x_crossdomain.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.9), constrained_layout=True)

    # (a) closure is too late under ice, and the reserve threshold that fixes it
    ax = axes[0]
    a = v2["v2_3_reserve_lead"]; u = v3["v3_7_under_ice"]
    sw = u["reserve_trigger_sweep"]
    th = [s["threshold_A"] for s in sw]
    p05 = [s["p05_warning_min"] for s in sw]
    med = [s["median_warning_min"] for s in sw]
    ax.fill_between(th, p05, med, color=C[2], alpha=0.20)
    ax.plot(th, med, "-o", color=C[2], markersize=4, label="median warning")
    ax.plot(th, p05, "--s", color=C[2], markersize=3.5, alpha=0.8, label="5th percentile")
    ax.axhline(u["transit_min"], color=RED, ls="--", lw=1.1)
    ax.text(th[-1], u["transit_min"] * 1.08, f"{u['transit_min']:.0f} min transit to a hole",
            fontsize=5.6, color=RED, ha="right")
    ct = u["closure_trigger"]
    ax.plot([0], [ct["lead_median_min"]], "X", color=AMBER, markersize=8, zorder=5)
    ax.text(1.0, ct["lead_median_min"] + 3.0,
            f"waiting for closure: {ct['lead_median_min']:.0f} min, too late",
            fontsize=5.4, color=AMBER, va="bottom")
    need = u["threshold_for_transit_A"]
    if need is not None:
        ax.axvline(need, color=C[0], lw=1.0, ls=":")
        ax.text(need * 1.05, 1.2, f"abort at {need:.0f} A", fontsize=5.8, color=C[0])
    ax.set_xlabel("abort threshold on the reserve  [A]")
    ax.set_ylabel("warning before closure  [min]")
    ax.legend(loc="lower right", fontsize=5.8)
    tag(ax, "(a)  closure is the wrong trigger")

    # (b) two clocks, and which one each platform runs on
    ax = axes[1]
    pc = x["x4_reserve"]["per_case"]
    names = list(pc)
    w_rho = [pc[k]["width_vs_endurance"]["rho"] for k in names]
    e_rho = [pc[k]["energy_clock_vs_endurance"]["rho"] for k in names]
    y = np.arange(len(names))
    ax.barh(y - 0.19, w_rho, 0.36, color=C[0], label=r"width:  $t_{envelope}$")
    ax.barh(y + 0.19, e_rho, 0.36, color=C[3], label=r"charge:  $t_{energy}$")
    for i, k in enumerate(names):
        best = max(w_rho[i], e_rho[i])
        ax.plot([best], [y[i] + (0.19 if e_rho[i] > w_rho[i] else -0.19)], "*",
                color=C[2], markersize=7, zorder=5,
                label="better clock" if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n.replace("-", " ") for n in names], fontsize=5.6)
    ax.set_xlabel(r"Spearman $\rho$ against endurance"); ax.set_xlim(-0.65, 1.15)
    ax.legend(loc="upper left", fontsize=5.2, framealpha=0.95)
    ax.text(0.98, 0.03, "act on the minimum of the two", transform=ax.transAxes,
            fontsize=5.0, color=MUTED, ha="right")
    tag(ax, "(b)  two clocks, not one")

    # (c) breaches with no warning, everywhere: all zero
    ax = axes[2]
    src = [("quadrotor", a["breaches_with_no_warning"], a["trials"], C[1]),
           ("gusts", v2["v2_6_gusts"]["total_unsafe"],
            v2["v2_6_gusts"]["total_trials"], C[1]),
           ("motor-out", v2["v2_8_motor_out"]["total_unsafe"],
            v2["v2_8_motor_out"]["total_trials"], C[1]),
           ("under-ice", u["breaches_with_no_warning"], u["trials"], C[2]),
           ("AUV bursts", v3["v3_8_transients"]["total_unsafe"],
            v3["v3_8_transients"]["total_trials"], C[2]),
           ("adversary", v2["v2_11_adversary"]["violations"],
            v2["v2_11_adversary"]["trials"], C[5])]
    y = np.arange(len(src))
    ub = [100 * (1 - 0.05 ** (1.0 / n)) if k == 0 else 100 * k / n for _, k, n, _ in src]
    for i, (lab, k, n, col) in enumerate(src):
        ax.barh(i, ub[i], color=col, height=0.6, alpha=0.9)
        ax.text(ub[i] * 1.15, i, f"{k}/{n:,}", fontsize=5.8, va="center", color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in src], fontsize=6)
    ax.set_xscale("log"); ax.set_xlim(0.05, 90)
    ax.axvline(1.0, color=INK, ls="--", lw=1.0)
    ax.text(1.15, -0.72, "certifies below 1 %", fontsize=5.8, color=INK)
    ax.set_xlabel("Clopper-Pearson 95 % upper bound  [%]")
    tag(ax, "(c)  breaches with no prior warning")
    fig.suptitle("The reserve warns, and it is one of two clocks the certificate carries",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV2_reserve")


# =======================================================================================
def fv3_sortie():
    """A delivery sortie, phase by phase: the interval opens in cruise and closes over the drop."""
    v2 = load("v2_aerial.json")
    s = v2["v2_4_sortie"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.7), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    names = [p["name"] for p in s["profile"]]
    mult = [p["power_mult"] for p in s["profile"]]
    secs = [p["seconds"] for p in s["profile"]]
    res = [s["phase_reserve"][n]["mean_width"] for n in names]
    edges = np.concatenate([[0], np.cumsum(secs)])
    mid = 0.5 * (edges[:-1] + edges[1:])
    ax.step(np.append(edges, edges[-1]), np.append(np.append(mult, mult[-1]), mult[-1]),
            where="post", color=C[0], lw=1.6, label="power demand / hover")
    ax.set_xlabel("mission time  [s]"); ax.set_ylabel("power demand  [x hover]")
    ax.set_ylim(0, 2.1)
    ax2 = ax.twinx()
    ax2.plot(mid, res, "-o", color=C[1], lw=1.8, markersize=4.5, label="admissible width")
    ax2.set_ylabel("reserve  [A]", color=C[1]); ax2.tick_params(axis="y", colors=C[1])
    ax2.grid(False)
    for i, n in enumerate(names):
        ax.text(mid[i], 0.06, n, rotation=90, fontsize=5.4, ha="center", va="bottom",
                color=MUTED)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6)
    tag(ax, "(a)  reserve tracks the mission, not a plan")

    ax = axes[1]
    labs = ["completed", "closed early"]
    vals = [100 * s["completion_rate"], 100 * (1 - s["completion_rate"])]
    ax.bar([0], [vals[0]], 0.55, color=C[2], label="sortie completed")
    ax.bar([0], [vals[1]], 0.55, bottom=[vals[0]], color=AMBER,
           label="closed early (correctly)")
    ax.bar([1], [100 * s["unsafe_while_certified"] / s["trials"]], 0.55, color=RED)
    ax.text(1, 4, f"{s['unsafe_while_certified']}", ha="center", fontsize=8, weight="bold",
            color=RED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"outcome\n(n={s['trials']})", "unsafe while\ncertified"], fontsize=6)
    ax.set_ylabel("sorties  [%]"); ax.set_ylim(0, 108)
    ax.legend(loc="center right", fontsize=5.8)
    tag(ax, "(b)  closure is not violation")
    fig.suptitle("A whole delivery sortie: climb, cruise, hover over the drop, return",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV3_sortie")


# =======================================================================================
def fv4_environments():
    """Four media, four cooling laws, and the sweeps that straddle each failure point."""
    v1, v3, v4 = load("v1_ground.json"), load("v3_underwater.json"), load("v4_space.json")
    fig, axes = plt.subplots(1, 4, figsize=(DBL, 2.75), constrained_layout=True)

    # (a) ground: the passive ceiling, and what the loop buys
    ax = axes[0]
    amb = v1["v1_3_ambient"]
    for key, col, mk, lab in (("passive", C[1], "s", "coupled to ambient"),
                              ("active-25C", C[0], "o", "coolant held at 25 C")):
        r = amb[key]["rows"]
        ax.plot([x["T_amb"] for x in r], [x["mean_soc"] for x in r], "-", marker=mk,
                color=col, label=lab, markersize=4)
    thr = amb["predicted_passive_ceiling_C"]
    ax.axvline(thr, color=RED, ls="--", lw=1.0)
    ax.text(0.97, 0.97, f"$T_{{max}}-\\delta_T$ = {thr:.1f} C", transform=ax.transAxes,
            fontsize=5.4, color=RED, ha="right", va="top")
    ax.set_xlabel("ambient  [C]"); ax.set_ylabel("SOC delivered")
    ax.legend(loc="lower left", fontsize=5.6)
    tag(ax, "(a)  ground")

    # (b) underwater: plating binds everywhere, so the informative split is *within* it
    ax = axes[1]
    rows = v3["v3_2_cold_binding"]["rows"]
    T = [r["T_water"] for r in rows]
    cap = [100 * r["plating_cap_fraction"] for r in rows]
    marg = [100 * (r["plating_fraction"] - r["plating_cap_fraction"]) for r in rows]
    th = [100 * r["thermal_fraction"] for r in rows]
    ax.stackplot(T, cap, marg, th, colors=[C[3], C[4], C[0]], alpha=0.92,
                 labels=["plating: current cap", "plating: margin", "thermal"])
    ax.set_xlabel("water temperature  [C]"); ax.set_ylabel("binding constraint  [%]")
    ax.set_ylim(0, 100); ax.set_xlim(min(T), max(T))
    ax.legend(loc="lower left", fontsize=5.2)
    ax.text(0.97, 0.94, "thermal never binds\nin a sealed hull", transform=ax.transAxes,
            fontsize=5.2, color=INK, ha="right", va="top")
    tag(ax, "(b)  underwater")

    # (c) space: radiator degradation to the vacuum limit
    ax = axes[2]
    rows = v4["v4_10_radiator"]["rows"]
    f = [r["radiator_fraction"] for r in rows]
    ax.plot(f, [r["mean_soc"] for r in rows], "-o", color=C[3], markersize=4)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"radiator $\varepsilon A$  [x nominal]"); ax.set_ylabel("SOC delivered")
    tv = v4["v4_10_radiator"]["total_violations"]
    ax.text(0.97, 0.06, f"{tv} violations in "
                        f"{v4['v4_10_radiator']['total_trials']:,}",
            transform=ax.transAxes, fontsize=5.4, color=C[2], ha="right")
    ax.text(0.03, 0.94, "cold-limited: a smaller\nradiator delivers more",
            transform=ax.transAxes, fontsize=5.2, color=MUTED, va="top")
    tag(ax, "(c)  space")

    # (d) lunar night: capacity does nothing, insulation does everything
    ax = axes[3]
    ln = v4["v4_5_lunar_night"]
    ins = ln["insulation"]
    eps = [r["eps"] for r in ins]
    hrs = [r["median_survival_h"] for r in ins]
    ax.plot(eps, hrs, "-o", color=C[3], markersize=4.5, label="45 W, 2.9 kWh")
    packs = sorted({r["pack_mult"] for r in ln["rows"]})
    base = [r for r in ln["rows"] if r["heater_W"] == 45.0]
    base.sort(key=lambda r: r["pack_kWh"])
    ax.plot([0.72] * len(base), [r["median_survival_h"] for r in base], "s",
            color=C[1], markersize=4, label="4x the pack, same $\\varepsilon$")
    ax.axhline(ln["night_hours"], color=RED, ls="--", lw=1.0)
    ax.text(0.03, 0.95, "354 h lunar night", transform=ax.transAxes, fontsize=5.2,
            color=RED, va="top")
    ax.axhline(ln["energy_limit_h"], color=MUTED, ls=":", lw=1.0)
    ax.text(0.03, 0.06, f"energy ceiling {ln['energy_limit_h']:.0f} h",
            transform=ax.transAxes, fontsize=5.2, color=MUTED)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_yscale("log")
    # an inverted log axis over 0.02-0.72 puts a decade of minor ticks on top of each other;
    # label the sweep's own nodes instead
    ax.set_xticks(eps); ax.set_xticklabels([f"{e:.2f}" for e in eps], fontsize=5.0)
    ax.minorticks_off()
    ax.set_xlabel(r"radiator emissivity $\varepsilon$")
    ax.set_ylabel("survival  [h]")
    ax.legend(loc="center left", fontsize=5.2)
    tag(ax, "(d)  lunar night")

    fig.suptitle("Four media, four cooling laws, one unmodified projection",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV4_environments")


# =======================================================================================
def fv5_missions():
    """Duty cycle: the certificate has to hold on the last cycle, not the first."""
    v1, v2 = load("v1_ground.json"), load("v2_aerial.json")
    v3, v4 = load("v3_underwater.json"), load("v4_space.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.9), constrained_layout=True)

    ax = axes[0]
    src = [("robotaxi 5 y", v1["v1_6_duty_cycle"]["duties"]["robotaxi"], "ground"),
           ("drone turnaround", v2["v2_9_turnaround"], "aerial"),
           ("AUV 6 months", v3["v3_4_deployment"], "underwater"),
           ("LEO 5 y", v4["v4_1_leo"], "space")]
    y = np.arange(len(src))
    for i, (lab, d, dom) in enumerate(src):
        ax.barh(i, d["trials"], color=DOMAIN_C[dom], height=0.62, alpha=0.9)
        ax.text(d["trials"] * 1.1, i, f"{d['violations']} viol", fontsize=6, va="center",
                color=C[2] if d["violations"] == 0 else RED)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in src], fontsize=6.2)
    ax.set_xscale("log"); ax.set_xlim(100, 3e5)
    ax.set_xlabel("charge cycles, never recalibrated")
    tag(ax, "(a)  mission length")

    ax = axes[1]
    for i, (lab, d, dom) in enumerate(src):
        a, b = d["first_quarter_soc"], d["last_quarter_soc"]
        ax.plot([0, 1], [a, b], "-", marker=DOMAIN_M[dom], color=DOMAIN_C[dom],
                markersize=4.5, label=lab)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["first quarter", "last quarter"], fontsize=6.2)
    ax.set_ylabel("SOC delivered per session")
    ax.legend(loc="best", fontsize=5.8)
    tag(ax, "(b)  what ageing costs")

    ax = axes[2]
    src2 = [("AUV, 180 d", v3["v3_9_no_recalibration"], "underwater"),
            ("deep space, 5 y", v4["v4_8_no_ground_loop"], "space")]
    x = np.arange(len(src2))
    ax.bar(x - 0.19, [100 * s[1]["bound"]["mean_soc"] for s in src2], 0.36,
           color=C[0], label="fixed datasheet bound")
    ax.bar(x + 0.19, [100 * s[1]["oracle"]["mean_soc"] for s in src2], 0.36,
           color=C[3], label="oracle, recalibrated every cycle")
    for i, s in enumerate(src2):
        ax.text(i, 4, f"{s[1]['bound']['violations']} / {s[1]['oracle']['violations']}\n"
                      f"violations", ha="center", fontsize=5.6, color=C[2])
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in src2], fontsize=6.2)
    ax.set_ylabel("SOC delivered  [%]")
    ax.legend(loc="upper center", fontsize=5.8)
    tag(ax, "(c)  the estimator buys charge, not safety")
    fig.suptitle("Nobody recalibrates anything: 5 years of LEO, 6 months under water",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV5_missions")


# =======================================================================================
def fv6_connectedness():
    """What makes this one method: structure, reduction, and a fixed cost everywhere."""
    x = load("x_crossdomain.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.9), constrained_layout=True)

    ax = axes[0]
    pd = x["x1_structure"]["per_domain"]
    doms = list(pd)
    si = [pd[d]["single-interval"] for d in doms]
    dc = [pd[d]["disconnected"] for d in doms]
    em = [pd[d]["empty"] for d in doms]
    xx = np.arange(len(doms))
    ax.bar(xx, si, 0.6, color=C[2], label="single interval")
    ax.bar(xx, em, 0.6, bottom=si, color=AMBER, label="empty (closure)")
    ax.bar(xx, dc, 0.6, bottom=[a + b for a, b in zip(si, em)], color=RED,
           label="disconnected")
    ax.set_xticks(xx); ax.set_xticklabels(doms, fontsize=6.2, rotation=20)
    ax.set_ylabel("dense scans")
    ax.set_ylim(0, max(a + b + c for a, b, c in zip(si, em, dc)) * 1.22)
    ax.legend(loc="upper left", fontsize=5.6, framealpha=0.96, handlelength=1.2)
    # the count goes in the title rather than the panel: an annotation placed anywhere inside
    # these axes collides with either the legend or the tallest bar
    tag(ax, f"(a)  an interval in all {x['x1_structure']['scans']:,} scans")

    ax = axes[1]
    rows = x["x3_cost"]["rows"]
    ch = [r for r in rows if r["mode"] == "charge"]
    di = [r for r in rows if r["mode"] == "discharge"]
    for grp, col, mk, lab in ((ch, C[0], "o", "charge  ($u_a=0$)"),
                              (di, C[1], "^", "discharge  ($u_a>0$)")):
        ax.scatter([r["max_evals"] for r in grp], [r["us_p99"] for r in grp],
                   s=30, color=col, marker=mk, label=lab, zorder=3)
    ax.axvline(x["x3_cost"]["theoretical_charge"], color=C[0], ls="--", lw=1.0)
    ax.axvline(x["x3_cost"]["theoretical_discharge"], color=C[1], ls="--", lw=1.0)
    ax.set_xlabel("worst-case model evaluations"); ax.set_ylabel(r"p99 latency  [$\mu$s]")
    ax.legend(loc="upper left", fontsize=5.8)
    tag(ax, "(b)  the cost is fixed, and small")

    ax = axes[2]
    ad = x["x5_adequacy"]["claims"]
    y = np.arange(len(ad))
    for i, r in enumerate(ad):
        col = C[2] if r["enough_for_0p1pct"] else (AMBER if r["enough_for_1pct"] else C[1])
        ax.barh(i, r["trials"], color=col, height=0.6, alpha=0.9)
    for t, lab in ((stats_n(0.01), "1 %"), (stats_n(0.001), "0.1 %")):
        ax.axvline(t, color=INK, ls="--", lw=0.9)
        ax.text(t * 1.1, len(ad) - 0.4, lab, fontsize=5.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r["claim"] for r in ad], fontsize=5.2)
    ax.set_xscale("log"); ax.set_xlim(100, 1e6)
    ax.set_xlabel("trials")
    tag(ax, "(c)  is the evidence enough?")
    fig.suptitle("Four domains, one method: same structure, same reduction, same fixed cost",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV6_connectedness")


def stats_n(target, alpha=0.05):
    import math
    return int(math.ceil(math.log(alpha) / math.log(1 - target)))


# =======================================================================================
def fv7_case_studies():
    """Six named vehicles, side by side, in the units their operators use."""
    v1, v2 = load("v1_ground.json"), load("v2_aerial.json")
    v3, v4 = load("v3_underwater.json"), load("v4_space.json")
    fig, axes = plt.subplots(2, 3, figsize=(DBL, 4.6), constrained_layout=True)

    # 1 robotaxi: deadline hit-rate over SOH x ambient
    ax = axes[0, 0]
    g = v1["v1_4_deadline"]["grid"]
    sohs = sorted({r["soh"] for r in g}, reverse=True)
    ambs = sorted({r["T_amb"] for r in g})
    Z = np.array([[next(r["hit_rate"] for r in g if r["soh"] == s and r["T_amb"] == a)
                   for a in ambs] for s in sohs])
    im = ax.imshow(Z, cmap=SEQ, aspect="auto", vmin=0, vmax=1, origin="upper")
    ax.set_xticks(range(len(ambs))); ax.set_xticklabels([f"{a:.0f}" for a in ambs], fontsize=5.6)
    ax.set_yticks(range(len(sohs))); ax.set_yticklabels([f"{s:.2f}" for s in sohs], fontsize=5.6)
    ax.set_xlabel("ambient [C]", fontsize=6.4); ax.set_ylabel("SOH", fontsize=6.4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=5.4); cb.set_label("80 % in 30 min", fontsize=5.6)
    ax.set_title("robotaxi: the dispatch question", fontsize=7, loc="left")

    # 2 haul truck: weakest cell exactness
    ax = axes[0, 1]
    rows = v1["v1_5_truck_pack"]["rows"]
    N = [r["N"] for r in rows]; dev = [max(r["abs_dev"], 1e-18) for r in rows]
    ax.plot(N, dev, "-o", color=C[0], markersize=4.5)
    ax.axhline(v1["v1_5_truck_pack"]["search_resolution_A"], color=RED, ls="--", lw=1.0)
    ax.text(N[0] * 1.4, v1["v1_5_truck_pack"]["search_resolution_A"] * 1.6,
            "bisection resolution", fontsize=5.6, color=RED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cells in pack", fontsize=6.4)
    ax.set_ylabel(r"$|u^*_{pack}-\min_i u^*_i|$  [A]", fontsize=6.4)
    ax.set_title("haul truck: 33 120 cells", fontsize=7, loc="left")

    # 3 eVTOL: reserve vs endurance
    ax = axes[0, 2]
    e = v2["v2_10_evtol"]
    ax.bar([0, 1, 2], [e["endurance_min_p05"], e["endurance_min_median"],
                       e["endurance_min_p95"]], 0.55, color=[C[1], C[0], C[2]])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["p05", "median", "p95"], fontsize=6)
    ax.set_ylabel("hover endurance  [min]", fontsize=6.4)
    ax.text(0.03, 0.92, f"$\\rho$={e['reserve_vs_endurance']['rho']:+.2f} reserve-endurance",
            transform=ax.transAxes, fontsize=5.8, color=MUTED, va="top")
    ax.set_title(f"eVTOL: {e['hover_c_rate']:.1f}C hover", fontsize=7, loc="left")

    # 4 glider vs AUV
    ax = axes[1, 0]
    gp = v3["v3_10_glider_vs_auv"]["platforms"]
    ks = list(gp)
    xx = np.arange(len(ks))
    ax.bar(xx - 0.19, [gp[k]["mean_anchor_A"] for k in ks], 0.36, color=C[0], label="anchor")
    ax.bar(xx + 0.19, [gp[k]["mean_width_A"] for k in ks], 0.36, color=C[1], label="reserve")
    ax.set_yscale("log")
    ax.set_xticks(xx); ax.set_xticklabels([k.replace("-", "\n") for k in ks], fontsize=5.6)
    ax.set_ylabel("current  [A]", fontsize=6.4)
    ax.legend(fontsize=5.6, loc="upper left")
    ax.set_title(f"glider vs AUV: {v3['v3_10_glider_vs_auv']['load_ratio']:.0f}x load",
                 fontsize=7, loc="left")

    # 5 GEO eclipse season
    ax = axes[1, 1]
    rows = v4["v4_2_geo"]["rows"]
    ax.plot([r["eclipse_min"] for r in rows],
            [100 * r["median_depth_of_discharge"] for r in rows], "-o", color=C[3],
            markersize=4.5, label="depth of discharge")
    ax.plot([r["eclipse_min"] for r in rows],
            [100 * r["completion_rate"] for r in rows], "-s", color=C[2],
            markersize=4.5, label="completed")
    ax.axvline(v4["v4_2_geo"]["max_eclipse_min"], color=INK, ls="--", lw=1.0)
    ax.text(v4["v4_2_geo"]["max_eclipse_min"] - 2, 50, "seasonal\nmax", fontsize=5.4,
            color=INK, ha="right")
    ax.set_xlabel("eclipse duration  [min]", fontsize=6.4); ax.set_ylabel("[%]", fontsize=6.4)
    ax.legend(fontsize=5.6, loc="center left")
    ax.set_title("GEO comsat: eclipse season", fontsize=7, loc="left")

    # 6 SEU containment
    ax = axes[1, 2]
    s = v4["v4_7_seu"]
    parts = [("caught by\nrange check", 100 * s["caught_fraction"], C[2]),
             ("benign or\nconservative", 100 * s["benign_or_conservative_fraction"], C[0]),
             ("more\naggressive", 100 * s["aggressive_fraction"], RED)]
    bottom = 0.0
    for lab, v, col in parts:
        ax.bar([0], [v], 0.5, bottom=[bottom], color=col)
        if v > 4:
            ax.text(0, bottom + v / 2, f"{lab}\n{v:.1f}%", ha="center", va="center",
                    fontsize=5.4, color="white", weight="bold")
        bottom += v
    ax.text(0.42, bottom - parts[-1][1] / 2,
            f"{parts[-1][0]}\n{parts[-1][1]:.2f}%", fontsize=5.4, color=RED, va="center")
    ax.set_xticks([]); ax.set_ylabel("bit flips  [%]", fontsize=6.4); ax.set_ylim(0, 105)
    ax.set_title(f"orbiter: {s['flips']:,} SEU flips", fontsize=7, loc="left")
    fig.suptitle("Six named vehicles, in the units their operators use",
                 fontsize=9, weight="bold", color=INK)
    save(fig, "FV7_case_studies")


# =======================================================================================
def fv8_envelope():
    """The two-sided envelope itself: both edges, the anchor between them, closing."""
    from zeroguard import anchored as A, platforms as P, vexp as V
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.9), constrained_layout=True)

    for ax, (case, dt, hor, lab) in zip(axes, (
            ("delivery-quadrotor", 2.0, 700, "delivery quadrotor, hover"),
            ("under-ice-auv", 60.0, 700, "under-ice AUV, transit"),
            ("lunar-night-lander", 300.0, 700, "lunar lander, night"))):
        est = V.pessimistic(case)
        plant = P.build(case, scale=dict(R=1.35, Q=0.9, plate=1.25))
        marg = V.margins(est)
        s = est.init(0.98, {"delivery-quadrotor": 20.0, "under-ice-auv": 4.0,
                            "lunar-night-lander": 10.0}[case])
        t, lo_s, hi_s, an_s = [], [], [], []
        for k in range(hor):
            lo, hi, st = A.interval(est, s, dt, est.w_nominal, marg)
            a = est.anchor(s)
            if st != "ok":
                t.append(k * dt / 60.0); lo_s.append(a); hi_s.append(a); an_s.append(a)
                break
            t.append(k * dt / 60.0); lo_s.append(lo); hi_s.append(hi); an_s.append(a)
            s, _o = plant.step(s, float(lo), dt, est.w_nominal)
        ax.fill_between(t, lo_s, hi_s, color=C[0], alpha=0.22, label="admissible")
        ax.plot(t, hi_s, "-", color=C[0], lw=1.4, label=r"$u_{hi}$: thermal / voltage")
        ax.plot(t, lo_s, "-", color=C[1], lw=1.4, label=r"$u_{lo}$: the load")
        ax.plot(t, an_s, ":", color=INK, lw=1.1, label=r"anchor $u_a$")
        ax.axvline(t[-1], color=AMBER, lw=1.2, ls="--")
        ax.text(t[-1], max(hi_s) * 0.94, " closure", fontsize=5.8, color=AMBER,
                ha="right", va="top", rotation=90)
        ax.set_xlabel("time  [min]"); ax.set_ylabel("pack current  [A]")
        ax.set_title(lab, fontsize=7, loc="left")
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=5.4)
    fig.suptitle("The two-sided envelope: the upper edge falls, the lower edge rises, "
                 "and the gap is the reserve", fontsize=9, weight="bold", color=INK)
    save(fig, "FV8_envelope")


# =======================================================================================
def main():
    style()
    matplotlib.rcParams["pdf.fonttype"] = 42       # TrueType, for IEEE PDF eXpress
    matplotlib.rcParams["ps.fonttype"] = 42
    os.makedirs(FIG, exist_ok=True)
    todo = [("FV1", fv1_domains), ("FV2", fv2_reserve), ("FV3", fv3_sortie),
            ("FV4", fv4_environments), ("FV5", fv5_missions),
            ("FV6", fv6_connectedness), ("FV7", fv7_case_studies), ("FV8", fv8_envelope)]
    print("rendering vehicle figures")
    for name, fn in todo:
        try:
            fn()
            print(f"  {name}  ok")
        except FileNotFoundError as e:
            print(f"  {name}: skipped, missing {e}")
        except Exception as e:
            print(f"  {name}: FAILED -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()


# =======================================================================================
def fv0_geometry():
    """The one-step geometry: why the admissible set is an interval.

    Every other figure in this file reports an outcome. This one shows the mechanism, and it is
    drawn from the real constrained signals rather than sketched: each panel evaluates the
    actual cell model over the whole input range at one fixed state, plots every constrained
    signal against its own limit, and marks where each constraint stops being satisfied.

    Left, a cell being charged: three caps, no floors, and the admissible set runs from zero to
    whichever cap binds first. Right, the same machinery on a vehicle in flight: the caps still
    come down from above, but the load's power requirement now comes *up* from below, and the
    certificate is the gap between them.
    """
    from zeroguard import anchored as A, platforms as P, vexp as V
    fig, axes = plt.subplots(2, 2, figsize=(DBL, 4.3), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 1]))

    for col, (case, mode, T0, soc0, lab) in enumerate((
            ("robotaxi-urban", "charge", 30.0, 0.55, "charging a pack: every constraint is a cap"),
            ("delivery-quadrotor", "discharge", 25.0, 0.55,
             "flying: the load is a floor the caps must clear"))):
        est = V.pessimistic(case)
        marg = V.margins(est)
        s = est.init(soc0, T0)
        dt, w = est.dt_nominal, est.w_nominal
        lo_b, hi_b = A._bounds(est, s)
        g = np.linspace(lo_b, hi_b, 500)
        vals = np.array([est.probe(s, float(u), dt, w) for u in g])
        u_lo, u_hi, st = A.interval(est, s, dt, w, marg)

        # -- top: the constrained signals against their own limits --------------------
        ax = axes[0, col]
        hi_f, lo_f = A.split_cached(est)
        series = []
        for i, idx, sense, val in hi_f:
            m = marg[i]
            lim = val - m if sense == "<=" else val + m
            series.append((idx, lim, sense, "cap"))
        for i, idx, sense, val in lo_f:
            m = marg[i]
            lim = val + m if sense == ">=" else val - m
            series.append((idx, lim, sense, "floor"))
        names = {0: r"$V$  [V]", 1: r"$T$  [$^\circ$C]", 2: r"$\varphi_{an}$  [V]",
                 3: r"$P$  [kW]", 4: "SOC"}
        for k, (idx, lim, sense, kind) in enumerate(series):
            y = vals[:, idx] / (1000.0 if idx == 3 else 1.0)
            L = lim / (1000.0 if idx == 3 else 1.0)
            yn = (y - L) / (np.abs(y).max() + 1e-12)     # normalised distance to the limit
            ax.plot(g, yn, "-", color=C[k % len(C)], lw=1.5,
                    label=f"{names[idx]} ({kind})")
        ax.axhline(0, color=INK, lw=1.1)
        ax.text(0.995, 0.06, "limit", transform=ax.transAxes, ha="right", fontsize=5.6,
                color=INK)
        # Not every binding constraint is one of the probed signals. On charge the
        # temperature-dependent plating *current cap* bounds the search directly, and on
        # discharge the actuator ceiling does; neither appears as a curve crossing zero, so
        # without drawing them the interval appears to end for no reason.
        capv = est.cap(s)
        if capv is not None and capv < hi_b * 0.999:
            ax.axvline(capv, color=MUTED, ls=":", lw=1.2)
            ax.text(capv, ax.get_ylim()[0] * 0.86, " plating current cap", fontsize=5.4,
                    color=MUTED, rotation=90, va="bottom")
        elif abs(hi_b - est.u_max) < 1e-9:
            ax.axvline(est.u_max, color=MUTED, ls=":", lw=1.2)
            ax.text(est.u_max, ax.get_ylim()[0] * 0.86, " actuator ceiling", fontsize=5.4,
                    color=MUTED, rotation=90, va="bottom", ha="right")
        if st == "ok":
            ax.axvspan(u_lo, u_hi, color=C[2], alpha=0.13, zorder=0)
        ax.set_ylabel("distance to limit\n(normalised)", fontsize=6.6)
        ax.legend(loc="lower left", fontsize=5.4, ncol=2, framealpha=0.95)
        tag(ax, f"({'ab'[col]})  {lab}")

        # -- bottom: the resulting admissible set -------------------------------------
        ax = axes[1, col]
        ok = np.array([A.feasible(est, s, float(u), dt, w, marg) for u in g])
        ax.fill_between(g, 0, ok.astype(float), step="mid", color=C[2], alpha=0.30)
        ax.plot(g, ok.astype(float), drawstyle="steps-mid", color=C[2], lw=1.4)
        if st == "ok":
            for e, c, nm, ha in ((u_lo, C[1], r"$u_{lo}$", "left"),
                                 (u_hi, C[0], r"$u_{hi}$", "right")):
                ax.axvline(e, color=c, lw=1.3, ls="--")
                ax.text(e, 1.10, nm, color=c, fontsize=7, ha=ha)
            a = A.effective_anchor(est, s)
            ax.plot([a], [0.5], "v", color=INK, markersize=6)
            ax.text(a, 0.60, r"anchor $u_a$" + ("$\\,=0$" if mode == "charge" else ""),
                    fontsize=5.8, ha="center", color=INK)
            # name the constraint that actually set the upper edge, so the panel explains
            # its own boundary instead of leaving the reader to infer it
            binder = None
            for i, idx, sense, val in hi_f:
                m = marg[i]
                v = est.probe(s, min(u_hi * 1.01 + 1e-6, hi_b), dt, w)[idx]
                bad = (v > val - m) if sense == "<=" else (v < val + m)
                if bad:
                    binder = {0: "voltage", 1: "temperature", 2: "plating margin"}.get(idx)
                    break
            if binder is None and abs(u_hi - hi_b) < 1e-6:
                binder = "actuator / plating current cap"
            if binder:
                ax.text(u_hi, 0.86, f"set by {binder} ", fontsize=5.6, color=C[0],
                        ha="right", va="top")
            ax.annotate("", xy=(u_lo, 0.28), xytext=(u_hi, 0.28),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
            ax.text(0.5 * (u_lo + u_hi), 0.33,
                    f"reserve = {u_hi-u_lo:.1f} A", fontsize=6, ha="center", color=INK)
        ax.set_ylim(0, 1.35); ax.set_yticks([0, 1])
        ax.set_yticklabels(["unsafe", "admissible"], fontsize=6)
        ax.set_xlabel("pack current $u$  [A]")
        tag(ax, f"({'cd'[col]})  the admissible set is one interval")

    fig.suptitle("Why the search collapses: monotone constraints make the admissible set an "
                 "interval", fontsize=9, weight="bold", color=INK)
    save(fig, "FV0_geometry")
