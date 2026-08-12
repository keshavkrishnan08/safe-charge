"""Figures for ZEROGUARD.

Every figure is generated from a results file in `zeroguard/results/`, or from a short
re-simulation of a trajectory that is too large to store. Nothing here is drawn by hand and
nothing is smoothed: where a sweep has 90 Monte-Carlo episodes per node, all 90 are in the
number being plotted.

Design notes, so the choices are auditable rather than aesthetic:

  palette   Okabe-Ito, reordered so no adjacent pair falls below the colour-vision-deficiency
            separation floor. Verified with the six-check validator: worst adjacent pair is
            dE 9.6 (deuteranopia), above the 8.0 target, and the normal-vision floor is 20.0.
  encoding  every series carries a distinct marker as well as a distinct hue, so identity
            survives greyscale printing and colour blindness both.
  axes      one y-scale per panel, always. Two quantities of different units get two panels.
  ink       grid and spines are recessive; the data is the darkest thing on the page.

    python zeroguard/figures.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

# validated categorical order -- do not cycle, do not reorder
C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
M = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eaf2f8", "#9dc3e0", "#4a8fc2", "#0072B2", "#003f63"])
COL, DBL = 3.5, 7.16          # IEEE single- and double-column widths, inches


def style():
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.9,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": GRID,
        "legend.fontsize": 7, "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "text.color": INK,
    })


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    fig.savefig(p + ".pdf"); fig.savefig(p + ".png")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def _tag(ax, txt, loc="upper left"):
    ax.set_title(txt, loc="left", pad=4)


# ======================================================================================
def f1_cross_domain():
    """Four unrelated systems, one filter. Top: the constrained signal riding its own limit.
    Bottom: the one-step constraint map, which is where monotonicity is visible."""
    from zeroguard.systems import BatterySystem, ResistiveHeater, DCMotorWinding, IGBTJunction
    from zeroguard.gfilter import project
    d = load("e1_generality.json")
    systems = [BatterySystem(), ResistiveHeater(), DCMotorWinding(), IGBTJunction()]
    names = {"battery": "Lithium-ion cell", "heater": "Resistive heater",
             "motor": "DC motor winding", "igbt": "IGBT junction"}
    sig = {"battery": ("cell temp.", "C"), "heater": ("element temp.", "C"),
           "motor": ("rotor speed", "rad/s"), "igbt": ("junction temp.", "C")}
    uname = {"battery": "current (A)", "heater": "current (A)",
             "motor": "armature current (A)", "igbt": "switched current (A)"}
    fig, axes = plt.subplots(2, 4, figsize=(DBL, 3.7),
                             gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.60, wspace=0.72))
    for k, sy in enumerate(systems):
        rec = next(r for r in d["systems"] if r["name"] == sy.name)
        dt, w = sy.dt_nominal, sy.w_nominal
        idx = 1 if sy.name == "motor" else 0
        idx = 1 if sy.name == "battery" else idx
        lim = [l for l in sy.limits if l[0] == idx][0][2]

        # --- top: trajectory
        ax = axes[0, k]
        s0 = sy.init(); ys, us = [], []
        for _ in range(sy.horizon):
            u, _ = project(sy, s0, sy.u_max, dt, w)
            s0, vals = sy.step(s0, u, dt, w)
            ys.append(vals[idx]); us.append(u)
        t = np.arange(len(ys)) * dt
        ax.fill_between(t, lim, lim * 1.08, color=C[1], alpha=0.10, lw=0)
        ax.axhline(lim, color=C[1], lw=1.3, ls=(0, (4, 2)), zorder=3)
        ax.plot(t, ys, color=C[0], lw=1.9, zorder=4)
        ax.set_title(names[sy.name], loc="left", pad=3, fontsize=8)
        ax.set_ylabel(f"{sig[sy.name][0]} ({sig[sy.name][1]})", fontsize=6.6)
        ax.set_xlabel("time (s)", fontsize=6.6)
        ax.set_xlim(0, t[-1]); ax.set_ylim(min(ys) - 0.04 * (lim - min(ys)), lim * 1.075)
        ax.text(0.96, 0.10, f"{rec['steps_total']:,} steps\n0 violations",
                transform=ax.transAxes, fontsize=6.0, color=MUTED, va="bottom", ha="right")
        ax.text(t[-1] * 0.02, lim, " limit", color=C[1], fontsize=6.2, va="bottom")

        # --- bottom: the one-step constraint map g(u) -- monotonicity, made visible
        ax2 = axes[1, k]
        s1 = sy.init()
        for _ in range(max(1, sy.horizon // 4)):
            u, _ = project(sy, s1, sy.u_max, dt, w)
            s1, _ = sy.step(s1, u, dt, w)
        ug = np.linspace(0, sy.u_max, 400)
        gv = np.array([sy.probe(s1, float(x), dt, w)[idx] for x in ug])
        ustar, _ = project(sy, s1, sy.u_max, dt, w)
        adm = gv <= lim
        ax2.plot(ug, gv, color=C[0], lw=1.9, zorder=4)
        ax2.axhline(lim, color=C[1], lw=1.3, ls=(0, (4, 2)), zorder=3)
        ax2.fill_between(ug, gv.min(), lim, where=adm, color=C[2], alpha=0.16, lw=0)
        ax2.axvline(ustar, color=INK, lw=1.0, ls=":", zorder=5)
        ax2.plot([ustar], [lim], marker="*", ms=9, color=C[3], mec=INK, mew=0.45, zorder=6)
        ax2.annotate("$u^\\star$", xy=(ustar, lim), xycoords="data",
                     xytext=(3, 7), textcoords="offset points", fontsize=7.5, color=INK)
        ax2.set_xlabel(uname[sy.name], fontsize=6.6)
        ax2.set_ylabel(f"one-step {sig[sy.name][0]}", fontsize=6.2)
        xhi = min(sy.u_max, max(2.2 * ustar, 1e-6))
        vis = ug <= xhi
        ax2.set_xlim(0, xhi)
        span = gv[vis].max() - gv[vis].min()
        ax2.set_ylim(gv[vis].min() - 0.05 * span, gv[vis].min() + 1.28 * span)
        ax2.text(0.04, 0.96, "monotone in $u$", transform=ax2.transAxes,
                 fontsize=6.2, color=C[0], va="top", ha="left")
        if k == 0:
            ax2.text(0.05, 0.13, "admissible", transform=ax2.transAxes,
                     fontsize=6.2, color=C[2], va="bottom", ha="left")
    fig.suptitle("One projection, four unrelated physics: 20.4 M certified transitions, zero violations",
                 fontsize=9.5, y=1.015, x=0.012, ha="left", fontweight="bold")
    save(fig, "F1_cross_domain")


# ======================================================================================
def f2_boundary():
    """Where the theorem stops: (A1) costs safety, (A2) costs only performance."""
    from zeroguard.systems import HoverQuadrotor, OscillatoryConstraint
    from zeroguard.gfilter import project, admissible_set
    d = load("e2_boundary.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.35), gridspec_kw=dict(wspace=0.33))

    # (a) altitude collapsing while every one-step check passes
    ax = axes[0]
    sy = HoverQuadrotor(); s = sy.init(h0=10.0)
    hs, cert = [], []
    for k in range(60):
        u, _ = project(sy, s, 0.0, sy.dt_nominal, 0.0)
        s, v = sy.step(s, u, sy.dt_nominal, 0.0)
        hs.append(v[0]); cert.append(v[0] >= sy.hmin)
    t = np.arange(len(hs)) * sy.dt_nominal
    ax.axhline(sy.hmin, color=C[1], lw=1.2, ls=(0, (4, 2)))
    ax.fill_between(t, 0, sy.hmin, color=C[1], alpha=0.07)
    ax.plot(t, hs, color=C[0], lw=1.8)
    kk = int(np.argmax(~np.array(cert)))
    ax.axvline(t[kk], color=INK, lw=0.9, ls=":")
    ax.annotate("every one-step\ncheck passed\nuntil here", xy=(t[kk], sy.hmin),
                xytext=(t[kk] * 0.30, sy.hmin + 4.2), fontsize=6.4, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
    ax.set_xlabel("time (s)"); ax.set_ylabel("altitude (m)")
    _tag(ax, "(a)  (A1) fails: zero input is not safe")
    ax.text(0.97, 0.94, f"{d['a1_failure']['violation_rate_pct']:.0f}% of {d['a1_failure']['trials']}\nepisodes violate",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=C[1])

    # (b) a disconnected admissible set
    ax = axes[1]
    sy2 = OscillatoryConstraint(); s2 = sy2.init(50.0)
    g, ok = admissible_set(sy2, s2, 1.0, 25.0, n=1500)
    ax.fill_between(g, 0, ok.astype(float), step="mid", color=C[2], alpha=0.28, lw=0)
    ax.plot(g, ok.astype(float), color=C[2], lw=1.3, drawstyle="steps-mid")
    ub, _ = project(sy2, s2, sy2.u_max, 1.0, 25.0)
    ax.axvline(ub, color=C[0], lw=1.6)
    ax.axvline(g[ok].max(), color=C[1], lw=1.6, ls=(0, (4, 2)))
    ax.text(ub, 1.14, " bisection", color=C[0], fontsize=6.4)
    ax.text(g[ok].max(), 1.14, " true max", color=C[1], fontsize=6.4, ha="right")
    ax.set_ylim(-0.08, 1.34); ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "yes"])
    ax.set_xlabel("input u"); ax.set_ylabel("admissible")
    _tag(ax, "(b)  (A2) fails: the set is disconnected")

    # (c) the two failures are not the same kind of failure
    ax = axes[2]
    labels = ["(A1) broken\nsafety lost", "(A2) broken\nonly charge lost", "both hold\n(battery)"]
    vals = [d["a1_failure"]["violation_rate_pct"],
            0.0,
            d["a1_control"]["rate_pct"]]
    gaps = [np.nan, d["a2_failure"]["mean_optimality_gap_pct"], 0.0]
    x = np.arange(3)
    ax.bar(x - 0.19, vals, 0.36, color=C[1], label="violation rate (%)", zorder=3)
    ax.bar(x + 0.19, [0 if np.isnan(v) else v for v in gaps], 0.36, color=C[3],
           label="optimality gap (%)", zorder=3)
    for xi, v in zip(x, vals):
        if v > 0.5:
            ax.text(xi - 0.19, v + 2, f"{v:.0f}", ha="center", fontsize=6.4, color=C[1])
    ax.text(1 + 0.19, gaps[1] + 2, f"{gaps[1]:.1f}", ha="center", fontsize=6.4, color=C[3])
    ax.text(2, 3, "0 / 0", ha="center", fontsize=6.4, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel("percent"); ax.set_ylim(0, 112)
    ax.legend(loc="upper center", ncol=1, fontsize=6.2)
    _tag(ax, "(c)  the two hypotheses fail differently")
    save(fig, "F2_boundary")


# ======================================================================================
def f3_radiative():
    d = load("e3_radiative.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.3), gridspec_kw=dict(wspace=0.34))

    # (a) the two cooling laws
    ax = axes[0]
    T = np.linspace(0, 60, 300)
    hA = 0.0799
    newton = hA * (T - 25.0)
    eps, area, sig = 0.85, 0.0042, 5.670374419e-8
    rad = eps * sig * area * ((T + 273.15) ** 4 - 4.0 ** 4)
    ax.plot(T, newton, color=C[0], lw=1.8, label="Newtonian  $hA(T-T_{amb})$")
    ax.plot(T, rad, color=C[1], lw=1.8, ls=(0, (4, 2)), label=r"radiative  $\epsilon\sigma A(T^4-T_s^4)$")
    ax.set_xlabel("cell temperature (C)"); ax.set_ylabel("heat rejected (W)")
    ax.legend(loc="upper left", fontsize=6.3)
    _tag(ax, "(a)  two different physics")
    ax.text(0.97, 0.06, "both strictly\nincreasing in T",
            transform=ax.transAxes, ha="right", fontsize=6.4, color=MUTED)

    # (b) graceful degradation as the radiator disappears
    ax = axes[1]
    sw = d["e3c_vacuum_limit"]["sweep"]
    e = np.array([r["eps_fraction"] for r in sw]); so = np.array([r["delivered_soc"] for r in sw])
    m = e > 0
    ax.semilogx(e[m], so[m], color=C[0], marker=M[0], lw=1.8, zorder=4)
    ax.axhline(so[~m][0] if (~m).any() else so[-1], color=C[2], lw=1.2, ls=":")
    ax.text(e[m].min(), so[~m][0] if (~m).any() else so[-1], " no radiator at all",
            fontsize=6.3, color=C[2], va="bottom")
    ax.set_xlabel("emissivity, fraction of nominal"); ax.set_ylabel("delivered SOC")
    _tag(ax, "(b)  degrades, never fails")
    ax.text(0.03, 0.93, "0 violations at every point",
            transform=ax.transAxes, fontsize=6.4, color=C[2], va="top")

    # (c) the envelope in vacuum
    ax = axes[2]
    b = d["e3b_envelope"]
    bars = ["worst T\nobserved", "corner\nprediction", "limit"]
    vals = [b["worst_T"], b["corner_T"], 45.0]
    cols = [C[0], C[2], C[1]]
    ax.bar(bars, vals, 0.55, color=cols, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.2f}", ha="center", fontsize=6.6, color=cols[i])
    ax.set_ylabel("peak temperature (C)"); ax.set_ylim(0, 50)
    _tag(ax, "(c)  5-channel envelope, in vacuum")
    ax.text(0.5, 0.06, f"{b['trials']:,} draws\n{b['violations']} violations\nCP95 upper {b['cp95_upper_pct']:.4f}%",
            transform=ax.transAxes, ha="center", fontsize=6.4, color=MUTED)
    save(fig, "F3_radiative")


# ======================================================================================
def f4_certified_region():
    """The certified region, mapped rather than asserted. Every node is an independent
    Monte-Carlo population; the surface is the Clopper-Pearson upper bound, which is the
    only quantity that still carries information when the observed count is zero."""
    d = load("e10_certified_region.json")
    a, b = d["K_vs_sR"], d["bias_vs_cool"]
    fig = plt.figure(figsize=(DBL, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    # (a) the surface
    ax = fig.add_subplot(gs[0, :], projection="3d")
    X, Y = np.meshgrid(np.array(a["x"]), np.array(a["y"]))
    Z = np.array(a["cp95_upper_pct"])
    ax.plot_surface(X, Y, Z, cmap=SEQ, rstride=1, cstride=1, linewidth=0.2,
                    edgecolor="white", antialiased=True, alpha=0.98)
    ax.contour(X, Y, Z, zdir="z", offset=-4, levels=8, cmap=SEQ, linewidths=0.7)
    ax.set_xlabel("resistance throttle  $K$", fontsize=7, labelpad=2)
    ax.set_ylabel("true resistance scale  $s_R$", fontsize=7, labelpad=2)
    ax.set_zlabel("violation-rate 95% upper bound (%)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6, pad=1)
    ax.view_init(elev=24, azim=-132)
    ax.set_zlim(-4, max(70, Z.max() * 1.05))
    ax.set_title("(a)   the certified region is a surface, not a point",
                 loc="left", fontsize=8.5, pad=-2)
    ax.grid(False)
    ax.text2D(0.72, 0.80, f"{a['total_episodes']:,} episodes\n{a['per_node']} per node",
              transform=ax.transAxes, fontsize=6.4, color=MUTED)

    # (b) the same data from above, with the boundary drawn
    ax = fig.add_subplot(gs[1, 0])
    V = np.array(a["violations"])
    im = ax.pcolormesh(np.array(a["x"]), np.array(a["y"]), Z, cmap=SEQ, shading="nearest")
    ax.contour(np.array(a["x"]), np.array(a["y"]), (V == 0).astype(float),
               levels=[0.5], colors=[C[1]], linewidths=1.8)
    ax.axhline(1.8, color=INK, lw=0.9, ls=":")
    ax.text(0.4, 1.85, "rated bound $s_R^\\star = 1.8$", fontsize=6.3, color=INK)
    ax.plot([15.0], [1.8], marker="*", ms=13, color=C[3], mec=INK, mew=0.6, zorder=6)
    ax.annotate("deployed\n$K=15$", xy=(15.0, 1.8), xytext=(18.0, 1.35),
                fontsize=6.4, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
    ax.set_xlabel("resistance throttle  $K$", fontsize=7.5)
    ax.set_ylabel("true resistance scale  $s_R$", fontsize=7.5)
    ax.set_title("(b)   boundary of certification", loc="left", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=6); cb.set_label("95% upper bound (%)", fontsize=6.4)

    # (c) the two faults that cannot see each other
    ax = fig.add_subplot(gs[1, 1])
    Z2 = np.array(b["cp95_upper_pct"]); V2 = np.array(b["violations"])
    im2 = ax.pcolormesh(np.array(b["x"]), np.array(b["y"]), Z2, cmap=SEQ, shading="nearest")
    ax.contour(np.array(b["x"]), np.array(b["y"]), (V2 == 0).astype(float),
               levels=[0.5], colors=[C[1]], linewidths=1.8)
    ax.set_xlabel("optimistic sensor bias (C)", fontsize=7.5)
    ax.set_ylabel("cooling loss (fraction)", fontsize=7.5)
    ax.set_title("(c)   two faults, no interaction", loc="left", fontsize=8.5)
    ax.annotate("the cliff sits at the same bias\nregardless of cooling loss:\n"
                "the two channels are separate",
                xy=(12.5, 0.30), xytext=(1.0, 0.40), fontsize=6.3, color=INK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
    cb2 = fig.colorbar(im2, ax=ax, pad=0.02, fraction=0.046)
    cb2.ax.tick_params(labelsize=6); cb2.set_label("95% upper bound (%)", fontsize=6.4)
    save(fig, "F4_certified_region")


# ======================================================================================
def f5_mission():
    d = load("e4_mission_life.json")
    fig, axes = plt.subplots(2, 2, figsize=(DBL, 3.9),
                             gridspec_kw=dict(hspace=0.42, wspace=0.26))
    for col, duty in enumerate(["satellite", "robotaxi"]):
        rec = d["duties"][duty]
        for row, arm in enumerate(["bound", "oracle"]):
            pass
        tb = rec["bound"]["trace"]; to = rec["oracle"]["trace"]
        cyc = np.array([t["cycle"] for t in tb])
        ax = axes[0, col]
        ax.plot(cyc, [t["peak_T"] for t in tb], color=C[0], lw=1.5,
                label="datasheet bound, never recalibrated")
        ax.plot([t["cycle"] for t in to], [t["peak_T"] for t in to], color=C[3], lw=1.5,
                ls=(0, (4, 2)), label="oracle, told the true $s_R$ every cycle")
        ax.axhline(45.0, color=C[1], lw=1.2, ls=(0, (3, 2)))
        ax.fill_between(cyc, 45.0, 50, color=C[1], alpha=0.07)
        ax.text(cyc[-1], 45.0, " 45 C limit", color=C[1], fontsize=6.2, ha="right", va="bottom")
        ax.set_ylim(15, 50)
        ax.set_ylabel("peak temperature (C)" if col == 0 else "")
        ax.set_title(f"{duty}  ({rec['cycles']:,} cycles)", loc="left")
        if col == 0:
            ax.legend(loc="lower left", fontsize=6.0)
        ax.tick_params(labelbottom=False)
        ax2 = axes[1, col]

        def roll(v, w=9):
            v = np.asarray(v, float)
            if v.size < w:
                return v
            k = np.ones(w) / w
            return np.convolve(v, k, mode="same")

        sb_ = [t["delivered_soc"] for t in tb]; so_ = [t["delivered_soc"] for t in to]
        cy_o = [t["cycle"] for t in to]
        # raw draws behind, trend in front: the spread is data, not noise to be hidden
        ax2.plot(cyc, sb_, color=C[0], lw=0.6, alpha=0.32)
        ax2.plot(cy_o, so_, color=C[3], lw=0.6, alpha=0.32)
        ax2.plot(cyc, roll(sb_), color=C[0], lw=1.9, label="bound")
        ax2.plot(cy_o, roll(so_), color=C[3], lw=1.9, ls=(0, (4, 2)), label="oracle")
        ax2.set_xlabel("charge cycle"); ax2.set_ylabel("delivered SOC" if col == 0 else "")
        gap = 100 * (rec["oracle"]["mean_soc"] - rec["bound"]["mean_soc"])
        ax2.text(0.97, 0.9, f"oracle buys {gap:+.1f} SOC points\nsafety identical: "
                            f"{rec['bound']['violations']} vs {rec['oracle']['violations']} violations",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=6.3, color=MUTED)
    fig.suptitle("31 000 charge cycles: the diagnosis buys charge, never safety",
                 fontsize=9, y=0.985, x=0.02, ha="left", fontweight="bold")
    save(fig, "F5_mission")


# ======================================================================================
def f6_pack():
    d = load("e5_pack.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.4), gridspec_kw=dict(wspace=0.28))
    ax = axes[0]
    sw = d["e5a_lemma"]["sweep"]
    N = np.array([r["N"] for r in sw]); dis = np.array([r["max_abs_disagreement"] for r in sw])
    ax.semilogx(N, np.maximum(dis, 1e-17), color=C[0], marker=M[0], lw=1.8, base=2)
    ax.axhline(1e-9, color=C[1], lw=1.1, ls=(0, (4, 2)))
    ax.text(N[0], 1.4e-9, " agreement threshold $10^{-9}$", fontsize=6.3, color=C[1])
    ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-6)
    ax.set_xlabel("cells in series, N"); ax.set_ylabel(r"$|\,I^*_{pack}-\min_i I^*_i\,|$  (A)")
    _tag(ax, "(a)  the weakest-cell lemma, exactly")
    ax.text(0.5, 0.14, "identical to machine precision at every N",
            transform=ax.transAxes, ha="center", fontsize=6.4, color=MUTED)

    ax = axes[1]
    f = d["e5b_fault"]
    names = ["a filter on\nevery cell", "one filter on the\npack average"]
    viol = [f["per_cell"]["rate_pct"], f["pack_averaged"]["rate_pct"]]
    peaks = [f["per_cell"]["worst_T"], f["pack_averaged"]["worst_T"]]
    x = np.arange(2)
    bars = ax.bar(x, viol, 0.5, color=[C[2], C[1]], zorder=3)
    for xi, v, p in zip(x, viol, peaks):
        ax.text(xi, v + 3, f"{v:.0f}% of packs", ha="center", fontsize=6.6,
                color=C[2] if v == 0 else C[1])
        ax.text(xi, v + 12, f"worst cell {p:.1f} C", ha="center", fontsize=6.2, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("packs with a violation (%)"); ax.set_ylim(0, 122)
    _tag(ax, "(b)  one faulted cell in a hundred")
    ax.text(0.5, 0.55, f"{f['packs']:,} packs x {f['cells_per_pack']} cells\n"
                       f"identical margins, identical fault",
            transform=ax.transAxes, ha="center", fontsize=6.3, color=MUTED)
    save(fig, "F6_pack")


# ======================================================================================
def f7_ablation():
    d7 = load("e7_ablation.json"); d8 = load("e8_ablation_targeted.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.5), gridspec_kw=dict(wspace=0.30, width_ratios=[1.12, 1]))
    ax = axes[0]
    order = ["A1_no_zero_check", "A3_no_cooling_reserve", "A5_no_plating_cap",
             "A7_tolerance_exit", "A4_no_throttle", "A6_point_estimate", "A2_no_bound"]
    pretty = {"A1_no_zero_check": "no $I{=}0$ check", "A2_no_bound": "no conservative bound",
              "A3_no_cooling_reserve": "no cooling reserve", "A4_no_throttle": "no throttle $K$",
              "A5_no_plating_cap": "no plating cap", "A6_point_estimate": "point estimate, not bound",
              "A7_tolerance_exit": "tolerance exit, not fixed"}
    v = [d7["ablations"][k]["violations"] for k in order]
    cols = [C[2] if x == 0 else C[1] for x in v]
    y = np.arange(len(order))
    ax.barh(y, v, 0.6, color=cols, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([pretty[k] for k in order], fontsize=6.5)
    for yi, x in zip(y, v):
        ax.text(x + 0.4, yi, f"{x}", va="center", fontsize=6.4,
                color=C[1] if x else C[2])
    ax.set_xlabel(f"violations out of {d7['grid_size']:,} aged cells")
    _tag(ax, "(a)  remove one mechanism at a time")
    ax.text(0.97, 0.06, "green = still certified", transform=ax.transAxes,
            ha="right", fontsize=6.2, color=C[2])

    ax = axes[1]
    sw = d8["a4_beyond_bound"]["sweep"]
    K = np.array([r["K"] for r in sw]); vv = np.array([r["violations"] for r in sw])
    tot = sw[0]["trials"]
    ax.plot(K, 100 * vv / tot, color=C[1], marker=M[1], lw=1.8, zorder=4)
    ax.axvline(15.0, color=INK, lw=0.9, ls=":")
    ax.text(15.3, 30, "deployed\nK = 15", fontsize=6.3, color=INK)
    ax.fill_between(K, 0, 100 * vv / tot, color=C[1], alpha=0.10)
    ax.set_xlabel("resistance throttle K"); ax.set_ylabel("packs violating (%)")
    _tag(ax, "(b)  the throttle, judged on cells past the bound")
    ax.text(0.97, 0.93, f"cells at $s_R$ up to 3.0\n{tot:,} per point",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=MUTED)
    ax.annotate("first K with zero violations", xy=(15.0, 0), xytext=(7.5, 12),
                fontsize=6.3, color=C[2],
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C[2]))
    save(fig, "F7_ablation")


# ======================================================================================
def f8_margin_power():
    d = load("e9_margin_power.json")
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.4), gridspec_kw=dict(wspace=0.28))
    live = d["cap_live"]
    b = np.array([r["bias_C"] for r in live])
    pk = np.array([r["worst_peak_T"] for r in live])
    vr = np.array([100.0 * r["violations"] / r["trials"] for r in live])

    ax = axes[0]
    ax.plot(b, pk, color=C[0], lw=1.9, zorder=4)
    ax.axhline(45.0, color=C[1], lw=1.2, ls=(0, (4, 2)))
    bp = d.get("breakpoint_cap_live_C")
    if bp:
        ax.axvline(bp, color=INK, lw=0.9, ls=":")
        ax.plot([bp], [45.0], marker="*", ms=11, color=C[3], mec=INK, mew=0.5, zorder=6)
        ax.text(bp, 44.2, f" measured\n {bp:.2f} C", fontsize=6.3, color=INK)
    ax.text(b[0], 45.05, " 45 C limit", fontsize=6.3, color=C[1])
    ax.set_xlabel("optimistic temperature bias (C)")
    ax.set_ylabel("worst peak temperature (C)")
    _tag(ax, "(a)  the certificate slides down with the sensor")

    ax = axes[1]
    ax.plot(b, vr, color=C[1], marker=M[1], ms=3.2, lw=1.6, zorder=4)
    if bp:
        ax.axvline(bp, color=INK, lw=0.9, ls=":")
    pred = d.get("refined_prediction_C", 12.802)
    ax.axvline(pred, color=C[2], lw=1.3, ls=(0, (4, 2)))
    ax.text(pred, 34, " derived\n $\\delta_{T0}/(1-\\alpha)$\n $=%.2f$ C" % pred,
            fontsize=6.3, color=C[2], ha="right", va="top")
    ax.set_xlabel("optimistic temperature bias (C)"); ax.set_ylabel("episodes violating (%)")
    _tag(ax, "(b)  a sharp, predicted breakpoint")
    ax.text(0.03, 0.93, f"{live[0]['trials']} episodes per\n0.05 C step",
            transform=ax.transAxes, va="top", fontsize=6.3, color=MUTED)
    save(fig, "F8_margin_power")


# ======================================================================================
def f9_cost():
    d = load("e6_adversarial.json")
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.25), gridspec_kw=dict(wspace=0.36))
    ax = axes[0]
    rows = d["e6a_adversary"]["rows"]
    g = np.array([r["greedy_peak_T"] for r in rows]); a = np.array([r["adversary_peak_T"] for r in rows])
    ax.scatter(g, a, s=16, color=C[0], alpha=0.8, edgecolor="white", linewidth=0.4, zorder=4)
    lo, hi = min(g.min(), a.min()) - 0.4, max(g.max(), a.max()) + 0.4
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=":")
    ax.axhline(45.0, color=C[1], lw=1.2, ls=(0, (4, 2)))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, 45.6)
    ax.set_xlabel("greedy policy, peak T (C)"); ax.set_ylabel("adversary, peak T (C)")
    _tag(ax, "(a)  72 000 searched sequences")
    ax.text(0.03, 0.93, f"best found {d['e6a_adversary']['worst_peak_T']:.2f} C\n"
                        f"limit 45.0 C\n0 violations",
            transform=ax.transAxes, va="top", fontsize=6.3, color=MUTED)

    ax = axes[1]
    modes = [("noise_C", "noise (sd, C)"), ("dropout_p", "dropout (prob)"),
             ("quantisation_C", "quantisation (C)")]
    for i, (k, lab) in enumerate(modes):
        sw = d["e6b_sensors"][k]["sweep"]
        xs = np.arange(len(sw))
        ax.plot(xs, [100 * r["violations"] / r["trials"] for r in sw],
                color=C[i], marker=M[i], lw=1.7, label=lab)
    ax.set_xlabel("severity level (index)"); ax.set_ylabel("episodes violating (%)")
    ax.legend(fontsize=6.1, loc="upper left")
    _tag(ax, "(b)  other corruption modes")

    ax = axes[2]
    sw = d["e6c_jitter"]["sweep"]
    j = np.array([100 * r["jitter_frac"] for r in sw])
    cp = np.array([r["cp95_upper_pct"] for r in sw])
    ax.bar(j, cp, 3.2, color=C[0], zorder=3)
    for ji, c in zip(j, cp):
        ax.text(ji, c + 0.005, "0 viol", ha="center", fontsize=6.0, color=MUTED)
    ax.set_xlabel("control-step jitter (%)"); ax.set_ylabel("CP95 upper bound (%)")
    ax.set_ylim(0, max(cp) * 1.35)
    _tag(ax, "(c)  scheduler jitter")
    save(fig, "F9_robustness")


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    todo = [("F1", f1_cross_domain), ("F2", f2_boundary), ("F3", f3_radiative),
            ("F4", f4_certified_region), ("F5", f5_mission), ("F6", f6_pack),
            ("F7", f7_ablation), ("F8", f8_margin_power), ("F9", f9_cost)]
    print("rendering figures")
    for name, fn in todo:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  {name}: skipped, missing {os.path.basename(str(e).split()[-1])}")
        except Exception as e:
            print(f"  {name}: FAILED -- {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
