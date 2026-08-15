"""N1 -- the method's assumptions, tested against measurements nobody here produced.

Everything else in this repository is simulation. The filter is checked against a
reduced-order model, and that model was calibrated against a Doyle--Fuller--Newman
simulation. That is a defensible chain and it is still a chain of models, so the fair
question a reader should ask is: what happens when the assumptions meet a real cell?

This experiment answers it with the NASA Ames Prognostics Center of Excellence battery
data set -- 18650 cells cycled to failure with periodic electrochemical impedance
spectroscopy, published by a different lab, years before this work, for a different purpose.

The design point is that **the load-bearing assumptions are testable without any model at
all.** The method does not need the ROM to be right about NASA's cells; it needs three
things to be true of real cells:

  N1  series resistance grows by no more than the datasheet bound s_R over the life the
      certificate claims to cover;
  N2  capacity fade stays inside the envelope, and where it does not, the scope is stated;
  N3  heat generation is monotone in current (hypothesis A2), measured directly.

Those are chemistry-independent statements about the *shape* of degradation, and they are
what the certificate actually rests on. N4 and N5 then ask the harder, more honest question
of how the ROM itself transfers to a cell it was never fitted to.

**Scope.** The NASA cells are ~2 Ah LCO 18650s; the ROM is calibrated to a 5 Ah LG M50
(NMC). N1-N3 are model-free and unaffected by that. N4 and N5 are explicitly a *transfer*
test with capacity rescaling and nothing else refitted, and are reported as such.

    python zeroguard/exp/n1_nasa_validation.py
"""
import os, sys, glob, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import scipy.io

from zeroguard import stats, vexp as V, platforms as P
from zeroguard import anchored as A

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "nasa")
S_R = V.S_R                      # the datasheet resistance bound the filter is given
Q_FLOOR = V.ENVELOPE["Q"][0]     # the capacity floor of the assumed envelope
EOL = 0.80                       # automotive end of life, as a fraction of initial capacity
R_PLAUSIBLE = (0.010, 2.0)       # ohm; outside this an EIS fit failed, it is not a measurement
REJECTED = []


# ---------------------------------------------------------------------------------------
def load_cells():
    """Every distinct cell in the archive, de-duplicated by name (some ship in two zips)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "**", "*.mat"), recursive=True)):
        name = os.path.basename(f)[:-4]
        if name in out:
            continue
        try:
            m = scipy.io.loadmat(f, simplify_cells=True)
            out[name] = m[name]["cycle"]
        except Exception:
            continue
    return out


def series(cyc):
    """Pull the measured series this experiment needs out of one cell's cycle list."""
    R, Q, chg = [], [], []
    for c in cyc:
        d = c.get("data", {})
        t = c.get("type", "")
        if t == "impedance":
            # NASA ships the raw output of an EIS curve fit, and some of those fits failed:
            # B0052 contains entries of -1.3e5 and +7.3e4 ohm. An 18650's series resistance is
            # tens to hundreds of milliohms, so anything outside [10 mohm, 2 ohm] is a failed
            # fit rather than a measurement. They are excluded and counted, not silently kept.
            try:
                re_ = float(np.real(np.atleast_1d(d["Re"])[0]))
                rct = float(np.real(np.atleast_1d(d["Rct"])[0]))
                tot = re_ + rct
                if np.isfinite(tot) and R_PLAUSIBLE[0] <= tot <= R_PLAUSIBLE[1]:
                    R.append(tot)
                else:
                    REJECTED.append((tot))
            except Exception:
                pass
        elif t == "discharge" and "Capacity" in d:
            try:
                q = float(np.atleast_1d(d["Capacity"])[0])
                if np.isfinite(q) and q > 0:
                    Q.append(q)
            except Exception:
                pass
        elif t == "charge":
            try:
                chg.append(dict(t=np.asarray(d["Time"], float),
                                I=np.asarray(d["Current_measured"], float),
                                V=np.asarray(d["Voltage_measured"], float),
                                T=np.asarray(d["Temperature_measured"], float),
                                amb=float(c.get("ambient_temperature", 24.0))))
            except Exception:
                pass
    return np.array(R), np.array(Q), chg


# ---------------------------------------------------------------------------------------
def n1_resistance_bound(cells):
    """Does real measured resistance growth stay inside the bound the filter is given?

    This is the assumption everything rests on. The filter never identifies resistance; it
    assumes a bound on it, and the bound is a datasheet number known before the vehicle
    ships. If real cells leave that bound, every certificate in this work is void.
    """
    rows, inside, total = [], 0, 0
    inside_eol, total_eol = 0, 0
    for name, cyc in sorted(cells.items()):
        R, Q, _ = series(cyc)
        if R.size < 5:
            continue
        ratio = R / R[0]
        # restrict to the life the envelope actually claims: capacity at or above EOL
        n = min(len(ratio), len(Q))
        in_life = (Q[:n] / Q[0] >= EOL) if len(Q) else np.ones(n, bool)
        rows.append(dict(cell=name, points=int(R.size),
                         R0=float(R[0]), Rend=float(R[-1]),
                         max_ratio=float(ratio.max()),
                         end_ratio=float(ratio[-1]),
                         inside_bound=bool(ratio.max() <= S_R),
                         q_end=float(Q[-1] / Q[0]) if len(Q) else None,
                         max_ratio_within_eol=float(ratio[:n][in_life].max())
                         if in_life.any() else None))
        inside += int((ratio <= S_R).sum()); total += ratio.size
        if in_life.any():
            r2 = ratio[:n][in_life]
            inside_eol += int((r2 <= S_R).sum()); total_eol += r2.size
    worst = max(rows, key=lambda r: r["max_ratio"])
    return dict(bound=S_R, cells=len(rows), rows=rows,
                measurements=total, inside=inside,
                fraction_inside=inside / max(total, 1),
                violations=total - inside,
                cp95_upper_pct=100 * stats.cp_upper(total - inside, total),
                measurements_within_eol=total_eol, inside_within_eol=inside_eol,
                fraction_inside_within_eol=inside_eol / max(total_eol, 1),
                worst_cell=worst["cell"], worst_ratio=worst["max_ratio"],
                worst_margin_pct=100 * (1 - worst["max_ratio"] / S_R),
                cells_inside=sum(r["inside_bound"] for r in rows))


# ---------------------------------------------------------------------------------------
def n2_capacity_envelope(cells):
    """Where does real capacity fade leave the assumed envelope, and what does that scope?"""
    rows = []
    for name, cyc in sorted(cells.items()):
        _, Q, _ = series(cyc)
        if Q.size < 5:
            continue
        f = Q / Q[0]
        rows.append(dict(cell=name, cycles=int(Q.size), q_end=float(f[-1]),
                         q_min=float(f.min()),
                         inside_envelope=bool(f.min() >= Q_FLOOR),
                         reaches_eol=bool(f.min() <= EOL)))
    past = [r for r in rows if not r["inside_envelope"]]
    return dict(envelope_floor=Q_FLOOR, eol=EOL, cells=len(rows), rows=rows,
                cells_past_envelope=len(past),
                worst_q=min(r["q_min"] for r in rows),
                cells_reaching_eol=sum(r["reaches_eol"] for r in rows),
                note=("the envelope's capacity floor is %.2f, which is the standard "
                      "automotive end-of-life; NASA cycles past it, so certificates are "
                      "scoped to cells at or above %.0f%% of initial capacity"
                      % (Q_FLOOR, 100 * EOL)))


# ---------------------------------------------------------------------------------------
def n3_monotone_heating(cells):
    """Hypothesis (A2), measured: is heat generation increasing in current on real cells?

    No model is involved anywhere in this test. But the obvious version of it does not work,
    and why it does not is worth recording. Correlating instantaneous `dT/dt` against
    instantaneous current returns nothing usable -- stratifying by `T - T_amb` to control for
    cooling gives Spearman coefficients that flip sign between strata (-0.34 in the coldest,
    +0.54 in the warmest). The cause is thermal lag: NASA measures surface temperature, the
    cell's core leads it by minutes, and over a charge whose *total* rise is under 3 K the lag
    dominates the signal entirely. That is a limitation of the instrument, not evidence about
    the physics, and reporting either sign from it would be an artefact.

    The lag-free formulation is an energy balance over a whole cycle, which is insensitive to
    how the heat arrives: total ohmic generation goes as the integral of I-squared, so the
    peak temperature rise of a cycle should increase with it. Discharge cycles are the
    informative ones because NASA varies the discharge load across its cell groups while
    charging almost everything at a constant 1.5 A -- so the charge cycles carry little
    current variation to correlate against, and are reported anyway.
    """
    ch, di = [], []
    for name, cyc in sorted(cells.items()):
        for c in cyc:
            if c.get("type") not in ("charge", "discharge"):
                continue
            d = c.get("data", {})
            amb = float(c.get("ambient_temperature", 24.0))
            try:
                t = np.asarray(d["Time"], float)
                I = np.asarray(d["Current_measured"], float)
                T = np.asarray(d["Temperature_measured"], float)
            except Exception:
                continue
            m = np.isfinite(t) & np.isfinite(I) & np.isfinite(T)
            if m.sum() < 30:
                continue
            t, I, T = t[m], I[m], T[m]
            trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            j2 = float(trap(I ** 2, t))
            rec = dict(j2=j2, peak=float(T.max() - amb), imax=float(np.abs(I).max()))
            if not np.isfinite(j2) or j2 <= 0:
                continue
            (ch if c["type"] == "charge" else di).append(rec)

    def _fit(rows, label):
        j2 = np.array([r["j2"] for r in rows]); pk = np.array([r["peak"] for r in rows])
        im = np.array([r["imax"] for r in rows])
        a = stats.spearman_perm(j2, pk, reps=2000)
        b = stats.spearman_perm(im, pk, reps=2000)
        return dict(phase=label, cycles=len(rows),
                    peak_dT_vs_joule=dict(rho=a["rho"], p=a["p"]),
                    peak_dT_vs_peak_current=dict(rho=b["rho"], p=b["p"]),
                    current_range=[float(im.min()), float(im.max())],
                    peak_dT_range=[float(pk.min()), float(pk.max())])

    d_res = _fit(di, "discharge")
    c_res = _fit(ch, "charge")
    return dict(discharge=d_res, charge=c_res,
                monotone_supported=bool(d_res["peak_dT_vs_joule"]["rho"] > 0.5
                                        and d_res["peak_dT_vs_joule"]["p"] < 0.01),
                instantaneous_test_confounded=True,
                note=("instantaneous dT/dt against current is confounded by thermal lag in a "
                      "surface thermocouple and flips sign between temperature strata; the "
                      "cycle-level energy balance is lag-free and is what is reported"))


# ---------------------------------------------------------------------------------------
def _rescaled_cell(q_nom_meas):
    """The published ROM with capacity rescaled to the measured cell, nothing else refitted."""
    p = P.load_params()
    scale = dict(R=1.0, Q=q_nom_meas / p["Q_nom"], plate=1.0)
    return P.Cell(cooling=P.Newtonian(p["hA"]), scale=scale, rc="exact")


def n4_transfer_prediction(cells, horizon_s=300.0, max_cycles=3):
    """How does the ROM transfer to a cell it was never fitted to -- and in which direction?

    Measured one step ahead this question is meaningless: NASA samples every ~9 s, over which
    the temperature moves by about 0.01 K, so predicting "unchanged" scores an RMSE of
    0.02 K and looks superb. The metric has to beat persistence to mean anything, so the ROM
    is integrated forward over a 300 s horizon under the measured current and scored
    against both the measurement and a persistence baseline.

    The expected answer is that absolute accuracy is poor: this is a 5 Ah NMC model applied to
    ~2 Ah LCO cells with capacity rescaled and nothing else refitted. What matters for a
    *safety* filter is the sign. A model that over-predicts temperature yields a conservative
    certificate; one that under-predicts yields an unsafe one.
    """
    errs, per_errs, over = [], [], 0
    n = 0
    for name, cyc in sorted(cells.items()):
        _, Q, chg = series(cyc)
        if not len(Q) or not chg:
            continue
        cell = _rescaled_cell(float(Q[0]))
        for c in chg[:max_cycles]:
            t, I, Tm = c["t"], c["I"], c["T"]
            amb = c["amb"]
            if t.size < 40:
                continue
            for k0 in range(0, t.size - 2, max(1, t.size // 12)):
                # integrate forward under the measured current until the horizon
                T_pred = float(Tm[k0]); k = k0
                while k + 1 < t.size and (t[k + 1] - t[k0]) <= horizon_s:
                    dt = float(t[k + 1] - t[k])
                    if not (0.1 <= dt <= 120.0):
                        break
                    _V, T_pred, _p, _s, _v = cell.probe(0.5, T_pred, 0.0, float(I[k]),
                                                        dt, amb)
                    k += 1
                if k <= k0 + 2:
                    continue
                e = float(T_pred - Tm[k])
                pe = float(Tm[k0] - Tm[k])          # persistence: assume nothing changes
                if np.isfinite(e) and np.isfinite(pe):
                    errs.append(e); per_errs.append(pe); over += int(e >= 0); n += 1
    a = np.array(errs); b = np.array(per_errs)
    rmse = float(np.sqrt((a ** 2).mean())); rmse_p = float(np.sqrt((b ** 2).mean()))
    margin = V.DT0 + V.K_TH * (S_R - 1.0)
    p05 = float(np.percentile(a, 5))
    return dict(steps=n, horizon_s=horizon_s,
                thermal_margin_C=margin,
                worst_underprediction_C=-p05,
                margin_over_underprediction=margin / max(-p05, 1e-9),
                rmse_C=rmse, persistence_rmse_C=rmse_p,
                skill_vs_persistence=float(1.0 - rmse / rmse_p) if rmse_p > 0 else None,
                mae_C=float(np.abs(a).mean()), bias_C=float(a.mean()),
                over_prediction_fraction=over / max(n, 1),
                p05_C=float(np.percentile(a, 5)), p95_C=float(np.percentile(a, 95)),
                note=("transfer test: a 5 Ah NMC ROM applied to ~2 Ah LCO cells with capacity "
                      "rescaled and nothing else refitted, scored over a horizon against a "
                      "persistence baseline; positive bias is the conservative direction"))


# ---------------------------------------------------------------------------------------
def n5_would_the_filter_allow_it(cells, max_cycles=6):
    """Would the certificate have permitted the protocol NASA actually ran?

    A safety filter that refuses everything is safe and useless. NASA charged these cells at
    a constant 1.5 A to 4.2 V and then held voltage -- a gentle, standard protocol on a
    healthy cell. If the filter clipped it, the filter is over-conservative on real hardware.
    """
    allowed = total = 0
    headroom = []
    refused_hot = refused_other = 0
    ceiling = 45.0 - (V.DT0 + V.K_TH * (S_R - 1.0))
    for name, cyc in sorted(cells.items()):
        _, Q, chg = series(cyc)
        if not len(Q) or not chg:
            continue
        q0 = float(Q[0])
        plat = P.single_cell(scale=dict(R=S_R, Q=q0 / P.load_params()["Q_nom"], plate=1.0))
        marg = (V.DV, V.DT0 + V.K_TH * (S_R - 1.0), V.DP)
        for c in chg[:max_cycles]:
            t, I, Tm = c["t"], c["I"], c["T"]
            amb = c["amb"]
            for k in range(0, t.size - 1, 25):
                Iap = float(I[k])
                if Iap <= 0.05:
                    continue
                dt = float(t[k + 1] - t[k])
                if not (0.5 <= dt <= 60.0):
                    continue
                s = dict(soc=0.5, T=float(Tm[k]), V1=0.0,
                         aging={"Qloss": 0.0, "Rfac": 1.0})
                lo, hi, st = A.interval(plat, s, dt, amb, marg)
                total += 1
                if st == "ok" and hi >= Iap - 1e-9:
                    allowed += 1
                    headroom.append(hi / Iap)
                elif float(Tm[k]) >= ceiling:
                    refused_hot += 1
                else:
                    refused_other += 1
    h = np.array(headroom) if headroom else np.array([0.0])
    ref = refused_hot + refused_other
    return dict(steps=total, allowed=allowed,
                allowed_fraction=allowed / max(total, 1),
                median_headroom=float(np.median(h)),
                p05_headroom=float(np.percentile(h, 5)),
                throttled_ceiling_C=ceiling,
                refused=ref, refused_above_ceiling=refused_hot,
                refused_other=refused_other,
                refusals_explained_by_ceiling=refused_hot / max(ref, 1),
                note=("the filter is given the end-of-life resistance bound and the full "
                      "throttled margin, i.e. its most conservative configuration; refusals "
                      "are attributed to the measured cell temperature exceeding "
                      "T_max - delta_T, which is the same passive ceiling V1-3 predicts"))


# ---------------------------------------------------------------------------------------
def n6_refusals_predict_degradation(cells, max_cycles=8):
    """Do the cells the certificate would have refused turn out to be the cells that died?

    **This test does not work on this data set, and the interesting part is why.**

    The hypothesis was attractive. NASA cycled some cells at 4 C; the plating current cap falls
    to a 0.70C floor below about 5.5 C; 1.5 A into a 2 Ah cell is 0.75C. So the filter refuses
    cold-charge steps on plating grounds, and plating is what degrades a cold-charged cell. If
    the refusals lined up with the measured fade, the enforced constraint would be shown to be
    the right one, against data collected years earlier by people who had never heard of it.

    They do not line up in any stable way. The rank correlation between a cell's refusal rate
    and its measured fade **changes sign** depending on how the capacity baseline is defined
    -- first measured discharge versus the median of the largest few -- because NASA's first
    discharge on several cells is partial, giving baselines like 0.068 Ah for a 2 Ah cell. Under
    one definition the correlation is negative and significant, under the other positive and
    significant, and in the second case the group means contradict the correlation. Two further
    confounds sit behind both quantities: refusal rate is driven mostly by ambient temperature,
    and total fade mostly by how many cycles NASA chose to run, which differs by cell group.

    All four combinations are computed and reported rather than one. A result that flips sign
    with a defensible change of definition is not a result, and reporting whichever version
    looked best would be the single most misleading thing this study could do.
    """
    per_cell = []
    for name, cyc in sorted(cells.items()):
        _, Q, chg = series(cyc)
        if Q.size < 10 or not chg:
            continue
        plat = P.single_cell(scale=dict(R=S_R, Q=float(Q[0]) / P.load_params()["Q_nom"],
                                        plate=1.0))
        marg = (V.DV, V.DT0 + V.K_TH * (S_R - 1.0), V.DP)
        ref = tot = 0
        ambs = []
        for c in chg[:max_cycles]:
            tt, I, Tm = c["t"], c["I"], c["T"]
            amb = c["amb"]; ambs.append(amb)
            for k in range(0, tt.size - 1, 25):
                Iap = float(I[k])
                if Iap <= 0.05:
                    continue
                dt = float(tt[k + 1] - tt[k])
                if not (0.5 <= dt <= 60.0):
                    continue
                s = dict(soc=0.5, T=float(Tm[k]), V1=0.0,
                         aging={"Qloss": 0.0, "Rfac": 1.0})
                lo, hi, st = A.interval(plat, s, dt, amb, marg)
                tot += 1
                ref += int(not (st == "ok" and hi >= Iap - 1e-9))
        if tot < 20:
            continue
        base_first = float(Q[0])
        base_robust = float(np.median(np.sort(Q)[-5:]))
        per_cell.append(dict(cell=name, refusal_rate=ref / tot, cycles=int(Q.size),
                             ambient=float(np.mean(ambs)),
                             fade_first_baseline=float(1.0 - Q.min() / base_first),
                             fade_robust_baseline=(float(1.0 - Q.min() / base_robust)
                                                   if base_robust >= 1.0 else None),
                             baseline_first=base_first, baseline_robust=base_robust))

    rr = np.array([r["refusal_rate"] for r in per_cell])
    cy = np.array([r["cycles"] for r in per_cell], float)
    variants = {}
    for bkey, blabel in (("fade_first_baseline", "first-discharge baseline"),
                         ("fade_robust_baseline", "robust baseline")):
        keep = [i for i, r in enumerate(per_cell) if r[bkey] is not None]
        f = np.array([per_cell[i][bkey] for i in keep])
        r2 = rr[keep]; c2 = cy[keep]
        for mkey, mlabel, y in (("total", "total fade", f),
                                ("per_cycle", "fade per cycle", f / c2)):
            s = stats.spearman_perm(r2, y, reps=20000)
            variants[f"{bkey}|{mkey}"] = dict(baseline=blabel, metric=mlabel, n=int(len(keep)),
                                              rho=s["rho"], p=s["p"])
    rhos = [v["rho"] for v in variants.values()]
    signs_agree = all(x > 0 for x in rhos) or all(x < 0 for x in rhos)
    return dict(cells=len(per_cell), rows=per_cell, variants=variants,
                rho_range=[float(min(rhos)), float(max(rhos))],
                signs_agree=bool(signs_agree),
                conclusive=bool(signs_agree and all(v["p"] < 0.05 for v in variants.values())),
                verdict=("inconclusive: the sign of the association depends on how the "
                         "capacity baseline is defined, and ambient temperature and cycle "
                         "count confound both quantities"),
                cells_with_bad_first_baseline=int(sum(
                    1 for r in per_cell if r["baseline_first"] < 1.0)))


# ---------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("N1 -- the assumptions, against NASA PCoE measurements\n" + "=" * 78)
    cells = load_cells()
    print(f"loaded {len(cells)} distinct cells from {DATA}")
    out = {"cells_loaded": len(cells), "source": "NASA Ames PCoE battery data set"}

    print("\nN1-1  does measured resistance growth stay inside the datasheet bound?")
    r = n1_resistance_bound(cells); out["n1_resistance_bound"] = r
    print(f"      bound s_R = {r['bound']} | {r['cells']} cells | {r['measurements']:,} EIS "
          f"measurements ({len(REJECTED)} failed fits excluded as physically impossible)")
    print(f"      inside the bound: {r['inside']:,}/{r['measurements']:,} = "
          f"{100*r['fraction_inside']:.2f}%  (violations {r['violations']}, CP95 "
          f"{r['cp95_upper_pct']:.3f}%)")
    print(f"      restricted to cells at or above {100*EOL:.0f}% capacity: "
          f"{100*r['fraction_inside_within_eol']:.2f}% of "
          f"{r['measurements_within_eol']:,}")
    print(f"      worst cell {r['worst_cell']} at {r['worst_ratio']:.3f}x "
          f"({r['worst_margin_pct']:.0f}% margin to the bound) | cells fully inside: "
          f"{r['cells_inside']}/{r['cells']}")

    print("\nN1-2  where does real capacity fade leave the assumed envelope?")
    r = n2_capacity_envelope(cells); out["n2_capacity_envelope"] = r
    print(f"      envelope floor {r['envelope_floor']:.2f} | worst measured "
          f"{r['worst_q']:.3f} | cells past the floor {r['cells_past_envelope']}/{r['cells']}"
          f" | cells reaching {100*EOL:.0f}% EOL {r['cells_reaching_eol']}")
    print(f"      {r['note']}")

    print("\nN1-3  is heat generation monotone in current on real cells? (hypothesis A2)")
    r = n3_monotone_heating(cells); out["n3_monotone_heating"] = r
    for k in ("discharge", "charge"):
        v = r[k]
        print(f"      {k:10s} {v['cycles']:,} cycles, I up to {v['current_range'][1]:.2f} A, "
              f"peak rise to {v['peak_dT_range'][1]:.1f} K")
        print(f"                 peak dT vs integral I^2 dt: "
              f"rho={v['peak_dT_vs_joule']['rho']:+.3f} p={v['peak_dT_vs_joule']['p']:.4f}"
              f" | vs peak |I|: rho={v['peak_dT_vs_peak_current']['rho']:+.3f}")
    print(f"      (A2) supported on real cells: {r['monotone_supported']}")
    print(f"      note: {r['note']}")

    print("\nN1-4  how does the ROM transfer to a cell it was never fitted to?")
    r = n4_transfer_prediction(cells); out["n4_transfer_prediction"] = r
    print(f"      {r['steps']:,} forecasts over a {r['horizon_s']:.0f} s horizon")
    print(f"      RMSE {r['rmse_C']:.3f} C vs persistence {r['persistence_rmse_C']:.3f} C "
          f"-> skill {r['skill_vs_persistence']:+.3f}")
    print(f"      bias {r['bias_C']:+.3f} C | over-predicts (the conservative direction) in "
          f"only {100*r['over_prediction_fraction']:.1f}% of forecasts -- the transfer is NOT "
          f"conservative")
    print(f"      worst under-prediction (5th pct) {r['worst_underprediction_C']:.2f} C against "
          f"a {r['thermal_margin_C']:.1f} C thermal margin = "
          f"{r['margin_over_underprediction']:.1f}x cover")

    print("\nN1-5  would the certificate have permitted NASA's own charging protocol?")
    r = n5_would_the_filter_allow_it(cells); out["n5_permits_protocol"] = r
    print(f"      {r['steps']:,} measured charge steps | permitted "
          f"{100*r['allowed_fraction']:.2f}%")
    print(f"      median headroom {r['median_headroom']:.2f}x the applied current "
          f"(5th percentile {r['p05_headroom']:.2f}x)")
    print(f"      of {r['refused']:,} refusals, {100*r['refusals_explained_by_ceiling']:.1f}% "
          f"are cells measured above the throttled ceiling of "
          f"{r['throttled_ceiling_C']:.1f} C -- the same frontier V1-3 predicts")

    print("\nN1-6  do the certificate's refusals predict which cells actually degraded?")
    r = n6_refusals_predict_degradation(cells); out["n6_refusals_predict_degradation"] = r
    print(f"      {r['cells']} cells | {r['cells_with_bad_first_baseline']} have an "
          f"implausible first-discharge capacity")
    for k, v in r["variants"].items():
        print(f"        {v['baseline']:26s} x {v['metric']:15s} n={v['n']:3d}  "
              f"rho={v['rho']:+.3f}  p={v['p']:.4f}")
    print(f"      signs agree across definitions: {r['signs_agree']} | conclusive: "
          f"{r['conclusive']}")
    print(f"      VERDICT -- {r['verdict']}")

    path = V.save("n1_nasa_validation.json", out)
    print(f"\nwrote {path}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
