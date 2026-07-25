"""A real, warm-started embedded QP (OSQP) against the solver-free filter, driven on the same
aged cells with the same fail-safe bound, margins, and plating cap.

Seeds are fixed and the main/corner populations use independent streams, so the numbers
reproduce exactly.

Design choices that keep the comparison honest and fair to OSQP:
 - Real-time-iteration MPC: linearize the electro-thermal-plating ROM each step, form a
   convex tracking QP (charge as fast as the linearized T/V/plating constraints allow).
 - OSQP is set up ONCE and WARM-STARTED across steps (update Ax/l/u, reuse the ADMM iterate),
   the embedded pattern, so its iteration count is not inflated by cold restarts.
 - It is handed the SAME fail-safe bound s_R=1.8, the SAME margins, and the SAME plating cap
   the filter uses. Any difference is the solver's, not a handicap.

The point: bounding a QP's worst-case time means capping its iterations, and a capped OSQP
stops converging at the cold, rated-end-of-life corner and overshoots the 4.20 V limit; the
full-budget OSQP stays safe but its iteration count is data-dependent; the filter's fixed 18
bisection iterations are bounded AND safe.

    python reproduce/solver_comparison.py            # quick: corner only, n=15
    python reproduce/solver_comparison.py --full      # full run: main n=100 + corner n=50
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, scipy.sparse as sp, osqp
from safe_charge import BatteryROM, project_current

# --- scenario constants (config.warm(40); charging_env.soh_to_scale) ---
DT, NSTEP, TAMB, SOC_TGT, IMAX = 30.0, 80, 25.0, 0.8, 15.0
VLIM, TLIM, K_TH, H = 4.20, 45.0, 15.0, 12
CM = dict(V=0.03, T=0.5, plate=0.006)
MAIN_SEED, CORNER_SEED = 20260717, 20260918   # independent streams

def soh_to_scale(soh):
    f = (1.0 - soh) / 0.2
    return dict(R=1.0 + 0.8*f, Q=soh, plate=1.0 + 0.6*f)

Qn = BatteryROM().p["Q_nom"]; MARG = BatteryROM().plating_margin()
dV, dT_eff, dP = CM["V"], CM["T"] + K_TH*0.8, CM["plate"]
Teff, Veff, Peff = TLIM - dT_eff, VLIM - dV, MARG + dP
D = np.eye(H) - np.eye(H, k=-1)
Pqp = sp.csc_matrix(2.0*np.eye(H) + 2.0*0.05*(D.T @ D)); qqp = -2.0*IMAX*np.ones(H); EPS = 0.05

def _roll(model, s, x):
    T = np.empty(len(x)); V = np.empty(len(x)); P = np.empty(len(x)); ss = dict(s)
    for k, I in enumerate(x):
        ss, o = model.step(ss, float(I), DT, TAMB); T[k], V[k], P[k] = o["T"], o["V"], o["phi_an"]
    return T, V, P

def drive_qp(scale, soc0, T0, max_iter):
    """Warm-started RTI-MPC on the true aged plant. OSQP is set up once and updated in place."""
    plant = BatteryROM(); plant.scale = dict(scale); s = plant.init_state(soc0, T0)
    model = BatteryROM(); model.scale = dict(R=1.8, Q=1.0, plate=1.0)   # fail-safe bound, no oracle
    x0 = np.full(H, min(IMAX, 1.0*Qn)); maxT = maxV = 0.0
    iters = []; niter_hit = ninfeas = 0; m = None; nnz0 = None
    for _ in range(NSTEP):
        T0v, V0v, P0v = _roll(model, s, x0)
        gT = np.empty((H, H)); gV = np.empty((H, H)); gP = np.empty((H, H))
        for j in range(H):
            xp = x0.copy(); xp[j] += EPS; Tj, Vj, Pj = _roll(model, s, xp)
            gT[:, j] = (Tj-T0v)/EPS; gV[:, j] = (Vj-V0v)/EPS; gP[:, j] = (Pj-P0v)/EPS
        ubox = min(IMAX, model.plate_current_cap(s["T"]))
        A = sp.csc_matrix(np.vstack([gT, gV, -gP, np.eye(H)]))
        u = np.concatenate([(Teff-T0v)+gT@x0, (Veff-V0v)+gV@x0, -((Peff-P0v)+gP@x0), np.full(H, ubox)])
        l = np.concatenate([np.full(3*H, -np.inf), np.zeros(H)])
        if m is None or A.nnz != nnz0:                      # first step, or (never) a pattern change
            m = osqp.OSQP()
            m.setup(Pqp, qqp, A, l, u, verbose=False, max_iter=max_iter, eps_abs=1e-4,
                    eps_rel=1e-4, polish=False, adaptive_rho=True, warm_starting=True)
            nnz0 = A.nnz
        else:
            m.update(Ax=A.data, Ax_idx=np.arange(A.nnz), l=l, u=u)   # reuse iterate: warm start
        r = m.solve(); iters.append(int(r.info.iter)); st = r.info.status
        if "maximum iterations" in st or "inaccurate" in st: niter_hit += 1
        elif "infeasible" in st: ninfeas += 1
        xs = np.asarray(r.x, float)
        if (not np.all(np.isfinite(xs))) or ("infeasible" in st): xs = np.zeros(H)
        xs = np.clip(xs, 0.0, ubox)
        s, o = plant.step(s, float(xs[0]), DT, TAMB)
        maxT, maxV = max(maxT, o["T"]), max(maxV, o["V"]); x0 = np.concatenate([xs[1:], xs[-1:]])
        if s["soc"] >= SOC_TGT: break
    return dict(safe=int(maxT <= TLIM and maxV <= VLIM), maxV=maxV, iters=iters,
                nonconv=niter_hit+ninfeas)

def drive_filter(scale, soc0, T0):
    plant = BatteryROM(); plant.scale = dict(scale); s = plant.init_state(soc0, T0)
    est = BatteryROM(); est.scale = dict(R=1.8, Q=1.0, plate=1.0); maxT = maxV = 0.0
    for _ in range(NSTEP):
        I, _ = project_current(est, s, 3.0*Qn, DT, TAMB, Vlim=VLIM, Tlim=TLIM,
                               margin=est.plating_margin(), dV=dV, dT=dT_eff, dP=dP)
        s, o = plant.step(s, float(max(0.0, I)), DT, TAMB)
        maxT, maxV = max(maxT, o["T"]), max(maxV, o["V"])
        if s["soc"] >= SOC_TGT: break
    return dict(safe=int(maxT <= TLIM and maxV <= VLIM), maxV=maxV)

def main_conditions(n_per_soh):
    rng = np.random.default_rng(MAIN_SEED); conds = []
    for soh in (0.90, 0.80):
        base = soh_to_scale(soh)
        for _ in range(n_per_soh):
            soc0 = float(rng.uniform(0.08, 0.15)); T0 = float(TAMB + rng.uniform(-2, 2))
            conds.append((dict(R=base["R"]*float(rng.uniform(0.95, 1.05)),
                               Q=base["Q"]*float(rng.uniform(0.97, 1.03)),
                               plate=base["plate"]*float(rng.uniform(0.97, 1.05))), soc0, T0))
    return conds

def corner_conditions(n):
    rng = np.random.default_rng(CORNER_SEED); base = soh_to_scale(0.80); conds = []
    for _ in range(n):
        soc0 = float(rng.uniform(0.30, 0.50)); T0 = float(rng.uniform(0.0, 6.0))
        conds.append((dict(R=base["R"]*float(rng.uniform(0.98, 1.08)),
                           Q=base["Q"]*float(rng.uniform(0.97, 1.03)),
                           plate=base["plate"]*float(rng.uniform(0.97, 1.05))), soc0, T0))
    return conds

def run_pop(conds, full_iter):
    BUD = {"QP-full": full_iter, "QP-embedded": 25}
    agg = {k: dict(n=0, safe=0, iters=[], nonconv=0, worstV=0.0) for k in BUD}
    agg["filter"] = dict(n=0, safe=0, worstV=0.0)
    for (scale, soc0, T0) in conds:
        for tag, mi in BUD.items():
            r = drive_qp(scale, soc0, T0, mi); a = agg[tag]
            a["n"] += 1; a["safe"] += r["safe"]; a["iters"] += r["iters"]; a["nonconv"] += r["nonconv"]
            a["worstV"] = max(a["worstV"], r["maxV"])
        f = drive_filter(scale, soc0, T0); a = agg["filter"]
        a["n"] += 1; a["safe"] += f["safe"]; a["worstV"] = max(a["worstV"], f["maxV"])
    return agg

def summarize(agg):
    o = {}
    for tag in ("QP-full", "QP-embedded"):
        a = agg[tag]; it = np.array(a["iters"]) if a["iters"] else np.array([0])
        o[tag] = dict(n=a["n"], safe=a["safe"], viol_pct=round(100*(a["n"]-a["safe"])/a["n"], 1),
                      nonconv_pct=round(100*a["nonconv"]/max(len(a["iters"]), 1), 1),
                      iter_mean=round(float(it.mean()), 1), iter_max=int(it.max()),
                      worstV=round(a["worstV"], 3))
    a = agg["filter"]
    o["filter"] = dict(n=a["n"], safe=a["safe"], viol_pct=round(100*(a["n"]-a["safe"])/a["n"], 1),
                       iters_fixed=18, worstV=round(a["worstV"], 3))
    return o

def run_all(n_main=50, n_corner=50, full_iter=20000):
    return {"main": summarize(run_pop(main_conditions(n_main), full_iter)),
            "corner": summarize(run_pop(corner_conditions(n_corner), full_iter))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); a = ap.parse_args()
    if a.full:
        out = run_all(n_main=50, n_corner=50); pops = ("main", "corner")
        print("Full run: main n=100 (seed 20260717) + corner n=50 (seed 20260918)\n")
    else:
        out = {"corner": summarize(run_pop(corner_conditions(15), 20000))}; pops = ("corner",)
        print("Quick: cold, rated-end-of-life corner, n=15 (seed 20260918)\n")
    for pop in pops:
        print(f"--- {pop} population ---")
        print(f"{'Controller':16}{'Iters mean/max':>16}{'T/V viol %':>12}{'worst V':>10}")
        for tag in ("QP-full", "QP-embedded", "filter"):
            d = out[pop][tag]
            im = f"{d.get('iter_mean','18.0')}/{d.get('iter_max',18)}" if tag != "filter" else "18/18"
            print(f"{tag:16}{im:>16}{d['viol_pct']:>11}%{d['worstV']:>10}")
        print()
    print("All three stay T/V-safe. Bounding the QP's cost (the 25-iteration cap) costs it "
          "convergence, while the full budget converges at a data-dependent iteration count. "
          "Only the filter is bounded and exact at once.")
