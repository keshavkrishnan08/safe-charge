"""E10 -- the certified region, as a surface.

Everything so far reports safety at a point in design space. This maps it. Two sweeps, each
a full two-dimensional grid with an independent Monte-Carlo population at every node, so the
boundary between "certified" and "not" is measured rather than interpolated from two runs.

  (a) throttle K   x   true resistance scale s_R      -- how far past the rated bound the
                                                         certificate still reaches, and what
                                                         K it costs to get there
  (b) sensor bias  x   cooling loss                   -- the two faults that are invisible to
                                                         each other, swept jointly

Each node reports a Clopper-Pearson 95 % upper bound on the violation rate, which is the
honest quantity to plot: at zero observed failures the point estimate is 0 everywhere and
tells you nothing, while the bound falls as evidence accumulates and can be compared across
nodes with different populations.

    python zeroguard/exp/e10_certified_region.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from zeroguard import stats
from safe_charge import BatteryROM, project_current

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DT, TLIM, VLIM, NSTEP = 30.0, 45.0, 4.20, 80
CM = dict(V=0.03, T=0.5, plate=0.006)
F_COOL, SR = 0.25, 1.8
QN = BatteryROM().p["Q_nom"]


def episode(sR_true, K, bias, cool_loss, Tamb, soc0, rng, sQ=None, sP=None):
    f = np.clip((sR_true - 1.0) / 0.8, 0.0, None)
    sQ = 1.0 - 0.20 * min(f, 1.0) if sQ is None else sQ
    sP = 1.0 + 0.6 * f if sP is None else sP
    plant = BatteryROM(cell_scale=dict(R=sR_true, Q=sQ, plate=sP))
    plant.p = dict(plant.p); plant.p["hA"] *= (1.0 - cool_loss)
    est = BatteryROM(cell_scale=dict(R=SR, Q=1.0, plate=1.0))
    dT0 = CM["T"] + K * (SR - 1.0)
    s = plant.init_state(soc0, Tamb)
    mT = mV = -1e9
    for _ in range(NSTEP):
        seen = s if bias == 0.0 else dict(s, T=s["T"] - bias)
        I, _ = project_current(est, seen, 3 * QN, DT, Tamb, Vlim=VLIM, Tlim=TLIM,
                               margin=est.plating_margin(), dV=CM["V"], dT=dT0,
                               dP=CM["plate"], cool_frac=F_COOL)
        s, o = plant.step(s, float(max(0.0, I)), DT, Tamb)
        mT = max(mT, o["T"]); mV = max(mV, o["V"])
        if s["soc"] >= 0.80:
            break
    return (mT > TLIM + 1e-9 or mV > VLIM + 1e-9), mT, s["soc"]


def grid_sweep(xs, ys, per_node, kind, seed=77):
    rng = np.random.default_rng(seed)
    viol = np.zeros((len(ys), len(xs)), int)
    cp = np.zeros((len(ys), len(xs)))
    soc = np.zeros((len(ys), len(xs)))
    pk = np.zeros((len(ys), len(xs)))
    t0 = time.time()
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            v, socs, peaks = 0, [], []
            for _ in range(per_node):
                Tamb = float(rng.uniform(26.0, 38.0))
                soc0 = float(rng.uniform(0.08, 0.20))
                if kind == "K_vs_sR":
                    bad, mT, sc = episode(y, x, 0.0, float(rng.uniform(0.0, 0.33)),
                                          Tamb, soc0, rng)
                else:                                     # bias vs cooling loss
                    bad, mT, sc = episode(float(rng.uniform(1.0, 1.8)), 15.0, x, y,
                                          Tamb, soc0, rng)
                v += int(bad); socs.append(sc); peaks.append(mT)
            viol[i, j] = v
            cp[i, j] = 100.0 * stats.cp_upper(v, per_node)
            soc[i, j] = float(np.mean(socs)); pk[i, j] = float(np.max(peaks))
    return dict(x=list(map(float, xs)), y=list(map(float, ys)), per_node=per_node,
                violations=viol.tolist(), cp95_upper_pct=cp.tolist(),
                mean_soc=soc.tolist(), worst_peak_T=pk.tolist(),
                total_episodes=int(len(xs) * len(ys) * per_node),
                seconds=round(time.time() - t0, 1))


def main():
    out = {}
    print("E10 -- mapping the certified region\n")

    Ks = np.arange(0.0, 26.0, 2.0)
    sRs = np.arange(1.0, 3.21, 0.20)
    print(f"(a) K x s_R : {len(Ks)} x {len(sRs)} nodes")
    a = grid_sweep(Ks, sRs, per_node=90, kind="K_vs_sR")
    out["K_vs_sR"] = a
    print(f"    {a['total_episodes']:,} episodes in {a['seconds']}s")
    cp = np.array(a["cp95_upper_pct"]); vi = np.array(a["violations"])
    print(f"    certified nodes (zero violations): {(vi==0).sum()}/{vi.size}")
    for i, s_ in enumerate(a["y"]):
        row = "".join("#" if vi[i, j] == 0 else "." for j in range(len(a["x"])))
        print(f"    s_R={s_:>4.1f}  {row}")
    print(f"    (columns K = {a['x'][0]:.0f} .. {a['x'][-1]:.0f};  # = certified)")

    biases = np.arange(0.0, 16.1, 1.0)
    cools = np.arange(0.0, 0.61, 0.05)
    print(f"\n(b) sensor bias x cooling loss : {len(biases)} x {len(cools)} nodes")
    b = grid_sweep(biases, cools, per_node=90, kind="bias_vs_cool")
    out["bias_vs_cool"] = b
    vi2 = np.array(b["violations"])
    print(f"    {b['total_episodes']:,} episodes in {b['seconds']}s")
    print(f"    certified nodes (zero violations): {(vi2==0).sum()}/{vi2.size}")
    for i, c_ in enumerate(b["y"]):
        row = "".join("#" if vi2[i, j] == 0 else "." for j in range(len(b["x"])))
        print(f"    loss={c_:>4.2f}  {row}")
    print(f"    (columns bias = {b['x'][0]:.0f} .. {b['x'][-1]:.0f} C;  # = certified)")

    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "e10_certified_region.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RES,'e10_certified_region.json')}")


if __name__ == "__main__":
    main()
