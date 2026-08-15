"""Statistical machinery. Every safety claim here is a Bernoulli proportion with zero
observed failures, which is exactly the case where a naive "0 %" is meaningless and a
one-sided Clopper-Pearson bound is the honest statement. Continuous outcomes get bootstrap
intervals, paired comparisons get Wilcoxon with an effect size, and monotonicity claims get
a rank correlation with a permutation p-value.

No p-value is returned without an effect size and an interval beside it.
"""
import numpy as np
from scipy.stats import beta as _beta


def cp_upper(k, n, alpha=0.05):
    """One-sided Clopper-Pearson upper bound on a failure rate, k failures in n trials.

    Computed from the Beta quantile rather than by summing binomial terms: the direct sum
    overflows for a few hundred trials once k is more than a handful, which is exactly the
    regime an unsafe arm lands in."""
    if n == 0:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    if k >= n:
        return 1.0
    return float(_beta.ppf(1.0 - alpha, k + 1, n - k))


def n_required(target_rate, alpha=0.05):
    """Trials needed to certify a failure rate below `target_rate` at 1-alpha, given zero
    observed failures. Inverts cp_upper: n = log(alpha) / log(1 - target)."""
    return int(np.ceil(np.log(alpha) / np.log(1.0 - target_rate)))


def bootstrap_ci(x, stat=np.mean, reps=10000, alpha=0.05, seed=0):
    """Percentile bootstrap interval for any statistic of a sample."""
    x = np.asarray(x, float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(reps, x.size))
    boot = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.percentile(boot, 100 * alpha / 2)), \
        float(np.percentile(boot, 100 * (1 - alpha / 2)))


def wilcoxon_paired(a, b, seed=0, reps=20000):
    """Paired comparison by signed-rank, with a permutation p-value (exact enough at these
    sample sizes and free of the normal approximation), the Hodges-Lehmann median shift, and
    the rank-biserial correlation as effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return dict(n=0, p=1.0, hodges_lehmann=0.0, rank_biserial=0.0, W=0.0)
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    Wp = float(r[d > 0].sum())
    Wm = float(r[d < 0].sum())
    W = min(Wp, Wm)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    null = np.minimum((r * (signs > 0)).sum(1), (r * (signs < 0)).sum(1))
    p = float((null <= W).mean())
    walsh = (d[:, None] + d[None, :]) / 2.0
    hl = float(np.median(walsh[np.triu_indices(n)]))
    rb = float((Wp - Wm) / (n * (n + 1) / 2.0))
    return dict(n=n, p=p, hodges_lehmann=hl, rank_biserial=rb, W=W)


def _rank(a):
    """Mid-ranks, so tied values share a rank instead of being ordered by array position.

    `argsort(argsort(a))` is the fast rank transform and it is wrong in the presence of ties:
    it breaks them by index, which turns a *constant* series into a perfectly ordered one. V3-5
    swept ten years of dormancy, every node returned the same delivered charge because a
    different constraint was binding throughout, and this function duly reported rho = +1.000,
    p = 0.0002 for a relationship that does not exist. Mid-ranks make a constant series
    constant, and `spearman_perm` then reports it as undefined rather than as perfect."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, float)
    r[order] = np.arange(a.size, dtype=float)
    # average the ranks within each run of equal values
    s = a[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman_perm(x, y, reps=20000, seed=0):
    """Spearman rho with a permutation p-value, for monotonicity claims across a sweep.

    Returns rho = nan when either series is constant: there is no monotone relationship to
    report, and reporting one would be worse than reporting nothing.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return dict(rho=float("nan"), p=float("nan"), n=int(x.size), degenerate=True)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    for i in range(reps):
        null[i] = np.corrcoef(rx, rng.permutation(ry))[0, 1]
    p = float((np.abs(null) >= abs(rho)).mean())
    return dict(rho=rho, p=p, n=int(x.size), degenerate=False)


def summarize_safety(name, k, n, extra=None):
    """Standard record for a zero-failure safety claim."""
    rec = dict(name=name, trials=int(n), violations=int(k),
               rate_pct=100.0 * k / n if n else float("nan"),
               cp95_upper_pct=100.0 * cp_upper(k, n))
    if extra:
        rec.update(extra)
    return rec
