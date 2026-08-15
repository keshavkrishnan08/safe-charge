# Paper source

LaTeX source and compiled PDF for the paper this code accompanies. The compiled
[`main_iecon.pdf`](main_iecon.pdf) is checked in, so you can read it without a TeX install.

## Build it yourself

Needs a TeX distribution with the `IEEEtran` class (TeX Live, MiKTeX, or TinyTeX).

```bash
pdflatex main_iecon.tex
bibtex   main_iecon
pdflatex main_iecon.tex
pdflatex main_iecon.tex
```

Output is `main_iecon.pdf` (6 pages).

## What's here

| File | Role |
|---|---|
| `main_iecon.tex` | the manuscript |
| `refs.bib` | bibliography |
| `results_macros.tex` | numeric results as macros (every number in the text) |
| `iecon_table.tex`, `safety_table.tex`, `appendix_tables.tex` | table bodies |
| `figures/` | the four figures (`fig_proj`, `fig5_pareto`, `fig3_aging`, `fig10_monotone`) |

The numbers in `results_macros.tex` are produced by the experiments in this repository; the reproduction
scripts under [`../reproduce/`](../reproduce/) regenerate the headline results.

## The 20-page vehicle paper

`zeroguard_sts.tex` is the long-form manuscript: the theory, and then four application
domains — ground, air, water, vacuum — across sixteen platforms. Applications are the bulk of
it by design.

```bash
python zeroguard/make_paper_macros.py     # regenerate numbers from results/*.json
cd paper && pdflatex zeroguard_sts && bibtex zeroguard_sts && pdflatex zeroguard_sts && pdflatex zeroguard_sts
```

Every number in the prose is a macro from `vehicle_macros.tex`, which is generated from the
result files. The paper cannot drift from the experiments; re-running an experiment and
rebuilding moves the manuscript with it.
