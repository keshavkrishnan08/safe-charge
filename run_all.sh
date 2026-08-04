#!/usr/bin/env bash
# Reproduce the artifact end to end. Uses python3; override with PYTHON=... bash run_all.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "======================================================================"
echo " 1/9  Safety-property checks (executable)"
echo "======================================================================"
"$PY" tests/test_safety.py

echo
echo "======================================================================"
echo " 2/9  Quickstart: charge an aged cell through the filter"
echo "======================================================================"
"$PY" examples/quickstart.py

echo
echo "======================================================================"
echo " 3/9  Software footprint: certified path vs the QP alternative"
echo "======================================================================"
"$PY" bench/footprint.py

echo
echo "======================================================================"
echo " 4/9  Real-time cost: latency, operation count, memory"
echo "======================================================================"
"$PY" bench/timing.py

echo
echo "======================================================================"
echo " 5/9  Solver comparison, quick n=15 (add --full for n=50)"
echo "======================================================================"
"$PY" reproduce/solver_comparison.py

echo
echo "======================================================================"
echo " 6/9  Dimension independence: one fixed certificate, a 4-channel envelope"
echo "======================================================================"
"$PY" reproduce/dimension_independence.py

echo
echo "======================================================================"
echo " 7/9  Price of oracle-freedom: localizes to the binding constraint"
echo "======================================================================"
"$PY" reproduce/price_of_oracle_freedom.py

echo
echo "======================================================================"
echo " 8/9  Bisection tolerance: what the iteration count buys"
echo "======================================================================"
"$PY" reproduce/tolerance_sweep.py

echo
echo "======================================================================"
echo " 9/9  Control step and robustness: step size, aging bound, margin gains"
echo "======================================================================"
"$PY" reproduce/step_size.py
echo
"$PY" reproduce/robustness.py

echo
echo "All artifact checks completed."
