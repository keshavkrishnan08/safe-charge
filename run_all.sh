#!/usr/bin/env bash
# Reproduce the artifact end to end. Uses python3; override with PYTHON=... bash run_all.sh
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "======================================================================"
echo " 1/6  Safety-property checks (executable)"
echo "======================================================================"
"$PY" tests/test_safety.py

echo
echo "======================================================================"
echo " 2/6  Quickstart: charge an aged cell through the filter"
echo "======================================================================"
"$PY" examples/quickstart.py

echo
echo "======================================================================"
echo " 3/6  Software footprint: certified path vs the QP alternative"
echo "======================================================================"
"$PY" bench/footprint.py

echo
echo "======================================================================"
echo " 4/6  Solver comparison, quick n=15 (add --full for n=50)"
echo "======================================================================"
"$PY" reproduce/solver_comparison.py

echo
echo "======================================================================"
echo " 5/6  Dimension independence: one fixed certificate, a 4-channel envelope"
echo "======================================================================"
"$PY" reproduce/dimension_independence.py

echo
echo "======================================================================"
echo " 6/6  Price of oracle-freedom: localizes to the binding constraint"
echo "======================================================================"
"$PY" reproduce/price_of_oracle_freedom.py

echo
echo "All artifact checks completed."
