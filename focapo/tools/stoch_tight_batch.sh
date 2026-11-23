#!/bin/bash
# Laptop batch (full Gurobi license): stochastic GNE, S=10, H2 floor on expectation,
# with the tight ADMM tolerance (1e-3) so the result is the converged equilibrium.
# Output names gne_tight_<week> in results/stoch_weeks.  Run from focapo/.
cd "$(dirname "$0")/.."
O=results/stoch_weeks; mkdir -p $O
P=${P:-4}
for w in 2024-07-08 2024-04-15 2024-10-14 2025-01-13 2025-06-23; do
  echo "gne_tight_$w --scenario gne --baseline fixed --commit-mode admm --start-date $w"
done | xargs -P $P -L 1 bash -c 'n=$0; [ -f '"$O"'/$n/financials.csv ] || python run_experiment.py "$@" --forecaster xgb --scenarios 10 --stoch-h2 expected --tol-rt 1e-3 --tol-da 1e-3 --out '"$O"'/$n > '"$O"'/$n.log 2>&1; echo "done $n"'
