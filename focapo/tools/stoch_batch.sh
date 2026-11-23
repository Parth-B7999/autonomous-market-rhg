#!/bin/bash
# Laptop batch (full Gurobi license): stochastic individual + GNE, S=10, H2 floor on expectation,
# for the season weeks.  5 runs at a time (2 Gurobi threads each).  Run from focapo/.
cd "$(dirname "$0")/.."
O=results/stoch_weeks; mkdir -p $O
for w in 2024-07-08 2024-04-15 2024-10-14 2025-01-13 2025-06-23; do
  echo "gne_$w --scenario gne --baseline fixed --commit-mode admm --start-date $w"
  echo "ind_$w --scenario individual --baseline fixed --start-date $w"
done | xargs -P 5 -L 1 bash -c 'n=$0; python run_experiment.py "$@" --forecaster xgb --scenarios 10 --stoch-h2 expected --out '"$O"'/$n > '"$O"'/$n.log 2>&1; echo "done $n: $(tail -1 '"$O"'/$n.log)"'
