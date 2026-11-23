#!/bin/bash
# Laptop batch (full Gurobi license): deterministic central benchmark for every week
# (the joint quadratic MIQP exceeds the pip license, so it cannot run on CHTC).
cd "$(dirname "$0")/.."
O=results/all_weeks; mkdir -p $O
cat chtc/params/weeks_ok.txt | xargs -P 2 -I{} bash -c 'n=central_fixed_{}; [ -f '"$O"'/$n/financials.csv ] || python run_experiment.py --scenario central --baseline fixed --forecaster xgb --start-date {} --out '"$O"'/$n > '"$O"'/$n.log 2>&1; echo "done $n"'
