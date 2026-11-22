#!/bin/bash
# Phase B: 7-day comparison set, seed 42 (run from focapo/). Runs 4 at a time.
cd "$(dirname "$0")/.."
O=results/phaseB
mkdir -p $O
run() { name=$1; shift; python run_experiment.py "$@" --out $O/$name > $O/$name.log 2>&1; echo "done $name: $(tail -1 $O/$name.log)"; }
export -f run; export O
cat <<LIST | xargs -P 4 -L 1 bash -c 'run "$@"' _
ind_paper        --scenario individual
gne_paper        --scenario gne
ind_b2           --scenario individual --fixes b2
ind_fixed        --scenario individual --baseline fixed
gne_paper_cadmm  --scenario gne --commit-mode admm
gne_b4           --scenario gne --fixes b4
gne_fixed_cadmm  --scenario gne --baseline fixed --commit-mode admm
central_paper    --scenario central
central_fixed    --scenario central --baseline fixed
LIST
