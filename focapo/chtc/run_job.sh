#!/bin/bash
# One HTCondor job = one simulation.  Usage: run_job.sh <name> <run_experiment args...>
set -e
NAME=$1; shift
tar -xzf bundle.tar.gz
export FOCAPO_DATA_DIR=$PWD/data/raw
export MPLBACKEND=Agg
python focapo/run_experiment.py "$@" --out out/$NAME
tar -czf $NAME.tar.gz -C out $NAME
rm -rf focapo data out bundle.tar.gz
