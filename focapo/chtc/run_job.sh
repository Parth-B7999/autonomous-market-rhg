#!/bin/bash
# One HTCondor job = one simulation.  Usage: run_job.sh <name> <week> <run_experiment args...>
set -e
NAME=$1; WEEK=$2; shift 2
tar -xzf bundle.tar.gz
[ -f "$WEEK.tar.gz" ] && tar -xzf "$WEEK.tar.gz"
export FOCAPO_DATA_DIR=$PWD/data/raw
export MPLBACKEND=Agg
python focapo/run_experiment.py --start-date "$WEEK" "$@" --out out/$NAME
tar -czf $NAME.tar.gz -C out $NAME
rm -rf focapo data out bundle.tar.gz "$WEEK.tar.gz"
