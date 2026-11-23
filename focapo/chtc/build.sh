#!/bin/bash
# Runs inside a CHTC build job: build focapo.sif and copy it to staging.
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export APPTAINER_TMPDIR=$PWD/tmp; mkdir -p $APPTAINER_TMPDIR
set -ex
apptainer build focapo.sif ./focapo.def
apptainer exec focapo.sif python -c "import numpy, pandas, scipy, xgboost, gurobipy; print('ok gurobi', gurobipy.gurobi.version(), 'xgb', xgboost.__version__)"
cp focapo.sif /staging/p/prbrahmbhatt/focapo.sif && rm -f focapo.sif
ls -la /staging/p/prbrahmbhatt/
