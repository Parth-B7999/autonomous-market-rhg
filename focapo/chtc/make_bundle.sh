#!/bin/bash
# Build the job input bundle (code + PJM data + trained forecaster models) on the laptop.
#   chtc/make_bundle.sh  ->  chtc/bundle.tar.gz   (copy to CHTC ~/focapo/)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"          # autonomous_market_rhg
cd "$REPO"
tar -czf focapo/chtc/bundle.tar.gz \
  --exclude='__pycache__' --exclude='results' --exclude='report' \
  --exclude='*.pdf' --exclude='*.png' --exclude='*.zip' --exclude='*.pptx' \
  --exclude='Focapo Paper overleaf' --exclude='FOCAPO_CPC_2027_Template*' \
  focapo/run_experiment.py focapo/med_scale_case_study/*.py \
  focapo/small_scale_case_study/*.py focapo/forecaster/*.py focapo/forecaster/models \
  data/raw/*.csv
ls -la focapo/chtc/bundle.tar.gz
