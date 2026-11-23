#!/bin/bash
# Build the job input bundle (code + PJM data + trained forecaster models) on the laptop.
#   chtc/make_bundle.sh  ->  chtc/bundle.tar.gz   (copy to CHTC ~/focapo/)
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"          # autonomous_market_rhg
cd "$REPO"
COPYFILE_DISABLE=1 tar --no-xattrs -czf focapo/chtc/bundle.tar.gz \
  --exclude='__pycache__' --exclude='results' --exclude='report' \
  --exclude='*.pdf' --exclude='*.png' --exclude='*.zip' --exclude='*.pptx' \
  --exclude='Focapo Paper overleaf' --exclude='FOCAPO_CPC_2027_Template*' \
  focapo/run_experiment.py focapo/med_scale_case_study/*.py \
  focapo/small_scale_case_study/*.py focapo/forecaster/*.py \
  data/raw/*.csv
ls -la focapo/chtc/bundle.tar.gz
# one small tarball of forecaster models per week (each job gets only its week)
mkdir -p focapo/chtc/models
for d in focapo/forecaster/models/*/; do
  w=$(basename "$d")
  COPYFILE_DISABLE=1 tar --no-xattrs -czf focapo/chtc/models/$w.tar.gz -C "$REPO" "focapo/forecaster/models/$w"
done
ls focapo/chtc/models | wc -l
