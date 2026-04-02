#!/bin/bash
# LR sweep: std=10, cov=50 — with and without cosine schedule
# 6 LRs × 2 = 12 runs total

LRS=(0.00001 0.00005 0.0001 0.0003 0.0005 0.001)
CFG="examples/cell_jepa/cfgs/subset4.yaml"
STD=10.0
COV=50.0

for LR in "${LRS[@]}"; do
  for COSINE in true false; do
    TAG="std10_cov50_lr${LR}_cosine${COSINE}"
    TAG="${TAG//./_}"
    OUTDIR="output/lr_sweep/${TAG}"
    echo ""
    echo "========================================================"
    echo "  std=${STD}  cov=${COV}  lr=${LR}  cosine=${COSINE}"
    echo "  → ${OUTDIR}"
    echo "========================================================"
    conda run -n eb_jepa python -m examples.cell_jepa.main \
      --fname="${CFG}" \
      --meta.output_dir="${OUTDIR}" \
      --loss.std_coeff=${STD} \
      --loss.cov_coeff=${COV} \
      --optim.lr=${LR} \
      --optim.use_cosine=${COSINE}
    if [ $? -ne 0 ]; then
      echo "[WARNING] Run failed: lr=${LR} cosine=${COSINE}"
    fi
  done
done

echo ""
echo "All 12 runs complete!"
