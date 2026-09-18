#!/bin/bash
# ==============================================================================
# Dioptra-DINO: End-to-End Ablation Retraining Runner
# Usage: ./scripts/run_kaggle_ablation.sh [no-ara | center-ray | no-ray | no-vnl | no-scale-loss | no-dynamic-crop] [EPOCHS]
# ==============================================================================

set -e

VARIANT=${1:-"no-ara"}
EPOCHS=${2:-40}
BATCH_SIZE=${3:-8}
ACCUM_STEPS=${4:-4}

echo "================================================================================"
echo " Launching Dioptra-DINO Ablation Retraining: [${VARIANT}]"
echo " Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE} | Accumulation: ${ACCUM_STEPS}"
echo "================================================================================"

python scripts/train_dino_ablation.py \
    --ablation "${VARIANT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --accum-steps "${ACCUM_STEPS}" \
    --train "auto" \
    --output-dir "outputs_ablations"

echo "================================================================================"
echo " Ablation [${VARIANT}] training finished successfully!"
echo " Checkpoint saved in: outputs_ablations/ablation_${VARIANT}/"
echo "================================================================================"
