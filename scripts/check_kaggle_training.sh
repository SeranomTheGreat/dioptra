#!/bin/bash
export KAGGLE_API_TOKEN="KGAT_606ae9d5fc752194af3ad226030798e0"

echo "============================================================"
echo " Kaggle GPU Kernel Status Monitor"
echo " Kernel: yumnamharryson/dioptra-dino-training"
echo " URL:    https://www.kaggle.com/code/yumnamharryson/dioptra-dino-training"
echo "============================================================"
/Users/krishnakant/Library/Python/3.9/bin/kaggle kernels status yumnamharryson/dioptra-dino-training
echo "============================================================"
echo "Commands:"
echo " • Live Web UI:      open https://www.kaggle.com/code/yumnamharryson/dioptra-dino-training"
echo " • Check Status:     bash scripts/check_kaggle_training.sh"
echo " • Download Outputs: /Users/krishnakant/Library/Python/3.9/bin/kaggle kernels output yumnamharryson/dioptra-dino-training -p ./outputs_dino_kaggle/"
echo "============================================================"
