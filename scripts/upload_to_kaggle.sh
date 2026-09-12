#!/bin/bash
set -e

export KAGGLE_API_TOKEN="KGAT_606ae9d5fc752194af3ad226030798e0"
export PYTHONUNBUFFERED=1

echo "========================================================"
echo " Starting Kaggle CLI Upload for TartanAir Stereo Suite"
echo " Dataset ID: yumnamharryson/tartanair-warehouse-stereo"
echo " File: data/kaggle_upload/tartanair_warehouse_stereo.zip (45.4 GB)"
echo "========================================================"

/Users/krishnakant/Library/Python/3.9/bin/kaggle datasets create -p data/kaggle_upload/
