#!/usr/bin/env bash

set -e

# ============================================================
# MassLinker Suite Quick Start
#
# This script runs a minimal MassLinker workflow:
#   1. Tokenize mzML files using the R tokenizer
#   2. Build an ExcelDataset object from tokenized outputs
#   3. Run downstream distance-based differential analysis
#
# Usage:
#   bash quick_start.sh
#
# Before running this script, please prepare:
#   - mzML files in ./data/mzML
#   - sample label file at ./data/target.xlsx
#   - pathway annotation file at ./metadata/pathway_compound_detail.csv
# ============================================================


# -----------------------------
# User-configurable parameters
# -----------------------------

WORK_DIR="./data/mzML"
TOKEN_OUTPUT_DIR="./data/MassLinker_tokens"
POLARITY="positive"

TARGET_FILE="./data/target.xlsx"
DATASET_OUTPUT="./data/processed_dataset.joblib"

METADATA_FILE="./metadata/pathway_compound_detail.csv"

CACHE_DIR="./cache"
RESULTS_DIR="./results"

# Optional analysis parameters
P_VALUE_CUTOFF=0.01
TOP_P_VALUE_N=20
TOP_PEAK_N=2000


# -----------------------------
# Step 0. Create directories
# -----------------------------

mkdir -p ./data
mkdir -p "${TOKEN_OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"
mkdir -p "${RESULTS_DIR}"


echo "============================================================"
echo "MassLinker Suite Quick Start"
echo "============================================================"
echo "Input mzML directory:      ${WORK_DIR}"
echo "Token output directory:    ${TOKEN_OUTPUT_DIR}"
echo "Polarity:                  ${POLARITY}"
echo "Target file:               ${TARGET_FILE}"
echo "Dataset output:            ${DATASET_OUTPUT}"
echo "Annotation file:           ${METADATA_FILE}"
echo "Results directory:         ${RESULTS_DIR}"
echo "============================================================"


# -----------------------------
# Step 1. Run MassLinker tokenizer
# -----------------------------

echo ""
echo "[Step 1/3] Running MassLinker tokenizer..."

Rscript ./R/MassLinker_tokenizer.R \
  "${WORK_DIR}" \
  "${TOKEN_OUTPUT_DIR}" \
  "${POLARITY}"

echo "[Step 1/3] Tokenization finished."


# -----------------------------
# Step 2. Build ExcelDataset object
# -----------------------------

echo ""
echo "[Step 2/3] Building ExcelDataset object..."

python ./python/build_dataset.py \
  --data_path "${TOKEN_OUTPUT_DIR}" \
  --target_path "${TARGET_FILE}" \
  --output_path "${DATASET_OUTPUT}" \
  --mode origin

echo "[Step 2/3] Dataset saved to ${DATASET_OUTPUT}."


# -----------------------------
# Step 3. Run downstream pipeline
# -----------------------------

echo ""
echo "[Step 3/3] Running downstream MassLinker analysis..."

python ./python/run_masslinker_pipeline.py \
  --dataset_path "${DATASET_OUTPUT}" \
  --annotation_path "${METADATA_FILE}" \
  --cache_dir "${CACHE_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --p_value_cutoff "${P_VALUE_CUTOFF}" \
  --top_p_value_n "${TOP_P_VALUE_N}" \
  --top_peak_n "${TOP_PEAK_N}"

echo "[Step 3/3] Downstream analysis finished."


echo ""
echo "============================================================"
echo "Quick Start completed successfully."
echo "Results are saved in: ${RESULTS_DIR}"
echo "============================================================"
