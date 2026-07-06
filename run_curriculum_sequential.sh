#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${BASE_CONFIG:-configs/default.yaml}"
CURR_DIR="${CURR_DIR:-configs/curriculum}"

run_stage () {
  local stage_yaml="$1"
  local ckpt="${2:-}"
  local extra=()

  if [[ -n "$ckpt" ]]; then
    extra+=("training.checkpoint=$ckpt" "training.resume_mode=weights_only")
  fi

  echo
  echo "====================================================================="
  echo "Running $stage_yaml"
  if [[ -n "$ckpt" ]]; then
    echo "Warm-starting from: $ckpt"
  fi
  echo "====================================================================="

  python train.py --config "$BASE_CONFIG" --override "$CURR_DIR/$stage_yaml" "${extra[@]}"
}

run_stage "01_chairs_baseline.yaml"
CKPT_01="checkpoints/curr_01_chairs_baseline/best.pth"

run_stage "02_things_pgma_init.yaml" "$CKPT_01"
CKPT_02="checkpoints/curr_02_things_pgma_init/best.pth"

run_stage "03_sintel_clean_refine.yaml" "$CKPT_02"
CKPT_03="checkpoints/curr_03_sintel_clean_refine/best.pth"

run_stage "04_sintel_final_refine.yaml" "$CKPT_03"
CKPT_04="checkpoints/curr_04_sintel_final_refine/best.pth"

run_stage "05_spring_detail_refine.yaml" "$CKPT_04"
CKPT_05="checkpoints/curr_05_spring_detail_refine/best.pth"

run_stage "06_hd1k_road_refine.yaml" "$CKPT_05"
CKPT_06="checkpoints/curr_06_hd1k_road_refine/best.pth"

run_stage "07_kitti2015_ft.yaml" "$CKPT_06"