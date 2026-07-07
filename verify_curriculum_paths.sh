#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/a/benchmark_data/optical_flow}"

check_dir () {
  local p="$1"
  if [[ -d "$p" ]]; then
    echo "OK   $p"
  else
    echo "MISS $p"
  fi
}

echo "Checking expected dataset roots under $ROOT"
check_dir "$ROOT/FlyingChairs_release/data"
check_dir "$ROOT/FlyingThings3D/FlyingThings3D_subset/train"
check_dir "$ROOT/FlyingThings3D/FlyingThings3D_subset/train/image_clean"
check_dir "$ROOT/FlyingThings3D/FlyingThings3D_subset/train/flow"
check_dir "$ROOT/FlyingThings3D/FlyingThings3D_subset/val"
check_dir "$ROOT/FlyingThings3D/FlyingThings3D_subset/val/image_clean"
check_dir "$ROOT/FlyingThings3D/FlyingThings3D_subset/val/flow"
check_dir "$ROOT/MPI-Sintel/MPI-Sintel-complete/training/clean"
check_dir "$ROOT/MPI-Sintel/MPI-Sintel-complete/training/final"
check_dir "$ROOT/MPI-Sintel/MPI-Sintel-complete/training/flow"
check_dir "$ROOT/hd1k/hd1k_full_package/hd1k_input/image_2"
check_dir "$ROOT/hd1k/hd1k_full_package/hd1k_flow_gt/flow_occ"
check_dir "$ROOT/Spring/spring/train"
check_dir "$ROOT/KITTI/KITTI 2015/data_scene_flow/training/image_2"
check_dir "$ROOT/KITTI/KITTI 2015/data_scene_flow/training/flow_occ"
check_dir "$ROOT/KITTI/KITTI 2015/data_scene_flow/testing/image_2"

echo
echo "Notes:"
echo "- If FlyingThings3D_subset is missing frames_cleanpass, edit stage 02 root to $ROOT/FlyingThings3D/FlyingThings3D"
echo "- Spring must be extracted; the current loader expects $ROOT/Spring/train/<scene>/frame_left and flow_FW_left."
echo "- Middlebury are not supported by the current data registry."
