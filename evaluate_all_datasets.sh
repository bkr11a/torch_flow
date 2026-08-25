#!/usr/bin/env bash

set -Eeuo pipefail

# =============================================================================
# Sequential benchmark evaluation harness for torch_flow
# =============================================================================
#
# Example:
#
#   ./evaluate_all_datasets.sh \
#       --run hqs_core_2_4_2_2_50000k_1_8_ap
#
# Full example:
#
#   ./evaluate_all_datasets.sh \
#       --run hqs_core_2_4_2_2_50000k_1_8_ap \
#       --stage curr_u04_universal_balanced_final \
#       --qualitative-samples 16 \
#       --device cuda \
#       --batch-size 1 \
#       --postproc-workers 2
#
# Select datasets:
#
#   ./evaluate_all_datasets.sh \
#       --run hqs_core_2_4_2_2_50000k_1_8_ap \
#       --datasets sintel_clean,sintel_final,kitti2015
#
# =============================================================================


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

RUN_NAME=""
STAGE="curr_u04_universal_balanced_final"

QUALITATIVE_SAMPLES="16"
DEVICE="cuda"
BATCH_SIZE="1"
POSTPROC_WORKERS="2"

CHECKPOINTS_ROOT="checkpoints"
RESULTS_ROOT="results"
EVAL_CONFIG_ROOT="configs/eval"

CONTINUE_ON_ERROR=false
DRY_RUN=false

DEFAULT_DATASETS=(
    chairs
    things_subset
    hd1k
    kitti2012
    kitti2015
    sintel_clean
    sintel_final
    spring_left
)

DATASETS=("${DEFAULT_DATASETS[@]}")


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------

separator() {
    printf '%*s\n' 78 '' | tr ' ' '='
}

subseparator() {
    printf '%*s\n' 78 '' | tr ' ' '-'
}

format_duration() {
    local total_seconds="$1"

    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    printf "%02d:%02d:%02d" \
        "$hours" \
        "$minutes" \
        "$seconds"
}


# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:

  ./evaluate_all_datasets.sh --run <run_name> [options]


Required:

  --run <name>
      Training run name under checkpoints/.


Options:

  --stage <name>
      Curriculum stage containing the checkpoint.

      Default:
        curr_u04_universal_balanced_final


  --datasets <dataset1,dataset2,...>
      Comma-separated list of datasets to evaluate.

      Default:
        chairs
        hd1k
        kitti2012
        kitti2015
        sintel_clean
        sintel_final
        spring_left


  --qualitative-samples <N|all>
      Number of qualitative samples to generate.

      Default:
        16


  --device <device>
      Evaluation device.

      Default:
        cuda


  --batch-size <N>
      Evaluation batch size.

      Default:
        1


  --postproc-workers <N>
      Number of post-processing workers.

      Default:
        2


  --checkpoints-root <path>
      Root checkpoint directory.

      Default:
        checkpoints


  --results-root <path>
      Root results directory.

      Default:
        results


  --eval-config-root <path>
      Directory containing *_native.yaml evaluation configs.

      Default:
        configs/eval


  --continue-on-error
      Continue evaluating remaining datasets if one evaluation fails.

      Without this option, evaluation stops at the first failure.


  --dry-run
      Print the commands that would be executed without running them.


  -h, --help
      Show this help message.


Examples:

  ./evaluate_all_datasets.sh \
      --run hqs_core_2_4_2_2_50000k_1_8_ap


  ./evaluate_all_datasets.sh \
      --run hqs_core_2_4_2_2_50000k_1_8_ap \
      --qualitative-samples all


  ./evaluate_all_datasets.sh \
      --run hqs_core_2_4_2_2_50000k_1_8_ap \
      --datasets sintel_clean,sintel_final,kitti2015


  ./evaluate_all_datasets.sh \
      --run hqs_core_2_4_2_2_50000k_1_8_ap \
      --datasets chairs,sintel_final \
      --continue-on-error

EOF
}


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in

        --run)
            RUN_NAME="${2:?ERROR: --run requires a value}"
            shift 2
            ;;

        --stage)
            STAGE="${2:?ERROR: --stage requires a value}"
            shift 2
            ;;

        --datasets)
            DATASET_STRING="${2:?ERROR: --datasets requires a value}"
            IFS=',' read -r -a DATASETS <<< "$DATASET_STRING"
            shift 2
            ;;

        --qualitative-samples)
            QUALITATIVE_SAMPLES="${2:?ERROR: --qualitative-samples requires a value}"
            shift 2
            ;;

        --device)
            DEVICE="${2:?ERROR: --device requires a value}"
            shift 2
            ;;

        --batch-size)
            BATCH_SIZE="${2:?ERROR: --batch-size requires a value}"
            shift 2
            ;;

        --postproc-workers)
            POSTPROC_WORKERS="${2:?ERROR: --postproc-workers requires a value}"
            shift 2
            ;;

        --checkpoints-root)
            CHECKPOINTS_ROOT="${2:?ERROR: --checkpoints-root requires a value}"
            shift 2
            ;;

        --results-root)
            RESULTS_ROOT="${2:?ERROR: --results-root requires a value}"
            shift 2
            ;;

        --eval-config-root)
            EVAL_CONFIG_ROOT="${2:?ERROR: --eval-config-root requires a value}"
            shift 2
            ;;

        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;

        --dry-run)
            DRY_RUN=true
            shift
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "ERROR: Unknown argument: $1"
            echo
            usage
            exit 2
            ;;
    esac
done


# -----------------------------------------------------------------------------
# Validate arguments
# -----------------------------------------------------------------------------

if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --run is required."
    echo
    usage
    exit 2
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "ERROR: No datasets were specified."
    exit 2
fi


# -----------------------------------------------------------------------------
# Resolve paths
# -----------------------------------------------------------------------------

RUN_CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/${RUN_NAME}/${STAGE}"

CONFIG="${RUN_CHECKPOINT_DIR}/config/resolved_config.yaml"
CHECKPOINT="${RUN_CHECKPOINT_DIR}/best.pth"

RUN_RESULTS_DIR="${RESULTS_ROOT}/${RUN_NAME}"
LOG_DIR="${RUN_RESULTS_DIR}/evaluation_logs"

SUMMARY_FILE="${LOG_DIR}/evaluation_summary.tsv"


# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

separator
echo "torch_flow benchmark evaluation"
separator

printf "%-24s %s\n" "Run:"                 "$RUN_NAME"
printf "%-24s %s\n" "Curriculum stage:"    "$STAGE"
printf "%-24s %s\n" "Config:"              "$CONFIG"
printf "%-24s %s\n" "Checkpoint:"          "$CHECKPOINT"
printf "%-24s %s\n" "Results directory:"   "$RUN_RESULTS_DIR"
printf "%-24s %s\n" "Device:"              "$DEVICE"
printf "%-24s %s\n" "Batch size:"          "$BATCH_SIZE"
printf "%-24s %s\n" "Postproc workers:"    "$POSTPROC_WORKERS"
printf "%-24s %s\n" "Qualitative samples:" "$QUALITATIVE_SAMPLES"
printf "%-24s %s\n" "Continue on error:"   "$CONTINUE_ON_ERROR"
printf "%-24s %s\n" "Dry run:"             "$DRY_RUN"

echo
echo "Datasets:"

for dataset in "${DATASETS[@]}"; do
    echo "  - ${dataset}"
done

echo
separator
echo "Pre-flight validation"
separator


# Main evaluation script

if [[ ! -f "evaluate_selected_dataset.py" ]]; then
    echo "ERROR: evaluate_selected_dataset.py was not found."
    echo
    echo "Run this script from the torch_flow repository root."
    exit 1
fi

echo "[OK] evaluate_selected_dataset.py"


# Resolved training config

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Resolved configuration does not exist:"
    echo
    echo "  ${CONFIG}"
    exit 1
fi

echo "[OK] ${CONFIG}"


# Checkpoint

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: Checkpoint does not exist:"
    echo
    echo "  ${CHECKPOINT}"
    exit 1
fi

echo "[OK] ${CHECKPOINT}"


# Dataset configs

CONFIG_ERROR=false

for dataset in "${DATASETS[@]}"; do

    DATA_CONFIG="${EVAL_CONFIG_ROOT}/${dataset}_native.yaml"

    if [[ ! -f "$DATA_CONFIG" ]]; then
        echo "[MISSING] ${DATA_CONFIG}"
        CONFIG_ERROR=true
    else
        echo "[OK] ${DATA_CONFIG}"
    fi

done

if [[ "$CONFIG_ERROR" == true ]]; then
    echo
    echo "ERROR: One or more evaluation configuration files are missing."
    exit 1
fi


# -----------------------------------------------------------------------------
# Dry-run helper
# -----------------------------------------------------------------------------

print_command() {

    local dataset="$1"
    local data_config="$2"
    local output_dir="$3"

    printf 'python evaluate_selected_dataset.py \\\n'
    printf '    --config %q \\\n' "$CONFIG"
    printf '    --checkpoint %q \\\n' "$CHECKPOINT"
    printf '    --data_config %q \\\n' "$data_config"
    printf '    --output_dir %q \\\n' "$output_dir"
    printf '    --device %q \\\n' "$DEVICE"
    printf '    --batch_size %q \\\n' "$BATCH_SIZE"
    printf '    --postproc_workers %q \\\n' "$POSTPROC_WORKERS"
    printf '    --qualitative_samples %q\n' "$QUALITATIVE_SAMPLES"
}


# -----------------------------------------------------------------------------
# Prepare logging
# -----------------------------------------------------------------------------

if [[ "$DRY_RUN" == false ]]; then

    mkdir -p "$LOG_DIR"

    printf "dataset\tstatus\texit_code\truntime_seconds\truntime\n" \
        > "$SUMMARY_FILE"

fi


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

declare -a SUCCESSFUL_DATASETS=()
declare -a FAILED_DATASETS=()

TOTAL_START=$SECONDS


for dataset in "${DATASETS[@]}"; do

    DATA_CONFIG="${EVAL_CONFIG_ROOT}/${dataset}_native.yaml"
    OUTPUT_DIR="${RUN_RESULTS_DIR}/${dataset}_native"
    LOG_FILE="${LOG_DIR}/${dataset}.log"

    echo
    separator
    echo "Dataset: ${dataset}"
    separator

    printf "%-20s %s\n" "Data config:" "$DATA_CONFIG"
    printf "%-20s %s\n" "Output:"      "$OUTPUT_DIR"

    if [[ "$DRY_RUN" == true ]]; then

        echo
        print_command \
            "$dataset" \
            "$DATA_CONFIG" \
            "$OUTPUT_DIR"

        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    DATASET_START=$SECONDS

    echo
    echo "Starting evaluation..."
    echo

    if python evaluate_selected_dataset.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data_config "$DATA_CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --postproc_workers "$POSTPROC_WORKERS" \
        --qualitative_samples "$QUALITATIVE_SAMPLES" \
        2>&1 | tee "$LOG_FILE"
    then
        EXIT_CODE=0
    else
        EXIT_CODE=${PIPESTATUS[0]}
    fi

    DATASET_ELAPSED=$((SECONDS - DATASET_START))
    DATASET_RUNTIME="$(format_duration "$DATASET_ELAPSED")"

    echo
    subseparator

    if [[ "$EXIT_CODE" -eq 0 ]]; then

        echo "SUCCESS: ${dataset}"
        echo "Runtime: ${DATASET_RUNTIME}"

        SUCCESSFUL_DATASETS+=("$dataset")

        printf "%s\tSUCCESS\t0\t%s\t%s\n" \
            "$dataset" \
            "$DATASET_ELAPSED" \
            "$DATASET_RUNTIME" \
            >> "$SUMMARY_FILE"

    else

        echo "FAILED: ${dataset}"
        echo "Exit code: ${EXIT_CODE}"
        echo "Runtime: ${DATASET_RUNTIME}"
        echo "Log: ${LOG_FILE}"

        FAILED_DATASETS+=("$dataset")

        printf "%s\tFAILED\t%s\t%s\t%s\n" \
            "$dataset" \
            "$EXIT_CODE" \
            "$DATASET_ELAPSED" \
            "$DATASET_RUNTIME" \
            >> "$SUMMARY_FILE"

        if [[ "$CONTINUE_ON_ERROR" == false ]]; then

            echo
            echo "Stopping because --continue-on-error was not specified."

            break
        fi
    fi

done


# -----------------------------------------------------------------------------
# Dry-run exit
# -----------------------------------------------------------------------------

if [[ "$DRY_RUN" == true ]]; then

    echo
    separator
    echo "Dry run complete. No evaluations were executed."
    separator

    exit 0
fi


# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
TOTAL_RUNTIME="$(format_duration "$TOTAL_ELAPSED")"

echo
echo
separator
echo "EVALUATION SUMMARY"
separator

printf "%-24s %s\n" "Run:"          "$RUN_NAME"
printf "%-24s %s\n" "Stage:"        "$STAGE"
printf "%-24s %s\n" "Total runtime:" "$TOTAL_RUNTIME"

echo

if [[ ${#SUCCESSFUL_DATASETS[@]} -gt 0 ]]; then

    echo "Successful datasets:"

    for dataset in "${SUCCESSFUL_DATASETS[@]}"; do
        echo "  [OK] ${dataset}"
    done

else
    echo "Successful datasets: none"
fi

echo

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then

    echo "Failed datasets:"

    for dataset in "${FAILED_DATASETS[@]}"; do
        echo "  [FAILED] ${dataset}"
    done

else
    echo "Failed datasets: none"
fi

echo

echo "Summary:"
echo "  ${SUMMARY_FILE}"

echo

separator


# Return a failing exit status if any benchmark failed.

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    exit 1
fi

exit 0