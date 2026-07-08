#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="conditional"
BASE_CONFIG="$PROJECT_DIR/configs/cifar10_unet_fm.yaml"
CONFIG_DIR="$PROJECT_DIR/configs/generated"
DATA_SOURCE="huggingface"
DATA_DIR="$PROJECT_DIR/datasets"
HF_CACHE_DIR=""
HF_ENDPOINT=""
RUN_ROOT="$PROJECT_DIR/runs/full_pipeline"
EPOCHS="100"
BATCH_SIZE="128"
NUM_WORKERS="4"
STEPS="50"
METHOD="euler"
NUM_SAMPLES="64"
SAMPLES_PER_CLASS="8"
SAMPLE_CLASS="3"
TRAJECTORY_EVERY="2"
DEVICE=""
SKIP_SYNC="0"
SKIP_DATA="0"
SKIP_TRAIN="0"
SKIP_SAMPLE="0"
NO_DOWNLOAD="0"
SAVE_TRAJECTORY="0"
REQUIRE_CUDA="1"

usage() {
  cat <<'EOF'
Run CIFAR-10 Flow Matching end to end with uv.

Usage:
  scripts/full_pipeline.sh [options]

Main options:
  --mode conditional|unconditional|both
  --epochs N
  --batch-size N
  --data-source huggingface|torchvision
  --data-dir PATH
  --hf-cache-dir PATH
  --hf-endpoint URL
  --run-root PATH
  --device cuda|cpu|cuda:0

Sampling options:
  --steps N
  --method euler|heun
  --num-samples N
  --samples-per-class N
  --sample-class N
  --save-trajectory
  --trajectory-every N

Control options:
  --skip-sync
  --skip-data
  --skip-train
  --skip-sample
  --no-download
  --allow-cpu
  -h, --help

Examples:
  scripts/full_pipeline.sh --mode conditional --epochs 100
  scripts/full_pipeline.sh --mode unconditional --epochs 100
  scripts/full_pipeline.sh --mode both --epochs 100 --data-source huggingface --data-dir /data/cifar10/datasets
EOF
}

abs_path() {
  case "$1" in
    /*) printf "%s" "$1" ;;
    *) printf "%s/%s" "$PROJECT_DIR" "$1" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --data-source) DATA_SOURCE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$(abs_path "$2")"; shift 2 ;;
    --hf-cache-dir) HF_CACHE_DIR="$(abs_path "$2")"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$(abs_path "$2")"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
    --samples-per-class) SAMPLES_PER_CLASS="$2"; shift 2 ;;
    --sample-class) SAMPLE_CLASS="$2"; shift 2 ;;
    --trajectory-every) TRAJECTORY_EVERY="$2"; shift 2 ;;
    --save-trajectory) SAVE_TRAJECTORY="1"; shift ;;
    --skip-sync) SKIP_SYNC="1"; shift ;;
    --skip-data) SKIP_DATA="1"; shift ;;
    --skip-train) SKIP_TRAIN="1"; shift ;;
    --skip-sample) SKIP_SAMPLE="1"; shift ;;
    --no-download) NO_DOWNLOAD="1"; shift ;;
    --allow-cpu) REQUIRE_CUDA="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MODE" in
  conditional|unconditional|both) ;;
  *) echo "--mode must be conditional, unconditional, or both" >&2; exit 2 ;;
esac

case "$METHOD" in
  euler|heun) ;;
  *) echo "--method must be euler or heun" >&2; exit 2 ;;
esac

case "$DATA_SOURCE" in
  huggingface|torchvision) ;;
  *) echo "--data-source must be huggingface or torchvision" >&2; exit 2 ;;
esac

if [[ "$DATA_SOURCE" == "huggingface" && -z "$HF_CACHE_DIR" ]]; then
  HF_CACHE_DIR="$DATA_DIR/huggingface"
fi

if [[ -n "$HF_ENDPOINT" ]]; then
  export HF_ENDPOINT
fi

if [[ "$NO_DOWNLOAD" == "1" ]]; then
  export HF_DATASETS_OFFLINE=1
fi

if [[ "$SKIP_SYNC" == "0" ]]; then
  uv sync
fi

uv run python - "$REQUIRE_CUDA" <<'PY'
import sys
import torch

require_cuda = sys.argv[1] == "1"
print(f"torch: {torch.__version__}")
print(f"torch cuda: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.version.cuda is None or not torch.version.cuda.startswith("12."):
    raise SystemExit("Expected a CUDA 12.x PyTorch build. Run `uv sync` and check pyproject.toml.")
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use --allow-cpu only for debugging.")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

mkdir -p "$CONFIG_DIR" "$RUN_ROOT"

if [[ "$SKIP_DATA" == "0" ]]; then
  data_args=(
    prepare_data.py
    --config "$BASE_CONFIG"
    --data-source "$DATA_SOURCE"
    --data-dir "$DATA_DIR"
  )
  if [[ -n "$HF_CACHE_DIR" ]]; then
    data_args+=(--hf-cache-dir "$HF_CACHE_DIR")
  fi
  if [[ -n "$HF_ENDPOINT" ]]; then
    data_args+=(--hf-endpoint "$HF_ENDPOINT")
  fi
  if [[ "$NO_DOWNLOAD" == "1" ]]; then
    data_args+=(--no-download)
  fi
  uv run python "${data_args[@]}"
fi

make_config() {
  local name="$1"
  local class_conditional="$2"
  local out_dir="$3"
  local config_path="$4"

  uv run python - \
    "$BASE_CONFIG" \
    "$config_path" \
    "$class_conditional" \
    "$DATA_SOURCE" \
    "$DATA_DIR" \
    "$HF_CACHE_DIR" \
    "$NO_DOWNLOAD" \
    "$out_dir" \
    "$EPOCHS" \
    "$BATCH_SIZE" \
    "$NUM_WORKERS" \
    "$STEPS" \
    "$METHOD" <<'PY'
from pathlib import Path
import sys
import yaml

(
    base_config,
    output_config,
    class_conditional,
    data_source,
    data_dir,
    hf_cache_dir,
    no_download,
    out_dir,
    epochs,
    batch_size,
    num_workers,
    steps,
    method,
) = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

cfg["data_dir"] = data_dir
cfg["data_source"] = data_source
cfg["download"] = no_download != "1"
if hf_cache_dir:
    cfg["hf_cache_dir"] = hf_cache_dir
else:
    cfg.pop("hf_cache_dir", None)
cfg["out_dir"] = out_dir
cfg["class_conditional"] = class_conditional == "true"
cfg["num_classes"] = 10
cfg["epochs"] = int(epochs)
cfg["batch_size"] = int(batch_size)
cfg["num_workers"] = int(num_workers)
cfg["num_steps_sampling"] = int(steps)
cfg["sampling_method"] = method

Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with open(output_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
  echo "Wrote config for $name: $config_path"
}

run_sample() {
  local config_path="$1"
  local checkpoint_path="$2"
  local output_path="$3"
  shift 3

  sample_args=(
    sample.py
    --config "$config_path"
    --ckpt "$checkpoint_path"
    --out "$output_path"
    --num-samples "$NUM_SAMPLES"
    --steps "$STEPS"
    --method "$METHOD"
  )
  if [[ -n "$DEVICE" ]]; then
    sample_args+=(--device "$DEVICE")
  fi
  if [[ "$SAVE_TRAJECTORY" == "1" ]]; then
    sample_args+=(--save-trajectory --trajectory-every "$TRAJECTORY_EVERY")
  fi

  uv run python "${sample_args[@]}" "$@"
}

run_experiment() {
  local name="$1"
  local class_conditional="$2"
  local out_dir="$RUN_ROOT/$name"
  local config_path="$CONFIG_DIR/${name}.yaml"
  local checkpoint_path="$out_dir/checkpoints/latest.pt"

  mkdir -p "$out_dir/samples"
  make_config "$name" "$class_conditional" "$out_dir" "$config_path"

  if [[ "$SKIP_TRAIN" == "0" ]]; then
    train_args=(train.py --config "$config_path")
    if [[ -n "$DEVICE" ]]; then
      train_args+=(--device "$DEVICE")
    fi
    uv run python "${train_args[@]}" 2>&1 | tee "$out_dir/train.log"
  fi

  if [[ "$SKIP_SAMPLE" == "1" ]]; then
    return
  fi

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "Missing checkpoint: $checkpoint_path" >&2
    echo "Run without --skip-train, or put a checkpoint at that path." >&2
    exit 1
  fi

  if [[ "$class_conditional" == "true" ]]; then
    run_sample \
      "$config_path" \
      "$checkpoint_path" \
      "$out_dir/samples/final_class_cycle_${METHOD}_${STEPS}.png"

    run_sample \
      "$config_path" \
      "$checkpoint_path" \
      "$out_dir/samples/final_class_grid_${METHOD}_${STEPS}.png" \
      --class-grid \
      --samples-per-class "$SAMPLES_PER_CLASS"

    run_sample \
      "$config_path" \
      "$checkpoint_path" \
      "$out_dir/samples/final_class_${SAMPLE_CLASS}_${METHOD}_${STEPS}.png" \
      --class-label "$SAMPLE_CLASS"
  else
    run_sample \
      "$config_path" \
      "$checkpoint_path" \
      "$out_dir/samples/final_unconditional_${METHOD}_${STEPS}.png"
  fi
}

case "$MODE" in
  conditional)
    run_experiment "conditional" "true"
    ;;
  unconditional)
    run_experiment "unconditional" "false"
    ;;
  both)
    run_experiment "conditional" "true"
    run_experiment "unconditional" "false"
    ;;
esac

echo "Done. Outputs are under: $RUN_ROOT"
