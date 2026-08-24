#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

device="${1:-cuda:0}"
qm9_tasks=(
  "mu"
  "alpha"
  "R^2"
  "ZPVE"
  "c_v"
  "Delta_epsilon"
  "epsilon_HOMO"
  "epsilon_LUMO"
  "U_0"
  "U"
  "H"
  "G"
)

for subtask in "${qm9_tasks[@]}"; do
  python train.py \
    --dataset qm9 \
    --subtask "$subtask" \
    --llm-model galactica-30b \
    --knowledge-type all \
    --num-samples 50 \
    --model-type gin \
    --feature-mode unimrl \
    --drop-ratio 0.0 \
    --runs 10 \
    --device "$device"
done
