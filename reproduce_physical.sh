#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

device="${1:-cuda:0}"

python train.py \
  --dataset esol \
  --llm-model galactica-6.7b \
  --knowledge-type all \
  --num-samples 30 \
  --model-type gin \
  --feature-mode unimrl \
  --drop-ratio 0.0 \
  --runs 10 \
  --device "$device"

python train.py \
  --dataset freesolv \
  --llm-model galactica-6.7b \
  --knowledge-type all \
  --num-samples 30 \
  --model-type gin \
  --feature-mode unimrl \
  --drop-ratio 0.3 \
  --runs 10 \
  --device "$device"

python train.py \
  --dataset lipophilicity \
  --llm-model galactica-6.7b \
  --knowledge-type all \
  --num-samples 30 \
  --model-type gin \
  --feature-mode unimrl \
  --drop-ratio 0.0 \
  --runs 10 \
  --device "$device"
