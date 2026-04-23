#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$ROOT_DIR/main.py"


# ===== Diginetica (4 组) =====
python $MAIN_PY --dataset=diginetica --comp_sub_pair_hyper_mix=0.4 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=1.5 --comp_sub_min_norm_weight=0.03 --sub_context_topk=15
python $MAIN_PY --dataset=diginetica --comp_sub_pair_hyper_mix=0.4 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=15
python $MAIN_PY --dataset=diginetica --comp_sub_pair_hyper_mix=0.5 --intent_align_weight=0.03 --comp_sub_topk=8 --comp_sub_min_support=1.5 --comp_sub_min_norm_weight=0.03 --sub_context_topk=20
python $MAIN_PY --dataset=diginetica --comp_sub_pair_hyper_mix=0.5 --intent_align_weight=0.03 --comp_sub_topk=8 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=20

# ===== Tmall (4 组) =====
python $MAIN_PY --dataset=Tmall --comp_sub_pair_hyper_mix=0.35 --intent_align_weight=0.01 --comp_sub_topk=4 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=10
python $MAIN_PY --dataset=Tmall --comp_sub_pair_hyper_mix=0.35 --intent_align_weight=0.01 --comp_sub_topk=4 --comp_sub_min_support=3.0 --comp_sub_min_norm_weight=0.05 --sub_context_topk=10
python $MAIN_PY --dataset=Tmall --comp_sub_pair_hyper_mix=0.45 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=15
python $MAIN_PY --dataset=Tmall --comp_sub_pair_hyper_mix=0.45 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=3.0 --comp_sub_min_norm_weight=0.05 --sub_context_topk=15
