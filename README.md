# MDHG

The code of our paper **Multi-Relation Enhanced Dynamic Hypergraph for Session-based Recommendation.** 

# Run

Requirements: Python 3.7, Pytorch 1.6.0, Numpy 1.18.1

```
python main.py --dataset=Tmall

python main.py --dataset=retailrocket 

python main.py --dataset=amazon
```

Reproducible runs with item-level comp/sub curriculum & reliability gating:

```
python main.py --dataset=Tmall --seed=2026 --comp_sub_warmup_epochs=1 --comp_sub_ramp_epochs=4

python main.py --dataset=retailrocket --seed=2027 --comp_sub_warmup_epochs=1 --comp_sub_ramp_epochs=5 --logit_sub_scale=0.30

python main.py --dataset=diginetica --seed=2028 --comp_sub_warmup_epochs=1 --comp_sub_ramp_epochs=4 --logit_comp_scale=0.22
```

Unified protocol (same seeds + early stop) and required ablations:

```
# seed protocol
--seed_list=2026,2027,2028 --early_stop_metric=mrr10 --early_stop_patience=3 --early_stop_min_epoch=6

# baseline
python main.py --dataset=Tmall --enable_comp_branch=0 --enable_sub_branch=0 --enable_logit_residual=0 --enable_rel_conf_gate=0 --seed_list=2026,2027,2028 --early_stop_metric=mrr10 --early_stop_patience=3 --early_stop_min_epoch=6

# +comp hypergraph
python main.py --dataset=Tmall --enable_comp_branch=1 --enable_sub_branch=0 --enable_logit_residual=0 --seed_list=2026,2027,2028 --early_stop_metric=mrr10 --early_stop_patience=3 --early_stop_min_epoch=6

# +sub hypergraph
python main.py --dataset=Tmall --enable_comp_branch=0 --enable_sub_branch=1 --enable_logit_residual=0 --seed_list=2026,2027,2028 --early_stop_metric=mrr10 --early_stop_patience=3 --early_stop_min_epoch=6

# +comp+sub (no logit residual)
python main.py --dataset=Tmall --enable_comp_branch=1 --enable_sub_branch=1 --enable_logit_residual=0 --seed_list=2026,2027,2028 --early_stop_metric=mrr10 --early_stop_patience=3 --early_stop_min_epoch=6

# +logit residual fusion (full model)
python main.py --dataset=Tmall --enable_comp_branch=1 --enable_sub_branch=1 --enable_logit_residual=1 --enable_rel_conf_gate=1 --seed_list=2026,2027,2028 --early_stop_metric=mrr10 --early_stop_patience=3 --early_stop_min_epoch=6
```
