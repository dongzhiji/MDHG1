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
