# MDHG

The code of our paper **Multi-Relation Enhanced Dynamic Hypergraph for Session-based Recommendation.** 

# Run

Requirements: Python 3.7, Pytorch 1.6.0, Numpy 1.18.1

```
python main.py --dataset=Tmall

python main.py --dataset=retailrocket 

python main.py --dataset=amazon
```

RetailRocket 推荐可先用如下商品级互补/替代超图配置：

```
python main.py --dataset=retailrocket --comp_max_gap=3 --comp_sub_topk=10 --comp_sub_min_support=1.5 --comp_sub_min_norm_weight=0.03 --sub_context_topk=30
```

Diginetica + Tmall 最小可跑网格（每个数据集 4 组）：

```
# ===== Diginetica (4 组) =====
python main.py --dataset=diginetica --comp_sub_pair_hyper_mix=0.4 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=1.5 --comp_sub_min_norm_weight=0.03 --sub_context_topk=15
python main.py --dataset=diginetica --comp_sub_pair_hyper_mix=0.4 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=15
python main.py --dataset=diginetica --comp_sub_pair_hyper_mix=0.5 --intent_align_weight=0.03 --comp_sub_topk=8 --comp_sub_min_support=1.5 --comp_sub_min_norm_weight=0.03 --sub_context_topk=20
python main.py --dataset=diginetica --comp_sub_pair_hyper_mix=0.5 --intent_align_weight=0.03 --comp_sub_topk=8 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=20

# ===== Tmall (4 组) =====
python main.py --dataset=Tmall --comp_sub_pair_hyper_mix=0.35 --intent_align_weight=0.01 --comp_sub_topk=4 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=10
python main.py --dataset=Tmall --comp_sub_pair_hyper_mix=0.35 --intent_align_weight=0.01 --comp_sub_topk=4 --comp_sub_min_support=3.0 --comp_sub_min_norm_weight=0.05 --sub_context_topk=10
python main.py --dataset=Tmall --comp_sub_pair_hyper_mix=0.45 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=2.0 --comp_sub_min_norm_weight=0.04 --sub_context_topk=15
python main.py --dataset=Tmall --comp_sub_pair_hyper_mix=0.45 --intent_align_weight=0.02 --comp_sub_topk=6 --comp_sub_min_support=3.0 --comp_sub_min_norm_weight=0.05 --sub_context_topk=15
```

也可直接运行：

```
bash run_minimal_grid.sh
```
