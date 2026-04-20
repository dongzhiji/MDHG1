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
