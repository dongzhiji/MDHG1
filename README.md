# MDHG

The code of our paper **Multi-Relation Enhanced Dynamic Hypergraph for Session-based Recommendation.** 

# Run

Requirements: Python 3.7, Pytorch 1.6.0, Numpy 1.18.1

```
python main.py --dataset=Tmall

python main.py --dataset=retailrocket 

python main.py --dataset=amazon

python main.py --dataset=diginetica
```

可选的商品级互补/替代超图挖掘参数：
```
--comp_max_gap      互补关系窗口（默认 3）
--comp_topk         互补超边每个锚点商品保留邻居数（默认 8）
--sub_topk          替代超边每个锚点商品保留邻居数（默认 8）
--rel_min_support   互补/替代关系最小支持度（默认 2）
--sub_context_min   替代上下文最小计数（默认 2）
--comp_sub_conflict_margin 互补/替代冲突判定边际（默认 0.05）
```
