# MDHG

The code of our paper **Multi-Relation Enhanced Dynamic Hypergraph for Session-based Recommendation.** 

# Run

Requirements: Python 3.7, Pytorch 1.6.0, Numpy 1.18.1

```
python main.py --dataset=Tmall

python main.py --dataset=retailrocket 

python main.py --dataset=amazon
```

For RetailRocket quick run with a prepared `test.txt` in repository root:
```
python main.py --dataset=retailrocket --test_file=test.txt
```
If `datasets/retailrocket/train.txt` or `all_train_seq.txt` is missing, code now auto-falls back to runnable bootstrap mode.
