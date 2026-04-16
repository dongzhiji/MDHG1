# MDHG

The code of our paper **Multi-Relation Enhanced Dynamic Hypergraph for Session-based Recommendation.** 

# Run

Requirements: Python 3.7, Pytorch 1.6.0, Numpy 1.18.1

```
python main.py --dataset=Tmall

python main.py --dataset=retailrocket 

python main.py --dataset=amazon

# RetailRocket + custom test set (e.g. repo-root test.txt)
python main.py --dataset=RetailRocket --test_file=/home/runner/work/MDHG1/MDHG1/test.txt
```
