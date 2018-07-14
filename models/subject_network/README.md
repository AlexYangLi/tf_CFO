## subject prediction

### reference
[CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases](http://aclweb.org/anthology/P16-1076)

### models
- sub_transe
employ the TransE model tp pretrain subject embedding

- sub_typevec
use fixed type vector to represent subject using type infomation

### get started

go to any of the folders corresponding to a subject prediction model:

- train
```python
python train.py
```
Note: To train sub_transe module, `sub_transe_embed.npy` is required. Go to ``TransE`` directory & follow the instruction on README.md, train & get it.

- test
```python
python test.py model_path data_path # eg. python test.py ./models/Sub-TransE-5 test.csv
```

### results of valid data
|                  | hits@1           | hits@3         | hits@5      |  hits@10   |
|------------------|------------------|----------------|-------------|------------|
|sub_transe        | 0.263849         | 0.408397       | 0.486359    |  0.610191  |
|sub_typevec       | 0.195320         | 0.331730       | 0.416906    |  0.513179  |


### results of test data
|                  | hits@1           | hits@3         | hits@5      |  hits@10   |
|------------------|------------------|----------------|-------------|------------|
|sub_transe        | 0.269462         | 0.406481       | 0.487426    |  0.610438  |
|sub_typevec       | 0.196792         | 0.334551       | 0.413277    |  0.509199  |

