## relation prediction

### reference
[CFO: Conditional Focused Neural Question Answering with Large-scale Knowledge Bases](http://aclweb.org/anthology/P16-1076)

### models
- relation_classify
treat relation prediction as a classification task.

- relation_rank
Treat relation prediction as a rank task. Train a triplet loss based
ranking model to score the similarity between question and relation.
Rank the candidate relations by their score

### get started
go to any of the folders corresponding to a relation prediction model:
- train
```python
python train.py
```

- test
```python
python test.py model_path data_path # eg. python test.py ./models/RelationRNN-5 test.csv
```

### results of valid data
|                             | hits@1           | hits@3         | hits@5      |  hits@10   |
|-----------------------------|------------------|----------------|-------------|------------|
|relation_classsify           | 0.607140         |      -         |     -       |     -      |
|relation_rank/random         | 0.736983         | 0.906594       | 0.936928    | 0.961343   |
|relation_rank/restrict       | 0.765005         | 0.893924       | 0.925183    | 0.953482   |


### results of test data
|                             | hits@1           | hits@3         | hits@5      |  hits@10   |
|-----------------------------|------------------|----------------|-------------|------------|
|relation_classify            | 0.593519         |      -         |     -       |     -      |
|relation_rank/random         | 0.737010         | 0.904216       | 0.936252    | 0.959967   |
|relation_rank/restrict       | 0.766966         | 0.892151       | 0.921921    | 0.949982   |
