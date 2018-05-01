## entity detection
Treat entity detection in a question as a sequence labeling task. Use
lstm + crf to solve the problem

### reference
[Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)

### get started
- train
```python
python train.py
```

- test
```python
python test.py model_type model_path # eg. python test.py 'bilstm_crf' './models/bilstm_crf-5'
```

### results of valid data
|                  | macro precision  | macro recall   | average f1  |
|------------------|------------------|----------------|-------------|
|lstm              | 0.911498         | 0.909205       | 0.898200    |
|lstm + crf        | 0.904018         | 0.917804       | 0.899736    |
|bilstm            | 0.910219         | 0.942874       | 0.914270    |
|bilstm + crf      | 0.905599         | 0.927418       | 0.902926    |

### results of test data
|                  | macro precision  | macro recall   | average f1  |
|------------------|------------------|----------------|-------------|
|lstm              | 0.910324         | 0.911293       | 0.898157    |
|lstm + crf        | 0.904132         | 0.920272       | 0.900484    |
|bilstm            | 0.909580         | 0.943186       | 0.913903    |
|bilstm + crf      | 0.904338         | 0.929689       | 0.902737    |

