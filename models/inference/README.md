## inference

### get started
```
sh start.sh
```

### results
| EntDect   | RelNet  | SubNet  | sub acc  | rel acc  | (sub, rel) acc |
|-----------|---------|---------|----------|----------|----------------|
| lstm      | relrank | typevec | 0.702848 | 0.718473 | 0.609837       |
| lstm_crf  | relrank | typevec | 0.716947 | 0.722587 | 0.621117       |
| bilstm    | relrank | typevec | 0.720460 | 0.729660 | 0.625925       |
| bilstm_crf| relrank | typevec | 0.697069 | 0.721847 | 0.606047       |
| lstm      | relrank | transe  | 0.713156 | 0.740662 | 0.620100       |
| lstm_crf  | relrank | transe  | 0.725915 | 0.744083 | **0.630501**   |
| bilstm    | relrank | transe  | 0.730584 | 0.752959 | 0.636464       |
| bilstm_crf| relrank | transe  | 0.708349 | 0.745285 | 0.617465       |

Sadly, I didn't achive the results as stated in the paper.

