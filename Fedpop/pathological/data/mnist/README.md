 # MNIST Dataset

## Introduction

Split MNIST dataset among `n_clients` as follows:
1.  classes (labels) are grouped into `n_components`
2.  the number of clients across `components` are distributed evenly;
3.  for clients within the same `component`, they have the equal number of samples.

Inspired by the split in [Factorized-FL: Personalized Federated Learning with Parameter Factorization & Similarity Matching](https://openreview.net/forum?id=Ql75oqz1npy)

## Instructions

Run generate_data.py with a choice of the following arguments:

- ```--n_tasks```: number of tasks/clients, written as integer
- ```--n_components```: number of domain-heterogeneous components, written as integer; default=``4``
- ```--s_frac```: fraction of the dataset to be used; default=``1.0``  
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--test_tasks_frac```: fraction of test tasks; default=``0.0``
- ```--val_frac```: fraction of validation set (from train set); default: ``0.0``
- ```--seed```: seed to be used before random sampling of data; default=``12345``

  
## Paper Experiments


In order to generate the data split for our experiment without validation set, run

```
python generate_data.py \
    --n_tasks 100 \
    --n_components 4 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345    
```