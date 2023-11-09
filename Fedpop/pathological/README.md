Implement PPFL in the pathological setting. (based on [Federated Multi-Task Learning under a Mixture of Distributions](https://github.com/omarfoq/FedEM) and [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://github.com/felisat/clustered-federated-learning) repositories.)

## Usage

### Data Generation

For generating data, see the `README.md` files of respective dataset, i.e., `data/$DATASET`. 

We provide three federated benchmark datasets: handwritten character recognition (MNIST), image classification (CIFAR10), and a synthetic dataset

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| MNIST   |     Handwritten character recognition       |     2-layer FFN  |
| CIFAR10   |     Image classification        |      MobileNet-v2 |
| Synthetic dataset| Binary classification | Linear model | 

### Training

Run on one dataset, with a specific  choice of federated learning method.
Specify the name of the dataset (experiment), the used method, and configure all other
hyper-parameters (see all hyper-parameters values in the appendix of the paper)

```bash
python run_experiment.py mnist FedAvg --n_learners 1 \ 
--n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step \ 
--log_freq 5 --device cuda --optimizer sgd --seed 1233 \
--logs_root ./logs --verbose 1
```

The test and training accuracy and loss will be saved in the specified log path.

>We provide example scripts to run paper experiments under `scripts/` directory. 
>
>For clustered FL on MNIST and CIFAR 10 datasets, we provide the main modified Jupyter Notebook files supported by [Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints](https://github.com/felisat/clustered-federated-learning).


