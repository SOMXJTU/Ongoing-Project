#!/bin/bash
cd ../

python run_experiment.py synthetic FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1233 --verbose 1 --input_dimension 150 --output_dimension 2

python run_experiment.py synthetic local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1233 --verbose 1 --input_dimension 150 --output_dimension 2

python run_experiment.py synthetic pFedMe --n_learners 1 --n_rounds 201 --bz 128 --lr 0.1 --mu 1.0 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1233 --verbose 1 --input_dimension 150 --output_dimension 2

python run_experiment.py synthetic clustered --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1233 --verbose 1 --input_dimension 150 --output_dimension 2

python run_experiment.py synthetic FedEM --n_learners 3 --n_rounds 201 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1233 --verbose 1 --input_dimension 150 --output_dimension 2

# LG is same as Fedavg

python run_experiment.py synthetic Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1233 --verbose 1 --exclude_keyword canonical --n_canonical 3 --client_lr 0.9 --client_scheduler multi_step --input_dimension 150 --output_dimension 2

python run_experiment.py synthetic Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1233 --verbose 1 --exclude_keyword canonical --n_canonical 3 --client_lr 0.9 --client_scheduler multi_step --input_dimension 150 --output_dimension 2 --interpolate