#!/bin/bash
cd ../

python run_experiment.py mnist FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1

python run_experiment.py mnist local --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1

python run_experiment.py mnist pFedMe --n_learners 1 --n_rounds 200 --bz 128 --lr 0.03 --mu 1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer prox_sgd --seed 1233 --verbose 1

python run_experiment.py mnist FedEM --n_learners 4 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1

python run_experiment.py mnist FedLG --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 --include_keyword classifier

python run_experiment.py mnist Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.3 --client_scheduler multi_step

python run_experiment.py mnist Fedpop --n_learners 1 --n_rounds 200 --bz 128 --lr 0.1 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1233 --verbose 1 --exclude_keyword canonical --n_canonical 4 --client_lr 0.5 --client_scheduler multi_step --interpolate
