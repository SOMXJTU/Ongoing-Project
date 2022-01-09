import sys, os
sys.path.append("..")

import pickle
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from models import base_model

def normalization(x):
    return x / np.sum(x)

def generate_fc_weights(income_unit, outcome_unit, name):
    weight_initializer = tf.initializers.GlorotUniform()
    bias_initializer = tf.keras.initializers.zeros()
    w = tf.Variable(weight_initializer(shape=(income_unit, outcome_unit), dtype=tf.float32), name=name+"_kernel", trainable=True)
    b = tf.Variable(bias_initializer(shape=(outcome_unit, ), dtype=tf.float32), name=name+"_bias", trainable=True)
    return (w, b)


def generate_W(basis_num, num_clients, positive_value=1.0, negative_value=1.0):
    quotient, remainder = num_clients // int(basis_num), num_clients % int(basis_num)
    W_list = []
    row_index = 0
    for i in range(int(basis_num)):
        if i < (int(basis_num) -1):
            w_temp = np.zeros((quotient, num_clients)) - negative_value 
            w_temp[:, row_index:(i+1)*quotient] = positive_value
            W_list.append(w_temp)
            row_index += quotient
        else:
            w_temp = np.zeros((quotient+remainder, num_clients)) - negative_value
            w_temp[:, row_index:] = positive_value
            W_list.append(w_temp)
    W = np.concatenate(W_list, axis=0)
    D = np.diag(np.sum(W, axis=1))
    return W, D

def find_model(parser):
    """find the correspoding model based on the args

    Args:
      parser: the instance of argparser 
    
    Return: A instance of deep network model.
    """
    if parser.algorithm != "PFedL":
        if parser.dataset.startswith("Mnist"):
            return base_model.Mnist()
        elif parser.dataset.startswith("Cifar10"):
            return base_model.Cifar()
        else:
            raise ValueError("unsupported dataset")
    else:
        dataset_name = parser.dataset
        num_basis = re.findall(re.compile(r'basis([0-9]?)'), dataset_name)[0]
        if dataset_name.startswith("Mnist"):
            if parser.experts:
                from models.experts import mnist_model
                if parser.pfedl_share:
                    if num_basis == '2':
                        return mnist_model.Mnist_basis2_share()
                    elif num_basis == "3":
                        return mnist_model.Mnist_basis3_share()
                    elif num_basis == "4":
                        return mnist_model.Mnist_basis4_share()
                    else:
                        raise ValueError("Unsupported basis number")
                else:
                    if num_basis == '2':
                        return mnist_model.Mnist_basis2()
                    elif num_basis == "3":
                        return mnist_model.Mnist_basis3()
                    elif num_basis == "4":
                        return mnist_model.Mnist_basis4()
                    else:
                        raise ValueError("Unsupported basis number")
            else:
                from models.mixture import mnist_model
                return mnist_model.Mnist_basis(int(num_basis))
        elif dataset_name.startswith("Cifar10"):
            if parser.experts:
                from models.experts import cifar_model
                if num_basis == '2':
                    return cifar_model.Cifar_basis2()
                elif num_basis == "3":
                    return cifar_model.Cifar_basis3()
                elif num_basis == "4":
                    return cifar_model.Cifar_basis4()
                else:
                    raise ValueError("Unsupported basis number")
            else:
                from models.mixture import cifar_model
                return cifar_model.Cifar_basis(int(num_basis))
        else:
            raise ValueError("unsupported dataset")

def get_data(data_path):
    train_path = os.path.join(data_path, 'train_array.pkl')
    test_path = os.path.join(data_path, 'test_array.pkl')
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    train_data = train_data['user_data']
    test_data = test_data['user_data']
    return train_data, test_data