import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

NUM_USER = 100
SAVE = True
IMAGE_DATA = False
np.random.seed(8)

def generate_data(train_idcs, train_labels, n_clients, group_distribution):
    n_classes = train_labels.max() + 1
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    client_ydistribution = [[0]*n_classes for _ in range(n_clients)]
    flag = 0
    for c, fracs in zip(class_idcs, group_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]
            client_ydistribution[i][flag] = len(idcs)
        flag += 1

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return (client_idcs, client_ydistribution)

def main(n_basis):
    (train_x, train_label), (test_x, test_label) = mnist.load_data()
    train_x, test_x = train_x.reshape(-1, 784)/255., test_x.reshape(-1, 784)/255.
    n_classes = train_label.max() + 1
    quotient_clients, remainder_clients = NUM_USER // n_basis, NUM_USER % n_basis
    quotient_labels, remainder_labels = n_classes // n_basis, n_classes % n_basis
    group_list = []
    flag = 0
    for basis in range(n_basis):
        if basis < (n_basis-1):
            temp_list = np.zeros((quotient_labels, NUM_USER))
            # temp_distribution = np.random.uniform(0, 1, (quotient_labels, quotient_clients))
            # temp_distribution = temp_distribution / np.sum(temp_distribution, axis=1, keepdims=True)
            temp_distribution = np.random.dirichlet([1] * quotient_clients, quotient_labels)
            temp_list[:, flag:(basis+1)*quotient_clients] = temp_distribution
            group_list.append(temp_list)
            flag += quotient_clients
        else:
            temp_list = np.zeros((quotient_labels+remainder_labels, NUM_USER))
            # temp_distribution = np.random.uniform(0, 1, (quotient_labels+remainder_labels, quotient_clients+remainder_clients))
            # temp_distribution = temp_distribution / np.sum(temp_distribution, axis=1, keepdims=True)
            temp_distribution = np.random.dirichlet([1] * (quotient_clients+remainder_clients), quotient_labels+remainder_labels)
            temp_list[:, flag:] = temp_distribution
            group_list.append(temp_list)
    group_distribution = np.concatenate(group_list, axis=0)
    train_index, trainlabel_distribution = generate_data(np.arange(len(train_label)), train_label, NUM_USER, group_distribution)
    test_index, testlabel_distribution = generate_data(np.arange(len(test_label)), test_label, NUM_USER, group_distribution)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    trainindex_data, testindex_data = [], []
    for i in range(NUM_USER):
        train_data['users'].append(i)
        train_data['user_data'][i] = {'x': train_x[train_index[i]], 'y': train_label[train_index[i]]}
        train_data['num_samples'].append(len(train_index[i]))
        test_data['users'].append(i)
        test_data['user_data'][i] = {'x': test_x[test_index[i]], 'y': test_label[test_index[i]]}
        test_data['num_samples'].append(len(test_index[i]))
        print("In {:d}-th client's train set, the label distribution is {}".format(i+1, trainlabel_distribution[i]))
        print("In {:d}-th client's test set, the label distribution is {}".format(i+1, testlabel_distribution[i]))
        trainindex_data.append(train_index[i])
        testindex_data.append(test_index[i])

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))
    with open('train_array.pkl', 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open('test_array.pkl', 'wb') as outfile:
        pickle.dump(test_data, outfile)
    with open('train_array_index.pkl', 'wb') as outfile:
        pickle.dump(trainindex_data, outfile)
    with open('test_array_index.pkl', 'wb') as outfile:
        pickle.dump(testindex_data, outfile)

    print('>>> Save data.')


if __name__ == "__main__":
    n_basis = 4
    main(n_basis)