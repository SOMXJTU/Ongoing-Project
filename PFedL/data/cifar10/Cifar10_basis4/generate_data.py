import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import pickle
import os
cpath = os.path.dirname(__file__)

NUM_USER = 20
SAVE = True
IMAGE_DATA = False
np.random.seed(8)


class ImageDataset(object):
    def __init__(self, images, labels):
        self.data = images.astype(np.float64)/255.
        self.target = labels.astype(np.int64)

    def __len__(self):
        return len(self.target)

def data_split(data_index, num_split):
    delta,r = len(data_index) // num_split, len(data_index)%num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data_index):
        if used_r < r:
            data_lst.append(data_index[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data_index[i:i+delta])
            i += delta
    return data_lst

def main():
    print('>>> Get cifar10 data.')
    (train_x, train_label), (test_x,test_label) = cifar10.load_data()
    train_x, test_x = train_x/255., test_x/255.
    n_classes = train_label.max() + 1

    trainclass_idx = [np.argwhere(train_label == y)[:, 0].flatten() for y in range(n_classes)]
    testclass_idx = [np.argwhere(test_label == y)[:, 0].flatten() for y in range(n_classes)]

    split_trainclass_idx = []
    split_testclass_idx = []

    # 这里是n_basis不同会有所区别
    vehicle1_index = [0, 1]
    animal1_index = [2 ,3 ,4]
    animal2_index = [5, 6, 7]
    vehicle2_index = [8, 9]
    for label in range(n_classes):
        split_trainclass_idx.append(data_split(trainclass_idx[label], 5))
        split_testclass_idx.append(data_split(testclass_idx[label], 5))


    data_distribution = np.array([len(v) for v in split_trainclass_idx])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_trainclass_idx])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_testclass_idx])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    train_index = [[] for _ in range(NUM_USER)]
    test_index = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(NUM_USER):
        print(user, np.array([len(v) for v in split_trainclass_idx]))
        if user < 5:
            for d in vehicle1_index:
                selected_trainindex = split_trainclass_idx[d].pop().tolist()
                train_index[user] += selected_trainindex

                selected_testindex = split_testclass_idx[d].pop().tolist()
                test_index[user] += selected_testindex
        elif user < 10:
            for d in animal1_index:
                selected_trainindex = split_trainclass_idx[d].pop().tolist()
                train_index[user] += selected_trainindex

                selected_testindex = split_testclass_idx[d].pop().tolist()
                test_index[user] += selected_testindex
        elif user < 15:
            for d in animal2_index:
                selected_trainindex = split_trainclass_idx[d].pop().tolist()
                train_index[user] += selected_trainindex

                selected_testindex = split_testclass_idx[d].pop().tolist()
                test_index[user] += selected_testindex
        else:
            for d in vehicle2_index:
                selected_trainindex = split_trainclass_idx[d].pop().tolist()
                train_index[user] += selected_trainindex

                selected_testindex = split_testclass_idx[d].pop().tolist()
                test_index[user] += selected_testindex

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
if __name__ == '__main__':
    main()
