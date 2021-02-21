import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import pickle
import os
cpath = os.path.dirname(__file__)

NUM_USER = 100
SAVE = True
IMAGE_DATA = False
np.random.seed(8)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, tf.Tensor):
            if not IMAGE_DATA:
                self.data = images.set_shape(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images.reshape(-1, 784)/255.
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def choose_two_digit(split_data_lst, user_index):
    available_digit = []
    if user_index <50:
        for i, digit in enumerate(split_data_lst[0:5]):
            if len(digit) > 0:
                available_digit.append(i)
        try:
            if user_index <49:
                lst = np.random.choice(available_digit, 4, replace=False).tolist()
            if user_index == 49:
                lst = [2, 2, 3, 3]
        except:
            print(available_digit)
    else:
        for i, digit in enumerate(split_data_lst[5:10]):
            if len(digit) > 0:
                available_digit.append(i+5)
        try:
            if user_index == 97:
                lst = [6, 6, 7, 8]
            elif user_index == 98:
                lst = [6, 7, 7, 8]
            elif user_index == 99:
                lst = [6, 6, 7, 8]
            else:
                lst = np.random.choice(available_digit, 4, replace=False).tolist()
        except:
            print(available_digit)
    return lst

def main():
    # Get MNIST data, normalize, and divide by level
    print('>>> Get MNIST data.')
    (train_data, train_labels), (test_data,test_labels) = mnist.load_data()

    train_mnist = ImageDataset(train_data, train_labels)
    test_mnist = ImageDataset(test_data, test_labels)

    mnist_traindex = []
    for number in range(10):
        idx = np.argwhere(train_mnist.target == number).ravel()
        # mnist_traindata.append(train_mnist.data[idx])
        mnist_traindex.append(idx)
    min_number = min([len(dig) for dig in mnist_traindex])
    for number in range(10):
        mnist_traindex[number] = mnist_traindex[number][:min_number-1]

    split_mnist_traindex = []
    for digit in mnist_traindex:
        split_mnist_traindex.append(data_split(digit, 40))

    mnist_testindex = []
    for number in range(10):
        idx = np.argwhere(test_mnist.target == number).ravel()
        mnist_testindex.append(idx)
    split_mnist_testindex = []
    for digit in mnist_testindex:
        split_mnist_testindex.append(data_split(digit, 40))

    data_distribution = np.array([len(v) for v in mnist_traindex])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_mnist_traindex])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_mnist_testindex])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(NUM_USER):
        print(user, np.array([len(v) for v in split_mnist_traindex]))

        for d in choose_two_digit(split_mnist_traindex, user):
            l = len(split_mnist_traindex[d][-1])
            train_X[user] += split_mnist_traindex[d].pop().tolist()  # index
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_mnist_testindex[d][-1])
            test_X[user] += split_mnist_testindex[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    # Setup directory for train/test data
    print('>>> Set data path for MNIST.')
    # Create data structure
    # train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    # test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    train_data = []
    test_data = []

    # Setup 1000 users
    for i in range(NUM_USER):
        # uname = i

        # train_data['users'].append(uname)
        # train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data.append(train_X[i])

        # test_data['users'].append(uname)
        # test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data.append(test_X[i])

    # print('>>> User data distribution: {}'.format(train_data['num_samples']))
    # print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    # print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data

    with open('train_array_index.pkl', 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open('test_array_index.pkl', 'wb') as outfile:
        pickle.dump(test_data, outfile)

    print('>>> Save data.')


if __name__ == '__main__':
    main()

