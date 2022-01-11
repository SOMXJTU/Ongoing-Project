import sys
import os
parent_path = os.path.dirname(os.path.dirname(__file__))
top_path = os.path.dirname(parent_path)
sys.path.append(top_path)

import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import normalization

class Cifar_basis2(tf.keras.Model):
    def __init__(self):
        super(Cifar_basis2, self).__init__()

        # bottom layer
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPool2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPool2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(64, activation='relu')
        self.fc_2 = layers.Dense(10, activation='softmax')

        # moe layer
        self.fc0_0 = layers.Dense(64, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(64, activation='relu', name="basis1-0")
        self.fc0_1 = layers.Dense(10, activation='softmax', name="basis0-1")
        self.fc1_1 = layers.Dense(10, activation='softmax', name="basis1-1")
        self.c = tf.Variable(normalization(tf.random.uniform([2, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)
        

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = layers.Flatten()(x)

        z0 = self.fc0_1(self.fc0_0(x))
        z1 = self.fc1_1(self.fc1_0(x))

        output = layers.Add()([tf.multiply(self.c[0], z0), tf.multiply(self.c[1], z1)])
        return output


class Cifar_basis3(tf.keras.Model):
    def __init__(self):
        super(Cifar_basis2, self).__init__()

        # bottom layer
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPool2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPool2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(64, activation='relu')
        self.fc_2 = layers.Dense(10, activation='softmax')

        # moe layer
        self.fc0_0 = layers.Dense(64, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(64, activation='relu', name="basis1-0")
        self.fc2_0 = layers.Dense(64, activation='relu', name="basis2-0")
        self.fc0_1 = layers.Dense(10, activation='softmax', name="basis0-1")
        self.fc1_1 = layers.Dense(10, activation='softmax', name="basis1-1")
        self.fc2_1 = layers.Dense(10, activation='softmax', name="basis2-1")
        self.c = tf.Variable(normalization(tf.random.uniform([3, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)
        

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = layers.Flatten()(x)

        z0 = self.fc0_1(self.fc0_0(x))
        z1 = self.fc1_1(self.fc1_0(x))
        z2 = self.fc2_1(self.fc2_0(x))
        output = layers.Add()([tf.multiply(self.c[0], z0), tf.multiply(self.c[1], z1),
                               tf.multiply(self.c[2], z2)])
        return output

class Cifar_basis4(tf.keras.Model):
    def __init__(self):
        super(Cifar_basis2, self).__init__()

        # bottom layer
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPool2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPool2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(64, activation='relu')
        self.fc_2 = layers.Dense(10, activation='softmax')

        # moe layer
        self.fc0_0 = layers.Dense(64, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(64, activation='relu', name="basis1-0")
        self.fc2_0 = layers.Dense(64, activation='relu', name="basis2-0")
        self.fc3_0 = layers.Dense(64, activation='relu', name="basis3-0")
        self.fc0_1 = layers.Dense(10, activation='softmax', name="basis0-1")
        self.fc1_1 = layers.Dense(10, activation='softmax', name="basis1-1")
        self.fc2_1 = layers.Dense(10, activation='softmax', name="basis2-1")
        self.fc3_1 = layers.Dense(10, activation='softmax', name="basis3-1")
        self.c = tf.Variable(normalization(tf.random.uniform([4, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)
        

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = layers.Flatten()(x)

        z0 = self.fc0_1(self.fc0_0(x))
        z1 = self.fc1_1(self.fc1_0(x))
        z2 = self.fc2_1(self.fc2_0(x))
        z3 = self.fc3_1(self.fc3_0(x))
        output = layers.Add()([tf.multiply(self.c[0], z0), tf.multiply(self.c[1], z1),
                               tf.multiply(self.c[2], z2), tf.multiply(self.c[3], z3)])
        return output
