import sys
import os
parent_path = os.path.dirname(os.path.dirname(__file__))
top_path = os.path.dirname(parent_path)
sys.path.append(top_path)

import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import normalization


class Mnist_basis5(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis5, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(100, activation='relu', name="basis1-0")
        self.fc2_0 = layers.Dense(100, activation='relu', name="basis2-0")
        self.fc3_0 = layers.Dense(100, activation='relu', name='basis3-0')
        self.fc4_0 = layers.Dense(100, activation='relu', name='basis4-0')

        self.fc0_1 = layers.Dense(category_num, activation="softmax", name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, activation="softmax", name="basis1-1")
        self.fc2_1 = layers.Dense(category_num, activation="softmax", name="basis2-1")
        self.fc3_1 = layers.Dense(category_num, activation="softmax", name="basis3-1")
        self.fc4_1 = layers.Dense(category_num, activation="softmax", name="basis4-1")
        self.c = tf.Variable(normalization(tf.random.uniform([5, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)
        z1_0 = self.fc1_0(x)
        z2_0 = self.fc2_0(x)
        z3_0 = self.fc3_0(x)
        z4_0 = self.fc4_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z1_0)
        z2_1 = self.fc2_1(z2_0)
        z3_1 = self.fc3_1(z3_0)
        z4_1 = self.fc4_1(z4_0)
        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1),
                              tf.multiply(self.c[2], z2_1), tf.multiply(self.c[3], z3_1),
                               tf.multiply(self.c[4], z4_1)])
        return output

class Mnist_basis4(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis4, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(100, activation='relu', name="basis1-0")
        self.fc2_0 = layers.Dense(100, activation='relu', name="basis2-0")
        self.fc3_0 = layers.Dense(100, activation='relu', name='basis3-0')

        self.fc0_1 = layers.Dense(category_num, activation="softmax", name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, activation="softmax", name="basis1-1")
        self.fc2_1 = layers.Dense(category_num, activation="softmax", name="basis2-1")
        self.fc3_1 = layers.Dense(category_num, activation="softmax", name="basis3-1")
        self.c = tf.Variable(normalization(tf.random.uniform([4, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)
        z1_0 = self.fc1_0(x)
        z2_0 = self.fc2_0(x)
        z3_0 = self.fc3_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z1_0)
        z2_1 = self.fc2_1(z2_0)
        z3_1 = self.fc3_1(z3_0)
        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1),
                              tf.multiply(self.c[2], z2_1), tf.multiply(self.c[3], z3_1)])
        return output

class Mnist_basis3(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis3, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(100, activation='relu', name="basis1-0")
        self.fc2_0 = layers.Dense(100, activation='relu', name="basis2-0")

        self.fc0_1 = layers.Dense(category_num, activation="softmax", name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, activation="softmax", name="basis1-1")
        self.fc2_1 = layers.Dense(category_num, activation="softmax", name="basis2-1")
        self.c = tf.Variable(normalization(tf.random.uniform([3, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)
        z1_0 = self.fc1_0(x)
        z2_0 = self.fc2_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z1_0)
        z2_1 = self.fc2_1(z2_0)
        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1),
                              tf.multiply(self.c[2], z2_1)])
        return output

class Mnist_basis2(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis2, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu', name="basis0-0")
        self.fc1_0 = layers.Dense(100, activation='relu', name="basis1-0")

        self.fc0_1 = layers.Dense(category_num, activation="softmax", name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, activation="softmax", name="basis1-1")
        self.c = tf.Variable(normalization(tf.random.uniform([2, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)
        z1_0 = self.fc1_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z1_0)
        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1)])
        return output

class Mnist_basis5_share(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis5_share, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu')

        self.fc0_1 = layers.Dense(category_num, name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, name="basis1-1")
        self.fc2_1 = layers.Dense(category_num, name="basis2-1")
        self.fc3_1 = layers.Dense(category_num, name="basis3-1")
        self.fc4_1 = layers.Dense(category_num, name="basis4-1")
        self.c = tf.Variable(normalization(tf.random.uniform([5, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z0_0)
        z2_1 = self.fc2_1(z0_0)
        z3_1 = self.fc3_1(z0_0)
        z4_1 = self.fc3_1(z0_0)

        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1),
                              tf.multiply(self.c[2], z2_1), tf.multiply(self.c[3], z3_1),
                               tf.multiply(self.c[4], z4_1)])
        output = layers.Softmax()(output)
        return output


class Mnist_basis4_share(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis4_share, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu')

        self.fc0_1 = layers.Dense(category_num, name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, name="basis1-1")
        self.fc2_1 = layers.Dense(category_num, name="basis2-1")
        self.fc3_1 = layers.Dense(category_num, name="basis3-1")
        self.c = tf.Variable(normalization(tf.random.uniform([4, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z0_0)
        z2_1 = self.fc2_1(z0_0)
        z3_1 = self.fc3_1(z0_0)

        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1),
                              tf.multiply(self.c[2], z2_1), tf.multiply(self.c[3], z3_1)])
        output = layers.Softmax()(output)
        return output

class Mnist_basis3_share(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis3_share, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu')

        self.fc0_1 = layers.Dense(category_num,  name="basis0-1")
        self.fc1_1 = layers.Dense(category_num,  name="basis1-1")
        self.fc2_1 = layers.Dense(category_num,  name="basis2-1")
        self.c = tf.Variable(normalization(tf.random.uniform([3, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z0_0)
        z2_1 = self.fc2_1(z0_0)
        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1),
                              tf.multiply(self.c[2], z2_1)])
        output = layers.Softmax()(output)
        return output

class Mnist_basis2_share(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist_basis2_share, self).__init__()
        self.fc0_0 = layers.Dense(100, activation='relu')

        self.fc0_1 = layers.Dense(category_num, name="basis0-1")
        self.fc1_1 = layers.Dense(category_num, name="basis1-1")
        self.c = tf.Variable(normalization(tf.random.uniform([2, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)

    def call(self, x):
        z0_0 = self.fc0_0(x)

        z0_1 = self.fc0_1(z0_0)
        z1_1 = self.fc1_1(z0_0)
        output = layers.Add()([tf.multiply(self.c[0], z0_1), tf.multiply(self.c[1], z1_1)])
        output = layers.Softmax()(output)
        return output