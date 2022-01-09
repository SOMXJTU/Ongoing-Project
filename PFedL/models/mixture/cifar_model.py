import sys
import os
parent_path = os.path.dirname(os.path.dirname(__file__))
top_path = os.path.dirname(parent_path)
sys.path.append(top_path)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import normalization, generate_fc_weights


class Cifar_basis(tf.keras.Model):
    def __init__(self, num_basis, category_num=10) -> None:
        super(Cifar_basis, self).__init__()

        self.num_basis = num_basis
        self.c = tf.Variable(normalization(tf.random.uniform([num_basis, 1], maxval=1, dtype=tf.float32)),
                             trainable=False, dtype=tf.float32)
        
        # bottom layer
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPool2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPool2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        
        self.fc0_list = [generate_fc_weights(1024, 64, "fc0basis_"+str(i)) for i in range(num_basis)]
        self.fc1_list = [generate_fc_weights(64, category_num, "fc1basis_"+str(i)) for i in range(num_basis)]

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = layers.Flatten()(x)
        
        # mixing the first fc layer
        self.weights_0, self.bias_0 = (layers.Add()([self.fc0_list[i][0]*self.c[i] for i in range(self.num_basis)]), 
        layers.Add()([self.fc0_list[i][1]*self.c[i] for i in range(self.num_basis)]))
        
        x = tf.nn.relu(x@self.weights_0 + self.bias_0)

        # mixing the second fc layer
        self.weights_1, self.bias_1 = (layers.Add()([self.fc1_list[i][0]*self.c[i] for i in range(self.num_basis)]), 
        layers.Add()([self.fc1_list[i][1]*self.c[i] for i in range(self.num_basis)]))

        x = tf.nn.softmax(x@self.weights_1 + self.bias_1)
        return x
