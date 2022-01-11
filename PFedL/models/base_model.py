import tensorflow as tf
from tensorflow.keras import layers

class Mnist(tf.keras.Model):
    def __init__(self, category_num=10):
        super(Mnist, self).__init__()
        self.fc_0 = layers.Dense(100, activation='relu')
        self.fc_1 = layers.Dense(category_num, activation='softmax')

    def call(self, x):
        x = self.fc_0(x)
        output = self.fc_1(x)
        return output

class Cifar(tf.keras.Model):
    '''
    Referece https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn
    '''
    def __init__(self):
        super(Cifar, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPool2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = layers.MaxPool2D((2, 2))
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(64, activation='relu')
        self.fc_2 = layers.Dense(10, activation='softmax')

    def __call__(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        x = layers.Flatten()(x)
        output = self.fc_2(self.fc_1(x))
        return output
