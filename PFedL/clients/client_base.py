import sys, os
sys.path.append('..')

import tensorflow as tf
from tensorflow.keras import metrics as tf_metrics
import numpy as np


class Client_base():
    def __init__(self, id, optimizer, loss_fn, metrics, E, batch_size, train_data={'x':[], 'y':[]}, test_data={'x':[], 'y':[]}, **kwargs):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data['y'])
        self.test_samples = len(test_data['y'])
        self.index = np.arange(self.train_samples)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.batch_size = batch_size
        self.E = E
        self.kwargs = kwargs
        
        self.train_loss = tf_metrics.Mean(name='train_loss')
        self.train_acc = tf_metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf_metrics.Mean(name='test_loss')
        self.test_acc = tf_metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def synchronize(self, model, broadcast_weights):
        tf.nest.map_structure(lambda x, y: x.assign(y), model.trainable_variables, broadcast_weights)
        return model

    def forward(self, model, broadcast_weights):
        # broadcast weights before local training.
        model = self.synchronize(model, broadcast_weights)

        # prepara data
        np.random.shuffle(self.index)
        self.selected_index = self.index[:self.batch_size*self.E]
        train_set = tf.data.Dataset.from_tensor_slices((self.train_data["x"][self.selected_index],
                                                        self.train_data["y"][self.selected_index])).batch(self.batch_size)

        for feature, label in train_set:
            label = tf.cast(label, dtype=tf.int64)
            with tf.GradientTape() as tape:
                logits = model(feature)
                loss = self.loss_fn(label, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        variables_list = self.get_updated_parameters(model)
        return variables_list

    def get_updated_parameters(self, model):
        return [variable.numpy() for variable in model.trainable_variables]

    def test(self, model, broadcast_weights):
        model = self.synchronize(model, broadcast_weights)

        test_set = tf.data.Dataset.from_tensor_slices((self.test_data["x"], self.test_data["y"])).batch(200)
        train_set = tf.data.Dataset.from_tensor_slices((self.train_data["x"], self.train_data["y"])).batch(200)

        self.train_loss.reset_states()
        self.train_acc.reset_states()
        self.test_loss.reset_states()
        self.test_acc.reset_states()

        for feature, label in train_set:
            logits = model(feature)
            loss = self.loss_fn(label, logits)
            self.train_loss.update_state(loss, sample_weight=len(label))
            self.train_acc.update_state(label, logits, sample_weight=len(label))

        for feature, label in test_set:
            logits = model(feature)
            loss = self.loss_fn(label, logits)
            self.test_loss.update_state(loss, sample_weight=len(label))
            self.test_acc.update_state(label, logits, sample_weight=len(label))
        return (self.train_loss.result().numpy(), self.train_acc.result().numpy(), self.test_loss.result().numpy(), self.test_acc.result().numpy(), self.train_samples, self.test_samples)