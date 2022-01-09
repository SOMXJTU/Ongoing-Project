import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf

from clients.client_base import Client_base
from models.base_model import Mnist, Cifar

class Client_APFL(Client_base):
    def __init__(self, id, optimizer, loss_fn, metrics, E, batch_size, dataset_name, alpha, train_data={'x':[], 'y':[]}, test_data={'x':[], 'y':[]}, **kwargs):
        super(Client_APFL, self).__init__(id, optimizer, loss_fn, metrics, E, batch_size, train_data, test_data, **kwargs)

        self.local_model = self.build_model(dataset_name)
        self.local_weights = self.get_localweights()

        if alpha:
            self.alpha = alpha
            self.trainable = False
        else:
            self.alpha = tf.Variable(tf.random.uniform([1, 1], maxval=1), dtype=tf.float32)
            self.trainable = True
    
    def build_model(self, dataset_name):
        if dataset_name.startswith("Mnist"):
            local_model = Mnist()
            input_shape = (None, 784)
        elif dataset_name.startswith("Cifar"):
            local_model = Cifar()
            input_shape = (None, 32, 32 ,3)
        else:
            raise ValueError("None corrosponding model for dataset %s"%(dataset_name))
        local_model.build(input_shape)
        return local_model
    
    def get_localweights(self):
        return [variable.numpy() for variable in self.local_model.trainable_variables]
    
    def assign_weights(self, weights):
        tf.nest.map_structure(lambda x, y: x.assign(y), self.local_model.trainable_variables, weights)
    
    def forward(self, model, broadcast_weights):
        model = self.synchronize(model, broadcast_weights)

        self.mixture_weight = self.weight_model(model)

        np.random.shuffle(self.index)
        self.selected_index = self.index[:self.batch_size*self.E]
        train_set = tf.data.Dataset.from_tensor_slices((self.train_data["x"][self.selected_index],
                                                        self.train_data["y"][self.selected_index])).batch(self.batch_size)

        for feature, label in train_set:
            label = tf.cast(label, dtype=tf.int64)

            old_global_weights = self.get_updated_parameters(model)

            # optimize the global model
            with tf.GradientTape() as tape:
                predict = model(feature)
                loss_g = self.loss_fn(label, predict)
            grads_g = tape.gradient(loss_g, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads_g, model.trainable_variables))

            # optimize the local model
            self.assign_weights(self.mixture_weight)
            with tf.GradientTape() as tape:
                predict = self.local_model(feature)
                loss_m = self.loss_fn(label, predict)
            grads_m = tape.gradient(loss_m, self.local_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads_m, self.local_model.trainable_variables))

            if self.trainable:
                self.optimizer.apply_gradients(zip([self.alpha_grad(old_global_weights, grads_m)], [self.alpha]))
                self.alpha.assign(tf.clip_by_value(self.alpha, 0, 1))
            
            self.local_weights = self.get_localweights()  # the local weight at (t-1) round
            # interpolate
            self.mixture_weight = self.weight_model(model)  # the interpolate weight at (t-1) round

        variables_list = self.get_updated_parameters(model)
        return variables_list
        
    def weight_model(self, model):
        result = []
        for i in range(len(model.trainable_variables)):
            mixture_value = self.alpha * self.local_model.trainable_variables[i].numpy() + (1.0-self.alpha) * model.trainable_variables[i].numpy()
            result.append(mixture_value)
        return result

    def alpha_grad(self, old_global_weights, grads):
        cum_ = 0.0
        for v, w, grad in zip(self.local_weights, old_global_weights, grads):
            cum_ += tf.reshape(v - w, (1, -1)) @ tf.reshape(grad, (-1, 1))
        return cum_

    def test(self, model, broadcast_weights):
        model = self.synchronize(model, broadcast_weights)

        self.mixture_weight = self.weight_model(model)

        model = self.synchronize(model, self.mixture_weight)

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

