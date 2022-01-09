import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
from clients.client_base import Client_base

class Client_PFedL(Client_base):
    def __init__(self, id, optimizer, loss_fn, metrics, E, batch_size, E_c, lr_c, beta, experts_flags, basis_num, train_data={'x':[], 'y':[]}, test_data={'x':[], 'y':[]}, **kwargs):
        super(Client_PFedL, self).__init__(id, optimizer, loss_fn, metrics, E, batch_size, train_data, test_data)

        self.E_c = E_c
        self.lr_c = lr_c
        self.beta = beta
        self.experts_flags = experts_flags
        self.kwargs = kwargs
        self.basis_num = basis_num
        self.epoch = 0

        self.c =  tf.Variable(np.asarray([1/self.basis_num]*self.basis_num).reshape(-1, 1), trainable=False, dtype=tf.float32)

    def synchronize(self, model, broadcast_weights):
        model.c.assign(self.c)
        return super().synchronize(model, broadcast_weights)
    
    def forward(self, model, broadcast_weights, D_row, W_row):
        self.synchronize(model, broadcast_weights)

        np.random.shuffle(self.index)
        self.selected_index = self.index[:self.batch_size * self.E]
        train_set = tf.data.Dataset.from_tensor_slices((self.train_data["x"][self.selected_index],
                                                        self.train_data["y"][self.selected_index])).batch(self.batch_size)
        for e in range(self.E_c):
            lr = self.lr_c * self.beta **(self.epoch*self.E_c+e)
            self.md_c(model, D_row, W_row, lr=lr)        

        model.c.assign(self.c.numpy())

        for feature, label in train_set:
            with tf.GradientTape() as tape:
                logits = model(feature)
                loss = self.loss_fn(label, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        variables_list = self.get_updated_parameters(model)

        name_list = [variable.name for variable in model.trainable_variables]
        self.epoch += 1

        return (variables_list, self.c.numpy().ravel(), name_list)

    def md_c(self, model, D_row, W_row, lr):
        batch_x = self.train_data['x'][self.selected_index]
        batch_y = self.train_data['y'][self.selected_index]        
        batch_x = tf.cast(batch_x, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(model.c)
            logits = model(batch_x)
            loss = self.loss_fn(batch_y, logits)

        grads_c = tape.gradient(loss, model.c)
        grads_c += tf.reshape(tf.cast(D_row - W_row, dtype=tf.float32), model.c.shape)
        grads_c = model.c * tf.exp(-lr * grads_c)

        self.c.assign(grads_c / tf.reduce_sum(grads_c))
