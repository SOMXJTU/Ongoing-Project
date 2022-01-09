import sys
sys.path.append("..")

import numpy as np
import  tensorflow as tf

from clients.client_base import Client_base

class Client_FedMe(Client_base):
    def __init__(self, id, optimizer, loss_fn, metrics, E, batch_size, K, old_lr, lamb, train_data={'x':[], 'y':[]}, test_data={'x':[], 'y':[]}, **kwargs):
        super(Client_FedMe, self).__init__(id, optimizer, loss_fn, metrics, E, batch_size, train_data, test_data, **kwargs)
        self.K = K
        self.old_lr = old_lr
        self.lamb = lamb

    def forward(self, model, broadcast_weights):
        model = self.synchronize(model, broadcast_weights)

        old_parameters = broadcast_weights
        for e in range(self.E):
            np.random.shuffle(self.index)
            fresh_index = self.index[-self.batch_size*self.K:]
            fresh_feature, fresh_label = tf.cast(self.train_data["x"][fresh_index], dtype=tf.float32), tf.cast(self.train_data["y"][fresh_index], dtype=tf.int64)
            for k in range(self.K):
                with tf.GradientTape() as tape:
                    logits = model(fresh_feature)
                    loss = self.loss_fn(fresh_label, logits)
                grads = tape.gradient(loss, model.trainable_variables)

                new_grads = self.moreau_envelop(model, old_parameters, grads)
                self.optimizer.apply_gradients(zip(new_grads, model.trainable_variables))

            old_parameters = self.update_oldparameters(old_parameters, model)

        return old_parameters    

    def moreau_envelop(self, model, old_parameters,grads):
        result = []
        for new_variable, old_variable, grad in zip(model.trainable_variables, old_parameters, grads):
            new_grad = grad.numpy() + self.lamb * (new_variable.numpy() - old_variable)
            result.append(new_grad)
        return result

    def update_oldparameters(self, old_parameters, model):
        result = []
        for old_variable, new_variable in zip(old_parameters, model.trainable_variables):
            temp = old_variable - self.lamb * self.old_lr * (old_variable - new_variable.numpy())
            result.append(temp)
        return result