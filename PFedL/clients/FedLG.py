import sys
sys.path.append("..")

import copy 
import tensorflow as tf

from clients.client_base import Client_base

class Client_FedLG(Client_base):
    def __init__(self, id, optimizer, loss_fn, metrics, E, batch_size, g_slot, train_data={'x':[], 'y':[]}, test_data={'x':[], 'y':[]}, **kwargs):
        super(Client_FedLG, self).__init__(id, optimizer, loss_fn, metrics, E, batch_size, train_data, test_data, **kwargs)
        self.g_slot = g_slot
    

    def resigter_local(self, local_parameters):
        self.local_parameters = copy.deepcopy(local_parameters)
    
    def synchronize(self, model, broadcast_weights):
        variables = model.trainable_variables
        tf.nest.map_structure(lambda x, y: x.assign(y), variables[:self.g_slot], self.local_parameters)
        tf.nest.map_structure(lambda x, y: x.assign(y), variables[self.g_slot:], broadcast_weights)
        return model

    def get_updated_parameters(self, model):
        variable_list = [variable.numpy() for variable in model.trainable_variables[self.g_slot:]]
        self.local_parameters = [variable.numpy() for variable in model.trainable_variables[:self.g_slot]]
        return variable_list


