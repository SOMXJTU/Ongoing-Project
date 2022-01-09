import sys
sys.path.append('..')

from tqdm import trange,tqdm
import numpy as np
import tensorflow as tf

from servers.server_base import Server_base
from clients.FedLG import Client_FedLG

class Server_FedLG(Server_base):
    def __init__(self, model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, g_slot, **kwargs):
        super(Server_FedLG, self).__init__(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, **kwargs)
        self.g_slot = g_slot

        self.latest_parameters = self.global_parameters()
        self.clients = self.setup_clients(train_data, test_data)

    def global_parameters(self):
        global_variable = self.model.trainable_variables[self.g_slot:]
        variable_list = [variable.numpy() for variable in global_variable]
        return variable_list

    def setup_clients(self, train_data, test_data):
        num_clients = len(train_data)
        clients_list = []
        for i in range(num_clients):
            clients_list.append(Client_FedLG(i, self.optimizer, self.loss_fn, self.metrics, self.E, self.batch_size, self.g_slot, train_data[i], test_data[i]))
        return clients_list

    def local_parameters(self):
        """
        Get the local parameter for initiating  client local parameters
        """
        local_variable = self.model.trainable_variables[:self.g_slot]
        variable_list = [variable.numpy() for variable in local_variable]
        return variable_list
    
    def init_local(self):
        local_parameter = self.local_parameters()
        for client in self.clients:
            client.resigter_local(local_parameter)  # deepcopy the initial local value

    def train(self):
        self.init_local()
        super().train()
