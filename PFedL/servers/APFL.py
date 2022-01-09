import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np

from servers.server_base import Server_base
from clients.APFL import Client_APFL

class Server_APFL(Server_base):
    def __init__(self, model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, dataset_name, apfl_alpha):
        super(Server_APFL, self).__init__(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape)
        
        self.dataset_name = dataset_name
        if apfl_alpha:
            self.train_alpha = False
            self.alpha = apfl_alpha
        else:
            self.train_alpha = True
            self.alpha = None

        self.clients = self.setup_clients(train_data, test_data)


    def setup_clients(self, train_data, test_data):
        num_clients = len(train_data)
        clients_list = []
        for i in range(num_clients):
            clients_list.append(Client_APFL(i, self.optimizer, self.loss_fn, self.metrics, self.E, self.batch_size, self.dataset_name, self.alpha,
                                            train_data[i], test_data[i]))
        return clients_list