import sys
sys.path.append('..')

from servers.server_base import Server_base
from clients.FedMe import Client_FedMe


class Server_FedMe(Server_base):
    def __init__(self, model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, K, old_lr, lamb, **kwargs):
        super(Server_FedMe, self).__init__(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, **kwargs)
        self.K = K
        self.old_lr = old_lr
        self.lamb = lamb

        self.clients = self.setup_clients(train_data, test_data)

    def setup_clients(self,train_data, test_data):
        num_clients = len(train_data)
        clients_list = []
        for i in range(num_clients):
            clients_list.append(Client_FedMe(i, self.optimizer, self.loss_fn, self.metrics, self.E,  self.batch_size, self.K, self.old_lr, self.lamb, train_data[i], test_data[i]))
        return clients_list
