import sys
sys.path.append('..')

from servers.server_base import Server_base
from clients.Fedavg import Client_Fedavg

class Server_fedavg(Server_base):
    def __init__(self, model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, **kwargs):
        super(Server_fedavg, self).__init__(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, **kwargs)
        self.clients = self.setup_clients(train_data, test_data)
    
    def setup_clients(self, train_data, test_data):
        num_clients = len(train_data)
        clients_list = []
        for i in range(num_clients):
            clients_list.append(Client_Fedavg(i, self.optimizer, self.loss_fn, self.metrics,  self.E, self.batch_size, train_data[i], test_data[i]))
        return clients_list
