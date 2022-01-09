import sys
sys.path.append("..")
from clients.client_base import Client_base

class Client_Fedavg(Client_base):

    def __init__(self, id, optimizer, loss_fn, metrics, E, batch_size, train_data={'x':[], 'y':[]}, test_data={'x':[], 'y':[]}, **kwargs):
        super(Client_Fedavg, self).__init__(id, optimizer, loss_fn, metrics, E, batch_size, train_data, test_data, **kwargs)
    
    
