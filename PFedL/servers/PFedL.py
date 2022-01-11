import sys
sys.path.append("..")
import os

import re
import time
from tqdm import trange,tqdm
import numpy as np

from servers.server_base import Server_base
from clients.PFedL import Client_PFedL

class Server_PFedL(Server_base):
    def __init__(self, model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, E_c, lr_c, beta, pfedl_lamb, W, D, experts_flags, basis_num, **kwargs):
        super(Server_PFedL, self).__init__(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape)

        self.E_c = E_c
        self.lr_c = lr_c
        self.beta = beta
        self.lamb = pfedl_lamb
        self.W = W
        self.D = D
        self.experts_flags = experts_flags  # interpolate(True) or mix the experts(False)
        self.basis_num = basis_num

        self.clients = self.setup_clients(train_data, test_data)
        self.C = np.repeat([[1.0/self.basis_num] *self.basis_num], len(train_data), axis=0)  # initial C
        # self.C = self.aggragate_c()
        # self.correct_c = self.generate_c()


    def setup_clients(self, train_data, test_data):
        num_clients = len(train_data)
        clients_list = []
        for i in range(num_clients):
            clients_list.append(Client_PFedL(i, self.optimizer, self.loss_fn, self.metrics, self.E, self.batch_size, self.E_c, self.lr_c, 
                                            self.beta, self.experts_flags, self.basis_num, train_data[i], test_data[i]))
        return clients_list

    def aggragate_c(self):
        c_list = [client.c.numpy().ravel() for client in self.clients]
        C = np.asarray(c_list).reshape(len(self.clients), -1)
        return C

    def train(self):
        for epoch in trange(self.epoch):

            client_solution = [self.clients[i].forward(self.model, self.latest_parameters,
                                                       self.lamb * self.D[i, i] * self.C[i, :],
                                                       self.lamb * self.W.dot(self.C)[i, :])
                                                       for i in range(len(self.clients))]
        
            self.latest_parameters, self.C = self.aggragate(client_solution)

            train_loss, train_acc, test_loss, test_acc = self.test()

            with open(self.path, 'a+') as f:
                f.write('At round {}, the training loss is {:.4f}, the training accuracy is {:.4f}, the test loss is: {:.4f}, the test accuracy is {:.4f}'.format(epoch,
                                                                                                           train_loss,
                                                                                                           train_acc,
                                                                                                           test_loss,
                                                                                                           test_acc))
                f.write("\n")

            tqdm.write(
                'At round {}, the training loss is {:.4f}, the training accuracy is {:.4f}, the test loss is: {:.4f}, the test accuracy is {:.4f}'.format(epoch,
                                                                                                           train_loss,
                                                                                                           train_acc,
                                                                                                           test_loss,
                                                                                                           test_acc))

            time_stamp = int(time.time())

            dir_path = os.path.dirname(self.path)
            base_name = os.path.basename(self.path)
            seed = int(os.path.splitext(base_name)[0].split("_")[1])
            if (epoch+1)%50 == 0:
                c_path = os.path.join(dir_path, "c_"+"seed"+str(seed)+str(epoch+1)+str(time_stamp)+".txt")
                np.savetxt(c_path, self.C)

    def aggragate(self, client_solution):
        concatenate_c = []
        concatenate_v = [np.zeros_like(x) for x in self.latest_parameters]
        basis_weight = [0] * len(concatenate_v)
        for i, solution in enumerate(client_solution):
            client_variables, client_c, name_list = solution
            for j in range(len(client_variables)):
                name = name_list[j]
                variable = client_variables[j]

                if 'basis' in name:
                    if self.experts_flags:
                        correspond_basis = int(re.split('basis|-', name)[1])
                    else:
                        correspond_basis = int(name[name.index("basis")+6])
                    concatenate_v[j] += variable * client_c[correspond_basis]
                    basis_weight[j] += client_c[correspond_basis]
                else:
                    concatenate_v[j] += variable / len(self.clients)
            concatenate_c.append(client_c)
        C = np.concatenate(concatenate_c).reshape(self.C.shape)
        V = [concatenate_v[i]/basis_weight[i] if basis_weight[i] >0 else concatenate_v[i] for i in range(len(concatenate_v)) ]
        return (V,C)
