import os
import time

from tqdm import trange,tqdm
import numpy as np
import tensorflow as tf

class Server_base():
    def __init__(self, model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.epoch = epoch
        self.E = E
        self.batch_size = batch_size

        self.model.build((input_shape))

        self.latest_parameters = self.get_parameters()
        self.path = result_path

        # self.clients = self.setup_clients(train_data, test_data)

    def get_parameters(self):
        return [variable.numpy() for variable in self.model.trainable_variables]

    def setup_clients(self, train_data, test_data):
        pass
    
    '''
    def broadcast(self):
        for client in self.clients:
            client.update_variable(self.latest_parameters)
    '''

    # TODO: change the training mode to a single model form. latest_parameter should be the [list numpy] and forward func should receive the new model parameter.(also the test)
    # TODO: C should be the private parameter, also assign in the forward and test function.
    def train(self):
        forwardtime_list = []
        testtime_list = []
        for epoch in trange(self.epoch):
            forward_start = time.time()
            # broadcast & local computation
            client_solution = [client.forward(self.model, self.latest_parameters) for client in self.clients]
            forward_stop = time.time()
            local_time = round(forward_stop - forward_start, 3)
            forwardtime_list.append(local_time)

            # aggregation
            self.latest_parameters = self.aggragate(client_solution)
            
            # broadcast & local test
            test_start = time.time()
            train_loss, train_acc, test_loss, test_acc = self.test()
            test_stop = time.time()
            test_time = round(test_stop - test_start, 3)
            testtime_list.append(test_time)

            with open(self.path, 'a+') as f:
                f.write('At round {}, the training loss is {:.4f}, the training accuracy is {:.4f}, the test loss is: {:.4f}, the test accuracy is {:.4f}'.format(epoch,
                                                                                                           train_loss,
                                                                                                           train_acc,
                                                                                                           test_loss,
                                                                                                           test_acc))
                f.write('\n')
            tqdm.write(
                'At round {}, the training loss is {:.4f}, the training accuracy is {:.4f}, the test loss is: {:.4f}, the test accuracy is {:.4f}'.format(epoch,
                                                                                                           train_loss,
                                                                                                           train_acc,
                                                                                                           test_loss,
                                                                                                           test_acc))
        dir_path = os.path.dirname(self.path)
        nptime_stamp = str(int(time.time()))
        forwardsave_name = "local computation" + nptime_stamp + ".out"
        testsave_name = "test" + nptime_stamp + ".out"
        np.savetxt(os.path.join(dir_path, forwardsave_name), np.asarray(forwardtime_list))
        np.savetxt(os.path.join(dir_path, testsave_name), np.asarray(testtime_list))

    def aggragate(self, client_solution):
        concatenate_v = [np.zeros_like(x) for x in self.latest_parameters]
        for i, solution in enumerate(client_solution):
            for j in range(len(solution)):
                concatenate_v[j] += solution[j]
        concatenate_v = [variable/len(self.clients) for variable in concatenate_v]
        return concatenate_v

    def test(self):
        train_loss_list = []
        train_acc_list = []

        # metrics_list = [[]*len(self.metrics)]
        test_loss_list = []
        test_acc_list = []

        num_train_weights = []
        num_test_weights = []

        for client in self.clients:
            train_loss, train_acc, test_loss, test_acc, train_samples, test_samples = client.test(self.model, self.latest_parameters)
            # client_loss, client_metrics, client_trainloss, 
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)

            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            num_train_weights.append(train_samples)
            num_test_weights.append(test_samples)

        return [np.average(train_loss_list, weights=num_train_weights), 
                np.average(train_acc_list, weights=num_train_weights),
                np.average(test_loss_list, weights=num_test_weights), 
                np.average(test_acc_list, weights=num_train_weights)]


