import os
import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics

from utils.utils import generate_W, find_model, get_data

def arg_parse():
    parser = argparse.ArgumentParser()

    # common parameters
    parser.add_argument("--dataset", type=str, default="Mnist_basis4", choices=["Mnist_basis4", "Mnist_basis3",
                                                                                "Mnist_basis2", "Mnist_basis2_4",
                                                                                "Mnist_basis2_3", "Cifar10_basis4",
                                                                                "Cifar10_basis3", "Cifar10_basis2"])

    parser.add_argument("--algorithm", type=str, default="PFedL", choices=["Fedavg", "APFL", "FedLG", "FedMe", "PFedL"])
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--epoches", type=int, default=200)
    parser.add_argument("--E", type=int, default=2, help="the number of inner steps of theta")

    # parameters for PFedL
    parser.add_argument("--pfedl_lamb", type=float, default=5e-2, help="the penalty parameters")
    parser.add_argument("--E_c", type=int, default=1, help="the number of inner steps rate for c")
    parser.add_argument("--lr_c", type=float, default=1.0, help="the learning rate of c")
    parser.add_argument("--beta", type=float, default=0.99, help="the exponential learning decay rate")
    parser.add_argument("--postive_w", type=float, default=1.0, help="the value of w within the same group")
    parser.add_argument("--negative_w", type=float, default=-1.0, help="the value of w within the different group")
    parser.add_argument("--experts", action="store_true", default=False, help="mixing the prediction or interprolating the model, the default is mixing the prediction.")
    parser.add_argument("--pfedl_share", action="store_true", default=False, help="whether share the bottom layer")

    # parameters for FedMe
    parser.add_argument("--fedme_lamb", type=int, default=15, help="the penalty parameter of FedMe")
    parser.add_argument("--K", type=int, default=5, help="the inner step in FedMe")
    parser.add_argument("--lr_fedme", type=float, default=0.01, help="the inner learning rate in FedMe")

    # parameter for APFL
    parser.add_argument("--apfl_alpha", type=float, default=0.25, help="the alpha value in APFL")
    parser.add_argument("--apfl_trainable_alpha", action="store_true", default=False, help="whether train the alpha in apfl, the default is false")

    # default parameters
    parser.add_argument("--seed_numpy", type=int, default=12)
    parser.add_argument("--seed_tf", type=int, default=123)
    parser.add_argument("--num_repeat", type=int, default=10)

    args = parser.parse_args()
    return args

def main(parser, result_path):
    optimizer = optimizers.SGD(lr=parser.learning_rate)
    loss_fn = losses.SparseCategoricalCrossentropy()
    metrics_list = [metrics.Accuracy()]
    epoch = parser.epoches
    E = parser.E
    batch_size = parser.batch_size

    # load basic model
    model = find_model(parser)

    # load dataset
    if parser.dataset.startswith("Mnist"):
        data_path = os.path.join("./data/mnist", parser.dataset)
        input_shape = (None, 784)
        g_slot = 2
    elif parser.dataset.startswith("Cifar"):
        data_path = os.path.join("./data/cifar10", parser.dataset)
        input_shape = (None, 32, 32, 3)
        g_slot = 6
    train_data, test_data = get_data(data_path)
    num_clients = len(train_data)


    # load server
    if parser.algorithm == "Fedavg":
        from servers.Fedavg import Server_fedavg
        server = Server_fedavg(model, optimizer, loss_fn, metrics_list, epoch, E, batch_size, train_data, test_data, result_path, input_shape)
    elif parser.algorithm == "FedLG":
        from servers.FedLG import Server_FedLG
        server = Server_FedLG(model, optimizer, loss_fn, metrics_list, epoch, E, batch_size, train_data, test_data, result_path, input_shape, g_slot)
    elif parser.algorithm == "FedMe":
        fedme_K = parser.K
        fedme_lr = parser.lr_fedme
        fedme_lambda = parser.fedme_lamb
        from servers.FedMe import Server_FedMe
        server = Server_FedMe(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, fedme_K, fedme_lr, fedme_lambda)
    elif parser.algorithm == "APFL":
        if parser.apfl_trainable_alpha:
            apfl_alpha = None
        else:
            apfl_alpha = parser.apfl_alpha
        dataset_name = parser.dataset
        from servers.APFL import Server_APFL
        server = Server_APFL(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, dataset_name, apfl_alpha)
    elif parser.algorithm == "PFedL":
        basis_num = int(parser.dataset[-1])
        W, D = generate_W(basis_num, num_clients, parser.postive_w, parser.negative_w)

        pfedl_lamb = parser.pfedl_lamb
        E_c = parser.E_c
        lr_c = parser.lr_c
        beta = parser.beta

        from servers.PFedL import Server_PFedL
        server = Server_PFedL(model, optimizer, loss_fn, metrics, epoch, E, batch_size, train_data, test_data, result_path, input_shape, E_c, lr_c, beta, pfedl_lamb, W, D, parser.experts, basis_num)
    else:
        raise ValueError("Unimplement")

    '''
    if parser.dataset.startswith("Mnist"):

        num_clients = len(train_data)

        if parser.algorithm == "PFedL":
            dataset_name = parser.dataset
            basis_num = re.findall(re.compile(r'basis([0-9]?)'), dataset_name)[0]
            quotient, remainder = num_clients // int(basis_num), num_clients % int(basis_num)
            if parser.pfedl_share:
                if basis_num == '2':
                    model = model_src.Mnist_basis2_share()
                    model_class = model_src.Mnist_basis2_share
                elif basis_num == '3':
                    model = model_src.Mnist_basis3_share()
                    model_class = model_src.Mnist_basis3
                elif basis_num == "4":
                    model = model_src.Mnist_basis4_share()
                    model_class = model_src.Mnist_basis4_share
                else:
                    raise ValueError("Unsupported basis number")
            else:
                if basis_num == '2':
                    model = model_src.Mnist_basis2()
                    model_class = model_src.Mnist_basis2
                elif basis_num == '3':
                    model = model_src.Mnist_basis3()
                    model_class = model_src.Mnist_basis3
                elif basis_num == "4":
                    model = model_src.Mnist_basis4()
                    model_class = model_src.Mnist_basis4
                else:
                    raise ValueError("Unsupported basis number")

            # TODO: find the suitable value 
            W, D = generate_W(basis_num, num_clients, positive_value=1.0, negative_value=1.0)

            if parser.pfedl_syn:
                client_class = client_src.PFedL.Client_PFedL_syn
                server_class = server_src.PFedL.Server_PFedL_syn
            else:
                client_class = client_src.PFedL.Client_PFedL_nonsyn
                server_class = server_src.PFedL.Server_PFedL_nonsyn
            lamb = parser.pfedl_lamb
            E_c = parser.E_c
            lr_c = parser.lr_c
            beta = parser.beta
            server = server_class(model, client_class, optimizer, E, loss_fn,  metrics_list, batch_size, train_data, test_data, epoch, result_path,
                                  lamb, E_c, lr_c, W, D, model_class, beta)
        # pfedl, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path,
        #                  lamb, E_c, lr_c, W, D, model_class, beta=0.99, **kwargs):
        else:
            model = model_src.Mnist()
            model_class = model_src.Mnist
            if parser.algorithm == "Fedavg":
                client_class = client_src.Fedavg.Client_Fedavg
                server = server_src.Fedavg.Server_fedavg(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                         epoch, result_path, model_class)
            # fedavg, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path
            elif parser.algorithm == "APFL":
                alpha = parser.apfl_alpha
                client_class = client_src.APFL.Client_APFL
                server = server_src.APFL.Server_APFL(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                     epoch, result_path, model_class, alpha)
            # apfl, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path, model_class, **kwargs):
            elif parser.algorithm == "FedLG":
                client_class = client_src.FedLG.Client_FedLG_mnist
                g_slot = 2
                server = server_src.FedLG.Server_FedLG_mnist(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                             epoch, result_path, model_class, g_slot)
            # fedlg, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path, g_slot, **kwargs):
            elif parser.algorithm == "FedMe":
                client_class = client_src.FedMe.Client_FedMe
                K = parser.K
                old_lr = parser.lr_fedme
                lamb = parser.fedme_lamb
                server = server_src.FedMe.Server_FedMe(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                       epoch, result_path, model_class, K, old_lr, lamb)
                # fedme, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path, K, old_lr, lamb, **kwargs):
            else:
                raise AttributeError("Unsupported Algorithm")

    elif parser.dataset.startswith("Cifar10"):
        data_path = os.path.join("./data/cifar10", parser.dataset)
        train_path = os.path.join(data_path, 'train_array.pkl')
        test_path = os.path.join(data_path, 'test_array.pkl')
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)
        train_data = train_data['user_data']
        test_data = test_data['user_data']
        num_clients = len(train_data)

        if parser.algorithm == "PFedL":
            
            dataset_name = parser.dataset
            basis_num = re.findall(re.compile(r'basis([0-9]?)'), dataset_name)[0]
            quotient, remainder = num_clients // int(basis_num), num_clients % int(basis_num)
            if parser.pfedl_share:
                if basis_num == '2':
                    model = model_src.Mnist_basis2_share()
                    model_class = model_src.Mnist_basis2_share
                elif basis_num == '3':
                    model = model_src.Mnist_basis3_share()
                    model_class = model_src.Mnist_basis3
                elif basis_num == "4":
                    model = model_src.Mnist_basis4_share()
                    model_class = model_src.Mnist_basis4_share
                else:
                    raise ValueError("Unsupported basis number")
            else:
                if basis_num == '2':
                    model = model_src.Mnist_basis2()
                    model_class = model_src.Mnist_basis2
                elif basis_num == '3':
                    model = model_src.Mnist_basis3()
                    model_class = model_src.Mnist_basis3
                elif basis_num == "4":
                    model = model_src.Mnist_basis4()
                    model_class = model_src.Mnist_basis4
                else:
                    raise ValueError("Unsupported basis number")

            W_list = []
            row_index = 0
            for i in range(int(basis_num)):
                if i < (int(basis_num) -1):
                    w_temp = np.zeros((quotient, num_clients))
                    w_temp[:, row_index:(i+1)*quotient] = 1
                    W_list.append(w_temp)
                    row_index += quotient
                else:
                    w_temp = np.zeros((quotient+remainder, num_clients))
                    w_temp[:, row_index:] = 1
                    W_list.append(w_temp)
            W = np.concatenate(W_list, axis=0)
            D = np.diag(np.sum(W, axis=1))

            if parser.pfedl_syn:
                client_class = client_src.PFedL.Client_PFedL_syn
                server_class = server_src.PFedL.Server_PFedL_syn
            else:
                client_class = client_src.PFedL.Client_PFedL_nonsyn
                server_class = server_src.PFedL.Server_PFedL_nonsyn
            lamb = parser.pfedl_lamb
            E_c = parser.E_c
            lr_c = parser.lr_c
            beta = parser.beta
            server = server_class(model, client_class, optimizer, E, loss_fn,  metrics_list, batch_size, train_data, test_data, epoch, result_path,
                                  lamb, E_c, lr_c, W, D, model_class, beta)
        # pfedl, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path,
        #                  lamb, E_c, lr_c, W, D, model_class, beta=0.99, **kwargs):
            
            raise AttributeError("Loading")
        else:
            model = model_src.Cifar()
            model_class = model_src.Cifar
            if parser.algorithm == "Fedavg":
                client_class = client_src.Fedavg.Client_Fedavg
                server = server_src.Fedavg.Server_fedavg(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                         epoch, result_path, model_class)
            # fedavg, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path
            elif parser.algorithm == "APFL":
                alpha = parser.apfl_alpha
                client_class = client_src.APFL.Client_APFL
                server = server_src.APFL.Server_APFL(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                     epoch, result_path, model_class, alpha)
            # apfl, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path, model_class, **kwargs):
            elif parser.algorithm == "FedLG":
                client_class = client_src.FedLG.Client_FedLG_mnist
                g_slot = 4
                server = server_src.FedLG.Server_FedLG_mnist(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                             epoch, result_path, model_class, g_slot)
            # fedlg, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path, g_slot, **kwargs):
            elif parser.algorithm == "FedMe":
                client_class = client_src.FedMe.Client_FedMe
                K = parser.K
                old_lr = parser.lr_fedme
                lamb = parser.fedme_lamb
                server = server_src.FedMe.Server_FedMe(model, client_class, optimizer, E, loss_fn, metrics_list, batch_size, train_data, test_data,
                                                       epoch, result_path, model_class, K, old_lr, lamb)
                # fedme, def __init__(self, model, clients_class, optimizer, E, loss_fn, metrics, batch_size, train_data, test_data, epoch, result_path, K, old_lr, lamb, **kwargs):
            else:
                raise AttributeError("Unsupported Algorithm")
    else:
        raise AttributeError("Unsupported dataset")
    '''
    return server

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    parser = arg_parse()
    seed_np = parser.seed_numpy
    seed_tf = parser.seed_tf
    data_result_path = os.path.join('./result', parser.dataset)
    if not os.path.exists(data_result_path):
        os.makedirs(data_result_path)
    algorithm_result_path = os.path.join(data_result_path, parser.algorithm)
    if not os.path.exists(algorithm_result_path):
        os.makedirs(algorithm_result_path)
    for i in range(parser.num_repeat):
        seed_np = seed_np + 1
        seed_tf = seed_tf + 1
        np.random.seed(seed_np)
        tf.random.set_seed(seed_tf)
        result_path = os.path.join(algorithm_result_path, 'result_'+str(seed_np)+str(int(time.time()))+'.txt')
        server = main(parser, result_path)
        server.train()
    '''
    np.random.seed(seed_np)
    tf.random.set_seed(seed_tf)
    result_path = os.path.join(algorithm_result_path, 'result_'+str(seed_np)+'.txt')
    server = main(parser, result_path)
    server.train()
    '''