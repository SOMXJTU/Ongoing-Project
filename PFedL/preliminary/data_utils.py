import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from torchvision import transforms as T

def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]
  
    return client_idcs


class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y


class CustomDataset_mnist(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x.reshape(-1, 784)/255.).float()
        self.y = torch.from_numpy(y).long()
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    (train_x, train_lables), (test_x, test_labels) = mnist.load_data()
    train_dataset = CustomDataset_mnist(train_x, train_lables)
    print("There are {:d} samples in training set".format(len(train_dataset)))
    print("The shape of top 50 raws of train_dataset is {}".format(train_dataset[np.arange(50)][0].shape))