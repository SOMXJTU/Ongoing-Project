from tqdm import tqdm
from rich.progress import Progress, track
import pandas as pd
import numpy as np
import tensorflow_federated as tff
from collections import defaultdict, Counter
import torch.nn.functional as F
import torch

def main():
    # load simulation dataset
    train_dataset, test_dataset = tff.simulation.datasets.emnist.load_data(only_digits=False, cache_dir="/checkpoint/pillutla/data")
    
    # read the client id
    train_available_client = pd.read_csv("emnist_client_ids_train.csv",dtype=str).to_numpy().reshape(-1).tolist()
    test_available_client = pd.read_csv("emnist_client_ids_test.csv",dtype=str).to_numpy().reshape(-1).tolist()
    train_available_client_set, test_available_client_set = set(train_available_client), set(test_available_client)


    # create client dataset 
    client_label_d = defaultdict(list)
    for client_id in track(train_available_client_set, description="Process train label distritbuion"):
        if client_id not in client_label_d.keys():
            tmp = [0] * 62
            client_train_dataset = train_dataset.create_tf_dataset_for_client(client_id)
            client_label_count = Counter(client_train_dataset.map(lambda x:x["label"]).as_numpy_iterator())

            for key, val in client_label_count.items():
                tmp[key] = val

            client_label_d[client_id] = tmp
    
    for client_id in track(test_available_client_set, description="Process test label distritbuion"):
        if client_id not in client_label_d.keys():
            tmp = [0] * 62
            client_test_dataset = test_dataset.create_tf_dataset_for_client(client_id)
            client_label_count = Counter(client_test_dataset.map(lambda x:x["lable"].as_numpy_iterator()))

            for key, val in client_label_count.items():
                tmp[key] = val
            
            client_label_d[client_id] = tmp
    
    # distribution matrix
    d = pd.DataFrame.from_dict(client_label_d, orient="index")
    d_tensor = torch.tensor(d.values, dtype=torch.float)
    similarity = F.cosine_similarity(d_tensor.unsqueeze(1), d_tensor.unsqueeze(0), dim=-1)
    print("*"*10, "w shape is {}".format(similarity.shape), "*"*10)
    w = pd.DataFrame(similarity.numpy(), dtype=np.float32, index=d.index, columns=d.index)
    w.to_csv("./eminist_similarity.csv")

if __name__ == "__main__":
    main()