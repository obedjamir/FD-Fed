import os
import shutil
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10

def check(config_path, train_path, val_path, test_path, num_clients, num_classes):
    if os.path.exists(config_path):
        print(f"Deleting existing dataset configuration at {config_path}...")
        os.remove(config_path)

    for path in [train_path, val_path, test_path]:
        dir_path = os.path.dirname(path)
        if os.path.exists(dir_path):
            print(f"Deleting existing directory {dir_path}...")
            shutil.rmtree(dir_path)

        print(f"Creating new directory {dir_path}...")
        os.makedirs(dir_path)

    return False

def separate_data(data, num_clients, num_classes, dataset, concentration):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    # Shuffle data and labels together
    shuffled_indices = np.random.permutation(len(dataset_label))
    dataset_content = dataset_content[shuffled_indices]
    dataset_label = dataset_label[shuffled_indices]

    dataidx_map = {}

    # Split data by class
    idx_for_each_class = [np.where(dataset_label == i)[0] for i in range(num_classes)]
    
    #alpha parameter for Dirichlet distribution

    # Set probability ranges based on dataset
    if dataset == "cifar10":
        selection_prob = 0.7
    elif dataset == "cifar100":
        selection_prob = 0.5
    elif dataset == "cinic10":
        selection_prob = 0.7
    elif dataset == "nihchestxray":
        selection_prob = 0.7
    elif dataset == "chexpert":
        selection_prob = 0.7
    elif dataset == "mimic":
        selection_prob = 0.7


    # Distribute data to clients
    for i in range(num_classes):
        selected_clients = []
        for j, client in enumerate(range(num_clients)):
            if np.random.rand() < selection_prob:
                selected_clients.append(client)

        if len(selected_clients) == 0:
            selected_clients = [np.random.choice(range(num_clients))]

        proportions = np.random.dirichlet(np.repeat(concentration, len(selected_clients)))
        num_samples_list = (proportions * len(idx_for_each_class[i])).astype(int)
        num_samples_list[-1] = len(idx_for_each_class[i]) - sum(num_samples_list[:-1])
        idx = 0
        for client, num_sample in zip(selected_clients, num_samples_list):
            if client not in dataidx_map:
                dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
            else:
                dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
            idx += num_sample

    # Assign data to each client
    for client in range(num_clients):
        if client in dataidx_map:
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

            for label in np.unique(y[client]):
                statistic[client].append((int(label), int(sum(y[client] == label))))

    # Print statistics
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic

def split_data(X, y, train_size=0.8, val_size=0.1):
    train_data, val_data, test_data = [], [], []
    num_samples = {'train': [], 'val': [], 'test': []}

    for i in range(len(y)):
        X_train, X_temp, y_train, y_temp = train_test_split(X[i], y[i], train_size=train_size, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size/(1-train_size), shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        val_data.append({'x': X_val, 'y': y_val})
        test_data.append({'x': X_test, 'y': y_test})

        num_samples['train'].append(len(y_train))
        num_samples['val'].append(len(y_val))
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['val'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of validation samples:", num_samples['val'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, val_data, test_data

def save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, num_clients,
              num_classes, statistic):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'Size of samples for labels in clients': statistic,
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)

    for idx, val_dict in enumerate(val_data):
        with open(val_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=val_dict)

    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)

    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finished generating dataset.\n")
