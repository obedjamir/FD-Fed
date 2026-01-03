import ujson
import numpy as np
import os
import torch

def batch_data(data, batch_size):
    data_x = data['x']
    data_y = data['y']
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)

def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x) // batch_size + 1
    if len(data_x) > batch_size:
        batch_idx = np.random.choice(range(num_parts + 1))
        sample_index = batch_idx * batch_size
        if sample_index + batch_size > len(data_x):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index + batch_size], data_y[sample_index: sample_index + batch_size])
    else:
        return (data_x, data_y)

def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[:batch_size]
    batched_y = data_y[:batch_size]
    return (batched_x, batched_y)

def read_data(dataset, idx, dataset_id, data_split='train'):
    data_dir = os.path.join('../dataset', dataset, f'{data_split}{dataset_id}/')
    file_path = os.path.join(data_dir, f'{idx}.npz')
    
    with open(file_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    
    return data

def read_client_data(dataset, idx, dataset_id, data_split='train'):
    train_data = read_data(dataset, idx, dataset_id, 'train')
    test_data = read_data(dataset, idx, dataset_id, 'test')
    val_data = read_data(dataset, idx, dataset_id, 'val')

    X_train, y_train = train_data['x'], train_data['y']
    X_test, y_test = test_data['x'], test_data['y']
    X_val, y_val = val_data['x'], val_data['y']

    unique_labels = set(y_train).union(set(y_test)).union(set(y_val))
    label_map = {label: i for i, label in enumerate(unique_labels)}

    if data_split == 'train':
        return [(x, label_map[y]) for x, y in zip(X_train, y_train)], unique_labels
    elif data_split == 'val':
        return [(x, label_map[y]) for x, y in zip(X_val, y_val)], unique_labels
    else:
        return [(x, label_map[y]) for x, y in zip(X_test, y_test)], unique_labels
