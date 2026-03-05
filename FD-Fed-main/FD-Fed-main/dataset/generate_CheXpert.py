import numpy as np
import os
import sys
import random
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from utils.dataset_utils import check, separate_data, split_data, save_file


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_clients = 3
runs = 5
num_classes = 14
dir_path = "chexpert/"

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.filter_data()
        self.unique_labels = []

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        label = self.data_frame.iloc[idx, -1]
        if label not in self.unique_labels:
            self.unique_labels.append(label)

        label = torch.tensor(self.unique_labels.index(label))
        return img_name, label

    def filter_data(self):
        # Exclude "No Finding" and keep only rows with one valid label
        label_columns = self.data_frame.columns[6:]

        # Replace 0.0 and -1.0 with NaN to avoid considering them as valid labels
        self.data_frame[label_columns] = self.data_frame[label_columns].replace([0.0, -1.0], pd.NA)

        # Convert columns to numeric (if they aren't already) to ensure idxmax works correctly
        self.data_frame[label_columns] = self.data_frame[label_columns].apply(pd.to_numeric, errors='coerce')

        # Filter rows with exactly one non-NaN label
        self.data_frame['LabelCount'] = self.data_frame[label_columns].notna().sum(axis=1)
        self.data_frame = self.data_frame[self.data_frame['LabelCount'] == 1]

        # Add a new 'Label' column with the first non-NaN value
        def find_label(row):
            return row[row.notna()].idxmax()

        self.data_frame['Label'] = self.data_frame[label_columns].apply(find_label, axis=1)
    def print_unique_labels(self):
        print("Unique labels:", self.unique_labels)

class CustomDataset(Dataset):
    def __init__(self, subset):
        self.data = [sample for sample, _ in subset]
        self.targets = [label for _, label in subset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, target = self.data[idx], self.targets[idx]
        return sample, target


def generate_chexpert(dir_path, num_clients, num_classes, id):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train" + id + "/"
    val_path = dir_path + "val" + id + "/"
    test_path = dir_path + "test" + id + "/"

    if check(config_path, train_path, val_path, test_path, num_clients, num_classes):
        return


    dataset_dir = './' # PUT THE DATASET DIRECTORY PATH HERE.
    trainset_alt = CheXpertDataset(csv_file = dataset_dir + 'CheXpert-v1.0-small/train.csv', root_dir=dataset_dir)
    trainset = CustomDataset(trainset_alt)
    
    print("Trainset Loaded...")
    testset = CheXpertDataset(csv_file = dataset_dir + 'CheXpert-v1.0-small/valid.csv', root_dir=dataset_dir)
    testset = CustomDataset(testset)
    print("Valset Loaded...")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    trainset_alt.print_unique_labels()
    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data)
    dataset_image.extend(testset.data)
    dataset_label.extend(trainset.targets)
    dataset_label.extend(testset.targets)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    print("Partitioning...")
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, "chexpert", 0.6)
    train_data, val_data, test_data = split_data(X, y)
    save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, num_clients,
              num_classes, statistic)


if __name__ == "__main__":
    for i in range(runs):
        set_seed(i)
        generate_chexpert(dir_path, num_clients, num_classes, str(i))
