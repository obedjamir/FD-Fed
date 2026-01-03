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
dir_path = "nihchestxray/"

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dirs):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dirs = root_dirs
        self.filter_data()
        self.unique_labels = []

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.find_image(self.data_frame.iloc[idx, 0])
        label = self.data_frame.iloc[idx, 1]
        if label not in self.unique_labels:
            self.unique_labels.append(label)

        label = torch.tensor(self.unique_labels.index(label))
        return img_name, label

    def filter_data(self):
        self.data_frame = self.data_frame[~self.data_frame['Finding Labels'].str.contains('No Finding|\|')]

    def find_image(self, img_name):
        for root_dir in self.root_dirs:
            img_path = os.path.join(root_dir, img_name)
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"Image {img_name} not found in any of the specified directories.")
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

def generate_nihchestxray(dir_path, num_clients, num_classes, id):
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
    root_dirs = [f'{dataset_dir}images_{str(i).zfill(3)}/images/' for i in range(1, 13)]
    dataset = NIHChestXrayDataset(csv_file = dataset_dir + 'Data_Entry_2017.csv', root_dirs=root_dirs)
    print("Datset Loaded...")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset_alt, testset = random_split(dataset, [train_size, test_size])
    del dataset
    trainset = CustomDataset(trainset_alt)
    testset = CustomDataset(testset)
    print("Custom Datset Loaded...")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data
    
    trainset_alt.dataset.print_unique_labels()
    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data)
    dataset_image.extend(testset.data)
    dataset_label.extend(trainset.targets)
    dataset_label.extend(testset.targets)
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    print("Partitioning...")
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, "nihchestxray", 0.6)
    train_data, val_data, test_data = split_data(X, y)
    save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, num_clients,
              num_classes, statistic)


if __name__ == "__main__":
    for i in range(runs):
        set_seed(i)
        generate_nihchestxray(dir_path, num_clients, num_classes, str(i))
