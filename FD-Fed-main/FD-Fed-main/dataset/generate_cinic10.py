import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os
from utils.dataset_utils import check, separate_data, split_data, save_file
from torch.utils.data import Dataset
from PIL import Image

num_clients = 5
num_classes = 10
dir_path = "cinic10/"


class CINIC10(Dataset):
    def __init__(self, root, data_type="train", transform=None):
        self.root = root
        self.data_type = data_type
        self.transform = transform
        self.data = []
        self.targets = []
        
        if self.data_type=="train":
            print("Train Data...")
            data_dir = os.path.join(root, 'train')
        elif self.data_type=="valid":
            print("Valid Data...")
            data_dir = os.path.join(root, 'valid')
        else:
            print("Test Data...")
            data_dir = os.path.join(root, 'test')

        class_folders = os.listdir(data_dir)
        class_folders.sort()
        for i, folder in enumerate(class_folders):
            class_dir = os.path.join(data_dir, folder)
            images = os.listdir(class_dir)
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.targets.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

def generate_cinic10(dir_path, num_clients, num_classes, id):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train" + id + "/"
    val_path = dir_path + "val" + id + "/"
    test_path = dir_path + "test" + id + "/"

    if check(config_path, train_path, val_path, test_path, num_clients, num_classes):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CINIC10(root=dir_path, data_type="train", transform=transform)
    valset = CINIC10(root=dir_path, data_type="valid", transform=transform)
    testset = CINIC10(root=dir_path, data_type="test", transform=transform)

    # Loaders for train, validation, and test sets
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=len(valset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    # Assign data and targets for train, validation, and test sets
    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, val_data in enumerate(valloader, 0):
        valset.data, valset.targets = val_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    # Add trainset, valset, and testset data
    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(valset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())

    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(valset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())

    # Convert to numpy arrays
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, "cinic10")
    train_data, val_data, test_data = split_data(X, y)
    save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, num_clients,
              num_classes, statistic)

if __name__ == "__main__":
    for i in range(5):
        random.seed(i)
        np.random.seed(i)
        generate_cinic10(dir_path, num_clients, num_classes, str(i))
