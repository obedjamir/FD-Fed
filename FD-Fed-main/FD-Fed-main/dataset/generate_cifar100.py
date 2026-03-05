import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file

num_clients = 5
num_classes = 100
dir_path = "cifar100/"

def generate_cifar100(dir_path, num_clients, num_classes, id):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train" + id + "/"
    val_path = dir_path + "val" + id + "/"
    test_path = dir_path + "test" + id + "/"

    if check(config_path, train_path, val_path, test_path, num_clients, num_classes):
        return
        
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, "cifar100", 0.6)
    train_data, val_data, test_data = split_data(X, y)
    save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, num_clients,
              num_classes, statistic)

if __name__ == "__main__":
    for i in range(5):
        random.seed(i)
        np.random.seed(i)
        generate_cifar100(dir_path, num_clients, num_classes, str(i))
