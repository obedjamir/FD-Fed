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
import glob
from utils.dataset_utils import check, separate_data, split_data, save_file
import shutil

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def duplicate_filtered_images(dataset, output_dir="filtered"):
    filtered_dir = os.path.join("./", output_dir)
    os.makedirs(filtered_dir, exist_ok=True)

    for img_path in dataset.data_frame["Image_Path"]:
        src_path = os.path.join("", img_path)
        dest_path = os.path.join(filtered_dir, img_path[7:])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
        else:
            print(f"Warning: {src_path} does not exist.")

    print(f"Filtered images have been copied to {filtered_dir}")

num_clients = 3
runs = 5
num_classes = 13
dir_path = "mimic/"

class MIMICDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.filter_data()
        self.unique_labels = []

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx]["Image_Path"]
        label = self.data_frame.iloc[idx]["Label"]
        if label not in self.unique_labels:
            self.unique_labels.append(label)

        label = torch.tensor(self.unique_labels.index(label))
        return img_name, label

    def filter_data(self):
        # Determine the keys for subject and study IDs
        subject_key = "subject_id"
        study_key = "study_id"
        
        # List of pathology columns (assuming first two columns are IDs)
        label_columns = self.data_frame.columns[2:]
        
        # Remove "No Finding" cases
        self.data_frame = self.data_frame[self.data_frame["No Finding"] != 1]
        
        # Replace 0 (negative) and -1 (uncertain) with NaN
        self.data_frame[label_columns] = self.data_frame[label_columns].replace([pd.NA, -1], 0)
        
        # Count how many positive labels exist per row
        self.data_frame["LabelCount"] = self.data_frame[label_columns].notna().sum(axis=1)
        
        # Keep only rows with exactly one positive label (single-class samples)
        self.data_frame = self.data_frame[self.data_frame["LabelCount"] == 1]
        
        # Extract the label name for each row using dropna() to avoid empty rows
        self.data_frame["Label"] = self.data_frame[label_columns].apply(
            lambda row: row.dropna().idxmax(), axis=1
        )
        
        # Dynamically map images based on subject & study IDs (JPG format)
        image_paths = []
        labels = []
        for _, row in self.data_frame.iterrows():
            subject_id = f"p{str(row[subject_key])[:2]}/p{row[subject_key]}"
            study_id = f"s{row[study_key]}"
            
            # Construct directory path (adjust this if your directory structure differs)
            study_path = os.path.join(self.root_dir, "files", subject_id, study_id)
            
            # Find JPG images in this folder
            jpg_files = glob.glob(os.path.join(study_path, "*.jpg"))
            
            for jpg_path in jpg_files:
                image_paths.append(jpg_path)
                labels.append(row["Label"])
        
        # Update the dataframe with image paths and labels
        self.data_frame = pd.DataFrame({"Image_Path": image_paths, "Label": labels})
        print(self.data_frame)

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

def generate_mimic(dir_path, num_clients, num_classes, id):
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
    csv_path = os.path.join(dataset_dir, 'mimic-cxr-2.0.0-negbio.csv')

    dataset = MIMICDataset(csv_file=csv_path, root_dir=dataset_dir)
    print("Datset Loaded...")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset_alt, testset = random_split(dataset, [train_size, test_size])
    del dataset
    trainset = CustomDataset(trainset_alt)
    testset = CustomDataset(testset)


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
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, "mimic", 0.6)
    train_data, val_data, test_data = split_data(X, y)
    save_file(config_path, train_path, val_path, test_path, train_data, val_data, test_data, num_clients,
                  num_classes, statistic)

if __name__ == "__main__":
    for i in range(runs):
        set_seed(i)
        generate_mimic(dir_path, num_clients, num_classes, str(i))
