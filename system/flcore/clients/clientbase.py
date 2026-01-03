import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from ..trainmodel.models import *
from PIL import Image, ImageFile, UnidentifiedImageError
import struct
import torchvision.transforms as transforms
import torch
from flcore.clients.grad_cam_module import GradCAMModule
class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, local_labels, dataset_id, **kwargs):
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.dataset_id = dataset_id
        self.local_labels = local_labels
        self.num_classes = len(local_labels)

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.model = LocalModel(copy.deepcopy(args.model), self.num_classes).to(args.device)
        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
            
        self.sample_rate = self.batch_size / self.train_samples

    def load_train_data(self, batch_size=None, shuffle=True):
        if batch_size is None:
            batch_size = self.batch_size
        
        # Load and cache train data if not already loaded
        if self.train_data is None:
            self.train_data, _ = read_client_data(self.dataset, self.id, self.dataset_id, data_split='train')

        batch_size = min(batch_size, len(self.train_data))
        return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=shuffle)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        # Load and cache test data if not already loaded
        if self.test_data is None:
            self.test_data, _ = read_client_data(self.dataset, self.id, self.dataset_id, data_split='test')

        batch_size = min(batch_size, len(self.test_data))
        return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=False)

    def load_val_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        # Load and cache val data if not already loaded
        if self.val_data is None:
            self.val_data, _ = read_client_data(self.dataset, self.id, self.dataset_id, data_split='val')

        batch_size = min(batch_size, len(self.val_data))
        return DataLoader(self.val_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model, init=False):
        for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()

    def get_eval_model(self, temp_model=None):
        model = self.model_per if hasattr(self, "model_per") else self.model
        return model
    
    def test_metrics(self, temp_model=None, val=True):
        if val:
            testloaderfull = self.load_val_data()
        else:
            testloaderfull = self.load_test_data()
        model = self.get_eval_model(temp_model)
        model.eval()
        
        test_correct = 0
        test_num = 0
        test_loss = 0.0
        y_prob = []
        y_true = []
        
        xrayData = ["chexpert", "nihchestxray", "mimic", "nihchestxrayalt"]
        with torch.no_grad():
            for x, y in testloaderfull:
                if self.args.dataset in xrayData:
                    x = torch.stack(self.load_images(x)).to(self.device)
                elif type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                test_loss += (self.criterion(output, y.long()) * y.shape[0]).item()

                test_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        try:
            test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            test_loss /= test_num
        except ValueError:
            test_auc, test_loss = 0.0, 0.0
        test_acc = test_correct / test_num
        return test_acc, test_auc, test_loss, test_num
    
    def get_layers(self, model):
        conv_layers = []
        layer_index = []
        for i, layer in enumerate(model.layer_list):
            print(i)
            for component in layer.modules():
                if self.is_depthwise_conv(component):
                    conv_layers.append(component)
                    layer_index.append(i)
        return conv_layers, layer_index

    def is_depthwise_conv(self, layer):
        return isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels

    def load_image(self, image_name, transform):
        image = Image.open(image_name)
        if image.mode != 'L':
            image = image.convert('L')
        return transform(image)

    def load_images(self, image_names
        , transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        , max_workers=10):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            images = list(executor.map(self.load_image, image_names, [transform] * len(image_names)))
        return images
