import os
import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.grad_cam_module import GradCAMModule
from flcore.clients.clientbase import Client
from ..trainmodel.models import *

class clientFDFed(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.finetune_optimizer = torch.optim.SGD(self.model.parameters(), lr=args.alpha, momentum=0.9)
        self.general_layer_index = self.model.layer_count - 1
        self.prev_predictor = None
        self.cam_trainloader = None

    def train(self, adapt=False):
        self.model.train()
        trainloader = self.load_train_data()
        if adapt:
            self.model.unfreeze_layers()
            self.model.freeze_layers(0, self.general_layer_index)
            max_local_steps = self.args.plocal_steps
            optimizer = self.finetune_optimizer
        else:
            self.model.unfreeze_layers()
            max_local_steps = self.local_steps
            optimizer = self.base_optimizer

        xrayData = ["chexpert", "nihchestxray", "mimic"]
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if self.args.dataset in xrayData:
                    x = torch.stack(self.load_images(x)).to(self.device)
                elif type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                optimizer.step()

    def replace_layers(self, global_model):
        layers_replaced = 0
        for layer_index, global_layer in enumerate(global_model.layer_list):
            if layer_index < round(global_model.layer_count//2) or layer_index <= self.general_layer_index:
                self.model.layer_list[layer_index].load_state_dict(global_layer.state_dict())
                layers_replaced += 1

        print(f"Layer Replaced: {layers_replaced}")
        print(f"Layer Preserved: {global_model.layer_count - layers_replaced}")

    def set_parameters(self, model, init=False):
        cam_model = copy.deepcopy(self.model)
        layer, layer_index = self.get_layer(cam_model)
        if self.cam_trainloader == None:
            self.cam_trainloader = self.load_train_data(shuffle=False)

        if layer is not None:
            if not init:
                gradcam_module = GradCAMModule(cam_model, layer, self.device, self.args, self.cam_trainloader, self.id, load_images_func=self.load_images)
                mean_cluster_count, mean_active_proportion = gradcam_module.get_gradcam_results(layer_index)
                print(f"Mean Cluster Count: {mean_cluster_count}")
                print(f"Mean Active Proportion: {mean_active_proportion}")
                del gradcam_module

                if 0 < round(mean_cluster_count) < self.args.theta and 0 < round(mean_active_proportion, 4) < self.args.lambdaa:
                    self.general_layer_index = layer_index - 1

            print(f"New General Layer Index {self.general_layer_index}")
            self.replace_layers(model)

    def get_layer(self, model):
        for i, layer in enumerate(reversed(model.layer_list[:(self.general_layer_index + 1)])):
            index = self.general_layer_index - i
            for component in layer.modules():
                if self.is_depthwise_conv(component):
                    print(f"Fetched Depthwise Layer From Block {index}")
                    return component, index
        return None, None

    def is_depthwise_conv(self, layer):
        return isinstance(layer, nn.Conv2d) and layer.groups == layer.in_channels
