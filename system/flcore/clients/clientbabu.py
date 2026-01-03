import copy
import torch
import torch.nn as nn
import numpy as np
from flcore.clients.clientbase import Client

class clientBABU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

        self.fine_tuning_steps = args.fine_tuning_steps
        self.alpha = args.alpha

    def train_one_iter(self, x, y, optimizer):
        optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()

    def get_training_optimizer(self, **kwargs):
        return torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate, momentum=0.9)

    def get_fine_tuning_optimizer(self, **kwargs):
        return torch.optim.SGD(self.model.parameters(), lr=self.alpha, momentum=0.9)

    def prepare_training(self, **kwargs):
        pass

    def prepare_fine_tuning(self, **kwargs):
        pass

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        optimizer = self.get_training_optimizer()
        self.prepare_training()

        for param in self.model.predictor.parameters():
            param.requires_grad = False

        max_local_steps = self.local_steps

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
                self.train_one_iter(x, y, optimizer)

    def fine_tune(self):
        trainloader = self.load_train_data()
        self.model.train()
        self.prepare_fine_tuning()
        optimizer = self.get_fine_tuning_optimizer()

        for param in self.model.predictor.parameters():
            param.requires_grad = True

        xrayData = ["chexpert", "nihchestxray", "mimic"]
        for step in range(self.fine_tuning_steps):
            for i, (x, y) in enumerate(trainloader):
                if self.args.dataset in xrayData:
                    x = torch.stack(self.load_images(x)).to(self.device)
                elif type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.train_one_iter(x, y, optimizer)
