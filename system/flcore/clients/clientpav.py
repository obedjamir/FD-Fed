import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.clients.clientbase import Client

class clientPav(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.distance = 1.0
        self.global_model_prev = None

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        max_local_steps = self.local_steps
        xrayData = ["chexpert", "nihchestxray", "mimic"]

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if self.args.dataset in xrayData:
                    x = torch.stack(self.load_images(x)).to(self.device)
                elif isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

        if self.global_model_prev is not None:
            cos_dist_score = self.compute_cosine_distance(self.model.base, self.global_model_prev)
            print(cos_dist_score)
            self.distance = cos_dist_score

    def compute_cosine_distance(self, local_model, reference_model):
        loader = self.load_train_data()
        data_iter = iter(loader)
        x, _ = next(data_iter)
        xrayData = ["chexpert", "nihchestxray", "mimic"]

        if self.args.dataset in xrayData:
            x_in = torch.stack(self.load_images(x)).to(self.device)
        elif isinstance(x, list):
            x_in = x[0].to(self.device)
        else:
            x_in = x.to(self.device)

        local_model.eval()
        reference_model.eval()

        with torch.no_grad():
            logits_local = local_model(x_in)
            logits_ref = reference_model(x_in)

        logits_local = logits_local.view(logits_local.size(0), -1)
        logits_ref = logits_ref.view(logits_ref.size(0), -1)

        # Check for zero vectors
        if (logits_local.norm(dim=1) == 0).any() or (logits_ref.norm(dim=1) == 0).any():
            raise ValueError("Zero vectors detected in logits, causing NaN in cosine similarity.")

        # Check for NaN values
        if torch.isnan(logits_local).any() or torch.isnan(logits_ref).any():
            raise ValueError("NaN detected in logits outputs.")

        # Ensure shapes match
        assert logits_local.shape == logits_ref.shape, f"Logits shapes mismatch: {logits_local.shape} vs {logits_ref.shape}"

        # Compute cosine similarity
        cos_sims = F.cosine_similarity(logits_local, logits_ref, dim=1)
        cos_dist = 1.0 - cos_sims
        distance_score = cos_dist.mean().item()

        local_model.train()
        reference_model.train()

        return distance_score
