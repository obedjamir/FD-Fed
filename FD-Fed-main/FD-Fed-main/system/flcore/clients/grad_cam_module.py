
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from torchvision import transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.forward_outputs = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def forward_hook_function(module, input, output):
            self.forward_outputs = output

        def backward_hook_function(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook_function)
        self.target_layer.register_backward_hook(backward_hook_function)

    def __call__(self, x, y=None):
        output = self.model(x)
        self.model.zero_grad()
        if y is None:
            _, index = output.max(dim=1)
        else:
            index = y
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, index.unsqueeze(1), 1.0)
        output.backward(gradient=one_hot)
        gradients = self.gradients

        # Compute Grad-CAM
        activations = self.forward_outputs
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        grad_cam_map = torch.relu(torch.sum(weights * activations, dim=1))
        min_vals = grad_cam_map.view(grad_cam_map.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        max_vals = grad_cam_map.view(grad_cam_map.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        grad_cam_map = (grad_cam_map - min_vals) / (max_vals - min_vals + 1e-8)

        predictions = torch.argmax(output, dim=1)
        return grad_cam_map, predictions

class GradCAMModule:
    def __init__(self, model, target_layer, device, args, trainloader, client_id, load_images_func):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.args = args
        self.trainloader = trainloader
        self.id = client_id
        self.load_images = load_images_func

        os.makedirs(f'FDFed/{self.id}', exist_ok=True)

        self.transform_alt = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])


    def get_gradcam_results(self, layer_index):
        grad_cam = GradCAM(self.model, self.target_layer)
        clusters = []
        active_proportions = []
        start_time = time.time()

        xrayData = ["chexpert", "nihchestxray", "mimic"]
        for i, (image_data, y) in enumerate(self.trainloader):
            if self.args.dataset in xrayData:
                x = torch.stack(self.load_images(image_data)).to(self.device)
            elif isinstance(image_data, list):
                x[0] = image_data[0].to(self.device)
            else:
                x = image_data.to(self.device)
            y = y.to(self.device)

            cams, preds = grad_cam(x, y)
            results = self._dbscan_cams(cams)

            for num_clusters, _, active_proportion in results:
                clusters.append(num_clusters)
                active_proportions.append(active_proportion)

        end_time = time.time()
        print(f"Total time taken for Grad-CAM and clustering: {end_time - start_time:.2f} seconds")

        print("Clusters Accumulated...")
        mean_cluster_count = np.mean(np.array(clusters))
        mean_active_proportion = np.mean(np.array(active_proportions))
        return mean_cluster_count, mean_active_proportion

    def _dbscan_cams(self, cams, threshold=0.3, eps=1, min_samples=1):
        batch_size = cams.size(0)
        cams_cpu = cams.cpu().detach()
        cam_binaries = (cams > threshold).float().detach()
        results = []

        for idx in range(batch_size):
            cam_binary = cam_binaries[idx]
            coords = torch.nonzero(cam_binary, as_tuple=False).float()

            active_proportion = coords.size(0) / cam_binary.numel()

            if coords.size(0) == 0:
                num_clusters = 0
                labels_cluster = []
            else:
                labels_cluster = self.dbscan(coords.to(self.device), eps=eps, min_samples=min_samples)
                labels_np = labels_cluster.cpu().numpy()
                unique_labels = set(labels_np)
                num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            results.append((num_clusters, labels_cluster, active_proportion))

        return results

    def dbscan(self, X, eps=1.0, min_samples=1):
        X_np = X.cpu().detach().numpy()
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_np)
        labels = torch.from_numpy(labels).to(X.device)
        return labels


