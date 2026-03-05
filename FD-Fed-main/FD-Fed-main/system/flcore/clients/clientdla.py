import torch
import torch.nn as nn
import numpy as np
import copy
import sys
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from open_clip import create_model_from_pretrained, get_tokenizer
import segmentation_models_pytorch as smp
import cv2
# ==================================================
# Block-wise Running Statistics (Mean + Intra + Inter)
# ==================================================
class BlockRunningStats:
    def __init__(self, num_blocks, num_classes):
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        self.means = [
            {c: None for c in range(num_classes)}
            for _ in range(num_blocks)
        ]
        self.M2 = [
            {c: 0.0 for c in range(num_classes)}
            for _ in range(num_blocks)
        ]
        self.counts = [
            {c: 0 for c in range(num_classes)}
            for _ in range(num_blocks)
        ]

    @torch.no_grad()
    def update(self, block_features, labels):
        for bidx, feats in block_features.items():
            for c in labels.unique().tolist():
                mask = labels == c
                if mask.sum() == 0:
                    continue

                class_feats = feats[mask]              # (Nc, D)
                batch_mean = class_feats.mean(dim=0)

                if self.means[bidx][c] is None:
                    self.means[bidx][c] = batch_mean.clone()
                    self.counts[bidx][c] = class_feats.size(0)
                    self.M2[bidx][c] = (
                        (class_feats - batch_mean) ** 2
                    ).sum().item()
                else:
                    n_old = self.counts[bidx][c]
                    n_new = class_feats.size(0)
                    n_total = n_old + n_new

                    delta = batch_mean - self.means[bidx][c]
                    self.means[bidx][c] += delta * (n_new / n_total)

                    self.M2[bidx][c] += (
                        ((class_feats - batch_mean) ** 2).sum().item()
                        + delta.pow(2).sum().item() * n_old * n_new / n_total
                    )

                    self.counts[bidx][c] = n_total

    @torch.no_grad()
    def compute_inter_class_variance(self):
        inter_var = []

        for b in range(self.num_blocks):
            centroids = [
                self.means[b][c]
                for c in range(self.num_classes)
                if self.means[b][c] is not None
            ]

            if len(centroids) < 2:
                inter_var.append(0.0)
                continue

            centroids = torch.stack(centroids)
            global_mean = centroids.mean(dim=0)
            var = ((centroids - global_mean) ** 2).sum(dim=1).mean()
            inter_var.append(var.item())

        return inter_var

    @torch.no_grad()
    def compute_intra_class_variance(self):
        intra_var = []

        for b in range(self.num_blocks):
            vars_c = []
            for c in range(self.num_classes):
                if self.counts[b][c] > 1:
                    vars_c.append(self.M2[b][c] / self.counts[b][c])

            intra_var.append(sum(vars_c) / len(vars_c) if len(vars_c) > 0 else 0.0)

        return intra_var

# ==================================================
# Client with Block-wise Training Logic
# ==================================================
class clientDLAFed(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        self.block_inter_class_variance = None
        self.block_intra_class_variance = None

        # --------------------------------------------------
        # Build BioBERT class prototypes (ONCE per round)
        # --------------------------------------------------
        trainloader = self.load_train_data()
        self.class_prototypes = self.build_class_prototypes()

        self.seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        ).to(args.device)
        self.seg_model.load_state_dict(torch.load("unet_lung_segmentation.pth"))
        self.seg_model.eval()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        '''
        from torchvision.utils import save_image

        save_root = "saved_images"
        idx_to_class = {v: k for k, v in self.label_map.items()}
        for batch_idx, (x, y) in enumerate(trainloader):

            x = torch.stack(self.load_images(x)).to(self.device)

            for i in range(x.shape[0]):

                label_idx = y[i].item()
                class_name = idx_to_class[label_idx]

                img = x[i].cpu()
                #img = img * 0.5 + 0.5   # unnormalize
                #img = img.clamp(0,1)

                class_dir = os.path.join(save_root, class_name)
                os.makedirs(class_dir, exist_ok=True)

                img_path = os.path.join(class_dir, f"{self.id}_{batch_idx}_i{i}.png")

                save_image(x[i].cpu(), img_path)
        '''
    '''
    def build_class_prototypes(self):

        # ---- Hugging Face BiomedCLIP ----
        hf_model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        model, _ = create_model_from_pretrained(hf_model_id)
        tokenizer = get_tokenizer(hf_model_id)

        model = model.to(self.device)
        model.eval()

        # ---- class names ----
        idx_to_label = {v: k for k, v in self.label_map.items()}

        class_texts = [idx_to_label[i] for i in range(len(idx_to_label))]

        # ---- tokenize (IMPORTANT: context_length=256 for PubMedBERT) ----
        tokens = tokenizer(
            class_texts,
            context_length=256
        ).to(self.device)

        # ---- encode text ----
        with torch.no_grad():
            text_features = model.encode_text(tokens)  # [C, 512]

        # ---- normalize prototypes ----
        prototypes = F.normalize(text_features, dim=1)

        return prototypes
    '''

    def enhance_cxr(self, x, batch_idx, threshold=0.5):
        device = x.device

        # -------- ensure shape --------
        if x.dim() == 4:          # [B,1,H,W]
            img = x.squeeze(1)    # [B,H,W]
            seg_input = x
        else:                     # [B,H,W]
            img = x
            seg_input = x.unsqueeze(1)

        # -------- segmentation --------
        with torch.no_grad():
            mask = self.seg_model(seg_input)
            mask = torch.sigmoid(mask)
            mask = (mask > threshold).float()

        mask = mask.squeeze(1)    # [B,H,W]

        # -------- CLAHE --------
        clahe_imgs = []

        for i in range(img.shape[0]):

            img_np = img[i].detach().cpu().numpy()
            img_uint8 = (img_np * 255.0).clip(0,255).astype("uint8")

            clahe_img = self.clahe.apply(img_uint8)

            clahe_tensor = torch.from_numpy(clahe_img).float() / 255.0
            clahe_imgs.append(clahe_tensor)

        clahe_imgs = torch.stack(clahe_imgs).to(device)

        # -------- save first batch --------
        if batch_idx == 0:
            os.makedirs("debug_cxr", exist_ok=True)

            for i in range(min(5, img.shape[0])):

                img_save = (img[i].cpu().numpy() * 255).astype("uint8")
                mask_save = (mask[i].cpu().numpy() * 255).astype("uint8")
                clahe_save = (clahe_imgs[i].cpu().numpy() * 255).astype("uint8")

                cv2.imwrite(f"debug_cxr/img_{i}.png", img_save)
                cv2.imwrite(f"debug_cxr/mask_{i}.png", mask_save)
                cv2.imwrite(f"debug_cxr/clahe_{i}.png", clahe_save)

        # -------- stack channels --------
        x_out = torch.stack([img, clahe_imgs, mask], dim=1)

        return x_out

    def build_class_prototypes(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1"
        )
        text_model = AutoModel.from_pretrained(
            "dmis-lab/biobert-base-cased-v1.1"
        ).to(self.device)
        text_model.eval()

        idx_to_label = {v: k for k, v in self.label_map.items()}
        class_texts = [idx_to_label[i] for i in range(len(idx_to_label))]

        inputs = tokenizer(
            class_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = text_model(**inputs)
            prototypes = outputs.last_hidden_state[:, 0, :]  # [C, d_text]

        return F.normalize(prototypes, dim=1)
    
    # ==================================================
    # TRAIN (FINAL, WITH SE OVERLAY SAVING)
    # ==================================================
    def train(self, adapt=False):
        trainloader = self.load_train_data()
        self.model.train()
        self.model.collect_se = False

        stats = BlockRunningStats(
            num_blocks=self.model.num_blocks,
            num_classes=self.num_classes
        )

        # ---------------- Training mode ----------------
        if adapt:
            self.model.unfreeze_all_blocks()
            self.model.freeze_base_blocks()
            max_local_steps = self.args.plocal_steps
        else:
            self.model.unfreeze_all_blocks()
            max_local_steps = self.local_steps

        xray_datasets = ["chexpert", "nihchestxray", "mimic"]

        # ==================================================
        # TRAINING LOOP
        # ==================================================
        for epoch in range(max_local_steps):
            for batch_idx, (x, y) in enumerate(trainloader):

                # ---------------- Data handling ----------------
                if self.args.dataset in xray_datasets:
                    x = torch.stack(self.load_images(x)).to(self.device)
                    x = self.enhance_cxr(x, batch_idx)
                elif isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                y = y.to(self.device)

                # ---------------- Forward ----------------
                self.optimizer.zero_grad()

                # clear hook buffers
                self.model.se_attn = {}
                self.model.block_features = {}

                loss, logits = self.model(
                    x, y, self.class_prototypes, self.class_sample_count, self.num_classes, use_proto=True
                )

                # ---------------- Block stats ----------------
                stats.update(self.model.block_features, y)

                # ---------------- Backward ----------------
                loss.backward()
                self.optimizer.step()
                #break

        # ==================================================
        # FINAL BLOCK STATISTICS
        # ==================================================
        self.block_inter_class_variance = stats.compute_inter_class_variance()
        self.block_intra_class_variance = stats.compute_intra_class_variance()

        print("Inter-class variance:", self.block_inter_class_variance)
        print("Intra-class variance:", self.block_intra_class_variance)

'''
class clientDLAFed(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        # --------------------------------------------------
        # Block-based bookkeeping
        # --------------------------------------------------
        self.base_block_count = int(
            self.model.num_blocks * self.args.base_layers
        )

        self.prev_predictor = None

    def train(self, adapt=False):
        trainloader = self.load_train_data()
        print("Loaded Data!")

        self.model.train()

        # Save initial block states (for possible proximal regularization)
        initial_blocks = [
            copy.deepcopy(block) for block in self.model.block_list
        ]

        # --------------------------------------------------
        # Determine local training regime
        # --------------------------------------------------
        if adapt:
            self.model.unfreeze_all_blocks()
            self.model.freeze_base_blocks()
            max_local_steps = self.args.plocal_steps
        else:
            self.model.unfreeze_all_blocks()
            max_local_steps = self.local_steps

        xray_datasets = ["chexpert", "nihchestxray", "mimic"]

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        for step in range(max_local_steps):
            for batch_idx, (x, y) in enumerate(trainloader):

                # Device handling
                if self.args.dataset in xray_datasets:
                    x = torch.stack(self.load_images(x)).to(self.device)
                elif isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                y = y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)

                # --------------------------------------------------
                # (Optional) Block-wise proximal regularization
                # --------------------------------------------------
                # mu = getattr(self.args, 'mu', 0.01)
                # prox_term = 0.0
                # total_blocks = len(self.model.block_list)
                #
                # for i, (curr_block, init_block) in enumerate(
                #         zip(self.model.block_list, initial_blocks)):
                #
                #     weight = (total_blocks - i) / total_blocks
                #
                #     for p_curr, p_init in zip(
                #             curr_block.parameters(),
                #             init_block.parameters()):
                #
                #         if p_curr.requires_grad:
                #             p_init = p_init.detach().to(p_curr.device)
                #             prox_term += (
                #                 nn.functional.mse_loss(
                #                     p_curr, p_init, reduction='sum'
                #                 ) * weight
                #             )
                #
                # loss += (mu / 2) * prox_term

                loss.backward()
                self.optimizer.step()
'''
