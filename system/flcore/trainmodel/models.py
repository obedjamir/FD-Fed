import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        effnet = models.efficientnet_b0(pretrained=False)

        self.features = nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.out_features = effnet.classifier[1].in_features

        self.block_list = self._create_block_list()
        self.block_count = len(self.block_list)

    def _create_block_list(self):
        block_list = []
        for block in self.features[0]:
            if block.__class__.__name__ == 'Sequential':
                for sub_block in block:
                    block_list.append(sub_block)
            else:
                block_list.append(block)
        return nn.ModuleList(block_list)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class LocalModel(nn.Module):
    def __init__(self, base_model, num_classes, out_feats=1280):
        super().__init__()

        self.base = base_model
        self.out_feats = out_feats

        self.feat_to_text = nn.Linear(1280, self.out_feats, bias=False)
        self.predictor_block = nn.Linear(1280, num_classes)

        # --------------------------------------------------
        # Block bookkeeping
        # --------------------------------------------------
        self.base.block_list.append(self.feat_to_text)
        self.base.block_count += 1
        self.block_list = self.base.block_list
        self.num_blocks = self.base.block_count

        self.block_features = {}     # for variance stats
        self.se_attn = {}            # 🔥 SE attention per block

        self._register_block_hooks()

    # --------------------------------------------------
    # Prototype loss
    # --------------------------------------------------
    def prototype_loss(
        self,
        base_feats,
        labels,
        prototypes,
        class_sample_count,
        num_classes,
        tau=0.07,
        beta=0.2
    ):

        # ---- project features to text space ----
        v = self.feat_to_text(base_feats)
        v = F.normalize(v, dim=1)

        # ---- normalize prototypes ----
        proto = F.normalize(prototypes, dim=1)

        # ---- contrastive prototype logits ----
        proto_logits = torch.matmul(v, proto.T) / tau   # [B, C]

        # ---- predictor classification logits ----
        cls_logits = self.predictor_block(base_feats)   # [B, C]

        # ---- class-balanced weights ----
        counts = torch.tensor(
            [class_sample_count.get(i, 1) for i in range(num_classes)],
            device=labels.device,
            dtype=torch.float32
        )

        counts = torch.clamp(counts, min=1.0)

        class_weights = 1.0 / counts
        class_weights = class_weights / class_weights.mean()

        # ---- classification loss ----
        cls_loss = F.cross_entropy(
            cls_logits,
            labels,
            weight=class_weights
        )

        # ---- prototype contrastive loss ----
        proto_loss = F.cross_entropy(
            proto_logits,
            labels,
            weight=class_weights
        )

        # ---- total loss ----
        total_loss = cls_loss + beta * proto_loss

        return total_loss, cls_logits

    # --------------------------------------------------
    # Block feature hooks (already used by you)
    # --------------------------------------------------
    def _register_block_hooks(self):
        def make_hook(bidx):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    feat = out
                    if feat.dim() > 2:
                        feat = feat.mean(dim=(2, 3))
                    self.block_features[bidx] = feat.detach()
            return hook

        for bidx, block in enumerate(self.block_list):
            block.register_forward_hook(make_hook(bidx))

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x, y=None, prototypes=None, class_sample_count=None, num_classes=None, use_proto=False):
        # --- ensure 3 channels ---
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        base_feats = self.base(x)

        if use_proto:
            p_loss, logits = self.prototype_loss(
                base_feats, y, prototypes, class_sample_count, num_classes
            )
            return p_loss, logits

        logits = self.predictor_block(base_feats)
        return logits

    # --------------------------------------------------
    # Block control
    # --------------------------------------------------
    def freeze_base_blocks(self):
        for block in self.block_list:
            for p in block.parameters():
                p.requires_grad = False

    def unfreeze_all_blocks(self):
        for block in self.block_list:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.predictor_block.parameters():
            p.requires_grad = True
            
    def freeze_blocks(self, start, end):
        if end is None:
            end = self.num_blocks - 1
        blocks_frozen = 0
        for i, block in enumerate(self.block_list):
            if i>= start and i <= end and start < end:
                for param in block.parameters():
                    param.requires_grad = False
                blocks_frozen += 1

        print(f"Blocks Frozen: {blocks_frozen}")

