import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LocalModel(nn.Module):
    def __init__(self, base, num_classes):
        super(LocalModel, self).__init__()
        self.base = base
        self.predictor = nn.Linear(base.out_features, num_classes)
        self.layer_list = self.base.layer_list
        self.layer_count = self.base.layer_count
        print([module for module in self.predictor.modules()])

    def forward(self, x):
        out = self.base(x)
        out = self.predictor(out)
        return out

    def freeze_layers(self, start, end):
        if end is None:
            end = self.layer_count - 1
        layers_frozen = 0
        for i, layer in enumerate(self.layer_list):
            if i>= start and i <= end and start < end:
                for param in layer.parameters():
                    param.requires_grad = False
                layers_frozen += 1

        print(f"Layers Frozen: {layers_frozen}")

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        print(f"Base has been Frozen.")

    def freeze_predictor(self):
        for param in self.predictor.parameters():
            param.requires_grad = False
        print("Predictor has been frozen.")

    def unfreeze_layers(self):
        for param in self.base.parameters():
            param.requires_grad = True
        for param in self.predictor.parameters():
            param.requires_grad = True
        print("All Layers have been unfrozen.")

class DownsampleBlock(nn.Module):
    def __init__(self):
        super(DownsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class MobileNetV3Small(nn.Module):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()

        mobilenet_v3_small = models.mobilenet_v3_small(pretrained=False)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.features = mobilenet_v3_small.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_features = 576

        self.layer_list = self._create_block_list()
        self.layer_count = len(self.layer_list)
        print(f"MobileNetV3SmallGradCAM Block Count: {self.layer_count}")

    def forward(self, x):
        x = self.upsample(x)
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def _create_block_list(self):
        block_list = [self.upsample]
        for block in self.features:
            block_list.append(block)
        return nn.ModuleList(block_list)


class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        effnet = models.efficientnet_b0(pretrained=False)

        self.downsample = DownsampleBlock()
        self.features = nn.Sequential(
            effnet.features,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.out_features = effnet.classifier[1].in_features

        self.layer_list = self._create_block_list()
        self.layer_count = len(self.layer_list)
        print(self.layer_count)

    def forward(self, x):
        x = self.downsample(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _create_block_list(self):
        block_list = []
        block_list.append(self.downsample)
        for block in self.features[0]:
            if block.__class__.__name__ == 'Sequential':
                for sub_block in block:
                    block_list.append(sub_block)
            else:
                block_list.append(block)
        return nn.ModuleList(block_list)
