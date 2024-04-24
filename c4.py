import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# resnet 34 with bilinear upsampling
class ColorizationNet4(nn.Module):
    def __init__(self, input_channels=1):
        super(ColorizationNet4, self).__init__()

        resnet = models.resnet34(num_classes=1000)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.feature_extractor_resnet = nn.Sequential(*list(resnet.children())[:7])

        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 2, 3, padding=1)

    def forward(self, input):
        features_resnet = self.feature_extractor_resnet(input)

        x = F.relu(self.bn1(self.conv1(features_resnet)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2,mode='bilinear', align_corners=False)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x