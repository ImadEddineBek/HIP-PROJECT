from collections import OrderedDict
from itertools import chain

import torch
from IPython.core.debugger import set_trace
from torch import nn as nn
import torch.nn.functional as F


class LandmarkAdjuster(nn.Module):
    def __init__(self, image_size=20, n_classes=8):
        super(LandmarkAdjuster, self).__init__()
        print("Using 2D Convolutional Patched model")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 16, kernel_size=3, stride=1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(16, 32, kernel_size=5, stride=1)),
            ("relu2", nn.ReLU(inplace=True)),
            ("conv3", nn.Conv2d(32, 16, kernel_size=5)),
            ("relu3", nn.ReLU(inplace=True)),
        ]))
        self.fc_size = 144

        self.point_detectors = []
        for i in range(n_classes):
            self.point_detectors.append(nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.fc_size, self.fc_size // 4),
                nn.ReLU(),
                nn.Linear(self.fc_size // 4, 5),
            ))
        self.point_detectors = nn.ModuleList(self.point_detectors)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

    # def get_params(self, base_lr):
    #     return [{"params": self.features.parameters(), "lr": 0.1},
    #             {"params": chain((net.parameters() for net in self.point_detectors)), "lr": base_lr}]

    @staticmethod
    def is_patch_based():
        return True

    def forward(self, x, jig=False):
        # set_trace()
        B, L, H, W = x.size()

        # print(B, L, H, W)
        x_encoded = self.features(x.view(B * L, 1, H, W).float())
        # x = self.classifier(x.view(B, -1))
        # print("here", x_encoded.size())
        jig_out = self.jigsaw_classifier(x_encoded.view(B * L, -1))
        detected_points = torch.zeros([B, L, 2], dtype=torch.float32)
        if torch.cuda.is_available():
            detected_points = detected_points.cuda()
        if not jig:
            for i in range(L):
                detected_points[:, i] = self.point_detectors[i](x_encoded.view(B, L, -1)[:, i])

        return jig_out, detected_points
