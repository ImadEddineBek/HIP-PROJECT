from collections import OrderedDict
from itertools import chain

import torch
from IPython.core.debugger import set_trace
from torch import nn as nn


class ConvClassifier(nn.Module):
    def __init__(self, jigsaw_classes=100, n_classes=17, dropout=True):
        super(ConvClassifier, self).__init__()
        print("Using 2D Convolutional Classifier model")
        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(1, 16, kernel_size=3)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=2)),
            ("conv2", nn.Conv2d(16, 32, kernel_size=3)),
            ("relu2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=2)),
            ("conv3", nn.Conv2d(32, 64, kernel_size=3)),
            ("relu3", nn.ReLU(inplace=True)),
        ]))
        self.fc_size = 64
        self.point_detectors = nn.Linear(self.fc_size, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, .1)
                nn.init.constant_(m.bias, 0.)

    @staticmethod
    def is_patch_based():
        return True

    def forward(self, x):
        # set_trace()
        B, L, H, W = x.size()
        # print(B, L, H, W)
        x_encoded = self.features(x.view(B * L, 1, H, W).float())
        # x = self.classifier(x.view(B, -1))
        # print('here', x_encoded.size())
        detected_points = self.point_detectors(x_encoded.view(B * L, -1)).view(B, L, -1)
        # print(detected_points.size())
        return detected_points
