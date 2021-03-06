from collections import OrderedDict
from itertools import chain

import torch
from IPython.core.debugger import set_trace
from torch import nn as nn


class Conv2DPatches(nn.Module):
    def __init__(self, jigsaw_classes=100, n_classes=16, dropout=True, image_size=350):
        super(Conv2DPatches, self).__init__()
        print("Using 2D Convolutional Patched model")
        if image_size == 350:
            self.features = nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(1, 16, kernel_size=11, stride=3)),
                ("relu1", nn.ReLU(inplace=True)),
                ("pool1", nn.MaxPool2d(kernel_size=5, stride=3)),
                ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
                ("conv2", nn.Conv2d(16, 32, kernel_size=5, stride=2)),
                ("relu2", nn.ReLU(inplace=True)),
                ("pool2", nn.MaxPool2d(kernel_size=5, stride=2)),
                ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
                ("conv3", nn.Conv2d(32, 16, kernel_size=5)),
                ("relu3", nn.ReLU(inplace=True)),
            ]))
            self.fc_size = 144
            # self.y_classifier = nn.Sequential(OrderedDict([
            #     ("fc6", nn.Linear(256 * 6 * 6, self.fc_size)),
            #     ("relu6", nn.ReLU(inplace=True)),
            #     ("drop6", nn.Dropout()),
            #     ("fc7", nn.Linear(4096, 4096)),
            #     ("relu7", nn.ReLU(inplace=True)),
            #     ("drop7", nn.Dropout())
            # ]))

            self.jigsaw_classifier = nn.Sequential(
                nn.Linear(self.fc_size, jigsaw_classes),
            )

            self.point_detectors = []
            for i in range(n_classes):
                self.point_detectors.append(nn.Sequential(
                    nn.Linear(self.fc_size, 2),
                ))
            self.point_detectors = nn.ModuleList(self.point_detectors)
        elif image_size == 100:
            self.features = nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(1, 16, kernel_size=5, stride=2)),
                ("relu1", nn.ReLU(inplace=True)),
                ("pool1", nn.MaxPool2d(kernel_size=3, stride=2)),
                # ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
                ("conv2", nn.Conv2d(16, 32, kernel_size=5, stride=2)),
                ("relu2", nn.ReLU(inplace=True)),
                ("pool2", nn.MaxPool2d(kernel_size=5, stride=1)),
                # ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
                ("conv3", nn.Conv2d(32, 16, kernel_size=5)),
                ("relu3", nn.ReLU(inplace=True)),
            ]))
            self.fc_size = 64
            jigsaw_classes = 25
            self.jigsaw_classifier = nn.Sequential(
                nn.Linear(self.fc_size, jigsaw_classes),
            )

            self.point_detectors = []
            for i in range(n_classes):
                self.point_detectors.append(nn.Sequential(
                    nn.Linear(self.fc_size, 2),
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
