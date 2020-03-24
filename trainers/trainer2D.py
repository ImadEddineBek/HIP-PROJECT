import torch
import sys

import imageio as imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import skimage
import torch.nn as nn
from termcolor import colored
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataloaders.dataloader2D import get_dataloader2D
from models import conv2d


class Trainer2D:
    def __init__(self, config):
        self.config = config
        self.model = conv2d.Conv2DPatches()
        print(self.model)
        self.train_loader, self.test_loader = get_dataloader2D(config)
        self.net_optimizer = optim.Adam(self.model.parameters(), config.lr, [0.5, 0.9999])
        if torch.cuda.is_available():
            self.model.cuda()
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_d = nn.MSELoss()
        self.epochs = config.epochs

    def train(self):
        for epoch in range(self.epochs):
            print("Starting epoch {}".format(epoch))
            train_loader = iter(self.train_loader)
            for i in range(len(train_loader)):
                self.net_optimizer.zero_grad()
                data, landmarks = train_loader.next()
                # print(data.shape)
                # print(landmarks.shape)
                data, landmarks = self.to_var(data), self.to_var(landmarks)
                B, L, H, W = data.size()
                B, L, S = landmarks.size()
                y = landmarks[:, :, 1].view(B, L)
                y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
                for i in range(B):
                    y_slices[i] = data[i, y[i]]

                jig_out, detected_points = self.model(y_slices)
                landmarks = landmarks.float()
                loss = self.criterion_d(detected_points, landmarks[:, :, [0, 2]])
                loss.backward()
                self.net_optimizer.step()
                print('loss: {}'.format(loss.item()))

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=False)

    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()
