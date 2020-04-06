import torch
import sys

from comet_ml import Experiment
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
from dataloaders.dataloader2D import get_dataloader2D, get_dataloader2DJigSaw
from models import conv2d


class Trainer2D:
    def __init__(self, config):
        self.experiment = Experiment(api_key='CQ4yEzhJorcxul2hHE5gxVNGu', project_name='HIP')
        self.config = config
        self.model = conv2d.Conv2DPatches()
        print(self.model)
        self.train_loader, self.test_loader = get_dataloader2D(config)
        self.train_loader_jig, self.test_loader_jig = get_dataloader2DJigSaw(config)
        self.net_optimizer = optim.Adam(self.model.parameters(), config.lr, [0.5, 0.9999])
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_d = nn.MSELoss()
        self.epochs = config.epochs
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        #     self.model = self.model.cuda()

    def pre_train(self):
        print("Starting pre-training and solving the jigsaw puzzle")
        for epoch in range(self.epochs):
            print("Starting epoch {}".format(epoch))
            train_loader = iter(self.train_loader_jig)
            with self.experiment.train():
                for i in range(len(train_loader)):
                    self.net_optimizer.zero_grad()
                    data, indexes = train_loader.next()
                    # print(landmarks)
                    # print(landmarks.shape)
                    data, indexes = self.to_var(data), self.to_var(indexes).float()
                    B, L, H, W = data.size()
                    B, L, S = indexes.size()

                    jig_out, _ = self.model(data, True)
                    loss = self.criterion_d(jig_out, indexes.view(-1, S))
                    loss.backward()
                    self.net_optimizer.step()
                    # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
                    self.experiment.log_metric('pre-loss', loss.item())
                    print('loss: {}'.format(loss.item()))

    def train(self):
        print("Starting training")
        for epoch in range(self.epochs):
            print("Starting epoch {}".format(epoch))
            train_loader = iter(self.train_loader)
            with self.experiment.train():
                for i in range(len(train_loader)):
                    self.net_optimizer.zero_grad()
                    data, landmarks = train_loader.next()
                    # print(landmarks)
                    # print(landmarks.shape)
                    data, landmarks = self.to_var(data), self.to_var(landmarks)
                    B, L, H, W = data.size()
                    B, L, S = landmarks.size()
                    y = landmarks[:, :, 1].view(B, L)
                    y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
                    for i in range(B):
                        y_slices[i] = data[i, y[i]]

                    jig_out, detected_points = self.model(y_slices)
                    landmarks = landmarks.float() / 350.
                    loss = self.criterion_d(detected_points, landmarks[:, :, [0, 2]])
                    loss.backward()
                    self.net_optimizer.step()
                    # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
                    self.experiment.log_metric('loss', loss.item())
                    print('loss: {}'.format(loss.item()))
            with self.experiment.test():
                self.evaluate()
        self.experiment.end()

    def evaluate(self):
        test_loader = iter(self.test_loader)
        with self.experiment.test():
            loss = 0
            for i in range(len(test_loader)):
                self.net_optimizer.zero_grad()
                data, landmarks = test_loader.next()
                data, landmarks = self.to_var(data), self.to_var(landmarks)
                B, L, H, W = data.size()
                B, L, S = landmarks.size()
                y = landmarks[:, :, 1].view(B, L)
                y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
                for i in range(B):
                    y_slices[i] = data[i, y[i]]

                jig_out, detected_points = self.model(y_slices)
                landmarks = landmarks.float() / 350.
                loss += self.criterion_d(detected_points, landmarks[:, :, [0, 2]]).item()
                self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
            self.experiment.log_metric('loss', loss / len(test_loader))

    def plots(self, slices, real, predicted):
        figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
        slices = slices[0].cpu().detach().numpy()
        real = real[0].cpu().detach().numpy()
        predicted = predicted[0].cpu().detach().numpy()
        real *= 350
        predicted *= 350
        s = 0
        # print(real.size())
        # print(predicted.size())
        for i in range(4):
            for j in range(4):
                axes[i, j].imshow(slices[s])
                x, z = real[s]
                axes[i, j].scatter(x, z, color='red')
                x, z = predicted[s]
                axes[i, j].scatter(x, z, color='blue')
                s += 1
        self.experiment.log_figure(figure=plt)
        plt.savefig('artifacts/predictions/img.png')
        plt.show()

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

    def predict(self, ):
        pass
