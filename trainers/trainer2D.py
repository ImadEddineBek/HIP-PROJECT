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
from utils.evaluate import Evaluator


class Trainer2D:
    def __init__(self, config):
        self.experiment = Experiment(api_key='CQ4yEzhJorcxul2hHE5gxVNGu', project_name='HIP')
        self.experiment.log_parameters(vars(config))
        self.config = config
        self.log_step = config.log_step
        self.model = conv2d.Conv2DPatches(image_size=config.image_size)
        print(self.model)
        self.d = get_dataloader2D(config)
        self.train_loader, self.test_loader = self.d
        self.train_loader_jig, self.test_loader_jig = get_dataloader2DJigSaw(config)
        self.net_optimizer = optim.Adam(self.model.parameters(), config.lr, [0.5, 0.9999])
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_d = nn.MSELoss()
        self.epochs = config.epochs
        if torch.cuda.is_available():
            print('Using CUDA')
            self.model = self.model.cuda()
        #     self.model = self.model.cuda()
        self.pre_model_path = './artifacts/pre_models/' + str(config.lr) + '.pth'
        self.model_path = './artifacts/models/' + str(config.lr) + '.pth'
        self.image_size = config.image_size

    def pre_train(self):

        if os.path.isfile(self.pre_model_path):
            print("Using pre-trained model for solving the jigsaw puzzle")
            self.model = torch.load(self.pre_model_path)
        else:
            print("Starting pre-training and solving the jigsaw puzzle")
            for epoch in range(0):
                print("Starting epoch {}".format(epoch))
                train_loader = iter(self.train_loader_jig)
                with self.experiment.train():
                    for i in range(len(train_loader)):
                        self.net_optimizer.zero_grad()
                        data, indexes, _ = train_loader.next()
                        # print(landmarks)
                        # print(landmarks.shape)
                        data, indexes = self.to_var(data), self.to_var(indexes).float()
                        B, L, H, W = data.size()
                        B, L, S = indexes.size()
                        print(data.size())
                        print(indexes.size())

                        jig_out, _ = self.model(data, True)
                        loss = self.criterion_d(jig_out, indexes.view(-1, S))
                        loss.backward()
                        self.net_optimizer.step()
                        # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
                        self.experiment.log_metric('pre-loss', loss.item())
                        print('loss: {}'.format(loss.item()))

            torch.save(self.model, self.pre_model_path)

    def train(self):
        if os.path.isfile(self.model_path):
            print("Using pre-trained model")
            self.model = torch.load(self.model_path)
        if False:
            pass
        else:
            print("Starting training")
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            for epoch in range(self.epochs):
                print("Starting epoch {}".format(epoch))
                train_loader = iter(self.train_loader)
                with self.experiment.train():
                    for i in range(len(train_loader)):
                        self.net_optimizer.zero_grad()
                        data, landmarks, _ = train_loader.next()
                        # print(landmarks)
                        data, landmarks = self.to_var(data), self.to_var(landmarks)
                        B, L, H, W = data.size()
                        B, L, S = landmarks.size()
                        y = landmarks[:, :, 1].view(B, L)
                        y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
                        if torch.cuda.is_available():
                            y_slices = y_slices.cuda()
                        for i in range(B):
                            y_slices[i] = data[i, y[i]]

                        jig_out, detected_points = self.model(y_slices)
                        landmarks = landmarks.float() / self.image_size
                        loss = self.criterion_d(detected_points, landmarks[:, :, [0, 2]])
                        loss.backward()
                        self.net_optimizer.step()
                        # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
                        self.experiment.log_metric('loss', loss.item())
                        print('loss: {}'.format(loss.item()))
                if epoch % self.log_step == 0:
                    with self.experiment.test():
                        self.evaluate()
                        evaluator = Evaluator(self, self.test_loader)
                        evaluator.report()
            torch.save(self.model, self.model_path)
        evaluator = Evaluator(self, self.test_loader)
        evaluator.report()

    def evaluate(self):
        test_loader = iter(self.test_loader)
        with self.experiment.test():
            loss = 0
            for i in range(len(test_loader)):
                self.net_optimizer.zero_grad()
                data, landmarks, _ = test_loader.next()
                data, landmarks = self.to_var(data), self.to_var(landmarks)
                B, L, H, W = data.size()
                B, L, S = landmarks.size()
                y = landmarks[:, :, 1].view(B, L)
                y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
                if torch.cuda.is_available():
                    y_slices = y_slices.cuda()

                for i in range(B):
                    y_slices[i] = data[i, y[i]]

                jig_out, detected_points = self.model(y_slices)
                landmarks = landmarks.float() / self.image_size
                loss += self.criterion_d(detected_points, landmarks[:, :, [0, 2]]).item()
                self.plots(y_slices.cpu(), landmarks[:, :, [0, 2]], detected_points)
            self.experiment.log_metric('loss', loss / len(test_loader))

    def plots(self, slices, real, predicted):
        figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
        slices = slices[0].cpu().detach().numpy()
        real = real[0].cpu().detach().numpy()
        predicted = predicted[0].cpu().detach().numpy()
        real *= self.image_size
        predicted *= self.image_size
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

    def predict(self, x):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            x = x.cuda()
        _, x = self.model(x)
        return x
