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
from sklearn.metrics import accuracy_score
from termcolor import colored
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from dataloaders.dataloader2D import get_dataloader2D, get_dataloader2DJigSaw, get_dataloader2DClassifier
from models import conv2d, classifier2d


def accuracy_function(true_labels, predicted_labels):
    _, pred = torch.max(predicted_labels, 1)
    correct = np.squeeze(pred.eq(true_labels.data.view_as(pred)))
    return correct.float().mean()


class Trainer2DClassifier:
    def __init__(self, config):
        self.experiment = Experiment(api_key='CQ4yEzhJorcxul2hHE5gxVNGu', project_name='HIP')
        self.config = config
        self.experiment.log_parameters(vars(config))

        self.log_step = config.log_step
        self.model = classifier2d.ConvClassifier()
        print(self.model)
        self.train_loader, self.test_loader = get_dataloader2DClassifier(config)
        self.net_optimizer = optim.Adam(self.model.parameters(), config.lr, [0.5, 0.9999], amsgrad=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.criterion_c = nn.CrossEntropyLoss()
        self.criterion_d = nn.MSELoss()
        self.epochs = config.epochs
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.image_size = config.image_size

    def pre_train(self):
        pass

    def train(self):
        print("Starting training")
        for epoch in range(self.epochs):
            print("Starting epoch {}".format(epoch))
            train_loader = iter(self.train_loader)
            with self.experiment.train():
                for i in range(len(train_loader)):
                    self.net_optimizer.zero_grad()
                    data, classes = train_loader.next()
                    # print(landmarks)
                    # print(landmarks.shape)
                    data, classes = self.to_var(data), self.to_var(classes)
                    # B, L, H, W = data.size()
                    # B, L = classes.size()

                    detected_points = self.model(data)
                    # print(classes.size())
                    loss = self.criterion_c(detected_points, classes)
                    loss.backward()
                    self.net_optimizer.step()
                    # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
                    self.experiment.log_metric('loss', loss.item())
                    print('loss: {}'.format(loss.item()))
            if epoch % self.log_step == 0:
                with self.experiment.test():
                    self.evaluate()
                    # evaluator = Evaluator(self, self.test_loader)
                    # evaluator.report()
        self.experiment.end()

    # @staticmethod
    # def accuracy(detected_points, classes):
    #     return accuracy_score(classes, detected_points)

    def evaluate(self):
        test_loader = iter(self.test_loader)
        with self.experiment.test():
            loss = 0
            accuracy = 0
            for i in range(len(test_loader)):
                self.net_optimizer.zero_grad()
                data, classes = test_loader.next()
                # print(landmarks)
                # print(landmarks.shape)
                data, classes = self.to_var(data), self.to_var(classes)
                # B, L, H, W = data.size()
                # B, L, S = classes.size()

                detected_points = self.model(data)
                loss += self.criterion_c(detected_points, classes).item()
                accuracy += accuracy_function(classes.cpu().detach(), detected_points.cpu().detach()).item()
                # self.plots(y_slices, landmarks[:, :, [0, 2]], detected_points)
            print('loss', loss / len(test_loader))
            self.experiment.log_metric('loss', loss / len(test_loader))
            print('accuracy', accuracy / len(test_loader))
            self.experiment.log_metric('accuracy', accuracy / len(test_loader))

    def plots(self, slices, real, predicted):
        pass
        # figure, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
        # slices = slices[0].cpu().detach().numpy()
        # real = real[0].cpu().detach().numpy()
        # predicted = predicted[0].cpu().detach().numpy()
        # real *= self.image_size
        # predicted *= self.image_size
        # s = 0
        # # print(real.size())
        # # print(predicted.size())
        # for i in range(4):
        #     for j in range(4):
        #         axes[i, j].imshow(slices[s])
        #         x, z = real[s]
        #         axes[i, j].scatter(x, z, color='red')
        #         x, z = predicted[s]
        #         axes[i, j].scatter(x, z, color='blue')
        #         s += 1
        # self.experiment.log_figure(figure=plt)
        # plt.savefig('artifacts/predictions/img.png')
        # plt.show()

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
