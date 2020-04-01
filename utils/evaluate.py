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


class Evaluator:
    def __init__(self, trainer, test_loader):
        self.trainer = trainer
        self.loader = test_loader
        pass

    def l1_loss(self, ):
        test_loader = iter(self.loader)
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
        pass

    def l2_loss(self, ):
        test_loader = iter(self.loader)
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
        pass
