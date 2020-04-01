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
        losses = None
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

                jig_out, detected_points = self.trainer.predict(y_slices)
                landmarks = landmarks.float()
                l1_loss = np.abs((landmarks[:, :, [0, 2]] - detected_points * 350).detach().cpu().numpy())
                if losses is None:
                    losses = l1_loss
                else:
                    losses = np.concatenate((losses, l1_loss))

    def l2_loss(self, ):
        test_loader = iter(self.loader)
        losses = None
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

                jig_out, detected_points = self.trainer.predict(y_slices)
                landmarks = landmarks.float()
                l2_loss = np.abs(((landmarks[:, :, [0, 2]] - detected_points * 350) ** 2).detach().cpu().numpy())
                if losses is None:
                    losses = l2_loss
                else:
                    losses = np.concatenate((losses, l2_loss))

    def report(self,):
        pass
