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
        loss = 0
        for i in range(len(test_loader)):
            data, landmarks = test_loader.next()
            data, landmarks = self.trainer.to_var(data), self.trainer.to_var(landmarks)
            B, L, H, W = data.size()
            B, L, S = landmarks.size()
            y = landmarks[:, :, 1].view(B, L)
            y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
            if torch.cuda.is_available():
                y_slices = y_slices.cuda()
            for i in range(B):
                y_slices[i] = data[i, y[i]]

            detected_points = self.trainer.predict(y_slices)
            landmarks = landmarks.float()
            l1_loss = np.abs(
                (landmarks[:, :, [0, 2]] - detected_points * self.trainer.image_size).detach().cpu().numpy())
            if losses is None:
                losses = l1_loss
            else:
                losses = np.concatenate((losses, l1_loss))

        return losses

    def l2_loss(self, ):
        test_loader = iter(self.loader)
        losses = None
        loss = 0
        for i in range(len(test_loader)):
            data, landmarks = test_loader.next()
            data, landmarks = self.trainer.to_var(data), self.trainer.to_var(landmarks)
            B, L, H, W = data.size()
            B, L, S = landmarks.size()
            y = landmarks[:, :, 1].view(B, L)
            y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
            for i in range(B):
                y_slices[i] = data[i, y[i]]

            detected_points = self.trainer.predict(y_slices)
            landmarks = landmarks.float()
            l2_loss = np.abs(
                ((landmarks[:, :, [0, 2]] - detected_points * self.trainer.image_size) ** 2).detach().cpu().numpy())
            if losses is None:
                losses = l2_loss
            else:
                losses = np.concatenate((losses, l2_loss))
        return losses

    def report(self, ):
        l1 = self.l1_loss()
        l2 = self.l2_loss()

        def process_loss(losses):
            means = np.mean(losses, axis=2)
            stds = np.std(means, axis=0)
            mins = np.min(means, axis=0)
            maxs = np.max(means, axis=0)
            means = np.mean(means, axis=0)

            return means, stds, mins, maxs

        l1_means, l1_stds, l1_mins, l1_maxs = process_loss(l1)
        l2_means, l2_stds, l2_mins, l2_maxs = process_loss(l2)
        print('\t\t means, stds, mins, maxes')
        for i in range(len(l1_means)):
            print('%s: %.2f, %.2f, %.2f, %.2f' % (
                self.loader.dataset.idx_to_landmark[i], l1_means[i], l1_stds[i], l1_mins[i], l1_maxs[i]))

        for i in range(len(l2_means)):
            print('%s: %.2f, %.2f, %.2f, %.2f' % (
                self.loader.dataset.idx_to_landmark[i], l2_means[i], l2_stds[i], l2_mins[i], l2_maxs[i]))
        # print(self.loader.idx_to_landmark)
