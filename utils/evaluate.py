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
    def __init__(self, trainer, test_loader, dim=2):
        self.trainer = trainer
        self.loader = test_loader
        self.dim = dim
        pass

    def l1_loss(self, ):
        test_loader = iter(self.loader)
        losses = None
        loss = 0
        for _ in range(len(test_loader)):
            data, landmarks, mri_sequences = test_loader.next()
            print(mri_sequences)
            data, landmarks = self.trainer.to_var(data), self.trainer.to_var(landmarks)
            if len(data.size()) == 4:
                B, L, H, W = data.size()
                B, L, S = landmarks.size()
                y = landmarks[:, :, 1].view(B, L)
                y_slices = torch.zeros([B, L, H, W], dtype=torch.float32)
            else:
                B, L, C, H, W = data.size()
                B, L, S = landmarks.size()
                y = landmarks[:, :, 1].view(B, L)
                y_slices = torch.zeros([B, L, C, H, W], dtype=torch.float32)

            if torch.cuda.is_available():
                y_slices = y_slices.cuda()
            for i in range(B):
                y_slices[i] = data[i, y[i]]

            detected_points = self.trainer.predict(y_slices)
            landmarks = landmarks[:, :, [0, 2]]
            if self.dim == 3:
                detected_points = detected_points[:, :, [0, 2]]
            # print(detected_points.size(), landmarks.size())
            landmarks = landmarks.float()
            l1_loss = np.abs(
                (landmarks - detected_points * self.trainer.image_size).detach().cpu().numpy())
            if losses is None:
                losses = l1_loss
            else:
                losses = np.concatenate((losses, l1_loss))

        return losses

    def report(self, ):
        l1 = self.l1_loss()

        def process_loss(losses):
            means = np.mean(losses, axis=2)
            stds = np.std(means, axis=0)
            mins = np.min(means, axis=0)
            maxs = np.max(means, axis=0)
            means = np.mean(means, axis=0)

            return means, stds, mins, maxs

        l1_means, l1_stds, l1_mins, l1_maxs = process_loss(l1)

        self.trainer.experiment.log_metric('mean error', l1_means[0])
        print('\t\t means, stds, mins, maxes')
        for i in range(len(l1_means)):
            print('%s: %.2f, %.2f, %.2f, %.2f' % (
                self.loader.dataset.idx_to_landmark[i], l1_means[i], l1_stds[i], l1_mins[i], l1_maxs[i]))
        # print(self.loader.idx_to_landmark)
