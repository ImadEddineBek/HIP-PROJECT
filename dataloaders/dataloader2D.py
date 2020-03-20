import torch
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

from utils.utils import fix_path

warnings.filterwarnings("ignore")


class HipLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
             csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmark_to_idx = ['left_edge_x', 'left_edge_y', 'left_edge_z',
                                'left_head_x', 'left_head_y', 'left_head_z',
                                'left_neck_x1', 'left_neck_x2', 'left_neck_y', 'left_neck_z1', 'left_neck_z2',
                                'left_shaft_x1', 'left_shaft_x2', 'left_shaft_y', 'left_shaft_z1', 'left_shaft_z2',
                                'left_vertical_x1', 'left_vertical_x2', 'left_vertical_y', 'left_vertical_z1',
                                'left_vertical_z2',

                                'right_edge_x', 'right_edge_y', 'right_edge_z',
                                'right_head_x', 'right_head_y', 'right_head_z',
                                'right_neck_x1', 'right_neck_x2', 'right_neck_y', 'right_neck_z1', 'right_neck_z2',
                                'right_shaft_x1', 'right_shaft_x2', 'right_shaft_y', 'right_shaft_z1', 'right_shaft_z2',
                                'right_vertical_x1', 'right_vertical_x2', 'right_vertical_y', 'right_vertical_z1',
                                'right_vertical_z2']
        self.landmark_to_idx = {self.landmark_to_idx[i]: i for i in range(len(self.landmark_to_idx))}
        self.idx_to_landmark = {v: k for k, v in self.landmark_to_idx.items()}

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame['images'][idx])
        # y = self.landmarks_frame['right_head_y'][idx]
        image = np.load(img_name)

        # left
        landmarks = [self.landmarks_frame['left_edge_x'][idx], self.landmarks_frame['left_edge_y'][idx],
                     self.landmarks_frame['left_edge_z'][idx]]
        landmarks += [self.landmarks_frame['left_head_x'][idx], self.landmarks_frame['left_head_y'][idx],
                      self.landmarks_frame['left_head_z'][idx]]

        landmarks += [self.landmarks_frame['left_neck_x1'][idx], self.landmarks_frame['left_neck_y'][idx],
                      self.landmarks_frame['left_neck_z1'][idx]]
        landmarks += [self.landmarks_frame['left_neck_x2'][idx], self.landmarks_frame['left_neck_y'][idx],
                      self.landmarks_frame['left_neck_z2'][idx]]
        landmarks += [self.landmarks_frame['left_shaft_x1'][idx], self.landmarks_frame['left_shaft_y'][idx],
                      self.landmarks_frame['left_shaft_z1'][idx]]
        landmarks += [self.landmarks_frame['left_shaft_x2'][idx], self.landmarks_frame['left_shaft_y'][idx],
                      self.landmarks_frame['left_shaft_z2'][idx]]
        landmarks += [self.landmarks_frame['left_vertical_x1'][idx], self.landmarks_frame['left_vertical_y'][idx],
                      self.landmarks_frame['left_vertical_z1'][idx]]
        landmarks += [self.landmarks_frame['left_vertical_x2'][idx], self.landmarks_frame['left_vertical_y'][idx],
                      self.landmarks_frame['left_vertical_z2'][idx]]

        # right
        landmarks += [self.landmarks_frame['right_edge_x'][idx], self.landmarks_frame['right_edge_y'][idx],
                      self.landmarks_frame['right_edge_z'][idx]]
        landmarks += [self.landmarks_frame['right_head_x'][idx], self.landmarks_frame['right_head_y'][idx],
                      self.landmarks_frame['right_head_z'][idx]]

        landmarks += [self.landmarks_frame['right_neck_x1'][idx], self.landmarks_frame['right_neck_y'][idx],
                      self.landmarks_frame['right_neck_z1'][idx]]
        landmarks += [self.landmarks_frame['right_neck_x2'][idx], self.landmarks_frame['right_neck_y'][idx],
                      self.landmarks_frame['right_neck_z2'][idx]]
        landmarks += [self.landmarks_frame['right_shaft_x1'][idx], self.landmarks_frame['right_shaft_y'][idx],
                      self.landmarks_frame['right_shaft_z1'][idx]]
        landmarks += [self.landmarks_frame['right_shaft_x2'][idx], self.landmarks_frame['right_shaft_y'][idx],
                      self.landmarks_frame['right_shaft_z2'][idx]]
        landmarks += [self.landmarks_frame['right_vertical_x1'][idx], self.landmarks_frame['right_vertical_y'][idx],
                      self.landmarks_frame['right_vertical_z1'][idx]]
        landmarks += [self.landmarks_frame['right_vertical_x2'][idx], self.landmarks_frame['right_vertical_y'][idx],
                      self.landmarks_frame['right_vertical_z2'][idx]]

        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('int').reshape(-1, 3)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':

    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        for l in landmarks:
            plt.imshow(image[:, l[1], :])
            plt.scatter(l[0], l[2], s=10, marker='.', c='r')
            plt.show()
        # plt.pause(0.001)  # pause a bit so that plots are updated


    fix_path()
    data_csv = 'data/data.csv'
    data_root = ''
    dataset = HipLandmarksDataset(data_csv, data_root)
    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        # ax = plt.subplot(1, 4, i + 1)
        # plt.tight_layout()
        # ax.set_title('Sample #{}'.format(i))
        # ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break
