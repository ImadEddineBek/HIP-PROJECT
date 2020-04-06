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

from dataloaders.transforms2D import *
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
        self.landmark_to_idx = ['left_edge',
                                'left_head',
                                'left_neck_1',
                                'left_neck_2',
                                'left_shaft_1',
                                'left_shaft_2',
                                'left_vertical_1',
                                'left_vertical_2',

                                'right_edge',
                                'right_head',
                                'right_neck_1',
                                'right_neck_2',
                                'right_shaft_1',
                                'right_shaft_2',
                                'right_vertical_1',
                                'right_vertical_2']
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
        image = np.array(np.load(img_name), dtype=np.float32)
        image = image / image.max()
        # print(image.shape)

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

        return sample['image'], sample['landmarks']


class HipJigsaw(Dataset):
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

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame['images'][idx])
        # y = self.landmarks_frame['right_head_y'][idx]
        image = np.array(np.load(img_name), dtype=np.float32)
        image = image / image.max()

        sample = {'image': image, 'indexes': None}
        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['indexes']


def get_dataloader2D(config):
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(config.image_size),
        ToTensor(config.image_size),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(config.image_size),
        ToTensorTest(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = HipLandmarksDataset(config.data_csv_train, config.data_root, transform_train)
    dataset_test = HipLandmarksDataset(config.data_csv_test, config.data_root, transform_test)
    train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers)

    test_loader = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers)
    return train_loader, test_loader


def get_dataloader2DClassifier(config):
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(config.image_size),
        ToTensorTestClassifier(config.image_size),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(config.image_size),
        ToTensorTestClassifier(config.image_size),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = HipLandmarksDataset(config.data_csv_train, config.data_root, transform_train)
    dataset_test = HipLandmarksDataset(config.data_csv_test, config.data_root, transform_test)
    train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers)

    test_loader = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers)
    return train_loader, test_loader


def get_dataloader2DJigSaw(config):
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(config.image_size),
        ToTensorJigsaw(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(config.image_size),
        ToTensorJigsawTest(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = HipJigsaw(config.data_csv_train, config.data_root, transform_train)
    dataset_test = HipJigsaw(config.data_csv_test, config.data_root, transform_test)
    train_loader = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers)

    test_loader = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers)
    return train_loader, test_loader


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

        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break
