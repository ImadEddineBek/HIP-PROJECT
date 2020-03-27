# Ignore warnings
import warnings

# import cv2
import numpy as np
import torch
from skimage import filters
from skimage import transform

warnings.filterwarnings("ignore")


def thresholding_optimal(img):
    return img > filters.threshold_li(img)


def multi_threshholding(img):
    thresholds = filters.threshold_multiotsu(img)
    regions = np.digitize(img, bins=thresholds)
    return regions


def hysteresis_thrsholding(img):
    edges = filters.sobel(img)

    low = 0.1
    high = 0.6

    lowt = (edges > low).astype(int)
    hight = (edges > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(edges, low, high)
    return hight + hyst


def low_thrsholding(img):
    edges = filters.sobel(img)

    low = 0.1
    high = 0.6

    lowt = (edges > low).astype(int)
    return lowt


def identity(image):
    """Return the original image, ignoring any kwargs."""
    return image


fi = [filters.roberts, filters.sobel, filters.scharr,
      filters.prewitt, filters.sobel_h, filters.farid_h,
      filters.farid_v, filters.sobel_h, filters.sobel_v, filters.scharr_h,
      filters.scharr_v, filters.prewitt_h, filters.prewitt_v, thresholding_optimal,
      multi_threshholding,
      identity, filters.sato, filters.hessian]


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        landmarks[:, 0] = landmarks[:, 0] * 350 // H
        landmarks[:, 2] = landmarks[:, 2] * 350 // W
        image = image.transpose((1, 0, 2))
        image_ = np.zeros((L, 350, 350))
        for i in range(L):
            f = np.random.choice(fi, 1)[0]
            # print(f)
            image_[i] = f(transform.resize(image[i], (350, 350))) + np.random.normal(0, 0.1, (350, 350))
            image_[i][image_[i] < 0] = 0.
            image_[i][image_[i] > 1] = 1.
        # print('here2', image_.shape)
        return {'image': torch.from_numpy(image_),
                'landmarks': torch.from_numpy(landmarks)}


class ToTensorTest(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        landmarks[:, 0] = landmarks[:, 0] * 350 // H
        landmarks[:, 2] = landmarks[:, 2] * 350 // W
        image = image.transpose((1, 0, 2))
        image_ = np.zeros((L, 350, 350))
        for i in range(L):
            # f = np.random.choice(fi, 1)[0]
            # print(f)
            image_[i] = transform.resize(image[i], (350, 350))

        # print('here2', image_.shape)
        return {'image': torch.from_numpy(image_),
                'landmarks': torch.from_numpy(landmarks)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
