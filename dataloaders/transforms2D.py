# Ignore warnings
import warnings

# import cv2
import numpy as np
import torch
from skimage import filters
from skimage import transform
import random
from torchvision.utils import make_grid
import torchvision.transforms as transforms

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


def make_permutation(img, n_pieces=10):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = transforms.ToTensor()(img)
    C, H, W = img.size()
    # print(C, H, W)
    h = H // n_pieces
    c = W // n_pieces
    # print(h,c)
    pieces = []
    for i in range(n_pieces):
        for j in range(n_pieces):
            x = i * h
            y = j * c
            # print(x, x + h)
            # print(y, y + c)
            pieces.append(img[..., x:x + h, y:y + c])
    # print(len(pieces))
    # print(pieces[0].size())
    indexes = list(range(n_pieces * n_pieces))

    # indexes = np.random.shuffle(indexes)

    random.shuffle(indexes)
    # print(indexes)
    result = []
    for i in indexes:
        # print(pieces[i].size())
        result.append(pieces[i])
    return make_grid(result, nrow=n_pieces, padding=0).numpy().reshape(3, H, W)[0], indexes


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


class ToTensorJigsaw(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, image_size=350):
        self.image_size = image_size

    def __call__(self, sample):
        image, indexes = sample['image'], sample['indexes']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        image = image.transpose((1, 0, 2))
        image_ = np.zeros((L, self.image_size, self.image_size))
        indexes = np.zeros((L, 100))
        for i in range(L):
            f = np.random.choice(fi, 1)[0]
            image_[i], indexes[i] = make_permutation(
                f(transform.resize(image[i], (self.image_size, self.image_size))) + np.random.normal(0, 0.1, (
                    self.image_size, self.image_size)))
            image_[i][image_[i] < 0] = 0.
            image_[i][image_[i] > 1] = 1.
        return {'image': torch.from_numpy(image_),
                'indexes': torch.from_numpy(indexes / 100)}


class ToTensorJigsawTest(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, image_size=350):
        self.image_size = image_size

    def __call__(self, sample):
        image, indexes = sample['image'], sample['indexes']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        image = image.transpose((1, 0, 2))
        image_ = np.zeros((L, self.image_size, self.image_size))
        indexes = np.zeros((L, 100))
        for i in range(L):
            image_[i], indexes[i] = make_permutation(transform.resize(image[i], (self.image_size, self.image_size)))
            image_[i][image_[i] < 0] = 0.
            image_[i][image_[i] > 1] = 1.
        return {'image': torch.from_numpy(image_),
                'indexes': torch.from_numpy(indexes / 100)}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, image_size=350):
        self.image_size = image_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        landmarks[:, 0] = landmarks[:, 0] * self.image_size // H
        landmarks[:, 2] = landmarks[:, 2] * self.image_size // W
        image = image.transpose((1, 0, 2))
        image_ = np.zeros((L, self.image_size, self.image_size))
        for i in range(L):
            f = np.random.choice(fi, 1)[0]
            # print(f)
            image_[i] = f(transform.resize(image[i], (self.image_size, self.image_size))) + np.random.normal(0, 0.1, (
                self.image_size, self.image_size))
            image_[i][image_[i] < 0] = 0.
            image_[i][image_[i] > 1] = 1.
        # print('here2', image_.shape)
        return {'image': torch.from_numpy(image_),
                'landmarks': torch.from_numpy(landmarks)}


class ToTensorTest(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, image_size=350):
        self.image_size = image_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        landmarks[:, 0] = landmarks[:, 0] * self.image_size // H
        landmarks[:, 2] = landmarks[:, 2] * self.image_size // W
        image = image.transpose((1, 0, 2))
        image_ = np.zeros((L, self.image_size, self.image_size))
        for i in range(L):
            # f = np.random.choice(fi, 1)[0]
            # print(f)
            image_[i] = transform.resize(image[i], (self.image_size, self.image_size))

        # print('here2', image_.shape)
        return {'image': torch.from_numpy(image_),
                'landmarks': torch.from_numpy(landmarks)}


class ToTensorClassifier(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, image_size=350):
        self.image_size = image_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        landmarks[:, 0] = landmarks[:, 0] * self.image_size // H
        landmarks[:, 2] = landmarks[:, 2] * self.image_size // W
        landmarks += np.random.randint(-5, 5, size=landmarks.shape)
        landmarks = list(landmarks)
        landmarks.append([random.randint(10, self.image_size - 10), random.randint(0, L),
                          random.randint(10, self.image_size - 10)])
        landmarks = np.array(landmarks)
        classes = np.array(list(range(len(landmarks))))
        images = np.zeros(shape=(len(landmarks), 20, 20))
        image = image.transpose((1, 0, 2))
        for i in range(len(classes)):
            f = np.random.choice(fi, 1)[0]
            ii = transform.resize(image[i], (self.image_size, self.image_size)) + np.random.normal(0, 0.1, (
                self.image_size, self.image_size))
            landmark = landmarks[i]
            x, y, z = landmark[0], landmark[1], landmark[2]
            if z < 10:
                z = 10
            if z > self.image_size - 10:
                z = self.image_size - 10

            if x < 10:
                x = 10
            if x > self.image_size - 10:
                x = self.image_size - 10

            if y < 1:
                y = 1
            if y >= L:
                y = L - 1

            if z < 10 or z > self.image_size - 10:
                print(i, x, y, z)
            if x < 10 or x > self.image_size - 10:
                print(i, x, y, z)
            iii = image[y, x - 10:x + 10, z - 10:z + 10]

            images[i] = f(iii)

            # image_ = np.zeros((L, self.image_size, self.image_size))
            # for i in range(L):
            #     f = identity
            # # print(f)
            # image_[i] = f(transform.resize(image[i], (self.image_size, self.image_size))) + np.random.normal(0, 0.1, (
            #     self.image_size, self.image_size))
            # image_[i][image_[i] < 0] = 0.
            # image_[i][image_[i] > 1] = 1.

            # print('here2', image_.shape)
        return {'image': torch.from_numpy(images),
                'landmarks': torch.from_numpy(classes)}


class ToTensorTestClassifier(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, image_size=350):
        self.image_size = image_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        H, L, W = image.shape
        # print('here1', L, H, W)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # print('here2', landmarks.shape)
        landmarks[:, 0] = landmarks[:, 0] * self.image_size // H
        landmarks[:, 2] = landmarks[:, 2] * self.image_size // W
        # landmarks += np.random.randint(-5, 5, size=landmarks.shape)
        landmarks = list(landmarks)
        landmarks.append([random.randint(10, self.image_size - 10), random.randint(0, L),
                          random.randint(10, self.image_size - 10)])
        landmarks = np.array(landmarks)
        classes = np.array(list(range(len(landmarks))))
        images = np.zeros(shape=(len(landmarks), 20, 20))
        image = image.transpose((1, 0, 2))
        for i in range(len(classes)):
            f = np.random.choice(fi, 1)[0]
            ii = transform.resize(image[i], (self.image_size, self.image_size)) + np.random.normal(0, 0.1, (
                self.image_size, self.image_size))
            landmark = landmarks[i]
            x, y, z = landmark[0], landmark[1], landmark[2]
            if z < 10:
                z = 10
            if z > self.image_size - 10:
                z = self.image_size - 10

            if x < 10:
                x = 10
            if x > self.image_size - 10:
                x = self.image_size - 10

            if y < 1:
                y = 1
            if y >= L:
                y = L - 1

            if z < 10 or z > self.image_size - 10:
                print(i, x, y, z)
            if x < 10 or x > self.image_size - 10:
                print(i, x, y, z)
            iii = image[y, x - 10:x + 10, z - 10:z + 10]

            images[i] = iii

            # image_ = np.zeros((L, self.image_size, self.image_size))
            # for i in range(L):
            #     f = identity
            # # print(f)
            # image_[i] = f(transform.resize(image[i], (self.image_size, self.image_size))) + np.random.normal(0, 0.1, (
            #     self.image_size, self.image_size))
            # image_[i][image_[i] < 0] = 0.
            # image_[i][image_[i] > 1] = 1.

            # print('here2', image_.shape)
        return {'image': torch.from_numpy(images),
                'landmarks': torch.from_numpy(classes)}


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
