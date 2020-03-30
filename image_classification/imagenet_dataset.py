import os
import cv2
import math
import random
import numpy as np
from paddle.fluid.io import Dataset

from datasets.folder import DatasetFolder


def center_crop_resize(img):
    h, w = img.shape[:2]
    c = int(224 / 256 * min((h, w)))
    i = (h + 1 - c) // 2
    j = (w + 1 - c) // 2
    img = img[i:i + c, j:j + c, :]
    return cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)


def random_crop_resize(img):
    height, width = img.shape[:2]
    area = height * width

    for attempt in range(10):
        target_area = random.uniform(0.08, 1.) * area
        log_ratio = (math.log(3 / 4), math.log(4 / 3))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w <= width and h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            img = img[i:i + h, j:j + w, :]
            return cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)

    return center_crop_resize(img)


def random_flip(img):
    if np.random.randint(0, 2) == 1:
        img = img[:, ::-1, :]
    return img


def normalize_permute(img):
    # transpose and convert to RGB from BGR
    img = img.astype(np.float32).transpose((2, 0, 1))[::-1, ...]
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    invstd = 1. / std
    for v, m, s in zip(img, mean, invstd):
        v.__isub__(m).__imul__(s)
    return img


def compose(functions):
    def process(sample):
        img, label = sample
        for fn in functions:
            img = fn(img)
        return img, label

    return process


def image_folder(path):
    valid_ext = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp')
    classes = [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    ]
    classes.sort()
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    samples = []
    for dir in sorted(class_map.keys()):
        d = os.path.join(path, dir)
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                p = os.path.join(root, fname)
                if os.path.splitext(p)[1].lower() in valid_ext:
                    samples.append((p, [class_map[dir]]))
    return samples


class ImageNetDataset(DatasetFolder):
    def __init__(self, path, mode='train'):
        super(ImageNetDataset, self).__init__(path)
        self.mode = mode
        if self.mode == 'train':
            self.transform = compose([
                cv2.imread, random_crop_resize, random_flip, normalize_permute
            ])
        else:
            self.transform = compose(
                [cv2.imread, center_crop_resize, normalize_permute])

    def __getitem__(self, idx):

        return self.transform(self.samples[idx])

    def __len__(self):
        return len(self.samples)
