# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import random
import functools
import numpy as np
from threading import Thread
import subprocess
import time

from Queue import Queue
import paddle
from PIL import Image, ImageEnhance

random.seed(0)

DATA_DIM = 224

THREAD = 10
BUF_SIZE = 100000

DATA_DIR = '/mnt/ImageNet'
TRAIN_LIST = '/mnt/ImageNet/train.txt'
TEST_LIST = '/mnt/ImageNet/val.txt'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = random.randint(0, width - size)
        h_start = random.randint(0, height - size)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * random.uniform(scale_min,
                                                             scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = random.randint(0, img.size[0] - w)
    j = random.randint(0, img.size[1] - h)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def rotate_image(img):
    angle = random.randint(-10, 10)
    img = img.rotate(angle)
    return img


def distort_color(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)

    ops = [random_brightness, random_contrast, random_color]
    random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


def process_image(sample, mode, color_jitter, rotate):
    img_path = sample[0]

    img = Image.open(img_path)
    if mode == 'train':
        if rotate: img = rotate_image(img)
        img = random_crop(img, DATA_DIM)
    else:
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=DATA_DIM, center=True)
    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if random.randint(0, 1) == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, sample[1]
    elif mode == 'test':
        return [img]


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False,
                    xmap=True):
    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            if shuffle:
                random.shuffle(full_lines)
            if mode == 'train':
                trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
                trainer_count = int(os.getenv("PADDLE_TRAINERS"))
                per_node_lines = len(full_lines) / trainer_count
                lines = full_lines[trainer_id * per_node_lines:(trainer_id + 1)
                                   * per_node_lines]
                print(
                    "read images from %d, length: %d, lines length: %d, total: %d"
                    % (trainer_id * per_node_lines, per_node_lines, len(lines),
                       len(full_lines)))
            else:
                lines = full_lines

            for line in lines:
                if mode == 'train':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(DATA_DIR, "train", img_path)
                    yield (img_path, int(label))
                elif mode == 'val':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(DATA_DIR, "val", img_path)
                    yield (img_path, int(label))
                elif mode == 'test':
                    img_path = os.path.join(DATA_DIR, line)
                    yield [img_path]

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    # if xmap:
    return paddle.reader.xmap_readers(mapper, reader, THREAD, BUF_SIZE)
    # else:


# return paddle.reader.map_readers(mapper, reader)


def train(file_list=TRAIN_LIST, xmap=True):
    return _reader_creator(
        file_list,
        'train',
        shuffle=True,
        color_jitter=False,
        rotate=False,
        xmap=xmap)


def val(file_list=TEST_LIST, xmap=True):
    return _reader_creator(file_list, 'val', shuffle=False, xmap=xmap)


def test(file_list=TEST_LIST):
    return _reader_creator(file_list, 'test', shuffle=False)
