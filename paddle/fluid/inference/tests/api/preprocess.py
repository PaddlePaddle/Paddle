#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
import unittest
import os
import numpy as np
import time
import sys
import random
import functools
import contextlib
from PIL import Image, ImageEnhance
import math

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

SIZE_FLOAT32 = 4
SIZE_INT64 = 8

DATA_DIR = '/data/ILSVRC2012'

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
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(img_path, mode, color_jitter, rotate):
    img = Image.open(img_path)
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img


def reader():
    data_dir = DATA_DIR
    file_list = os.path.join(data_dir, 'val_list.txt')
    bin_file = os.path.join(data_dir, 'data.bin')
    with open(file_list) as flist:
        lines = [line.strip() for line in flist]
        num_images = len(lines)

        with open(bin_file, "w+b") as of:
            of.seek(0)
            num = np.array(int(num_images)).astype('int64')
            of.write(num.tobytes())
            for idx, line in enumerate(lines):
                img_path, label = line.split()
                img_path = os.path.join(data_dir, img_path)
                if not os.path.exists(img_path):
                    continue

                #save image(float32) to file
                img = process_image(
                    img_path, 'val', color_jitter=False, rotate=False)
                np_img = np.array(img)
                of.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 *
                        idx)
                of.write(np_img.astype('float32').tobytes())

                #save label(int64_t) to file
                label_int = (int)(label)
                np_label = np.array(label_int)
                of.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 *
                        num_images + idx * SIZE_INT64)
                of.write(np_label.astype('int64').tobytes())


if __name__ == '__main__':
    reader()
