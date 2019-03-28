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
from paddle.dataset.common import download

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

SIZE_FLOAT32 = 4
SIZE_INT64 = 8

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


def download_unzip():

    tmp_folder = 'int8/download'

    cache_folder = os.path.expanduser('~/.cache/' + tmp_folder)

    data_urls = []
    data_md5s = []

    data_urls.append(
        'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partaa'
    )
    data_md5s.append('60f6525b0e1d127f345641d75d41f0a8')
    data_urls.append(
        'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partab'
    )
    data_md5s.append('1e9f15f64e015e58d6f9ec3210ed18b5')

    file_names = []
    for i in range(0, len(data_urls)):
        download(data_urls[i], tmp_folder, data_md5s[i])
        file_names.append(data_urls[i].split('/')[-1])

    zip_path = os.path.join(cache_folder, 'full_imagenet_val.tar.gz')

    if not os.path.exists(zip_path):
        cat_command = 'cat'
        for file_name in file_names:
            cat_command += ' ' + os.path.join(cache_folder, file_name)
        cat_command += ' > ' + zip_path
        os.system(cat_command)

    if not os.path.exists(cache_folder):
        cmd = 'mkdir {0} && tar xf {1} -C {0}'.format(cache_folder, zip_path)

    cmd = 'rm -rf {3} && ln -s {1} {0}'.format("data", cache_folder, zip_path)

    os.system(cmd)

    data_dir = os.path.expanduser(cache_folder + 'data')

    return data_dir


def reader():
    data_dir = download_unzip()
    file_list = os.path.join(data_dir, 'val_list.txt')
    output_file = os.path.join(data_dir, 'int8_full_val.bin')
    with open(file_list) as flist:
        lines = [line.strip() for line in flist]
        num_images = len(lines)

        with open(output_file, "w+b") as of:
            #save num_images(int64_t) to file
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
