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

THREAD = 8
BUF_SIZE = 20000

# DATA_DIR = 'data/ILSVRC2012'
# TRAIN_LIST = 'data/ILSVRC2012/train_list.txt'
# TEST_LIST = 'data/ILSVRC2012/val_list.txt'

DATA_DIR = '/mnt/ImageNet'
TRAIN_LIST = '/mnt/ImageNet/train.txt'
TEST_LIST = '/mnt/ImageNet/val.txt'

# DATA_DIR = 'ImageNet'
# TRAIN_LIST = 'ImageNet/train.txt'
# TEST_LIST = 'ImageNet/val.txt'

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


class XmapEndSignal():
    pass


def xmap_readers(mapper, reader, process_num, buffer_size, order=False):
    """
    Use multiprocess to map samples from reader by a mapper defined by user.
    And this function contains a buffered decorator.
    :param mapper:  a function to map sample.
    :type mapper: callable
    :param reader: the data reader to read from
    :type reader: callable
    :param process_num: process number to handle original sample
    :type process_num: int
    :param buffer_size: max buffer size
    :type buffer_size: int
    :param order: keep the order of reader
    :type order: bool
    :return: the decarated reader
    :rtype: callable
    """
    end = XmapEndSignal()

    # define a worker to read samples from reader to in_queue
    def read_worker(reader, in_queue):
        for i in reader():
            in_queue.put(i)
        in_queue.put(end)

    # define a worker to read samples from reader to in_queue with order flag
    def order_read_worker(reader, in_queue):
        in_order = 0
        for i in reader():
            in_queue.put((in_order, i))
            in_order += 1
        in_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue
    def handle_worker(in_queue, out_queue, mapper):
        sample = in_queue.get()
        while not isinstance(sample, XmapEndSignal):
            r = mapper(sample)
            out_queue.put(r)
            sample = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    # define a worker to handle samples from in_queue by mapper
    # and put mapped samples into out_queue by order
    def order_handle_worker(in_queue, out_queue, mapper, out_order):
        ins = in_queue.get()
        while not isinstance(ins, XmapEndSignal):
            order, sample = ins
            r = mapper(sample)
            while order != out_order[0]:
                pass
            out_queue.put(r)
            out_order[0] += 1
            ins = in_queue.get()
        in_queue.put(end)
        out_queue.put(end)

    def status(in_queue, out_queue):
        while True:
            print('in queue size %d, out size: %d' %
                  (in_queue.qsize(), out_queue.qsize()))
            time.sleep(10)

    def xreader():
        in_queue = Queue(buffer_size)
        out_queue = Queue(buffer_size)
        out_order = [0]
        # start a read worker in a thread
        target = order_read_worker if order else read_worker
        t = Thread(target=target, args=(reader, in_queue))
        t.daemon = True
        t.start()
        st_thread = Thread(target=status, args=(in_queue, out_queue))
        st_thread.daemon = True
        st_thread.start()
        # start several handle_workers
        target = order_handle_worker if order else handle_worker
        args = (in_queue, out_queue, mapper, out_order) if order else (
            in_queue, out_queue, mapper)
        workers = []
        for i in xrange(process_num):
            worker = Thread(target=target, args=args)
            worker.daemon = True
            workers.append(worker)
        for w in workers:
            w.start()

        sample = out_queue.get()
        while not isinstance(sample, XmapEndSignal):
            yield sample
            sample = out_queue.get()
        finish = 1
        while finish < process_num:
            sample = out_queue.get()
            if isinstance(sample, XmapEndSignal):
                finish += 1
            else:
                yield sample

    return xreader


def _reader_creator(file_list,
                    mode,
                    shuffle=False,
                    color_jitter=False,
                    rotate=False):
    def reader():
        with open(file_list) as flist:
            full_lines = [line.strip() for line in flist]
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
            trainer_count = int(os.getenv("PADDLE_TRAINERS"))
            per_node_lines = len(full_lines) / trainer_count
            lines = full_lines[trainer_id * per_node_lines:(trainer_id + 1) *
                               per_node_lines]
            print("read images from %d, length: %d, lines length: %d" %
                  (trainer_id * per_node_lines, per_node_lines, len(lines)))
            if shuffle:
                random.shuffle(lines)
            for line in lines:
                if mode == 'train':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(DATA_DIR, "train", img_path)
                    yield img_path, int(label)
                elif mode == 'val':
                    img_path, label = line.split()
                    img_path = img_path.replace("JPEG", "jpeg")
                    img_path = os.path.join(DATA_DIR, "val", img_path)
                    yield img_path, int(label)
                elif mode == 'test':
                    img_path = os.path.join(DATA_DIR, line)
                    yield [img_path]

    mapper = functools.partial(
        process_image, mode=mode, color_jitter=color_jitter, rotate=rotate)

    return xmap_readers(mapper, reader, THREAD, BUF_SIZE)


def train(file_list=TRAIN_LIST):
    return _reader_creator(
        file_list, 'train', shuffle=True, color_jitter=False, rotate=False)


def val(file_list=TEST_LIST):
    return _reader_creator(file_list, 'val', shuffle=False)


def test(file_list=TEST_LIST):
    return _reader_creator(file_list, 'test', shuffle=False)
