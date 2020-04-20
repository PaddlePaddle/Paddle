from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from os import path
import random
import traceback
import copy
import math
import tarfile
from PIL import Image

import logging
logger = logging.getLogger(__name__)

import paddle
from paddle import fluid
from paddle.fluid.dygraph.parallel import ParallelEnv

DATA_MD5 = "7256b1d5420d8c3e74815196e58cdad5"
DATA_URL = "http://paddle-ocr-data.bj.bcebos.com/data.tar.gz"
CACHE_DIR_NAME = "attention_data"
SAVED_FILE_NAME = "data.tar.gz"
DATA_DIR_NAME = "data"
TRAIN_DATA_DIR_NAME = "train_images"
TEST_DATA_DIR_NAME = "test_images"
TRAIN_LIST_FILE_NAME = "train.list"
TEST_LIST_FILE_NAME = "test.list"


class Resize(object):
    def __init__(self, height=48):
        self.interp = Image.NEAREST  # Image.ANTIALIAS
        self.height = height

    def __call__(self, samples):
        shape = samples[0][0].size
        for i in range(len(samples)):
            im = samples[i][0]
            im = im.resize((shape[0], self.height), self.interp)
            samples[i][0] = im
        return samples


class Normalize(object):
    def __init__(self,
                 mean=[127.5],
                 std=[1.0],
                 scale=False,
                 channel_first=True):
        self.mean = mean
        self.std = std
        self.scale = scale
        self.channel_first = channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, samples):
        for i in range(len(samples)):
            im = samples[i][0]
            im = np.array(im).astype(np.float32, copy=False)
            im = im[np.newaxis, ...]
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            if self.scale:
                im = im / 255.0
            #im -= mean
            im -= 127.5
            #im /= std
            samples[i][0] = im
        return samples


class PadTarget(object):
    def __init__(self, SOS=0, EOS=1):
        self.SOS = SOS
        self.EOS = EOS

    def __call__(self, samples):
        lens = np.array([len(s[1]) for s in samples], dtype="int64")
        max_len = np.max(lens)
        for i in range(len(samples)):
            label = samples[i][1]
            if max_len > len(label):
                pad_label = label + [self.EOS] * (max_len - len(label))
            else:
                pad_label = label
            samples[i][1] = np.array([self.SOS] + pad_label, dtype='int64')
            # label_out
            samples[i].append(np.array(pad_label + [self.EOS], dtype='int64'))
            mask = np.zeros((max_len + 1)).astype('float32')
            mask[:len(label) + 1] = 1.0
            # mask
            samples[i].append(np.array(mask, dtype='float32'))
        return samples


class BatchSampler(fluid.io.BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 drop_last=True,
                 seed=None):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._random = np.random
        self._random.seed(seed)
        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank
        self._device_id = ParallelEnv().dev_id
        self._num_samples = int(
            math.ceil(len(self._dataset) * 1.0 / self._nranks))
        self._total_size = self._num_samples * self._nranks
        self._epoch = 0

    def __iter__(self):
        infos = copy.copy(self._dataset._sample_infos)
        skip_num = 0
        if self._shuffle:
            if self._batch_size == 1:
                self._random.RandomState(self._epoch).shuffle(infos)
            else:  # partial shuffle
                infos = sorted(infos, key=lambda x: x.w)
                skip_num = random.randint(1, 100)

        infos = infos[skip_num:] + infos[:skip_num]
        infos += infos[:(self._total_size - len(infos))]
        last_size = self._total_size % (self._batch_size * self._nranks)
        batches = []
        for i in range(self._local_rank * self._batch_size,
                       len(infos) - last_size,
                       self._batch_size * self._nranks):
            batches.append(infos[i:i + self._batch_size])

        if (not self._drop_last) and last_size != 0:
            last_local_size = last_size // self._nranks
            last_infos = infos[len(infos) - last_size:]
            start = self._local_rank * last_local_size
            batches.append(last_infos[start:start + last_local_size])

        if self._shuffle:
            self._random.RandomState(self._epoch).shuffle(batches)
            self._epoch += 1

        for batch in batches:
            batch_indices = [info.idx for info in batch]
            yield batch_indices

    def __len__(self):
        if self._drop_last:
            return self._total_size // self._batch_size
        else:
            return math.ceil(self._total_size / float(self._batch_size))


class SampleInfo(object):
    def __init__(self, idx, h, w, im_name, labels):
        self.idx = idx
        self.h = h
        self.w = w
        self.im_name = im_name
        self.labels = labels


class OCRDataset(paddle.io.Dataset):
    def __init__(self, image_dir, anno_file):
        self.image_dir = image_dir
        self.anno_file = anno_file
        self._sample_infos = []
        with open(anno_file, 'r') as f:
            for i, line in enumerate(f):
                w, h, im_name, labels = line.strip().split(' ')
                h, w = int(h), int(w)
                labels = [int(c) for c in labels.split(',')]
                self._sample_infos.append(SampleInfo(i, h, w, im_name, labels))

    def __getitem__(self, idx):
        info = self._sample_infos[idx]
        im_name, labels = info.im_name, info.labels
        image = Image.open(path.join(self.image_dir, im_name)).convert('L')
        return [image, labels]

    def __len__(self):
        return len(self._sample_infos)


def train(
        root_dir=None,
        images_dir=None,
        anno_file=None,
        shuffle=True, ):
    if root_dir is None:
        root_dir = download_data()
    if images_dir is None:
        images_dir = TRAIN_DATA_DIR_NAME
    images_dir = path.join(root_dir, TRAIN_DATA_DIR_NAME)
    if anno_file is None:
        anno_file = TRAIN_LIST_FILE_NAME
    anno_file = path.join(root_dir, TRAIN_LIST_FILE_NAME)
    return OCRDataset(images_dir, anno_file)


def test(
        root_dir=None,
        images_dir=None,
        anno_file=None,
        shuffle=True, ):
    if root_dir is None:
        root_dir = download_data()
    if images_dir is None:
        images_dir = TEST_DATA_DIR_NAME
    images_dir = path.join(root_dir, TEST_DATA_DIR_NAME)
    if anno_file is None:
        anno_file = TEST_LIST_FILE_NAME
    anno_file = path.join(root_dir, TEST_LIST_FILE_NAME)
    return OCRDataset(images_dir, anno_file)


def download_data():
    '''Download train and test data.
    '''
    tar_file = paddle.dataset.common.download(
        DATA_URL, CACHE_DIR_NAME, DATA_MD5, save_name=SAVED_FILE_NAME)
    data_dir = path.join(path.dirname(tar_file), DATA_DIR_NAME)
    if not path.isdir(data_dir):
        t = tarfile.open(tar_file, "r:gz")
        t.extractall(path=path.dirname(tar_file))
        t.close()
    return data_dir
