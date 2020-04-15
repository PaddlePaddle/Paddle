# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance

try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO

from paddle.io import Dataset

import logging
logger = logging.getLogger(__name__)

__all__ = ['KineticsDataset']

KINETICS_CLASS_NUM = 400


class KineticsDataset(Dataset):
    """
    Kinetics dataset

    Args:
        file_list (str): path to file list
        pickle_dir (str): path to pickle file directory
        label_list (str): path to label_list file, if set None, the
            default class number 400 of kinetics dataset will be
            used. Default None
        mode (str): 'train' or 'val' mode, segmentation methods will
            be different in these 2 modes. Default 'train'
        seg_num (int): segment number to sample from each video.
            Default 8
        seg_len (int): frame number of each segment. Default 1
        transform (callable): transforms to perform on video samples,
            None for no transforms. Default None.
    """

    def __init__(self,
                 file_list=None,
                 pickle_dir=None,
                 pickle_file=None,
                 label_list=None,
                 mode='train',
                 seg_num=8,
                 seg_len=1,
                 transform=None):
        assert str.lower(mode) in ['train', 'val', 'test'], \
                "mode can only be 'train' 'val' or 'test'"
        self.mode = str.lower(mode)

        if self.mode in ['train', 'val']:
            assert os.path.isfile(file_list), \
                    "file_list {} not a file".format(file_list)
            with open(file_list) as f:
                self.pickle_paths = [l.strip() for l in f]

            assert os.path.isdir(pickle_dir), \
                    "pickle_dir {} not a directory".format(pickle_dir)
            self.pickle_dir = pickle_dir
        else:
            assert os.path.isfile(pickle_file), \
                    "pickle_file {} not a file".format(pickle_file)
            self.pickle_dir = ''
            self.pickle_paths = [pickle_file]

        self.label_list = label_list
        if self.label_list is not None:
            assert os.path.isfile(self.label_list), \
                "label_list {} not a file".format(self.label_list)
            with open(self.label_list) as f:
                self.label_list = [int(l.strip()) for l in f]

        self.seg_num = seg_num
        self.seg_len = seg_len
        self.transform = transform

    def __len__(self):
        return len(self.pickle_paths)

    def __getitem__(self, idx):
        pickle_path = os.path.join(self.pickle_dir, self.pickle_paths[idx])

        if six.PY2:
            data = pickle.load(open(pickle_path, 'rb'))
        else:
            data = pickle.load(open(pickle_path, 'rb'), encoding='bytes')

        vid, label, frames = data

        if self.label_list is not None:
            label = self.label_list.index(label)
        imgs = self._video_loader(frames)

        if self.transform:
            imgs, label = self.transform(imgs, label)
        return imgs, np.array([label])

    @property
    def num_classes(self):
        return KINETICS_CLASS_NUM if self.label_list is None \
                else len(self.label_list)

    def _video_loader(self, frames):
        videolen = len(frames)
        average_dur = int(videolen / self.seg_num)
        
        imgs = []
        for i in range(self.seg_num):
            idx = 0
            if self.mode == 'train':
                if average_dur >= self.seg_len:
                    idx = random.randint(0, average_dur - self.seg_len)
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            else:
                if average_dur >= self.seg_len:
                    idx = (average_dur - self.seg_len) // 2
                    idx += i * average_dur
                elif average_dur >= 1:
                    idx += i * average_dur
                else:
                    idx = i
            
            for jj in range(idx, idx + self.seg_len):
                imgbuf = frames[int(jj % videolen)]
                img = self._imageloader(imgbuf)
                imgs.append(img)
        
        return imgs

    def _imageloader(self, buf):
        if isinstance(buf, str):
            img = Image.open(StringIO(buf))
        else:
            img = Image.open(BytesIO(buf))
        
        return img.convert('RGB')

