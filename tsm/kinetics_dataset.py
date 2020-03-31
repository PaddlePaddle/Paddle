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

from paddle.fluid.io import Dataset

import logging
logger = logging.getLogger(__name__)

__all__ = ['KineticsDataset']


class KineticsDataset(Dataset):
    """
    Kinetics dataset

    Args:
        filelist (str): path to file list, default None.
        num_classes (int): class number
    """

    def __init__(self,
                 filelist,
                 pickle_dir,
                 mode='train',
                 seg_num=8,
                 seg_len=1,
                 transform=None):
        assert os.path.isfile(filelist), \
                "filelist {} not a file".format(filelist)
        with open(filelist) as f:
            self.pickle_paths = [l.strip() for l in f]

        assert os.path.isdir(pickle_dir), \
                "pickle_dir {} not a directory".format(pickle_dir)
        self.pickle_dir = pickle_dir

        assert mode in ['train', 'val'], \
                "mode can only be 'train' or 'val'"
        self.mode = mode

        self.seg_num = seg_num
        self.seg_len = seg_len
        self.transform = transform

    def __len__(self):
        return len(self.pickle_paths)

    def __getitem__(self, idx):
        pickle_path = os.path.join(self.pickle_dir, self.pickle_paths[idx])

        try:
            if six.PY2:
                data = pickle.load(open(pickle_path, 'rb'))
            else:
                data = pickle.load(open(pickle_path, 'rb'), encoding='bytes')

            vid, label, frames = data
            if len(frames) < 1:
                logger.error("{} contains no frame".format(pickle_path))
                sys.exit(-1)
        except Exception as e:
            logger.error("Load {} failed: {}".format(pickle_path, e))
            sys.exit(-1)

        label_list = [0, 2, 3, 4, 6, 7, 9, 12, 14, 15]
        label = label_list.index(label)
        imgs = self._video_loader(frames)

        if self.transform:
            imgs, label = self.transform(imgs, label)
        return imgs, np.array([label])

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


if __name__ == "__main__":
    kd = KineticsDataset('/paddle/ssd3/kineteics_mini/val_10.list', '/paddle/ssd3/kineteics_mini/val_10')
    print("KineticsDataset length", len(kd))
    for d in kd:
        print(len(d[0]), d[0][0].size, d[1])
