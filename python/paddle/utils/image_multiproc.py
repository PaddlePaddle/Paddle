#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os, sys
import numpy as np
from PIL import Image
from cStringIO import StringIO
import multiprocessing
import functools
import itertools

from paddle.utils.image_util import *
from paddle.trainer.config_parser import logger

try:
    import cv2
except ImportError:
    logger.warning("OpenCV2 is not installed, using PIL to process")
    cv2 = None

__all__ = ["CvTransformer", "PILTransformer", "MultiProcessImageTransformer"]


class CvTransformer(ImageTransformer):
    """
    CvTransformer used python-opencv to process image.
    """

    def __init__(
            self,
            min_size=None,
            crop_size=None,
            transpose=(2, 0, 1),  # transpose to C * H * W
            channel_swap=None,
            mean=None,
            is_train=True,
            is_color=True):
        ImageTransformer.__init__(self, transpose, channel_swap, mean, is_color)
        self.min_size = min_size
        self.crop_size = crop_size
        self.is_train = is_train

    def resize(self, im, min_size):
        row, col = im.shape[:2]
        new_row, new_col = min_size, min_size
        if row > col:
            new_row = min_size * row / col
        else:
            new_col = min_size * col / row
        im = cv2.resize(im, (new_row, new_col), interpolation=cv2.INTER_CUBIC)
        return im

    def crop_and_flip(self, im):
        """
        Return cropped image.
        The size of the cropped image is inner_size * inner_size.
        im: (H x W x K) ndarrays
        """
        row, col = im.shape[:2]
        start_h, start_w = 0, 0
        if self.is_train:
            start_h = np.random.randint(0, row - self.crop_size + 1)
            start_w = np.random.randint(0, col - self.crop_size + 1)
        else:
            start_h = (row - self.crop_size) / 2
            start_w = (col - self.crop_size) / 2
        end_h, end_w = start_h + self.crop_size, start_w + self.crop_size
        if self.is_color:
            im = im[start_h:end_h, start_w:end_w, :]
        else:
            im = im[start_h:end_h, start_w:end_w]
        if (self.is_train) and (np.random.randint(2) == 0):
            if self.is_color:
                im = im[:, ::-1, :]
            else:
                im = im[:, ::-1]
        return im

    def transform(self, im):
        im = self.resize(im, self.min_size)
        im = self.crop_and_flip(im)
        # transpose, swap channel, sub mean
        im = im.astype('float32')
        ImageTransformer.transformer(self, im)
        return im

    def load_image_from_string(self, data):
        flag = cv2.CV_LOAD_IMAGE_COLOR if self.is_color else cv2.CV_LOAD_IMAGE_GRAYSCALE
        im = cv2.imdecode(np.fromstring(data, np.uint8), flag)
        return im

    def transform_from_string(self, data):
        im = self.load_image_from_string(data)
        return self.transform(im)

    def load_image_from_file(self, file):
        flag = cv2.CV_LOAD_IMAGE_COLOR if self.is_color else cv2.CV_LOAD_IMAGE_GRAYSCALE
        im = cv2.imread(file, flag)
        return im

    def transform_from_file(self, file):
        im = self.load_image_from_file(file)
        return self.transform(im)


class PILTransformer(ImageTransformer):
    """
    PILTransformer used PIL to process image.
    """

    def __init__(
            self,
            min_size=None,
            crop_size=None,
            transpose=(2, 0, 1),  # transpose to C * H * W
            channel_swap=None,
            mean=None,
            is_train=True,
            is_color=True):
        ImageTransformer.__init__(self, transpose, channel_swap, mean, is_color)
        self.min_size = min_size
        self.crop_size = crop_size
        self.is_train = is_train

    def resize(self, im, min_size):
        row, col = im.size[:2]
        new_row, new_col = min_size, min_size
        if row > col:
            new_row = min_size * row / col
        else:
            new_col = min_size * col / row
        im = im.resize((new_row, new_col), Image.ANTIALIAS)
        return im

    def crop_and_flip(self, im):
        """
        Return cropped image.
        The size of the cropped image is inner_size * inner_size.
        """
        row, col = im.size[:2]
        start_h, start_w = 0, 0
        if self.is_train:
            start_h = np.random.randint(0, row - self.crop_size + 1)
            start_w = np.random.randint(0, col - self.crop_size + 1)
        else:
            start_h = (row - self.crop_size) / 2
            start_w = (col - self.crop_size) / 2
        end_h, end_w = start_h + self.crop_size, start_w + self.crop_size
        im = im.crop((start_h, start_w, end_h, end_w))
        if (self.is_train) and (np.random.randint(2) == 0):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return im

    def transform(self, im):
        im = self.resize(im, self.min_size)
        im = self.crop_and_flip(im)
        im = np.array(im, dtype=np.float32)  # convert to numpy.array
        # transpose, swap channel, sub mean
        ImageTransformer.transformer(self, im)
        return im

    def load_image_from_string(self, data):
        im = Image.open(StringIO(data))
        return im

    def transform_from_string(self, data):
        im = self.load_image_from_string(data)
        return self.transform(im)

    def load_image_from_file(self, file):
        im = Image.open(file)
        return im

    def transform_from_file(self, file):
        im = self.load_image_from_file(file)
        return self.transform(im)


def job(is_img_string, transformer, (data, label)):
    if is_img_string:
        return transformer.transform_from_string(data), label
    else:
        return transformer.transform_from_file(data), label


class MultiProcessImageTransformer(object):
    def __init__(self,
                 procnum=10,
                 resize_size=None,
                 crop_size=None,
                 transpose=(2, 0, 1),
                 channel_swap=None,
                 mean=None,
                 is_train=True,
                 is_color=True,
                 is_img_string=True):
        """
        Processing image with multi-process. If it is used in PyDataProvider,
        the simple usage for CNN is as follows:
       
        .. code-block:: python

            def hool(settings, is_train,  **kwargs):
                settings.is_train = is_train
                settings.mean_value = np.array([103.939,116.779,123.68], dtype=np.float32)
                settings.input_types = [
                    dense_vector(3 * 224 * 224),
                    integer_value(1)]
                settings.transformer = MultiProcessImageTransformer(
                    procnum=10,
                    resize_size=256,
                    crop_size=224,
                    transpose=(2, 0, 1),
                    mean=settings.mean_values,
                    is_train=settings.is_train)


            @provider(init_hook=hook, pool_size=20480)
            def process(settings, file_list):
                with open(file_list, 'r') as fdata:
                    for line in fdata: 
                        data_dic = np.load(line.strip()) # load the data batch pickled by Pickle.
                        data = data_dic['data']
                        labels = data_dic['label']
                        labels = np.array(labels, dtype=np.float32)
                        for im, lab in settings.dp.run(data, labels):
                            yield [im.astype('float32'), int(lab)]

        :param procnum: processor number.
        :type procnum: int
        :param resize_size: the shorter edge size of image after resizing.
        :type resize_size: int
        :param crop_size: the croping size.
        :type crop_size: int
        :param transpose: the transpose order, Paddle only allow C * H * W order.
        :type transpose: tuple or list
        :param channel_swap: the channel swap order, RGB or BRG.
        :type channel_swap: tuple or list
        :param mean: the mean values of image, per-channel mean or element-wise mean.
        :type mean: array, The dimension is 1 for per-channel mean.
                    The dimension is 3 for element-wise mean. 
        :param is_train: training peroid or testing peroid.
        :type is_train: bool.
        :param is_color: the image is color or gray. 
        :type is_color: bool.
        :param is_img_string: The input can be the file name of image or image string.
        :type is_img_string: bool.
        """

        self.procnum = procnum
        self.pool = multiprocessing.Pool(procnum)
        self.is_img_string = is_img_string
        if cv2 is not None:
            self.transformer = CvTransformer(resize_size, crop_size, transpose,
                                             channel_swap, mean, is_train,
                                             is_color)
        else:
            self.transformer = PILTransformer(resize_size, crop_size, transpose,
                                              channel_swap, mean, is_train,
                                              is_color)

    def run(self, data, label):
        fun = functools.partial(job, self.is_img_string, self.transformer)
        return self.pool.imap_unordered(
            fun, itertools.izip(data, label), chunksize=100 * self.procnum)
