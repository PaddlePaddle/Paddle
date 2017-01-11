# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
Some class for prepare cifar-10 image data.
"""
import io
import random

import paddle.trainer.PyDataProvider2 as dataprovider
import paddle.utils.image_util as image_util
from paddle.trainer.PyDataProvider2 import *
from py_paddle import DataProviderConverter


class BatchPool(object):
    """A class to get all data in memory and do data shuffle."""

    def __init__(self, generator, batch_size):
        self.data = list(generator)
        self.batch_size = batch_size

    def __call__(self):
        random.shuffle(self.data)
        for offset in xrange(0, len(self.data), self.batch_size):
            limit = min(offset + self.batch_size, len(self.data))
            yield self.data[offset:limit]


class Cifar10Data(object):
    """
    Class to prepare cifar-10 data, have all the message to read and convert
    image data to the data the can send to paddle to train or test.
    """

    def __init__(self,
                 meta,
                 train_file_list,
                 test_file_list,
                 batch_size,
                 img_size=32,
                 mean_img_size=32,
                 num_classes=10,
                 is_color=True,
                 use_jpeg=True):
        self.mean_img_size = mean_img_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.train_file_list = train_file_list
        self.test_file_list = test_file_list
        self.is_color = is_color
        if self.is_color:
            self.img_raw_size = self.img_size * self.img_size * 3
        else:
            self.img_raw_size = self.img_size * self.img_size
        self.meta_path = meta
        self.use_jpeg = use_jpeg
        self.batch_size = batch_size
        self.img_mean = image_util.load_meta(self.meta_path, self.mean_img_size,
                                             self.img_size, self.is_color)

        # DataProvider Converter is a utility convert Python Object to Paddle C++
        # Input. The input format is as same as Paddle's DataProvider.
        # input_types = {
        #     'image': dp.dense_vector(data_size),
        #              'label': dp.integer_value(label_size)
        # }
        input_types = [
            dataprovider.dense_vector(self.img_raw_size),
            dataprovider.integer_value(self.num_classes)
        ]
        self.data_converter = DataProviderConverter(input_types)

    @staticmethod
    def _input_order_converter(generator):
        for item in generator:
            yield item['image'], item['label']

    @staticmethod
    def generator_to_batch(generator, batch_size):
        ret_val = list()
        for each_item in generator:
            ret_val.append(each_item)
            if len(ret_val) == batch_size:
                yield ret_val
                ret_val = list()
        if len(ret_val) != 0:
            yield ret_val

    def _read_data(self, is_train):
        """
        The main function for loading data.
        Load the batch, iterate all the images and labels in this batch.
        file_list: the batch file list.
        """
        if is_train:
            file_list = self.train_file_list
        else:
            file_list = self.test_file_list

        with open(file_list, 'r') as fdata:
            lines = [line.strip() for line in fdata]
            random.shuffle(lines)
            for file_name in lines:
                with io.open(file_name.strip(), 'rb') as file:
                    data = cPickle.load(file)
                    indexes = list(range(len(data['images'])))
                    if is_train:
                        random.shuffle(indexes)
                    for i in indexes:
                        if self.use_jpeg == 1:
                            img = image_util.decode_jpeg(data['images'][i])
                        else:
                            img = data['images'][i]
                        img_feat = image_util.preprocess_img(
                            img, self.img_mean, self.img_size, is_train,
                            self.is_color)
                        label = data['labels'][i]
                        yield {
                            'image': img_feat.astype('float32'),
                            'label': int(label)
                        }

    def train_data(self):
        """
        Get Train Data.
        TrainData will stored in a data pool. Currently implementation is not care
        about memory, speed. Just a very naive implementation.
        """
        train_data_generator = self._input_order_converter(
            self._read_data(True))
        train_data = BatchPool(train_data_generator, self.batch_size)
        return train_data

    def test_data(self):
        """
        Get Test Data.
        TrainData will stored in a data pool. Currently implementation is not care
        about memory, speed. Just a very naive implementation.
        """
        test_data_generator = self._input_order_converter(
            self._read_data(False))
        test_data = self.generator_to_batch(test_data_generator,
                                            self.batch_size)
        return test_data
