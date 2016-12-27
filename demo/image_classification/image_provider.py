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

import io
import random

import paddle.utils.image_util as image_util
from paddle.trainer.PyDataProvider2 import *


#
# {'img_size': 32,
# 'settings': a global object,
# 'color': True,
# 'mean_img_size': 32,
# 'meta': './data/cifar-out/batches/batches.meta',
# 'num_classes': 10,
# 'file_list': ('./data/cifar-out/batches/train_batch_000',),
# 'use_jpeg': True}
def hook(settings, img_size, mean_img_size, num_classes, color, meta, use_jpeg,
         is_train, **kwargs):
    settings.mean_img_size = mean_img_size
    settings.img_size = img_size
    settings.num_classes = num_classes
    settings.color = color
    settings.is_train = is_train

    if settings.color:
        settings.img_raw_size = settings.img_size * settings.img_size * 3
    else:
        settings.img_raw_size = settings.img_size * settings.img_size

    settings.meta_path = meta
    settings.use_jpeg = use_jpeg

    settings.img_mean = image_util.load_meta(settings.meta_path,
                                             settings.mean_img_size,
                                             settings.img_size, settings.color)

    settings.logger.info('Image size: %s', settings.img_size)
    settings.logger.info('Meta path: %s', settings.meta_path)
    settings.input_types = {
        'image': dense_vector(settings.img_raw_size),
        'label': integer_value(settings.num_classes)
    }

    settings.logger.info('DataProvider Initialization finished')


@provider(init_hook=hook, min_pool_size=0)
def processData(settings, file_list):
    """
    The main function for loading data.
    Load the batch, iterate all the images and labels in this batch.
    file_list: the batch file list.
    """
    with open(file_list, 'r') as fdata:
        lines = [line.strip() for line in fdata]
        random.shuffle(lines)
        for file_name in lines:
            with io.open(file_name.strip(), 'rb') as file:
                data = cPickle.load(file)
                indexes = list(range(len(data['images'])))
                if settings.is_train:
                    random.shuffle(indexes)
                for i in indexes:
                    if settings.use_jpeg == 1:
                        img = image_util.decode_jpeg(data['images'][i])
                    else:
                        img = data['images'][i]
                    img_feat = image_util.preprocess_img(
                        img, settings.img_mean, settings.img_size,
                        settings.is_train, settings.color)
                    label = data['labels'][i]
                    yield {
                        'image': img_feat.astype('float32'),
                        'label': int(label)
                    }
