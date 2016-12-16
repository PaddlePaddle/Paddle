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

from paddle.utils.image_util import *
from paddle.trainer.PyDataProvider2 import *


def hook(settings, image_size, crop_size, color, file_list, is_train, **kwargs):
    """
    Description: Init with a list of data file
    file_list is the name list of input files.
    kwargs["load_data_args"] is the value of 'load_data_args'
    which can be set in config.
    Each args is separated by a column.
    image_size: the crop image size.
    mean_meta: the path of the meta file to store the mean image.
    mean_value: can be mean value, not a file.
                can not set mean_meta and mean_value at the same time.
    color: 'color' means a color image. Otherwise, it means a gray image.
    is_train: whether the data provider is used for training.
              Data argumentation might be different for training and testing.
    """
    settings.img_size = image_size
    settings.crop_size = crop_size
    settings.mean_img_size = settings.crop_size
    settings.color = color  # default is color
    settings.is_train = is_train

    settings.is_swap_channel = kwargs.get('swap_channel', None)
    if settings.is_swap_channel is not None:
        settings.swap_channel = settings.is_swap_channel
        settings.is_swap_channel = True

    if settings.color:
        settings.img_input_size = settings.crop_size * settings.crop_size * 3
    else:
        settings.img_input_size = settings.crop_size * settings.crop_size

    settings.file_list = file_list
    settings.mean_meta = kwargs.get('mean_meta', None)
    settings.mean_value = kwargs.get('mean_value', None)
    # can not specify both mean_meta and mean_value.
    assert not (settings.mean_meta and settings.mean_value)
    if not settings.mean_meta:
        settings.mean_value = kwargs.get('mean_value')
        sz = settings.crop_size * settings.crop_size
        settings.img_mean = np.zeros(sz * 3, dtype=np.single)
        for idx, value in enumerate(settings.mean_value):
            settings.img_mean[idx * sz:(idx + 1) * sz] = value
        settings.img_mean = settings.img_mean.reshape(3, settings.crop_size,
                                                      settings.crop_size)

    else:
        settings.img_mean = load_meta(settings.mean_meta,
                                      settings.mean_img_size,
                                      settings.crop_size, settings.color)

    settings.input_types = [
        dense_vector(settings.img_input_size),  # image feature
        integer_value(1)
    ]  # labels

    settings.logger.info('Image short side: %s', settings.img_size)
    settings.logger.info('Crop size: %s', settings.crop_size)
    settings.logger.info('Meta path: %s', settings.mean_meta)
    if settings.is_swap_channel:
        settings.logger.info('swap channel: %s', settings.swap_channel)
    settings.logger.info('DataProvider Initialization finished')


@provider(init_hook=hook, should_shuffle=False)
def processData(settings, file_list):
    """
    The main function for loading data.
    Load the batch, iterate all the images and labels in this batch.
    file_name: the batch file name.
    """
    img_path, lab = file_list.strip().split(' ')
    img = Image.open(img_path)
    img.load()
    img = img.resize((settings.img_size, settings.img_size), Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    # swap channel
    if settings.is_swap_channel:
        img = img[settings.swap_channel, :, :]
    img_feat = preprocess_img(img, settings.img_mean, settings.crop_size,
                              settings.is_train, settings.color)
    yield img_feat.tolist(), int(lab.strip())
