#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from ..fluid.layer_helper import LayerHelper, unique_name
from ..fluid import core, layers
from ..fluid.layers import nn, utils
from ..fluid.framework import _non_static_mode

import paddle
from paddle.common_ops_import import *
from paddle import _C_ops

__all__ = [  #noqa
    'file_label_loader',
    'file_label_reader',
]


class _Sampler(object):
    def __init__(self, batch_size, num_samples, shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.start_idx = 0

        self.sample_ids = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(self.sample_ids)

    def __next__(self):
        if self.start_idx >= self.num_samples:
            self.reset()
            return self.__next__()

        batch_len = min(self.batch_size, self.num_samples - self.start_idx)
        indices = self.sample_ids[self.start_idx:self.start_idx + batch_len]
        self.start_idx += batch_len

        if self.drop_last and len(indices) < self.batch_size:
            self.reset()
            return self.__next__()

        return indices

    def reset(self):
        self.start_idx = 0
        if self.shuffle:
            np.random.shuffle(self.sample_ids)


class _SamplerManager(object):
    def __init__(self):
        self.samplers = {}

    def get(self,
            sample_id,
            batch_size,
            num_samples,
            shuffle=False,
            drop_last=False):
        if sample_id in self.samplers:
            return self.samplers[sample_id]

        sampler = _Sampler(batch_size, num_samples, shuffle, drop_last)
        self.samplers[sample_id] = sampler
        return sampler


_sampler_manager = _SamplerManager()


def file_label_loader(data_root, indices, batch_size, name=None):
    """
    Reads a batch of data, outputs the bytes contents of a file
    as a uint8 Tensor with one dimension.

    .. note::
        This API can only be used in Paddle GPU version.

    Args:
        data_root (str): root directory of ImageNet format dataset.
        indices (Tensor): A Tensor of batch indices of samples in shape of 
            [N], while N is the batch size.
        batch_size (int): The batch size, same as shape of indices.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        A list of image Tensor holds byte streams of a batch of images and
        A Tensor of label Tensor.

    Examples:
        .. code-block:: python
          :name: code-example

            import os
            import paddle
            from paddle.utils.download import get_path_from_url

            DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
            DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
            DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
            BATCH_SIZE = 16

            data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                          DATASET_MD5)
            indices = paddle.arange(BATCH_SIZE)
            images, labels = paddle.vision.reader.file_label_loader(
                                    data_root, indices, BATCH_SIZE)
            print(images[0].shape, labels.shape)

    """

    if _non_static_mode():
        image = [
            core.VarBase(core.VarDesc.VarType.UINT8, [],
                         unique_name.generate("file_label_loader"),
                         core.VarDesc.VarType.LOD_TENSOR, False)
            for i in range(batch_size)
        ]
        return _C_ops.file_label_loader(indices, image, 'data_root', data_root)

    inputs = {"Indices": indices}
    attrs = {'data_root': data_root, }

    helper = LayerHelper("file_label_loader", **locals())
    image = [
        helper.create_variable(
            name=unique_name.generate("file_label_loader"),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype='uint8') for i in range(batch_size)
    ]

    label = helper.create_variable(
        name=unique_name.generate("file_label_loader"),
        type=core.VarDesc.VarType.LOD_TENSOR,
        dtype='int')

    helper.append_op(
        type="file_label_loader",
        inputs=inputs,
        attrs=attrs,
        outputs={"Image": image,
                 "Label": label})

    return image, label


def file_label_reader(data_root,
                      batch_size=1,
                      shuffle=False,
                      drop_last=False,
                      seed=None,
                      name=None):
    """
    Reads batches of data iterably, outputs the bytes contents of a file
    as a uint8 Tensor with one dimension.

    This API will start a C++ thread to load data with
    :attr:`file_label_loader`, and yiled data iterably.

    .. note::
        This API can only be used in Paddle GPU version.

    Args:
        data_root (str): root directory of ImageNet dataset.
        batch_size (int, optional): The batch size of a mini-batch.
            Default 1.
        shuffle (bool, optional): Whether to shuffle samples. Default False.
        drop_last (bool, optional): Whether to drop the last incomplete
            batch. Default False.
        seed (int, optional): The seed for sample shuffling. Default None.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        A list of image Tensor holds byte streams of a batch of images and
        A Tensor of label Tensor.

    Examples:
        .. code-block:: python
          :name: code-example

            import os
            import paddle
            from paddle.utils.download import get_path_from_url

            DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
            DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
            DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
            BATCH_SIZE = 16

            data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                          DATASET_MD5)
            images, labels = paddle.vision.reader.file_label_reader(
                                    data_root, BATCH_SIZE)
            print(images[0].shape, labels.shape)

    """

    from paddle.vision.datasets import DatasetFolder
    data_folder = DatasetFolder(data_root)
    samples = [s[0] for s in data_folder.samples]
    targets = [s[1] for s in data_folder.samples]

    if _non_static_mode():
        sample_id = utils._hash_with_id(data_root, batch_size, shuffle,
                                        drop_last)
        sampler = _sampler_manager.get(sample_id,
                                       batch_size=batch_size,
                                       num_samples=len(samples),
                                       shuffle=shuffle,
                                       drop_last=drop_last)
        indices = paddle.to_tensor(next(sampler), dtype='int64')
        return file_label_loader(data_root, indices, batch_size)

    def _reader(indices):
        return file_label_loader(data_root, indices, batch_size)

    outs = paddle.io.data_reader(
        _reader,
        batch_size=batch_size,
        num_samples=len(samples),
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed)
    return outs[:-1], outs[-1]
