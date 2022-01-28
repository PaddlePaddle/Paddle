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

import paddle
from paddle.common_ops_import import *
from paddle import _C_ops

__all__ = [ #noqa
    'file_label_loader',
    'file_label_reader',
]


class _Sampler(object):
    def __init__(self, batch_size, num_samples,
                 shuffle=False, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = num_samples
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

    def get(self, sample_id, batch_size, num_samples,
            shuffle=False, drop_last=False):
        if sample_id in self.samplers:
            return self.samplers[sample_id]

        sampler = _Sampler(batch_size, num_samples,
                           shuffle, drop_last)
        self.samplers[sample_id] = sampler
        return sampler


_sampler_manager = _SamplerManager()


def file_label_loader(data_root, indices, name=None):
    """
    Reads a batch of data, outputs the bytes contents of a file
    as a uint8 Tensor with one dimension.

    Args:
        data_root (str): root directory of data
        indices (list of int): batch indices of samples
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    """
    from paddle.vision.datasets import DatasetFolder
    data_folder = DatasetFolder(data_root)
    samples = [s[0] for s in data_folder.samples]
    targets = [s[1] for s in data_folder.samples]

    if in_dygraph_mode():
        image = core.VarBase(core.VarDesc.VarType.UINT8, [],
                             unique_name.generate("file_label_loader"),
                             core.VarDesc.VarType.LOD_TENSOR_ARRAY, False)
        return _C_ops.file_label_loader(indices, image, 'files',
                                        samples, 'labels', targets)

    inputs = {"Indices": indices}
    attrs = {
        'files': samples,
        'labels': targets,
    }

    helper = LayerHelper("file_label_loader", **locals())
    image = helper.create_variable(
        name=unique_name.generate("file_label_loader"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype='uint8')
    
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


def file_label_reader(file_root,
                      batch_size=1,
                      shuffle=False,
                      drop_last=False):
    """
    Reads and outputs the bytes contents of a file as a uint8 Tensor
    with one dimension.

    Args:
        filename (str): Path of the file to be read.
        name (str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.

    Returns:
        A uint8 tensor.

    Examples:
        .. code-block:: python

            import cv2
            import paddle

            image = paddle.vision.ops.file_label_reader('/workspace/datasets/ILSVRC2012/val/', 2)

    """
    from paddle.vision.datasets import DatasetFolder
    data_folder = DatasetFolder(file_root)
    samples = [s[0] for s in data_folder.samples]
    targets = [s[1] for s in data_folder.samples]

    if in_dygraph_mode():
        sample_id = utils._hash_with_id(file_root, batch_size,
                                        shuffle, drop_last)
        sampler = _sampler_manager.get(sample_id,
                                       batch_size=batch_size,
                                       num_samples=len(samples),
                                       shuffle=shuffle,
                                       drop_last=drop_last)
        indices = paddle.to_tensor(next(sampler), dtype='int64')
        return file_label_loader(file_root, indices)

    def _reader(indices):
        return file_label_loader(file_root, indices)

    return paddle.io.data_reader(_reader,
                                 batch_size=batch_size,
                                 num_samples=len(samples),
                                 shuffle=shuffle,
                                 drop_last=drop_last)
    # inputs = dict()
    # attrs = {
    #     'root_dir': file_root,
    #     'batch_size': batch_size,
    #     'files': samples,
    #     'labels': targets,
    #     'reader_id': unq_reader_id,
    # }
    #
    # helper = LayerHelper("file_label_reader", **locals())
    # out = helper.create_variable(
    #     name=unique_name.generate("file_label_reader"),
    #     type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
    #     dtype='uint8')
    #
    # label = helper.create_variable(
    #     name=unique_name.generate("file_label_reader"),
    #     type=core.VarDesc.VarType.LOD_TENSOR,
    #     dtype='int')
    #
    # helper.append_op(
    #     type="file_label_reader",
    #     inputs=inputs,
    #     attrs=attrs,
    #     outputs={"Out": out,
    #              "Label": label
    #              })

    return out, label

