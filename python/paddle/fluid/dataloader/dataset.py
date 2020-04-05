#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle.dataset.common

__all__ = ["Dataset"]


class Dataset(object):
    """
    An abstract class to encapsulates methods and behaviors of datasets.

    All datasets in map-style(dataset samples can be get by a given key)
    should be a subclass of `fluid.io.Dataset`. All subclasses should
    implement following methods:

    :math:`__getitem__`: get sample from dataset with a given index. This
    method is required by reading dataset sample in `fluid.io.DataLoader`.
    :math:`__len__`: return dataset sample number. This method is required
    by some implements of `fluid.io.BatchSampler`

    see `fluid.io.DataLoader`.
    """

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))
