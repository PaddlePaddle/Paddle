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
# limitations under the License

from functools import reduce

import paddle
from paddle.fluid.framework import Variable
from paddle.distributed.auto_parallel.dist_tensor import DistributedTensor

from .base_cost import Cost


class TensorCost:
    def __init__(self, tensor=None, dist_tensor=None, shape=None, dtype=None):
        self._check_args(tensor, dist_tensor, shape, dtype)
        self._tensor = tensor
        self._dist_tensor = dist_tensor
        self._shape = shape
        self._dtype = dtype
        self._cost = self.calc_cost()

    @property
    def tensor(self):
        return self._tensor

    @property
    def dist_tensor(self):
        return self._dist_tensor

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def _check_args(self, tensor, dist_tensor, shape, dtype):
        if tensor is not None:
            assert shape is None and dist_tensor is None and dtype is None

            if not isinstance(tensor, Variable):
                raise TypeError(
                    "Please check tensor type is Variable, but got {}".format(
                        type(tensor)
                    )
                )

        elif dist_tensor is not None:
            assert tensor is None and shape is None
            if not isinstance(dist_tensor, DistributedTensor):
                raise TypeError(
                    "Please check dist_tensor type is DistributedTensor, but got {}".format(
                        type(dist_tensor)
                    )
                )

        elif shape is not None:
            assert tensor is None and dist_tensor is None and dtype is not None
            if not isinstance(shape, (list, set)):
                raise TypeError(
                    "Please check shape type is list or set, but got {}".format(
                        type(shape)
                    )
                )

        elif dtype is not None:
            assert tensor is None and dist_tensor is None and shape is not None

    @property
    def cost(self):
        return self._cost

    def calc_cost(self):
        dtype = None
        shape = None

        if self.dist_tensor:
            shape = self.dist_tensor.local_sizes()
            dtype = self.dist_tensor.serial_tensor.dtype
        elif self.tensor:
            shape = self.tensor.shape
            dtype = self.tensor.dtype
        elif self.shape and self.dtype:
            shape = self.shape
            dtype = self.dtype

        total_count = reduce(lambda x, y: x * y, shape)

        if dtype == paddle.float32 or dtype == paddle.int32:
            dtype_factor = 4
        elif dtype == paddle.int64:
            dtype_factor = 8
        elif dtype == paddle.uint8:
            dtype_factor = 1
        else:
            dtype_factor = 2

        memory = total_count * dtype_factor
        assert memory >= 0
        cost = Cost(memory=memory)

        return cost
