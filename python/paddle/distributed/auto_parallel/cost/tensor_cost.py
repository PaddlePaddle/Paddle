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

from .base_cost import Cost


class TensorCost:
    def __init__(self, tensor, dist_context):
        self.check_tensor(tensor)
        self._tensor = tensor
        self._cost = self.calc_cost()

    def check_tensor(self, tensor):
        if not isinstance(tensor, Variable):
            raise TypeError("Please check tensor type is Variable, but got {}".
                            format(type(tensor)))

    @property
    def cost(self):
        return self._cost

    def calc_cost(self):
        dtype = tensor.dtype
        shape = None
        dist_tensor = dist_context.get_dist_tensor_for_program(tensor)
        if dist_tensor:
            shape = dist_tensor.local_sizes()
        else:
            shape = tensor.shape

        total_count = reduce(lambda x, y: x * y, shape)

        if dtype == paddle.float32 or dtype == paddle.int32:
            dtype_factor = 4
        elif node.dtype == paddle.int64:
            dtype_factor = 8
        elif node.dtype == paddle.uint8:
            dtype_factor = 1
        else:
            dtype_factor = 2

        memory = total_count * dtype_factor
        assert memory > 0
        cost = Cost(memory=memory)

        return cost
