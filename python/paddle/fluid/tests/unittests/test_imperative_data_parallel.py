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

from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.nn import Linear
import paddle.fluid.core as core


class MLP(fluid.Layer):
    def __init__(self, param_attr=None, bias_attr=None):
        super(MLP, self).__init__()

        self._linear1 = Linear(784, 10)
        self._linear2 = Linear(10, 10)

    def forward(self, inputs):
        y = self._linear1(inputs)
        y = self._linear2(y)
        return y


class TestDataParallelStateDict(unittest.TestCase):
    def test_data_parallel_state_dict(self):
        with fluid.dygraph.guard():
            strategy = paddle.imperative.prepare_context()
            mlp = MLP()
            parallel_mlp = dygraph.parallel.DataParallel(mlp, strategy)

            single_state = mlp.state_dict()
            parallel_state = parallel_mlp.state_dict()

            base_para = {}
            place = fluid.CPUPlace() if not core.is_compiled_with_cuda(
            ) else fluid.CUDAPlace(0)
            for k, v in single_state.items():
                self.assertTrue(k in parallel_state)

                self.assertTrue(
                    np.array_equal(v.numpy(), parallel_state[k].numpy()))

                base_para[k] = v.numpy()

            for k, v in parallel_state.items():
                np_t = v.numpy()
                var = v.value().get_tensor()
                var.set(np.zeros_like(np_t), place)

                self.assertTrue(np.sum(np.abs(v.numpy())) == 0)

            parallel_mlp.set_dict(base_para)

            parallel_state = parallel_mlp.state_dict()

            for k, v in parallel_state.items():
                self.assertTrue(np.array_equal(v.numpy(), base_para[k]))

            parallel_mlp.load_dict(base_para)


if __name__ == '__main__':
    unittest.main()
