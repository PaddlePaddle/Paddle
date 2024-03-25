# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from collections import OrderedDict

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core


class MyLayer(paddle.nn.Layer):
    def __init__(self, name_scope):
        super().__init__(name_scope)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = paddle.multiply(x, x)
        x = paddle.sum(x)
        return [x]


class TestImperativeParallelCoalesceSplit(unittest.TestCase):
    def test_coalesce_split(self):
        from paddle.distributed.parallel import (
            _coalesce_tensors,
            _split_tensors,
        )

        with base.dygraph.guard():
            test_layer = MyLayer("test_layer")
            strategy = core.ParallelStrategy()
            test_layer = paddle.DataParallel(test_layer, strategy)

            # test variables prepare
            vars = []
            vars.append(
                paddle.to_tensor(np.random.random([2, 3]).astype("float32"))
            )
            vars.append(
                paddle.to_tensor(np.random.random([4, 9]).astype("float32"))
            )
            vars.append(
                paddle.to_tensor(np.random.random([10, 1]).astype("float32"))
            )
            var_groups = OrderedDict()
            var_groups.setdefault(0, vars)

            # record shapes
            orig_var_shapes = []
            for var in vars:
                orig_var_shapes.append(var.shape)

            # execute interface
            coalesced_vars = _coalesce_tensors(var_groups)
            _split_tensors(coalesced_vars)

            # compare
            for orig_var_shape, var in zip(orig_var_shapes, vars):
                self.assertEqual(orig_var_shape, var.shape)

    def test_reshape_inplace(self):
        from paddle.distributed.parallel import _reshape_inplace

        with base.dygraph.guard():
            test_layer = MyLayer("test_layer")
            strategy = core.ParallelStrategy()
            test_layer = paddle.DataParallel(test_layer, strategy)

            ori_shape = [2, 25]
            new_shape = [5, 10]
            x_data = np.random.random(ori_shape).astype("float32")
            x = paddle.to_tensor(x_data)
            _reshape_inplace(x, new_shape)
            self.assertEqual(x.shape, new_shape)


if __name__ == '__main__':
    unittest.main()
