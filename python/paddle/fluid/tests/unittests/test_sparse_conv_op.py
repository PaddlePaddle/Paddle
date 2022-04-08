# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import numpy as np
import paddle
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard


class TestSparseConv(unittest.TestCase):
    def test_conv3d(self):
        with _test_eager_guard():
            kernel = [[[[[1], [1], [1]], [[1], [1], [1]], [[1], [1], [1]]]]]
            dense_kernel = paddle.to_tensor(
                kernel, dtype='float32', stop_gradient=False)
            dense_kernel = paddle.reshape(dense_kernel, [1, 3, 3, 1, 1])
            paddings = [0, 0, 0]
            strides = [1, 1, 1]
            dilations = [1, 1, 1]

            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [1, 2, 3, 4]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            correct_out_values = [[4], [10]]
            sparse_input = core.eager.sparse_coo_tensor(indices, values,
                                                        dense_shape, False)
            out = _C_ops.final_state_sparse_conv3d(sparse_input, dense_kernel,
                                                   paddings, dilations, strides,
                                                   1, False)
            out.backward(out)
            #At present, only backward can be verified to work normally
            #TODO(zhangkaihuo): compare the result with dense conv
            print(sparse_input.grad.values())
            assert np.array_equal(correct_out_values, out.values().numpy())


#TODO: Add more test case
