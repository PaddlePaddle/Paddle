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
            bias = [1]

            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [1, 2, 3, 4]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            correct_out_values = [[5], [11]]
            sparse_input = core.eager.sparse_coo_tensor(indices, values,
                                                        dense_shape, False)
            out = paddle.sparse.functional.conv3d(
                sparse_input,
                dense_kernel,
                bias=paddle.to_tensor(
                    bias, dtype='float32'),
                stride=strides,
                padding=paddings,
                dilation=dilations,
                groups=1,
                data_format="NDHWC")
            out.backward(out)
            assert np.array_equal(correct_out_values, out.values().numpy())

    def test_subm_conv3d(self):
        with _test_eager_guard():
            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            sparse_x = paddle.sparse.sparse_coo_tensor(
                indices, values, dense_shape, stop_gradient=True)
            weight = paddle.randn((1, 3, 3, 1, 1), dtype='float32')
            y = paddle.sparse.functional.subm_conv3d(sparse_x, weight)
            assert np.array_equal(sparse_x.indices().numpy(),
                                  y.indices().numpy())

    def test_Conv3D(self):
        with _test_eager_guard():
            #(4, non_zero_num), 4-D:(N, D, H, W)
            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            #(non_zero_num, C)
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            correct_out_values = [[4], [10]]
            sparse_input = paddle.sparse.sparse_coo_tensor(indices, values,
                                                           dense_shape, False)

            sparse_conv3d = paddle.sparse.Conv3D(
                1, 1, (1, 3, 3), data_format='NDHWC')
            sparse_out = sparse_conv3d(sparse_input)
            #test errors
            with self.assertRaises(ValueError):
                #Currently, only support data_format='NDHWC'
                conv3d = paddle.sparse.SubmConv3D(
                    1, 1, (1, 3, 3), data_format='NCDHW')

    def test_SubmConv3D(self):
        with _test_eager_guard():
            indices = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 2], [1, 3, 2, 3]]
            values = [[1], [2], [3], [4]]
            indices = paddle.to_tensor(indices, dtype='int32')
            values = paddle.to_tensor(values, dtype='float32')
            dense_shape = [1, 1, 3, 4, 1]
            correct_out_values = [[4], [10]]
            sparse_input = paddle.sparse.sparse_coo_tensor(indices, values,
                                                           dense_shape, False)

            subm_conv3d = paddle.sparse.SubmConv3D(
                1, 1, (1, 3, 3), data_format='NDHWC')
            # test extra_repr
            print(subm_conv3d.extra_repr())

            sparse_out = subm_conv3d(sparse_input)
            # the output shape of subm_conv is same as input shape
            assert np.array_equal(indices, sparse_out.indices().numpy())

            #test errors
            with self.assertRaises(ValueError):
                #Currently, only support data_format='NDHWC'
                conv3d = paddle.sparse.SubmConv3D(
                    1, 1, (1, 3, 3), data_format='NCDHW')
