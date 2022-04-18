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
import paddle.fluid.core as core
from paddle import _C_ops
from paddle.fluid.framework import _test_eager_guard


class TestSparseMaxPool3D(unittest.TestCase):
    def test1(self):
        with _test_eager_guard():
            dense_x = paddle.randn((1, 1, 4, 4, 1))
            print(dense_x)
            sparse_x = dense_x.to_sparse_coo(4)
            kernel_sizes = [1, 3, 3]
            paddings = [0, 0, 0]
            strides = [1, 1, 1]
            dilations = [1, 1, 1]
            out = _C_ops.final_state_sparse_max_pool3d(
                sparse_x, kernel_sizes, paddings, dilations, strides)
            print(out)
