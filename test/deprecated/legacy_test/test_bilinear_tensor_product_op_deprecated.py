#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import paddle_static_guard

import paddle
from paddle import base


class TestDygraphBilinearTensorProductAPIError(unittest.TestCase):
    def test_errors(self):
        with paddle_static_guard():
            with base.program_guard(base.Program(), base.Program()):
                layer = paddle.nn.Bilinear(5, 4, 1000)
                # the input must be Variable.
                x0 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                self.assertRaises(TypeError, layer, x0)
                # the input dtype must be float32 or float64
                x1 = paddle.static.data(
                    name='x1', shape=[-1, 5], dtype="float16"
                )
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 4], dtype="float32"
                )
                self.assertRaises(TypeError, layer, x1, x2)
                # the dimensions of x and y must be 2
                paddle.enable_static()
                x3 = paddle.static.data("", shape=[0], dtype="float32")
                x4 = paddle.static.data("", shape=[0], dtype="float32")
                self.assertRaises(
                    ValueError,
                    paddle.static.nn.bilinear_tensor_product,
                    x3,
                    x4,
                    1000,
                )


if __name__ == "__main__":
    unittest.main()
