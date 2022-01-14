# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
import paddle
import numpy as np
from paddle.fluid.framework import _test_eager_guard
import unittest


class EagerOpAPIGenerateTestCase(unittest.TestCase):
    def test_elementwise_add(self):
        with _test_eager_guard():
            paddle.set_device("cpu")
            np_x = np.ones([4, 16, 16, 32]).astype('float32')
            np_y = np.ones([4, 16, 16, 32]).astype('float32')
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            out = paddle.add(x, y)
            out_arr = out.numpy()

            out_arr_expected = np.add(np_x, np_y)
            self.assertTrue(np.array_equal(out_arr, out_arr_expected))

    def test_sum(self):
        with _test_eager_guard():
            x_data = np.array(
                [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]]).astype('float32')
            x = paddle.to_tensor(x_data, 'float32')
            out = paddle.sum(x, axis=0)
            out_arr = out.numpy()
            out_arr_expected = np.sum(x_data, axis=0)
            self.assertTrue(np.array_equal(out_arr, out_arr_expected))

    def test_mm(self):
        with _test_eager_guard():
            np_input = np.random.random([16, 32]).astype('float32')
            np_mat2 = np.random.random([32, 32]).astype('float32')
            input = paddle.to_tensor(np_input)
            mat2 = paddle.to_tensor(np_mat2)
            out = paddle.mm(input, mat2)
            out_arr = out.numpy()
            out_arr_expected = np.matmul(np_input, np_mat2)
            self.assertTrue(np.allclose(out_arr, out_arr_expected))

    def test_sigmoid(self):
        with _test_eager_guard():
            np_x = np.array([-0.4, -0.2, 0.1, 0.3]).astype('float32')
            x = paddle.to_tensor(np_x)
            out = paddle.nn.functional.sigmoid(x)
            out_arr = out.numpy()
            out_arr_expected = np.array(
                [0.40131234, 0.450166, 0.52497919, 0.57444252]).astype(
                    'float32')
            self.assertTrue(np.allclose(out_arr, out_arr_expected))


if __name__ == "__main__":
    unittest.main()
