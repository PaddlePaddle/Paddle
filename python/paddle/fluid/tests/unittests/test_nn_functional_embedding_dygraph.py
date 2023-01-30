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

<<<<<<< HEAD
import unittest

import numpy as np

import paddle
=======
from __future__ import print_function

import unittest

import paddle
import paddle.nn as nn
import numpy as np
from paddle.fluid.framework import _test_eager_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.disable_static()


class EmbeddingDygraph(unittest.TestCase):
<<<<<<< HEAD
    def test_1(self):
=======

    def func_1(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)

        embedding = paddle.nn.Embedding(10, 3, sparse=True, padding_idx=9)

        w0 = np.full(shape=(10, 3), fill_value=2).astype(np.float32)
        embedding.weight.set_value(w0)

<<<<<<< HEAD
        adam = paddle.optimizer.Adam(
            parameters=[embedding.weight], learning_rate=0.01
        )
=======
        adam = paddle.optimizer.Adam(parameters=[embedding.weight],
                                     learning_rate=0.01)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        adam.clear_grad()

        out = embedding(x)
        out.backward()
        adam.step()

<<<<<<< HEAD
    def test_2(self):
=======
    def test_1(self):
        with _test_eager_guard():
            self.func_1()
        self.func_1()

    def func_2(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x_data = np.arange(3, 6).reshape((3, 1)).astype(np.int64)
        y_data = np.arange(6, 12).reshape((3, 2)).astype(np.float32)
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(x_data, stop_gradient=False)
        y = paddle.to_tensor(y_data, stop_gradient=False)

        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(10, 3, padding_idx=11, sparse=True)

        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(-1, 3, sparse=True)

        with self.assertRaises(ValueError):
            embedding = paddle.nn.Embedding(10, -3, sparse=True)

<<<<<<< HEAD
=======
    def test_2(self):
        with _test_eager_guard():
            self.func_2()
        self.func_2()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if __name__ == '__main__':
    unittest.main()
