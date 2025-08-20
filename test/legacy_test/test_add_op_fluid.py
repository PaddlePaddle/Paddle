#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import unittest

import numpy as np

os.environ['FLAGS_enable_pir_api'] = '0'
import paddle
from paddle.base import core


class TestPaddleAddNewFeatures(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([3, 5], dtype='float32')
        self.y_np = np.array([2, 3], dtype='float32')
        self.scalar = 2.0
        self.place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )

    def test_static_graph_add_with_alpha(self):
        """test static graph add with alpha and parameter aliases"""
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 2], dtype='float32')
        out1 = paddle.add(x, y, alpha=2)
        out2 = paddle.add(input=x, other=y, alpha=2)

        exe = paddle.static.Executor(self.place)
        res = exe.run(
            feed={
                'x': self.x_np.reshape(1, 2),
                'y': self.y_np.reshape(1, 2),
            },
            fetch_list=[out1, out2],
        )

        expected = self.x_np + self.y_np * 2
        for result in res:
            np.testing.assert_array_equal(result.flatten(), expected)
        paddle.disable_static()

    def test_static_graph_add_with_alpha_1(self):
        paddle.enable_static()
        """Test static graph add with alpha=1 (default behavior)"""
        x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        y = paddle.static.data(name='y', shape=[-1, 2], dtype='float32')
        out = paddle.add(x, y, alpha=1)

        exe = paddle.static.Executor(self.place)
        res = exe.run(
            feed={
                'x': self.x_np.reshape(1, 2),
                'y': self.y_np.reshape(1, 2),
            },
            fetch_list=[out],
        )

        expected = self.x_np + self.y_np
        np.testing.assert_array_equal(res[0].flatten(), expected)
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
