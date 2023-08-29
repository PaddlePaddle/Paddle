# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

paddle.enable_static()


class TestBuildModule(unittest.TestCase):
    def test_basic_network(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', [4, 4], dtype='float32')
            y = paddle.static.data('y', [4, 4], dtype='float32')
            divide_out = paddle.divide(x, y)
            sum_out = paddle.sum(divide_out)

            exe = paddle.static.Executor()
            x_feed = np.ones([4, 4], dtype=np.float32) * 10
            y_feed = np.ones([4, 4], dtype=np.float32) * 2
            (sum_value,) = exe.run(
                feed={'x': x_feed, 'y': y_feed}, fetch_list=[sum_out]
            )
            self.assertEqual(sum_value, 5 * 4 * 4)


if __name__ == "__main__":
    unittest.main()
