# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestPermuteApi(unittest.TestCase):
    def test_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[2, 3, 4], dtype='float32')

            # function: list / tuple / varargs
            y1 = paddle.permute(x, [1, 0, 2])
            y2 = paddle.permute(x, (2, 1, 0))
            y3 = paddle.permute(x, 1, 2, 0)
            y4 = paddle.permute(x, dims=[1, 2, 0])

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_np = np.random.random([2, 3, 4]).astype("float32")
            out1, out2, out3, out4 = exe.run(
                feed={"x": x_np}, fetch_list=[y1, y2, y3, y4]
            )

            expected1 = np.transpose(x_np, [1, 0, 2])
            expected2 = np.transpose(x_np, (2, 1, 0))
            expected3 = np.transpose(x_np, [1, 2, 0])

            np.testing.assert_array_equal(out1, expected1)
            np.testing.assert_array_equal(out2, expected2)
            np.testing.assert_array_equal(out3, expected3)
            np.testing.assert_array_equal(out4, expected3)

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.randn([2, 3, 4])
        x_np = x.numpy()

        y1 = paddle.permute(x, [1, 0, 2])
        y2 = paddle.permute(x, (2, 1, 0))
        y3 = paddle.permute(x, 1, 2, 0)
        y4 = paddle.permute(x, dims=[1, 2, 0])

        m1 = x.permute([1, 0, 2])
        m2 = x.permute((2, 1, 0))
        m3 = x.permute(1, 2, 0)
        m4 = x.permute(dims=[1, 2, 0])

        expected1 = np.transpose(x_np, [1, 0, 2])
        expected2 = np.transpose(x_np, (2, 1, 0))
        expected3 = np.transpose(x_np, [1, 2, 0])

        np.testing.assert_array_equal(y1.numpy(), expected1)
        np.testing.assert_array_equal(y2.numpy(), expected2)
        np.testing.assert_array_equal(y3.numpy(), expected3)
        np.testing.assert_array_equal(y4.numpy(), expected3)

        np.testing.assert_array_equal(m1.numpy(), expected1)
        np.testing.assert_array_equal(m2.numpy(), expected2)
        np.testing.assert_array_equal(m3.numpy(), expected3)
        np.testing.assert_array_equal(m4.numpy(), expected3)

        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
