#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0-Size Tensor indicates that the tensor's shape contains 0
# 0-Size Tensor's shape can be [2, 0, 3], [0, 2]...etc, numel is 0
# which can be created by paddle.rand([2, 0, 3])

import unittest

import numpy as np

import paddle


# Use to test zero-size of Sundry API, which is unique and can not be classified
# with others. It can be implemented here flexibly.
class TestSundryAPI(unittest.TestCase):
    def test_detach(self):
        x = paddle.rand([0, 2])
        out = x.detach()

        self.assertEqual(out.shape, [0, 2])
        self.assertEqual(out.size, 0)

    def test_numpy(self):
        x = paddle.rand([0, 2])
        out = x.numpy()

        self.assertEqual(out.shape, (0, 2))
        self.assertEqual(out.size, 0)

    def test_reshape_dygraph(self):
        # case 1
        x1 = paddle.rand([0, 2])
        x1.stop_gradient = False
        out1 = paddle.reshape(x1, [-1])

        self.assertEqual(out1.shape, [0])
        self.assertEqual(out1.size, 0)

        # case 2
        x2 = paddle.rand([0, 2])
        x2.stop_gradient = False
        out2 = paddle.reshape(x2, [2, -1])

        self.assertEqual(out2.shape, [2, 0])
        self.assertEqual(out2.size, 0)

        # case 3
        x3 = paddle.rand([0, 2])
        x3.stop_gradient = False
        out3 = paddle.reshape(x3, [2, 3, 0])

        self.assertEqual(out3.shape, [2, 3, 0])
        self.assertEqual(out3.size, 0)

        # case 4
        x4 = paddle.rand([0, 2])
        x4.stop_gradient = False
        out4 = paddle.reshape(x4, [0])

        self.assertEqual(out4.shape, [0])
        self.assertEqual(out4.size, 0)

        # 5
        x5 = paddle.rand([0])
        with self.assertRaises(ValueError):
            out4 = paddle.reshape(x5, [2, 0, -1])

    def test_reshape_static(self):
        paddle.enable_static()
        place = paddle.CPUPlace()
        if paddle.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)

        input_cases = [
            # (x, new_shape, desired_shape)
            (np.random.rand(0, 2), [-1], [0]),
            (np.random.rand(0, 2), [2, -1], [2, 0]),
            (np.random.rand(0, 2), [2, 3, 0], [2, 3, 0]),
            (np.random.rand(0, 2), [0], [0]),
        ]
        for case in input_cases:
            data_np, new_shape, desired_shape = case
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            executor = paddle.static.Executor(place)
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name="x", shape=data_np.shape, dtype='float64'
                )
                out = paddle.reshape(x, new_shape)
                fetch_list = [out]
                feeds = {'x': data_np}

                executor.run(startup_program)
                res = executor.run(
                    main_program, feed=feeds, fetch_list=fetch_list
                )
            np.testing.assert_equal(res[0], np.random.rand(*desired_shape))
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
