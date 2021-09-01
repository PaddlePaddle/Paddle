#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.static as static


class API_TestStaticCond(unittest.TestCase):
    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input1 = np.random.rand(10, 10).astype('float64')
            input2 = np.random.rand(6, 4, 4).astype('float32')
            input3 = np.random.rand(5, 5, 5).astype('float64')
            input4 = np.random.rand(7, 4, 3).astype('float32')
            input5 = np.random.rand(10, 1, 10).astype('float64')

            for dev in ("cpu", "gpu"):
                paddle.device.set_device(dev)
                for p in ("fro", 1, -1, np.inf, -np.inf):
                    for input in (input1, input2, input3):
                        with static.program_guard(static.Program(),
                                                static.Program()):
                            input_data = static.data(
                                "X", shape=input.shape, dtype=input.dtype)
                            output = paddle.linalg.cond(input_data, p)
                            exe = static.Executor()
                            result = exe.run(feed={"X": input}, fetch_list=[output])
                            expected_output = np.linalg.cond(input, p)
                            self.assertTrue(np.allclose(result, expected_output))

                # for p in (None, 2, -2):
                #     for input in (input4, input5):
                #         with static.program_guard(static.Program(), static.Program()):
                #             input_data = static.data("X", shape=input.shape, dtype=input.dtype)
                #             output = paddle.linalg.cond(input_data, p)
                #             exe = static.Executor()
                #             result = exe.run(feed={"X": input}, fetch_list=[output])
                #             expected_output = np.linalg.cond(input, p)
                #             self.assertTrue(np.allclose(result, expected_output))


class API_TestDygraphCond(unittest.TestCase):
    def test_out(self):
        paddle.disable_static()
        input1 = np.random.rand(10, 10).astype('float32')
        input2 = np.random.rand(4, 5, 5).astype('float32')
        input3 = np.random.rand(3, 6, 6).astype('float64')
        input4 = np.random.rand(3, 4, 5).astype('float32')
        input5 = np.random.rand(4, 3, 7).astype('float64')

        for dev in ("cpu", "gpu"):
            paddle.device.set_device(dev)
            for p in (None, "fro", "nuc", 1, -1, 2, -2, np.inf, -np.inf):
                for input in (input1, input2, input3):
                    input_tensor = paddle.to_tensor(input)
                    output = paddle.linalg.cond(input_tensor, p)
                    expected_output = np.linalg.cond(input, p)
                    self.assertTrue(np.allclose(output, expected_output))

            for p in (None, 2, -2):
                for input in (input4, input5):
                    input_tensor = paddle.to_tensor(input)
                    output = paddle.linalg.cond(input_tensor, p)
                    expected_output = np.linalg.cond(input, p)
                    self.assertTrue(np.allclose(output, expected_output))


class TestCondAPIError(unittest.TestCase):
    def test_api_error(self):
        x_data_1 = np.arange(32, dtype='float32').reshape((2, 4, 4))
        x_data_2 = np.arange(32, dtype='float64').reshape((2, 4, 4))
        x_data_3 = np.arange(24, dtype='float64').reshape((2, 3, 4))
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_1, 'fro_')
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_1, 1.5)
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_1, 0)
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_2, '_nuc')
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_2, -0.7)
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_2, 3)
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_3, np.inf)
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_3, -1)
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_3, 'fro')
        self.assertRaises(ValueError, paddle.linalg.cond, x_data_3, 'nuc')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()