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


class TestIsFloatPoint_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)

        self.test_cases = [
            {'shape': [3, 4], 'dtype': 'float32'},
            {'shape': [5], 'dtype': 'float64'},
            {'shape': [2, 3, 4], 'dtype': 'int32'},
        ]
        self.init_data()

    def init_data(self):
        self.data = []
        for case in self.test_cases:
            shape = case['shape']
            dtype = case['dtype']
            np_data = np.random.rand(*shape).astype(dtype)
            expected_result = 'float' in dtype

            self.data.append(
                {
                    'np_data': np_data,
                    'dtype': dtype,
                    'shape': shape,
                    'expected': expected_result,
                }
            )

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        for case in self.data:
            np_data = case['np_data']
            tensor = paddle.to_tensor(np_data)

            result_x = paddle.is_floating_point(x=tensor)
            result_input = paddle.is_floating_point(input=tensor)

            np.testing.assert_array_equal(result_x, result_input)
            np.testing.assert_array_equal(result_x, case['expected'])

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        for case in self.data:
            np_data = case['np_data']
            tensor = paddle.to_tensor(np_data)

            result_x = paddle.is_floating_point(x=tensor)
            result_input = paddle.is_floating_point(input=tensor)

            np.testing.assert_array_equal(result_x, result_input)
            np.testing.assert_array_equal(result_x, case['expected'])


if __name__ == '__main__':
    unittest.main()
