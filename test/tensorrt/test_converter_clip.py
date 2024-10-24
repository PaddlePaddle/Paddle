# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestClipTRTPatternCase1(TensorRTBaseTest):
    '''min/max is attr, and x/min/max is float'''

    def setUp(self):
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 0.3, 0.5, 0.9], [0.1, 0.2, 6, 7]]).astype(
                "float32"
            ),
            "min": 2.2,
            "max": 5.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase2(TensorRTBaseTest):
    def setUp(self):
        '''min/max is attr, and x is int, min/max is float'''
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 3, 5, 9], [1, 2, 6, 7]]).astype("int32"),
            "min": 2.2,
            "max": 5.5,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase3(TensorRTBaseTest):
    '''min/max is input, and x/min/max is float'''

    def setUp(self):
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 0.3, 0.5, 0.9], [0.1, 0.2, 6, 7]]).astype(
                "float32"
            ),
            "min": np.array([2.2]).astype("float32"),
            "max": np.array([5.2]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "min", "max"]}
        self.min_shape = {"x": [1, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


class TestClipTRTPatternCase4(TensorRTBaseTest):
    '''min/max is input, and x is int, min/max is float'''

    def setUp(self):
        self.python_api = paddle.clip
        self.api_args = {
            "x": np.array([[2, 3, 5, 9], [1, 2, 6, 7]]).astype("int32"),
            "min": np.array([2]).astype("float32"),
            "max": np.array([5]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "min", "max"]}
        self.min_shape = {"x": [1, 4]}
        self.max_shape = {"x": [5, 4]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == "__main__":
    unittest.main()
