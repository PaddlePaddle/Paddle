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

# class TestFlattenTRTPattern(TensorRTBaseTest):
#     def setUp(self):
#         self.python_api = paddle.full
#         self.api_args = {"shape": [3, 2], "fill_value": 2.5, "dytpe": "int32"}
#         self.program_config = {"feed_list": []}
#         self.min_shape = {}
#         self.max_shape = {}

#     def test_trt_result(self):
#         self.check_trt_result()


# class TestArangeTRTPattern(TensorRTBaseTest):
#     def setUp(self):
#         self.python_api = paddle.arange
#         self.api_args = {
#             "start": np.array([0]).astype("int32"),
#             "end": np.array([6]).astype("int32"),
#             "step": np.array([1]).astype("int32"),
#         }
#         self.program_config = {"feed_list": []}
#         self.min_shape = {}
#         self.max_shape = {}

#     def test_trt_result(self):
#         self.check_trt_result()


class TestFullLikeFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full_like
        self.api_args = {
            "input": np.random.randn(3, 2).astype(np.float32),
            "fill_value": 5,
        }
        self.program_config = {"feed_list": ["input"]}
        self.min_shape = {"input": [1, 2]}
        self.max_shape = {"input": [5, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullLikeIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full_like
        self.api_args = {
            "input": np.random.randn(3, 2).astype(np.int32),
            "fill_value": 5,
        }
        self.program_config = {"feed_list": ["input"]}
        self.min_shape = {"input": [1, 2]}
        self.max_shape = {"input": [5, 2]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == "__main__":
    unittest.main()
