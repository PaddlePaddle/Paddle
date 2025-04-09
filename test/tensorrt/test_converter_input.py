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


class TestOneHotCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.one_hot
        self.api_args = {
            "x": np.random.randint(0, 2, size=(3, 1)).astype("int64"),
            "num_classes": np.array([2], dtype="int64"),
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(0, 2, size=shape).astype(
                "int64"
            ),
            "num_classes": lambda shape: np.array([2], dtype="int64"),
        }
        self.program_config = {"feed_list": ["x", "num_classes"]}
        self.min_shape = {"x": [1, 1]}
        self.opt_shape = {"x": [3, 1]}
        self.max_shape = {"x": [6, 1]}

    def test_trt_result(self):
        self.check_trt_result()


class TestOneHotCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.one_hot
        self.num_classes = 2
        self.api_args = {
            "x": np.random.randint(0, 2, size=(3, 1)).astype(
                "int64"
            ),  # Random integers between 0 and num_classes
            "num_classes": self.num_classes,
        }
        self.dynamic_shape_data = {
            "x": lambda shape: np.random.randint(
                0, self.num_classes, size=shape
            )
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1]}
        self.opt_shape = {"x": [3, 1]}
        self.max_shape = {"x": [6, 1]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
