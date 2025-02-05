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
from paddle import _C_ops


class TestFlattenTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full
        self.api_args = {"shape": [3, 2], "fill_value": 1.0}
        self.program_config = {"feed_list": []}
        self.min_shape = {}
        self.opt_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


class TestAssignTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.assign
        self.api_args = {
            "x": np.random.random([2, 2]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 2]}
        self.opt_shape = {"x": [2, 2]}
        self.max_shape = {"x": [3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


def assign_value_api(input, dtype, values):
    output = paddle.zeros_like(input)
    return _C_ops.assign_value_(
        output,
        list(input.shape),
        dtype,
        values,
        paddle.framework._current_expected_place(),
    )


def assign_value_api_case2(input, dtype, values):
    return _C_ops.assign_value(
        list(input.shape),
        dtype,
        values,
        paddle.framework._current_expected_place(),
    )


class TestAssignValueInTRTPattern(TensorRTBaseTest):
    def test_trt_result(self):
        test_cases = [
            # Test case 1
            (
                assign_value_api,
                {
                    "x": np.random.random([2, 2]).astype("int32"),
                    "dtype": paddle.int32,
                    "values": [1, 1, 1, 1],
                },
            ),
            # Test case 2
            (
                assign_value_api_case2,
                {
                    "x": np.random.random([2, 2]).astype("float32"),
                    "dtype": paddle.float32,
                    "values": [1.0, 1.0, 1.0, 1.0],
                },
            ),
        ]

        for python_api, api_args in test_cases:
            with self.subTest(python_api=python_api, api_args=api_args):
                self.python_api = python_api
                self.api_args = api_args
                self.program_config = {"feed_list": ["x"]}
                self.min_shape = {}
                self.opt_shape = {}
                self.max_shape = {}
                self.check_trt_result()


class TestArangeTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.arange
        self.api_args = {
            "start": np.array([0]).astype("int64"),
            "end": np.array([6]).astype("int64"),
            "step": np.array([1]).astype("int64"),
        }
        self.program_config = {"feed_list": []}
        self.min_shape = {}
        self.opt_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


class TestAssignOutTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.assign
        self.api_args = {
            "x": np.random.random([2, 2]).astype("float32"),
            "output": np.zeros((2, 2), dtype="float32"),
        }
        self.program_config = {"feed_list": ["x", "output"]}
        self.min_shape = {"x": [1, 2], "output": [1, 2]}
        self.opt_shape = {"x": [2, 2], "output": [2, 2]}
        self.max_shape = {"x": [3, 2], "output": [3, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullLikeBoolTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full_like
        self.api_args = {
            "input": np.random.randn(3, 2).astype("bool"),
            "fill_value": True,
        }
        self.program_config = {"feed_list": ["input"]}
        self.min_shape = {"input": [1, 2]}
        self.opt_shape = {"input": [3, 2]}
        self.max_shape = {"input": [5, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullLikeFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full_like
        self.api_args = {
            "input": np.random.randn(3, 2).astype("float32"),
            "fill_value": 5.0,
        }
        self.program_config = {"feed_list": ["input"]}
        self.min_shape = {"input": [1, 2]}
        self.opt_shape = {"input": [3, 2]}
        self.max_shape = {"input": [5, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullLikeIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full_like
        self.api_args = {
            "input": np.random.randn(3, 2).astype("int64"),
            "fill_value": 5,
        }
        self.program_config = {"feed_list": ["input"]}
        self.min_shape = {"input": [1, 2]}
        self.opt_shape = {"input": [3, 2]}
        self.max_shape = {"input": [5, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullLikeDynamicTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.full_like
        self.api_args = {
            "input": np.random.randn(3, 2).astype("float32"),
            "fill_value": np.array([5.0]).astype("float32"),
        }
        self.program_config = {"feed_list": ["input", "fill_value"]}
        self.min_shape = {"input": [1, 2]}
        self.opt_shape = {"input": [3, 2]}
        self.max_shape = {"input": [5, 2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullWithTensorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tensor.fill_constant
        self.api_args = {
            "shape": np.array([1]).astype("int64"),
            "dtype": "float32",
            "value": np.array([0.0]).astype("float32"),
        }
        self.program_config = {"feed_list": ["value", "shape"]}
        self.min_shape = {}
        self.opt_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


class TestFullWithTensorCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.tensor.fill_constant
        self.api_args = {
            "shape": [1, 1],
            "dtype": np.float32,
            "value": np.array([1.0]).astype("float32"),
        }
        self.program_config = {"feed_list": ["value"]}
        self.min_shape = {}
        self.opt_shape = {}
        self.max_shape = {}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == "__main__":
    unittest.main()
