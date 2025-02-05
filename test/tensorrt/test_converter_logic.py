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


class TestGreaterThanFloat32TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.greater_than
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGreaterThanInt64TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.greater_than
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLessThanFloat32TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.less_than
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLessThanInt64TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.less_than
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEqualFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.equal
        self.api_args = {
            "x": np.random.randn(3).astype("float32"),
            "y": np.random.randn(3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestEqualIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.equal
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestNotEqualFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.not_equal
        self.api_args = {
            "x": np.random.randn(3).astype("float32"),
            "y": np.random.randn(3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestNotEqualIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.not_equal
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestAndRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_and
        self.api_args = {
            "x": np.random.randn(2, 3).astype("bool"),
            "y": np.random.randn(3).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestAndRTPatternDifferentShapes(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_and
        self.api_args = {
            "x": np.random.randn(4, 5).astype("bool"),
            "y": np.random.randn(1, 5).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 5], "y": [1, 5]}
        self.opt_shape = {"x": [2, 5], "y": [1, 5]}
        self.max_shape = {"x": [10, 5], "y": [1, 5]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestAndRTPatternDifferentShapes1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_and
        self.api_args = {
            "x": np.random.randint(0, 2, (2, 3)).astype("bool"),
            "y": np.random.randint(0, 2, (2, 3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"x": [2, 3], "y": [2, 3]}
        self.max_shape = {"x": [5, 3], "y": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestOrRTPatternBroadcast(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_or
        self.api_args = {
            "x": np.random.randn(2, 1).astype("bool"),
            "y": np.random.randn(2, 3).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [2, 1], "y": [2, 3]}
        self.opt_shape = {"x": [2, 1], "y": [2, 3]}
        self.max_shape = {"x": [2, 1], "y": [2, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestOrRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_or
        self.api_args = {
            "x": np.random.randn(2, 3).astype("bool"),
            "y": np.random.randn(3).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestNotRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_not
        self.api_args = {
            "x": np.random.randn(2, 3).astype("bool"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestNotRTPatternEdgeCase(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_not
        self.api_args = {
            "x": np.zeros((2, 3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLogicalOrTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_or

    def test_trt_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(3,)).astype("bool"),
            "y": np.random.choice([True, False], size=(3,)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}
        self.check_trt_result()

    def test_trt_diff_shape_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(2, 3)).astype("bool"),
            "y": np.random.choice([True, False], size=(3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [4, 3], "y": [3]}
        self.check_trt_result()


class TestAndRTPatternErrorType(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_and
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int32"),
            "y": np.random.randn(3).astype("int32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestOrRTPatternErrorType(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_or
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int32"),
            "y": np.random.randn(3).astype("int32"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [5, 3], "y": [3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestNotRTINT8(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_not
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int8"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestNotRTINT64(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.bitwise_not
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestLogicalOrMarker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_or
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.target_marker_op = "pd_op.logical_or"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestLogicalAndTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_and

    def test_trt_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(3,)).astype("bool"),
            "y": np.random.choice([True, False], size=(3,)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}
        self.check_trt_result()

    def test_trt_diff_shape_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(2, 3)).astype("bool"),
            "y": np.random.choice([True, False], size=(3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [4, 3], "y": [3]}
        self.check_trt_result()


class TestLogicalAndMarker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_and
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.target_marker_op = "pd_op.logical_and"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestLogicalOr_TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_or_

    def test_trt_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(3,)).astype("bool"),
            "y": np.random.choice([True, False], size=(3,)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}
        self.check_trt_result()

    def test_trt_diff_shape_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(2, 3)).astype("bool"),
            "y": np.random.choice([True, False], size=(3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [4, 3], "y": [3]}
        self.check_trt_result()


class TestLogicalOr_Marker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_or_
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.target_marker_op = "pd_op.logical_or_"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestLogicalNotTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_not
        self.api_args = {
            "x": np.random.choice([True, False], size=(2, 3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogicalNotCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_not
        self.api_args = {"x": np.random.random([2]).astype("bool")}
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [2]}
        self.opt_shape = {"x": [2]}
        self.max_shape = {"x": [2]}

    def test_trt_result(self):
        self.check_trt_result()


class TestLogicalXorTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_xor

    def test_trt_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(3,)).astype("bool"),
            "y": np.random.choice([True, False], size=(3,)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1], "y": [1]}
        self.opt_shape = {"x": [2], "y": [2]}
        self.max_shape = {"x": [5], "y": [5]}
        self.check_trt_result()

    def test_trt_diff_shape_result(self):
        self.api_args = {
            "x": np.random.choice([True, False], size=(2, 3)).astype("bool"),
            "y": np.random.choice([True, False], size=(3)).astype("bool"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.min_shape = {"x": [1, 3], "y": [3]}
        self.opt_shape = {"x": [2, 3], "y": [3]}
        self.max_shape = {"x": [4, 3], "y": [3]}
        self.check_trt_result()


class TestLogicalXorMarker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.logical_xor
        self.api_args = {
            "x": np.random.randn(3).astype("int64"),
            "y": np.random.randn(3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "y"]}
        self.target_marker_op = "pd_op.logical_xor"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


if __name__ == '__main__':
    unittest.main()
