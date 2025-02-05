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


class TestArgmaxCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmax
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestArgmaxCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmax
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.target_marker_op = "pd_op.argmax"

    def test_trt_result(self):
        # test input's dtype
        self.check_marker(expected_result=False)


class TestArgmaxCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmax
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.target_marker_op = "pd_op.argmax"

    def test_trt_result(self):
        # test axis
        self.check_marker(expected_result=False)


class TestArgmaxCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmin
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": np.random.randn(1).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "axis"]}
        self.target_marker_op = "pd_op.argmax"

    def test_trt_result(self):
        # test axis Value
        self.check_marker(expected_result=False)


class TestArgminCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmin
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestArgminCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmin
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.target_marker_op = "pd_op.argmin"

    def test_trt_result(self):
        # test input's dtype
        self.check_marker(expected_result=False)


class TestArgminCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmin
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.target_marker_op = "pd_op.argmin"

    def test_trt_result(self):
        # test axis
        self.check_marker(expected_result=False)


class TestArgminCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argmin
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": np.random.randn(1).astype("int64"),
        }
        self.program_config = {"feed_list": ["x", "axis"]}
        self.target_marker_op = "pd_op.argmin"

    def test_trt_result(self):
        # test axis Value
        self.check_marker(expected_result=False)


class TestWhereTRTPatternCase1(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.where
        self.api_args = {
            "condition": np.random.choice([True, False], size=(2, 3)),
            "x": np.random.randn(2, 3).astype("float32"),
            "y": np.random.randn(2, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["condition", "x", "y"]}
        self.min_shape = {"condition": [1, 3], "x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"condition": [2, 3], "x": [2, 3], "y": [2, 3]}
        self.max_shape = {"condition": [5, 3], "x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestArgsortCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argsort
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestArgsortCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argsort
        self.api_args = {
            "x": np.random.randn(2).astype("float32"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [2]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestArgsortCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argsort
        self.api_args = {
            "x": np.random.randn(2, 3).astype("int64"),
            "axis": -1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestArgsortCase4TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.argsort
        self.api_args = {
            "x": np.random.randn(2, 4000).astype("float32"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.target_marker_op = "pd_op.argsort"

    def test_trt_result(self):
        # test axis attr
        self.check_marker(expected_result=False)


class TestWhereTRTPatternCase2(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.where
        self.api_args = {
            "condition": np.random.choice([True, False], size=(2, 3)),
            "x": np.random.randn(2, 3).astype("int64"),
            "y": np.random.randn(2, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["condition", "x", "y"]}
        self.min_shape = {"condition": [1, 3], "x": [1, 3], "y": [1, 3]}
        self.opt_shape = {"condition": [2, 3], "x": [2, 3], "y": [2, 3]}
        self.max_shape = {"condition": [5, 3], "x": [5, 3], "y": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.random.randn(2, 3).astype("float32"),
            "k": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 3]}
        self.opt_shape = {"x": [2, 3]}
        self.max_shape = {"x": [5, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.random.randn(2).astype("int64"),
            "k": 1,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [2]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestTopkCase3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.topk
        self.api_args = {
            "x": np.random.randn(2).astype("int64"),
            "k": 1,
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1]}
        self.opt_shape = {"x": [2]}
        self.max_shape = {"x": [5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestIndexSelectCase1TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.index_select
        self.api_args = {
            "x": np.random.randn(2, 3, 3).astype("float32"),
            "index": np.array([0, 2], dtype="int64"),
            "axis": 1,
        }
        self.program_config = {"feed_list": ["x", "index"]}
        self.min_shape = {"x": [1, 3, 3], "index": [1]}
        self.opt_shape = {"x": [2, 3, 3], "index": [2]}
        self.max_shape = {"x": [5, 3, 3], "index": [5]}

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")

    def test_trt_result_fp32(self):
        self.check_trt_result()


class TestIndexSelectCase2TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.index_select
        self.api_args = {
            "x": np.random.randn(2, 3, 3).astype("int64"),
            "index": np.array([0, 1], dtype="int64"),
            "axis": 0,
        }
        self.program_config = {"feed_list": ["x", "index"]}
        self.min_shape = {"x": [1, 3, 3], "index": [1]}
        self.opt_shape = {"x": [2, 3, 3], "index": [2]}
        self.max_shape = {"x": [5, 3, 3], "index": [5]}

    def test_trt_result(self):
        self.check_trt_result()


if __name__ == '__main__':
    unittest.main()
