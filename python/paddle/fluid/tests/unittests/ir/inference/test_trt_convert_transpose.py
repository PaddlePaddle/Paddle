# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest
from program_config import TensorConfig
import unittest
import numpy as np
import paddle.inference as paddle_infer


class TrtConvertTransposeTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.set_params()
        self.ops_config = [{
            "op_type": "transpose",
            "op_inputs": {
                "X": ["transpose_input"]
            },
            "op_outputs": {
                "Out": ["transpose_output"]
            },
            "op_attrs": {
                "axis": self.axis
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def set_params(self):
        self.axis = [[0, 1, 3, 2]]
        self.trt = 1
        self.paddle = 2

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        transpose_input = TensorConfig(shape=[-1, 3, 64, 64])
        self.program_weights = {"conv2d_weight": filter}
        self.program_inputs = {"transpose_input": transpose_input}
        self.program_outputs = ["transpose_output"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"transpose_input": [1, 3, 32, 64]}
        self.dynamic_shape.max_input_shape = {"transpose_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"transpose_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"transpose_input": [1, 3, 32, 64]}
        self.dynamic_shape.max_input_shape = {"transpose_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"transpose_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-2)


class TrtConvertTransposeAxisTest(TrtConvertTransposeTest):
    def set_params(self):
        self.axis = [[0, 3, 2, 1]]
        self.trt = 1
        self.paddle = 2


class DynamicShapeTrtConvertTransposeTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.set_params()
        self.ops_config = [{
            "op_type": "transpose",
            "op_inputs": {
                "X": ["transpose_input"]
            },
            "op_outputs": {
                "Out": ["transpose_output"]
            },
            "op_attrs": {
                "axis": self.axis
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def set_params(self):
        self.axis = [[3, 2, 0, 1]]
        self.trt = 1
        self.paddle = 2

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        transpose_input = TensorConfig(shape=[-1, 3, 64, 64])
        self.program_weights = {"conv2d_weight": filter}
        self.program_inputs = {"transpose_input": transpose_input}
        self.program_outputs = ["transpose_output"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=0, paddle_op_num=3, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"transpose_input": [1, 3, 32, 64]}
        self.dynamic_shape.max_input_shape = {"transpose_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"transpose_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"transpose_input": [1, 3, 32, 64]}
        self.dynamic_shape.max_input_shape = {"transpose_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"transpose_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-2)


if __name__ == "__main__":
    unittest.main()
