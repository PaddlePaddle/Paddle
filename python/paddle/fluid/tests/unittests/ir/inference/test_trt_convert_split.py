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


class TrtConvertSplitDims2Test(TrtLayerAutoScanTest):
    def setUp(self):
        self.set_params()
        self.ops_config = [{
            "op_type": "split",
            "op_inputs": {
                "X": ["split_input"]
            },
            "op_outputs": {
                "Out": ["output_var0", "output_var1"]
            },
            "op_attrs": {
                "sections": self.sections,
                "num": self.num,
                "axis": self.axis
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def set_params(self):
        self.sections = [[2, 1]]
        self.num = [0]
        self.axis = [1]
        self.trt = 1
        self.paddle = 3

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        split_input = TensorConfig(shape=[1, 3, 64, 64])
        self.program_inputs = {"split_input": split_input}
        self.program_outputs = ["output_var0", "output_var1"]
        self.program_weights = {}

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
        self.dynamic_shape.min_input_shape = {"split_input": [1, 3, 64, 32]}
        self.dynamic_shape.max_input_shape = {"split_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"split_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"split_input": [1, 3, 64, 32]}
        self.dynamic_shape.max_input_shape = {"split_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"split_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-2)


class TrtConvertSplitDims2AxisTest(TrtConvertSplitDims2Test):
    def set_params(self):
        self.sections = [[10, 54]]
        self.num = [0]
        self.axis = [2]
        self.trt = 1
        self.paddle = 3


class TrtConvertSplitDims2NumTest(TrtConvertSplitDims2Test):
    def set_params(self):
        self.sections = [[]]
        self.num = [2]
        self.axis = [2]
        self.trt = 1
        self.paddle = 3


class TrtConvertSplitDims3Test(TrtLayerAutoScanTest):
    def setUp(self):
        self.set_params()
        self.ops_config = [{
            "op_type": "split",
            "op_inputs": {
                "X": ["split_input"]
            },
            "op_outputs": {
                "Out": ["output_var0", "output_var1", "output_var2"]
            },
            "op_attrs": {
                "sections": self.sections,
                "num": self.num,
                "axis": self.axis
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def set_params(self):
        self.sections = [[3, 7, 54]]
        self.num = [0]
        self.axis = [2]
        self.trt = 1
        self.paddle = 4

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        split_input = TensorConfig(shape=[1, 3, 64, 64])
        self.program_inputs = {"split_input": split_input}
        self.program_outputs = ["output_var0", "output_var1", "output_var2"]
        self.program_weights = {}

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
        self.dynamic_shape.min_input_shape = {"split_input": [1, 3, 64, 32]}
        self.dynamic_shape.max_input_shape = {"split_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"split_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"split_input": [1, 3, 64, 32]}
        self.dynamic_shape.max_input_shape = {"split_input": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"split_input": [1, 3, 64, 64]}
        self.run_test(
            trt_engine_num=self.trt, paddle_op_num=self.paddle, threshold=1e-2)


if __name__ == "__main__":
    unittest.main()
