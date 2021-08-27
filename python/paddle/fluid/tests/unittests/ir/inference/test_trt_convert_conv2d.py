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
import numpy as np
import paddle.inference as paddle_infer


class TrtConvertConv2dTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "conv2d",
            "op_inputs": {
                "Input": ["input_data"],
                "Filter": ["conv2d_weight"]
            },
            "op_outputs": {
                "Output": ["conv_output_data"]
            },
            "op_attrs": {
                "data_format": ["NCHW"],
                "dilations": [[1, 1]],
                "padding_algorithm": ["EXPLICIT"],
                "groups": [1],
                "paddings": [[0, 3], [3, 1]],
                "strides": [[1, 1], [2, 2]],
            }
        }, {
            "op_type": "relu",
            "op_inputs": {
                "X": ["conv_output_data"]
            },
            "op_outputs": {
                "Out": ["relu_output_data"]
            },
            "op_attrs": {}
        }]
        self.batch_size_set = [1, 2, 4]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        weight = np.random.randn(24, 3, 3, 3).astype("float32")
        filter = TensorConfig(shape=[24, 3, 3, 3], data=weight)
        if op_attr_list[0]["data_format"] == "NCHW":
            input_data = TensorConfig(shape=[-1, 3, 64, 64])
        else:
            input_data = TensorConfig(shape=[-1, 64, 64, 3])
        self.program_weights = {"conv2d_weight": filter}
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["relu_output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}
        self.run_test(trt_engine_num=1, paddle_op_num=2, threshold=1e-2)

    def test_trt_int8_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        self.run_test(
            trt_engine_num=1, paddle_op_num=2, quant=True, threshold=1e-1)


if __name__ == "__main__":
    unittest.main()
