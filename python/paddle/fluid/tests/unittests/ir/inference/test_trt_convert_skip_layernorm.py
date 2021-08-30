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
import unittest


class TrtConvertScaleTest(TrtLayerAutoScanTest):
    def setUp(self):
        self.ops_config = [{
            "op_type": "skip_layernorm",
            "op_inputs": {
                "X": ["skip_layernorm_inputX_data"],
                "Y": ["skip_layernorm_inputY_data"],
                "Bias": ["bias"],
                "Scale": ["scale"]
            },
            "op_outputs": {
                "Out": ["skip_layernorm_output_data"]
            },
            "op_attrs": {
                "epsilon": [1e-5],
                "begin_norm_axis": [0, 1, 2],
                "enable_int8": [False, True]
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        input_bias = np.random.randn(768).astype("float32")
        input_scale = np.random.randn(768).astype("float32")
        skip_layernorm_inputX_data = TensorConfig(shape=[1, 128, 768])
        skip_layernorm_inputY_data = TensorConfig(shape=[1, 128, 768])
        bias = TensorConfig(shape=[768], data=input_bias)
        scale = TensorConfig(shape=[768], data=input_scale)
        self.program_weights = {"bias": bias, "scale": scale}
        self.program_inputs = {
            "skip_layernorm_inputX_data": skip_layernorm_inputX_data,
            "skip_layernorm_inputY_data": skip_layernorm_inputY_data,
        }
        self.program_outputs = ["skip_layernorm_output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(trt_engine_num=0, paddle_op_num=4, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(trt_engine_num=0, paddle_op_num=4, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {
            "skip_layernorm_inputX_data": [1, 128, 768],
            "skip_layernorm_inputY_data": [1, 128, 768],
            "bias": [768],
            "scale": [768]
        }
        self.dynamic_shape.max_input_shape = {
            "skip_layernorm_inputX_data": [4, 768, 3072],
            "skip_layernorm_inputY_data": [4, 128, 3072],
            "bias": [3072],
            "scale": [3072]
        }
        self.dynamic_shape.opt_input_shape = {
            "skip_layernorm_inputX_data": [1, 128, 768],
            "skip_layernorm_inputY_data": [1, 128, 768],
            "bias": [768],
            "scale": [768]
        }
        self.run_test(trt_engine_num=1, paddle_op_num=3, threshold=1e-5)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {
            "skip_layernorm_inputX_data": [1, 128, 768],
            "skip_layernorm_inputY_data": [1, 128, 768],
            "bias": [768],
            "scale": [768]
        }
        self.dynamic_shape.max_input_shape = {
            "skip_layernorm_inputX_data": [4, 768, 3072],
            "skip_layernorm_inputY_data": [4, 128, 3072],
            "bias": [3072],
            "scale": [3072]
        }
        self.dynamic_shape.opt_input_shape = {
            "skip_layernorm_inputX_data": [1, 128, 768],
            "skip_layernorm_inputY_data": [1, 128, 768],
            "bias": [768],
            "scale": [768]
        }
        self.run_test(trt_engine_num=1, paddle_op_num=3, threshold=1e-2)


if __name__ == "__main__":
    unittest.main()
