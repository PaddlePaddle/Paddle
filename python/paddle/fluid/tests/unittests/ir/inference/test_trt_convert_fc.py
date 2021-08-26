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


class TrtConvertFcTest_col_dims1(TrtLayerAutoScanTest):
    def init(self):
        self.x_num_col_dims = 2
        self.y_num_col_dims = 1
        self.trt_engine_num = 1
        self.paddle_op_num = 2

    def setUp(self):
        self.init()
        self.ops_config = [{
            "op_type": "mul",
            "op_inputs": {
                "X": ["input_data"],
                "Y": ["mul_weight"]
            },
            "op_outputs": {
                "Out": ["mul_output"]
            },
            "op_attrs": {
                "x_num_col_dims": [self.x_num_col_dims],
                "y_num_col_dims": [self.y_num_col_dims]
            }
        }, {
            "op_type": "elementwise_add",
            "op_inputs": {
                "X": ["mul_output"],
                "Y": ["elementwise_add_weight"]
            },
            "op_outputs": {
                "Out": ["elementwise_add_output"]
            },
            "op_attrs": {
                "axis": [2]
            }
        }]
        self.batch_size_set = [1, 2, 4]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        input_data = TensorConfig(shape=[-1, 128, 256])
        mul_weight = TensorConfig(
            shape=[256, 256], data=np.random.randn(256, 256).astype("float32"))
        elementwise_add_weight = TensorConfig(
            shape=[256], data=np.random.randn(256).astype("float32"))

        self.program_weights = {
            "mul_weight": mul_weight,
            "elementwise_add_weight": elementwise_add_weight
        }
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["elementwise_add_output"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    # def test_dynamic_shape_fp32_check_output(self):
    #     self.trt_param.precision = paddle_infer.PrecisionType.Float32
    #     self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 8, 8]}
    #     self.dynamic_shape.max_input_shape = {"input_data": [4, 1, 8, 8]}
    #     self.dynamic_shape.opt_input_shape = {"input_data": [2, 1, 8, 8]}
    #     self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    # def test_dynamic_shape_fp16_check_output(self):
    #     self.trt_param.precision = paddle_infer.PrecisionType.Half
    #     self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 8, 8]}
    #     self.dynamic_shape.max_input_shape = {"input_data": [4, 1, 8, 8]}
    #     self.dynamic_shape.opt_input_shape = {"input_data": [2, 1, 8, 8]}
    #     self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-2)

    # def test_trt_int8_check_output(self):
    #     self.trt_param.precision = paddle_infer.PrecisionType.Int8
    #     self.run_test(
    #         self.trt_engine_num, self.paddle_op_num, quant=True, threshold=1e-1)

    # class TrtConvertFcTest_col_dims2(TrtConvertFcTest_col_dims1):
    #     def init(self):
    #         self.in_num_col_dims = 2
    #         self.trt_engine_num = 1
    #         self.paddle_op_num = 2

    #     def update_program_input_and_weight_with_attr(self, op_attr_list):
    #         weight = np.random.randn(64, 32).astype("float32")
    #         filter = TensorConfig(shape=[64, 32], data=weight)
    #         input_data = TensorConfig(shape=[-1, 1, 8, 8])
    #         bias_data = np.random.randn(1, 32).astype("float32")
    #         bias = TensorConfig(shape=[1, 32], data=bias_data)

    #         self.program_weights = {"fc_weight": filter, "bias_weight": bias}
    #         self.program_inputs = {"input_data": input_data}
    #         self.program_outputs = ["output_data"]

    #     def test_dynamic_shape_fp32_check_output(self):
    #         self.trt_param.precision = paddle_infer.PrecisionType.Float32
    #         self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 8, 8]}
    #         self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 8, 8]}
    #         self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 8, 8]}
    #         self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    #     def test_dynamic_shape_fp16_check_output(self):
    #         self.trt_param.precision = paddle_infer.PrecisionType.Half
    #         self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 8, 8]}
    #         self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 8, 8]}
    #         self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 8, 8]}
    #         self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-2)

    # class TrtConvertFcTest_col_dims3(TrtConvertFcTest_col_dims1):
    #     def init(self):
    #         self.in_num_col_dims = 3
    #         self.trt_engine_num = 1
    #         self.paddle_op_num = 2

    #     def update_program_input_and_weight_with_attr(self, op_attr_list):
    #         weight = np.random.randn(64, 32).astype("float32")
    #         filter = TensorConfig(shape=[64, 32], data=weight)
    #         input_data = TensorConfig(shape=[-1, 8, 8, 64])
    #         bias_data = np.random.randn(1, 32).astype("float32")
    #         bias = TensorConfig(shape=[1, 32], data=bias_data)

    #         self.program_weights = {"fc_weight": filter, "bias_weight": bias}
    #         self.program_inputs = {"input_data": input_data}
    #         self.program_outputs = ["output_data"]

    #     def test_dynamic_shape_fp32_check_output(self):
    #         self.trt_param.precision = paddle_infer.PrecisionType.Float32
    #         self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 1, 64]}
    #         self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 64, 64]}
    #         self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 32, 64]}
    #         self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    #     def test_dynamic_shape_fp16_check_output(self):
    #         self.trt_param.precision = paddle_infer.PrecisionType.Half
    #         self.dynamic_shape.min_input_shape = {"input_data": [1, 1, 1, 64]}
    #         self.dynamic_shape.max_input_shape = {"input_data": [4, 64, 64, 64]}
    #         self.dynamic_shape.opt_input_shape = {"input_data": [2, 32, 32, 64]}
    #         self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-2)


if __name__ == "__main__":
    unittest.main()
