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
import unittest
import paddle.inference as paddle_infer


class TrtConvertBilinearInterpV2Test_all(TrtLayerAutoScanTest):
    def init(self):
        self.data_layout = "NCHW"
        self.interp_method = "bilinear"
        self.align_corners = False
        self.scale = [2., 2.]
        self.out_h = 32
        self.out_w = 64
        self.trt_engine_num = 1
        self.paddle_op_num = 2

    def setUp(self):
        self.init()
        self.ops_config = [{
            "op_type": "bilinear_interp_v2",
            "op_inputs": {
                "X": ["input_data"],
                "Scale": ["input_scale"]
            },
            "op_outputs": {
                "Out": ["output_data"]
            },
            "op_attrs": {
                "data_layout": [self.data_layout],
                "interp_method": [self.interp_method],
                "align_corners": [self.align_corners],
                "scale": [self.scale],
                "out_h": [self.out_h],
                "out_w": [self.out_w]
            }
        }]

        self.batch_size_set = [1, 2, 4]

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        input_data = TensorConfig(shape=[-1, 3, 64, 64])
        alpha = np.random.uniform(low=0.5, high=6.0, size=(2)).astype("float32")
        input_scale = TensorConfig(shape=[2], data=alpha)

        self.program_weights = {"input_scale": input_scale}
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["output_data"]

    def test_check_fp32_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Float32
        # the fused tensorrt engine num is 1, and paddle op num is 2(feed and fetch).
        self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    def test_check_fp16_output(self):
        self.trt_param.precision == paddle_infer.PrecisionType.Half
        self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-2)

    def test_dynamic_shape_fp32_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}
        self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-5)

    def test_dynamic_shape_fp16_check_output(self):
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        self.dynamic_shape.min_input_shape = {"input_data": [1, 3, 32, 32]}
        self.dynamic_shape.max_input_shape = {"input_data": [4, 3, 64, 64]}
        self.dynamic_shape.opt_input_shape = {"input_data": [1, 3, 64, 64]}
        self.run_test(self.trt_engine_num, self.paddle_op_num, threshold=1e-2)


class TrtConvertBilinearInterpV2Test(TrtConvertBilinearInterpV2Test_all):
    def init(self):
        self.data_layout = "NCHW"
        self.interp_method = "bilinear"
        self.align_corners = False
        self.scale = [2., 2.]
        self.in_h = 16
        self.in_w = 32
        self.out_h = 32
        self.out_w = 64
        self.trt_engine_num = 1
        self.paddle_op_num = 2

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        input_data = TensorConfig(shape=[-1, 3, 64, 64])
        alpha = np.random.uniform(low=0.5, high=6.0, size=(2)).astype("float32")
        input_scale = TensorConfig(shape=[2], data=alpha)

        self.program_weights = {"input_scale": input_scale}
        self.program_inputs = {"input_data": input_data}
        self.program_outputs = ["output_data"]


if __name__ == "__main__":
    unittest.main()
