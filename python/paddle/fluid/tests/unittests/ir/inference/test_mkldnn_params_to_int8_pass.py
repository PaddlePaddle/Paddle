# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import unittest
from paddle import fluid

import hypothesis.strategies as st


class TestMkldnnParamToInt8Pass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_gpu=False, ir_optim=True, use_mkldnn=True)

        config.enable_quantizer()
        bsz = 1
        config.quantizer_config().set_quant_batch_size(bsz)
        infer_data = fluid.core.PaddleTensor()
        infer_data.name = "data"
        arr = np.ndarray([1, 1, 1, 1], dtype=np.float32)
        infer_data.lod = [[1]]
        infer_data.data = fluid.core.PaddleBuf(arr)
        infer_data.shape = arr.shape
        infer_data.dtype = fluid.core.PaddleDType.FLOAT32
        warmup_data = [infer_data]
        print("warmup_data:", infer_data.data.float_data())
        print("warmup data type:", type(warmup_data[0]))
        config.quantizer_config().set_quant_data(warmup_data)

        config.switch_use_feed_fetch_ops(True)

        config.pass_builder().append_pass("params_to_int8_pass")
        yield config, ["conv2d"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        return True

    def sample_program_config(self, draw):
        x_shape = [1, 1, 1, 4]

        data_format = "NCHW"

        f_shape = [1, 1, 1, 1]
        if data_format == "NCHW":
            f_shape[1] = x_shape[1]
        else:
            f_shape[1] = x_shape[3]

        strides = [1, 1]

        padding_algorithm = "EXPLICIT"

        padding = [0, 0, 0, 0]

        groups = 1

        dilations = [1, 1]

        has_bias = False

        def generate_one():
            return np.random.random([1]).astype(np.float32)

        conv_inputs = {
            "Input": ["input_x"],
            "Filter": ["filter"],
        }
        program_weights = {"filter": TensorConfig(shape=f_shape)}
        program_inputs = {"input_x": TensorConfig(shape=x_shape)}

        attrs = {
            "Scale_weights": [0.45],
            "Scale_in": generate_one(),
            "Sum_scale": generate_one(),
            "Output_shift_scale": generate_one(),
            "Activation_scale": generate_one(),
        }

        if has_bias:
            conv_inputs["Bias"] = ["bias"]
            program_weights["bias"] = TensorConfig(shape=f_shape[0])
            attrs["Bias_scales"] = np.random.random(f_shape[0]).astype(
                np.float32)

        conv2d_op = OpConfig(
            "conv2d",
            inputs=conv_inputs,
            outputs={"Output": ["conv2d_out"]},
            strides=strides,
            padding_algorithm=padding_algorithm,
            paddings=padding,
            groups=groups,
            dilations=dilations,
            data_format=data_format,
            use_mkldnn=True,
            mkldnn_data_type="int8",
            attrs=attrs)

        ops = [conv2d_op]

        program_config = ProgramConfig(
            ops=ops,
            weights=program_weights,
            inputs=program_inputs,
            outputs=["conv2d_out"])
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False, max_examples=1, passes=["params_to_int8_pass"])


if __name__ == "__main__":
    unittest.main()
