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
        minputs = fluid.create_lod_tensor(
            np.random.rand(bsz, 1, 1, 1).astype('float32'), [[1] * bsz],
            fluid.CPUPlace())
        infer_data = fluid.core.PaddleTensor()
        infer_data.data = fluid.core.PaddleBuf(np.array(minputs))
        infer_data.lod = minputs.lod()
        infer_data.shape = minputs.shape()
        infer_data.dtype = fluid.core.PaddleDType.FLOAT32
        warmup_data = [infer_data]
        print("warmup data type:", type(warmup_data[0]))
        config.quantizer_config().set_quant_data(warmup_data)

        config.switch_use_feed_fetch_ops(True)
        predictor = fluid.core.create_paddle_predictor(config)
        predictor.run(warmup_data)

        config.pass_builder().append_pass("params_to_int8_pass")
        yield config, ["conv2d"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        paddings = prog_config.ops[0].attrs["paddings"]
        strides = prog_config.ops[0].attrs["strides"]
        groups = prog_config.ops[0].attrs["groups"]
        padding_algorithm = prog_config.ops[0].attrs["padding_algorithm"]
        dilations = prog_config.ops[0].attrs["dilations"]
        data_format = prog_config.ops[0].attrs["data_format"]
        filter_shape = prog_config.weights["filter"].shape
        input_shape = prog_config.inputs["input_x"].shape
        if padding_algorithm == "VALID":
            if ((input_shape[2] - (dilations[0] * (filter_shape[2] - 1) + 1)) / strides[0] + 1) <= 1 or \
               ((input_shape[3] - (dilations[1] * (filter_shape[3] - 1) + 1)) / strides[1] + 1) <= 1:
                return False
        if padding_algorithm == "EXPLICIT":
            if ((input_shape[2] + paddings[0] + paddings[1] - (dilations[0] * (filter_shape[2] - 1) + 1)) / strides[0] + 1) <= 1 or \
               ((input_shape[3] + paddings[2] + paddings[3] - (dilations[1] * (filter_shape[3] - 1) + 1)) / strides[1] + 1) <= 1:
                return False
        if data_format == "NCHW":
            if input_shape[1] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False
        else:
            if input_shape[3] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False
        return True

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=1), min_size=4, max_size=4))
        x_shape[1] = draw(st.integers(min_value=1, max_value=1))

        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

        f_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=1), min_size=4, max_size=4))
        if data_format == "NCHW":
            f_shape[1] = x_shape[1]
        else:
            f_shape[1] = x_shape[3]

        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))

        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))

        padding = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=4, max_size=4))

        groups = draw(st.integers(min_value=1, max_value=3))

        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))

        has_bias = st.booleans()

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

        if draw(has_bias):
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
            quant=False, max_examples=100, passes=["params_to_int8_pass"])


if __name__ == "__main__":
    unittest.main()
