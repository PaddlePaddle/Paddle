# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestSqueezeExcitationFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["squeeze_excitation_block"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=12), min_size=4, max_size=4
            )
        )
        x_shape[1] = 24

        oc = 6
        conv2d_op1_w_shape = [oc, x_shape[1], 1, 1]
        conv2d_op1_b_shape = [oc]
        conv2d_op2_w_shape = [x_shape[1], oc, 1, 1]
        conv2d_op2_b_shape = [x_shape[1]]

        # Random choose if add a relu operator
        has_relu = draw(st.sampled_from([True, False]))

        pool2d_op = OpConfig(
            type="pool2d",
            inputs={"X": ["pool2d_x"]},
            outputs={"Out": ["pool2d_out"]},
            adaptive=True,
            data_format="NCHW",
            global_pooling=False,
            ksize=[1, 1],
            pooling_type="avg",
        )
        ops = [pool2d_op]

        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["pool2d_out"],
                "Filter": ["conv2d_weight"],
            },
            outputs={"Output": ["conv2d_out"]},
            data_format="NCHW",
            dilations=[1, 1],
            padding_algorithm="EXPLICIT",
            groups=1,
            paddings=[0, 0, 0, 0],
            strides=[1, 1],
            has_bias=False,
        )

        ew_bias_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["conv2d_out"], "Y": ["ew_bias"]},
            outputs={"Out": ["add_out"]},
            axis=1,
        )
        ops.extend([conv2d_op, ew_bias_op])
        conv2d_input = "add_out"
        # 3. activation
        if has_relu:
            relu_op = OpConfig(
                "relu", inputs={"X": ["add_out"]}, outputs={"Out": ["relu_out"]}
            )
            conv2d_input = "relu_out"
            ops.append(relu_op)

        conv2d_op2 = OpConfig(
            "conv2d",
            inputs={
                "Input": [conv2d_input],
                "Filter": ["conv2d_weight2"],
            },
            outputs={"Output": ["conv2d_out2"]},
            data_format="NCHW",
            dilations=[1, 1],
            padding_algorithm="EXPLICIT",
            groups=1,
            paddings=[0, 0, 0, 0],
            strides=[1, 1],
            has_bias=False,
        )

        ew_bias_op2 = OpConfig(
            "elementwise_add",
            inputs={"X": ["conv2d_out2"], "Y": ["ew_bias2"]},
            outputs={"Out": ["add_out2"]},
            axis=1,
        )
        ops.extend([conv2d_op2, ew_bias_op2])
        ele_mul_input = "add_out2"
        # 3. activation
        if has_relu:
            relu_op2 = OpConfig(
                "relu",
                inputs={"X": ["add_out2"]},
                outputs={"Out": ["relu_out2"]},
            )
            ele_mul_input = "relu_out2"
            ops.append(relu_op2)

        ew_mul_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["pool2d_x"], "Y": [ele_mul_input]},
            outputs={"Out": ["ew_mul_out"]},
            axis=-1,
        )
        ops.append(ew_mul_op)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_data, conv2d_op1_w_shape)
                ),
                "ew_bias": TensorConfig(shape=conv2d_op1_b_shape),
                "conv2d_weight2": TensorConfig(
                    data_gen=partial(generate_data, conv2d_op2_w_shape)
                ),
                "ew_bias2": TensorConfig(shape=conv2d_op2_b_shape),
            },
            inputs={
                "pool2d_x": TensorConfig(shape=x_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["squeeze_excitation_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
