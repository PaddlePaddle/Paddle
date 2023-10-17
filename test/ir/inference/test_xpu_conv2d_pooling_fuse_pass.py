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


class TestConv2dPoolingXPU(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["conv2d_pooling_xpu"], (1e-3, 1e-3)


    def sample_program_config(self, draw):
        data_format = draw(st.sampled_from(["NCHW"]))

        print("data_format:", data_format)

        x_shape=[3072,32,1,32]

        pooling_type = draw(st.sampled_from(["max", "avg"]))
        
        ceil_mode = draw(st.booleans())

        exclusive = draw(st.booleans())
        
        global_pooling = draw(st.booleans())

        # 3. Generate legal shape of input:Y of conv2d
        w_shape=[32,32,1,3]

        padding_algorithm = draw(st.sampled_from(["SAME", "VALID"]))

        print("padding_algo:",padding_algorithm)

        groups = draw(st.integers(min_value=1, max_value=1))

        axis = 1
        
        ew_bias_shape = [w_shape[0]]

        # Random choose if add a relu operator
        has_relu = True

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while runing
        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["conv2d_input"],
                "Filter": ["conv2d_weight"],
            },
            outputs={"Output": ["conv2d_out"]},
            data_format="NCHW",
            dilations=[1,1],
            padding_algorithm="EXPLICIT",
            groups=1,
            paddings=[0,1],
            strides=[1,1],
            has_bias=False,
        )
        
        ew_bias_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["conv2d_out"], "Y": ["ew_bias"]},
            outputs={"Out": ["add_out"]},
            axis=axis,
        )
        ops = [conv2d_op, ew_bias_op]
        
        # 3. activation
        if has_relu:
            relu_op = OpConfig(
                "relu", inputs={"X": ["add_out"]}, outputs={"Out": ["relu_out"]}
            )
            ops.append(relu_op)

        pool_op = OpConfig(
            "pool2d",
            inputs={"X": ["relu_out"]},
            outputs={"Out": ["pool_output"]},
            ksize=[1, 2],
            adaptive=False,
            pooling_type="avg",
            data_format="NCHW",
            strides1=[1,2],
            paddings1=[0,0],
            ceil_mode=True,
            global_pooling=False,
            padding_algorithm="EXPLICIT",
            exclusive=True,
        )
        ops.append(pool_op)

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "conv2d_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_data, w_shape)
                ),
                "ew_bias": TensorConfig(
                    data_gen=partial(generate_data, ew_bias_shape)
                ),
            },
            outputs=["pool_output"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["conv2d_pooling_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()