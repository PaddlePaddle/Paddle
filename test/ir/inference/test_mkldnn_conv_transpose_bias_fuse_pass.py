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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestConvTransposeMkldnnFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if attrs[0]['data_format'] == "NCHW" and attrs[1]["axis"] == 3:
            return False
        if attrs[0]['data_format'] == "NHWC" and attrs[1]["axis"] == 1:
            return False

        return True

    def sample_program_config(self, draw):
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        dilations = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        groups = draw(st.sampled_from([1, 2, 4, 8]))
        paddings = draw(st.sampled_from([[0, 3], [1, 2, 3, 4]]))
        strides = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        axis = draw(st.sampled_from([1, 3]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            if data_format == "NCHW":
                return np.random.random([batch_size, 16, 64, 64]).astype(
                    np.float32
                )
            else:
                return np.random.random([batch_size, 64, 64, 16]).astype(
                    np.float32
                )

        def generate_weight1():
            return np.random.random([16, 16, 3, 3]).astype(np.float32)

        def generate_weight2():
            return np.random.random([16 * groups]).astype(np.float32)

        conv2d_op = OpConfig(
            type="conv2d_transpose",
            inputs={"Input": ["input_data"], "Filter": ["conv2d_weight"]},
            outputs={"Output": ["conv_output"]},
            attrs={
                "data_format": data_format,
                "dilations": dilations,
                "padding_algorithm": padding_algorithm,
                "groups": groups,
                "paddings": paddings,
                "strides": strides,
                "output_size": [],
                "output_padding": [],
                "is_test": True,
            },
        )

        elt_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["conv_output"], "Y": ["elementwise_weight"]},
            outputs={"Out": ["elementwise_output"]},
            attrs={'axis': axis},
        )

        model_net = [conv2d_op, elt_op]

        program_config = ProgramConfig(
            ops=model_net,
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_weight1)
                ),
                "elementwise_weight": TensorConfig(
                    data_gen=partial(generate_weight2)
                ),
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["elementwise_output"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['conv2d_transpose_bias'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_duration=300,
            passes=["conv_transpose_bias_onednn_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
