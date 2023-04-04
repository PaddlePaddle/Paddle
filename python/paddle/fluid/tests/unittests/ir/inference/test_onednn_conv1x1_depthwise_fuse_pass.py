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


class TestOneDNNCon1x1vDepthwiseFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        groups = 1
        filter_channel = 1
        in_channel = groups * filter_channel
        out_channel = groups * 1

        groups_depthwise = 1
        in_channel_depthwise = groups_depthwise
        out_channel_depthwise = groups_depthwise

        data_format = draw(st.sampled_from(["NCHW"]))

        batch_size = 1
        dilations = draw(
            st.lists(
                st.integers(min_value=1, max_value=2), min_size=2, max_size=2
            )
        )
        paddings = draw(
            st.lists(
                st.integers(min_value=0, max_value=2), min_size=2, max_size=2
            )
        )
        paddings_depthwise = [1, 1]
        strides = [1, 1]

        x_shape = [batch_size, in_channel, 4, 4]
        filter1x1_shape = [out_channel, in_channel, 1, 1]
        filter_depthwise_shape = [
            out_channel_depthwise,
            in_channel_depthwise,
            3,
            3,
        ]

        def generate_input(shape):
            return np.ones(shape).astype(np.float32)

        def generate_filter(shape):
            return np.ones(shape).astype(np.float32)

        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["conv2d_input"],
                "Filter": ["conv2d_filter_1"],
            },
            outputs={"Output": ["conv2d_out"]},
            data_format=data_format,
            dilations=dilations,
            padding_algorithm=padding_algorithm,
            groups=groups,
            paddings=paddings,
            strides=strides,
            use_mkldnn=True,
            has_bias=False,
            is_test=True,
        )

        depthwise_conv2d_op = OpConfig(
            type="conv2d",
            inputs={
                "Input": ["conv2d_out"],
                "Filter": ["conv2d_filter_2"],
            },
            outputs={
                "Output": ["depth_conv2d_out"],
            },
            attrs={
                'data_format': data_format,
                'dilations': dilations,
                'padding_algorithm': padding_algorithm,
                'groups': groups_depthwise,
                'paddings': paddings_depthwise,
                'strides': strides,
                'use_mkldnn': True,
                'has_bias': False,
                'is_test': True,
            },
        )

        program_config = ProgramConfig(
            ops=[conv2d_op, depthwise_conv2d_op],
            inputs={
                "conv2d_input": TensorConfig(
                    data_gen=partial(generate_input, x_shape)
                ),
            },
            weights={
                "conv2d_filter_1": TensorConfig(
                    data_gen=partial(generate_filter, filter1x1_shape)
                ),
                "conv2d_filter_2": TensorConfig(
                    data_gen=partial(generate_filter, filter_depthwise_shape)
                ),
            },
            outputs=["depth_conv2d_out"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            use_mkldnn=True, passes=["conv1x1_depthwise_conv_mkldnn_fuse_pass"]
        )
        yield config, ['fused_conv2d'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=["conv1x1_depthwise_conv_mkldnn_fuse_pass"],
        )


if __name__ == '__main__':
    unittest.main()
