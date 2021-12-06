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

from auto_scan_test import PassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestConvAffineChannelFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        groups = draw(st.integers(min_value=1, max_value=3))
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        axis = draw(st.sampled_from([1]))
        filter_channel = draw(st.integers(min_value=1, max_value=16)) * 4
        filter_size = draw(st.integers(min_value=1, max_value=4))
        in_channel = groups * filter_channel
        out_channel_factor = draw(st.integers(min_value=1, max_value=16)) * 4
        out_channel = groups * out_channel_factor
        batch_size = draw(st.integers(min_value=1, max_value=4))
        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=2, max_size=2))
        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=2), min_size=2, max_size=2))
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=2), min_size=2, max_size=2))

        x_shape = [
            batch_size, in_channel, 64, 64
        ] if data_format == "NCHW" else [batch_size, 64, 64, in_channel]
        w_shape = [out_channel, filter_channel, filter_size, filter_size]
        scale_shape = [out_channel]
        bias_shape = [out_channel]

        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["input_data"],
                "Filter": ["conv2d_weight"],
                # "Bias": ["conv2d_bias"],
            },
            outputs={"Output": ["conv_output"]},
            data_format=data_format,
            dilations=dilations,
            padding_algorithm=padding_algorithm,
            groups=groups,
            paddings=paddings,
            strides=strides,
            is_test=True)
        ac_op = OpConfig(
            "affine_channel",
            inputs={
                "X": ["conv_output"],
                "Scale": ["affine_channel_scale"],
                "Bias": ["affine_channel_bias"]
            },
            outputs={"Out": ["affine_channel_ouput"]},
            data_layout=data_format)
        ops = [conv2d_op, ac_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "conv2d_weight": TensorConfig(shape=w_shape),
                "affine_channel_scale": TensorConfig(shape=scale_shape),
                "affine_channel_bias": TensorConfig(shape=bias_shape),
                # "conv2d_bias": TensorConfig(shape=bias_shape)
            },
            inputs={"input_data": TensorConfig(shape=x_shape), },
            outputs=ops[-1].outputs["Out"], )
        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['conv2d', 'elementwise_add'], (1e-4, 1e-4)

        # mkldnn Output has diff with bias!!!
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['conv2d', 'elementwise_add'], (1e-4, 1e-4)

    def add_ignore_pass_case(self):
        # If the problem has been fixed, the judgment 
        # in is_program_valid needs to be deleted!!!
        def teller1(program_config, predictor_config):
            if program_config.ops[0].attrs['data_format'] == "NHWC":
                return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PASS_ACCURACY_ERROR,
            "The output format of conv2d is wrong when data_format attribute is NHWC, \
            Operator(Conv2DFusion) only supports data format of channel first (NCHW) now."
        )

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=["conv_affine_channel_fuse_pass"], )


if __name__ == "__main__":
    unittest.main()
