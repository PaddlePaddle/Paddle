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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestConvBiasMkldnnFusePass(PassAutoScanTest):
    """
    x_var   f_var(persistable)
      \       /
        conv2d 
          |
      conv2d_var  bias_var(persistable)
              \      /
           elementwise_add
                |
         elementwise_add_var
    """

    def sample_predictor_configs(self, program_config):
        # MKLDNN
        config = self.create_inference_config(use_gpu=False)
        config.enable_mkldnn()
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
        # 1. Generate shape of input:X of conv2d
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=5, max_value=100), min_size=4, max_size=4))
        x_shape[1] = draw(st.integers(min_value=5, max_value=10))

        # 2. Generate legal attr:data_format of conv2d
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

        # 3. Generate legal shape of input:Y of conv2d
        f_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=4, max_size=4))
        if data_format == "NCHW":
            f_shape[1] = x_shape[1]
        else:
            f_shape[1] = x_shape[3]

        # 4. Generate legal attr:strides of conv2d
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))

        # 5. Generate legal attr:padding_algorithm of conv2d
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))

        # 6. Generate legal attr:padding of conv2d
        padding = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=4, max_size=4))

        # 7. Generate legal attr:groups of conv2d
        groups = draw(st.integers(min_value=1, max_value=3))

        # 8. Generate legal attr:dilations of conv2d
        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))

        # 9. Generate legal shape of input:bias of elementwise_add
        bias_shape = [f_shape[0]]

        # 10. Generate legal shape of attr:axis of elementwise_add
        axis = 1
        if data_format == "NCHW":
            axis = 1
        else:
            axis = 3

        # 11. Generate legal shape of input:bias of conv2d
        conv_bias_shape = []
        inputs = dict()
        weights = dict()
        use_mkldnn = None
        if draw(st.booleans()):
            conv_bias_shape = [f_shape[0]]
            inputs = {
                "Input": ["input_x"],
                "Filter": ["filter"],
                "Bias": ["conv_bias"],
            }
            weights = {
                "filter": TensorConfig(shape=f_shape),
                "bias": TensorConfig(shape=bias_shape),
                "conv_bias": TensorConfig(shape=conv_bias_shape)
            }
            use_mkldnn = True
        else:
            inputs = {
                "Input": ["input_x"],
                "Filter": ["filter"],
            }
            weights = {
                "filter": TensorConfig(shape=f_shape),
                "bias": TensorConfig(shape=bias_shape)
            }
            use_mkldnn = False

        conv2d_op = OpConfig(
            "conv2d",
            inputs=inputs,
            outputs={"Output": ["conv2d_out"]},
            strides=strides,
            padding_algorithm=padding_algorithm,
            paddings=padding,
            groups=groups,
            dilations=dilations,
            data_format=data_format,
            use_mkldnn=use_mkldnn)

        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["conv2d_out"],
                    "Y": ["bias"]},
            outputs={"Out": ["add_out"]},
            axis=axis)

        ops = [conv2d_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs={"input_x": TensorConfig(shape=x_shape)},
            outputs=ops[-1].outputs["Out"])
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=350,
            passes=["conv_bias_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
