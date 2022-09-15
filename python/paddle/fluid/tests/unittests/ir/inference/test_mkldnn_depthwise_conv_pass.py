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
import copy as cp
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class DepthwiseConvMKLDNNPass(PassAutoScanTest):
    '''
    conv_input   conv_weight_var(persistable)
      \       /
         conv_op
          |
      conv_out_var
    '''

    def test(self):
        self.run_and_statis(quant=False, passes=["depthwise_conv_mkldnn_pass"])

    def sample_program_config(self, draw):
        # generate random number
        random_batch_size = draw(st.integers(min_value=1, max_value=4))
        random_channel = draw(st.integers(min_value=2, max_value=10))
        random_input_dim1 = draw(st.integers(min_value=20, max_value=50))
        random_input_dim2 = draw(st.integers(min_value=20, max_value=50))
        random_out_channel = draw(st.integers(min_value=20, max_value=25))

        random_groups = draw(st.integers(min_value=1, max_value=3))
        random_dilations = draw(
            st.lists(st.integers(min_value=1, max_value=3),
                     min_size=2,
                     max_size=2))
        random_strides = draw(
            st.lists(st.integers(min_value=1, max_value=4),
                     min_size=2,
                     max_size=2))
        random_paddings = draw(
            st.lists(st.integers(min_value=0, max_value=4),
                     min_size=2,
                     max_size=2))
        random_padding_algorithm = draw(
            st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        random_data_layout = draw(st.sampled_from(["NCHW", "NHWC"]))
        random_filter = draw(
            st.lists(st.integers(min_value=1, max_value=4),
                     min_size=2,
                     max_size=2))

        def generate_conv2d_Input():
            shape = [random_input_dim1, random_input_dim2]
            if random_data_layout == "NCHW":
                shape.insert(0, random_channel * random_groups)
                shape.insert(0, random_batch_size)
            else:
                shape.append(random_channel)
                shape.insert(0, random_batch_size)
            return np.random.random(shape).astype(np.float32)

        def generate_conv2d_Filter():
            shape = cp.copy(random_filter)
            shape.insert(0, random_channel)
            shape.insert(0, random_out_channel * random_groups)
            return np.random.random(shape).astype(np.float32)

        # define op
        conv2d_op = OpConfig(type="depthwise_conv2d",
                             inputs={
                                 "Input": ["conv2d_Input"],
                                 "Filter": ["conv2d_Filter"],
                             },
                             outputs={
                                 "Output": ["conv2d_Out"],
                             },
                             attrs={
                                 'groups': random_groups,
                                 'dilations': random_dilations,
                                 'strides': random_strides,
                                 'paddings': random_paddings,
                                 'padding_algorithm': random_padding_algorithm,
                                 'data_format': random_data_layout,
                                 'use_mkldnn': True,
                             })

        # define model_net
        model_net = [conv2d_op]

        # set tensor
        program_config = ProgramConfig(
            ops=model_net,
            inputs={
                "conv2d_Input": TensorConfig(data_gen=generate_conv2d_Input),
            },
            weights={
                "conv2d_Filter": TensorConfig(data_gen=generate_conv2d_Filter),
            },
            outputs=["conv2d_Out"])

        return program_config

    def sample_predictor_configs(self, program_config):
        # for mkldnn
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['conv2d'], (1e-5, 1e-5)

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if attrs[0]['data_format'] == "NHWC":
            return False

        return True

    def add_ignore_pass_case(self):

        def teller1(program_config, predictor_config):
            if program_config.ops[0].attrs['data_format'] == "NHWC":
                return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PASS_ACCURACY_ERROR,
            "The output format of depthwise_conv2d is wrong when data_format attribute is NHWC"
        )
