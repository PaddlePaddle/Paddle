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


class TestConvTransposeEltwiseaddBnFusePass(PassAutoScanTest):
    '''
    conv_input   conv_weight_var(persistable)
      \       /
         conv_op     
          |
      conv_out_var  elementwise_add_y 
          |       /
    elementwise_add
          |
    elementwise_add_out (bn_scale_var, bn_bias_var, bn_mean_var,bn_variance_var)
                |            /
                batch_norm_op
                |            \
                bn_out_var     (bn_mean_out_var, bn_variance_out_var,bn_saved_mean_var, bn_saved_variance_var)
    '''

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=150,
            max_duration=250,
            passes=["conv_transpose_eltwiseadd_bn_fuse_pass"])

    def sample_program_config(self, draw):
        # generate random number
        random_batch_size = draw(st.integers(min_value=1, max_value=3))
        random_channel = draw(st.integers(min_value=2, max_value=10))
        random_input_dim1 = draw(st.integers(min_value=20, max_value=50))
        random_input_dim2 = draw(st.integers(min_value=20, max_value=50))
        random_groups = draw(st.integers(min_value=1, max_value=2))
        random_dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=3), min_size=2, max_size=2))
        random_strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))
        random_paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=4), min_size=2, max_size=2))
        random_padding_algorithm = draw(
            st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        random_data_layout = draw(st.sampled_from(["NCHW", "NHWC"]))
        random_use_mkldnn = draw(st.booleans())
        random_output_size = []
        random_filter = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))
        random_out_channel = draw(st.integers(min_value=20, max_value=25))
        random_epsilon = draw(st.floats(min_value=0.0, max_value=0.001))

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
            shape.insert(0, random_out_channel * random_groups)
            shape.insert(0, random_channel * random_groups)
            return np.random.random(shape).astype(np.float32)

        def generate_elementwise_add_Y():
            return np.random.random(
                [random_out_channel * random_groups * random_groups]).astype(
                    np.float32)

        def generate_batch_norm_Scale():
            return np.random.random(
                [random_out_channel * random_groups * random_groups]).astype(
                    np.float32)

        def generate_batch_norm_Bias():
            return np.random.random(
                [random_out_channel * random_groups * random_groups]).astype(
                    np.float32)

        def generate_batch_norm_Mean():
            return np.random.random(
                [random_out_channel * random_groups * random_groups]).astype(
                    np.float32)

        def generate_batch_norm_Variance():
            return np.random.random(
                [random_out_channel * random_groups * random_groups]).astype(
                    np.float32)

        # define op
        conv2d_op = OpConfig(
            type="conv2d_transpose",
            inputs={
                "Input": ["conv2d_Input"],
                "Filter": ["conv2d_Filter"],
            },
            outputs={"Output": ["conv2d_Out"], },
            attrs={
                'groups': random_groups,
                'dilations': random_dilations,
                'strides': random_strides,
                'paddings': random_paddings,
                'padding_algorithm': random_padding_algorithm,
                'data_format': random_data_layout,
                'output_size': random_output_size,
                'output_padding': random_output_size,
                'use_mkldnn': random_use_mkldnn,
                'is_test': True,
            })

        elementwise_op = OpConfig(
            type="elementwise_add",
            inputs={
                "X": ["conv2d_Out"],
                "Y": ["elementwise_add_Y"],
            },
            outputs={"Out": ["elementwise_add_Out"], },
            attrs={'axis': 1, })

        batch_norm_op = OpConfig(
            type="batch_norm",
            inputs={
                "X": ["elementwise_add_Out"],
                "Scale": ["batch_norm_Scale"],
                "Bias": ["batch_norm_Bias"],
                "Mean": ["batch_norm_Mean"],
                "Variance": ["batch_norm_Variance"],
            },
            outputs={
                "Y": ["batch_norm_Y"],
                "MeanOut": ["batch_norm_Mean"],
                "VarianceOut": ["batch_norm_Variance"],
                "SavedMean": ["batch_norm_SavedMean"],
                "SavedVariance": ["batch_norm_SavedVariance"],
                "ReserveSpace": ["batch_norm_ReserveSpace"],
            },
            attrs={
                'epsilon': random_epsilon,
                'is_test': True,
                'trainable_statistics': False,
                'data_layout': random_data_layout,
                'use_mkldnn': random_use_mkldnn,
            })

        # define model_net
        model_net = [conv2d_op, elementwise_op, batch_norm_op]

        # set tensor
        program_config = ProgramConfig(
            ops=model_net,
            inputs={
                "conv2d_Input": TensorConfig(data_gen=generate_conv2d_Input),
            },
            weights={
                "conv2d_Filter": TensorConfig(data_gen=generate_conv2d_Filter),
                "elementwise_add_Y":
                TensorConfig(data_gen=generate_elementwise_add_Y),
                "batch_norm_Scale":
                TensorConfig(data_gen=generate_batch_norm_Scale),
                "batch_norm_Bias":
                TensorConfig(data_gen=generate_batch_norm_Bias),
                "batch_norm_Mean":
                TensorConfig(data_gen=generate_batch_norm_Mean),
                "batch_norm_Variance":
                TensorConfig(data_gen=generate_batch_norm_Variance),
            },
            outputs=["batch_norm_Y"])

        return program_config

    def sample_predictor_configs(self, program_config):
        # for mkldnn
        config = self.create_inference_config()
        if program_config.ops[2].attrs['use_mkldnn']:
            config.enable_mkldnn()
            yield config, ['conv2d_transpose', 'elementwise_add'], (1e-5, 1e-5)
        # cpu
        else:
            yield config, ['conv2d_transpose', 'elementwise_add'], (1e-5, 1e-5)

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if attrs[0]['data_format'] == "NHWC":
            return False

        return True

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if program_config.ops[0].attrs['data_format'] == "NHWC":
                return True
            return False

        def teller2(program_config, predictor_config):
            if program_config.ops[0].attrs['groups'] != 1:
                return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PASS_ACCURACY_ERROR,
            "The output format of conv2d_transpose is wrong when data_format attribute is NHWC"
        )
        self.add_ignore_check_case(teller2, IgnoreReasons.PASS_ACCURACY_ERROR,
                                   "there is diff when group >1 in this pass")
