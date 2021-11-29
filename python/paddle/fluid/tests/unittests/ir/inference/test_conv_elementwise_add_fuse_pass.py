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
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestConvEltwiseAddFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if attrs[0]['data_format'] == "NHWC":
            return False

        return True

    def sample_program_config(self, draw):
        strides = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        paddings = draw(st.sampled_from([[0, 3], [1, 2, 3, 4]]))
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        groups = draw(st.sampled_from([1]))
        dilations = draw(st.sampled_from([[1, 1], [2, 2], [1, 2]]))
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        axis = draw(st.sampled_from([1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input(attrs):
            if attrs[0]['data_format'] == "NCHW":
                return np.random.random(
                    [attrs[2]['batch_size'], 16, 64, 64]).astype(np.float32)
            else:
                return np.random.random(
                    [attrs[2]['batch_size'], 64, 64, 16]).astype(np.float32)

        def generate_weight1():
            return np.random.random([16, 16, 3, 3]).astype(np.float32)

        def generate_weight2():
            return np.random.random([16]).astype(np.float32)

        def generate_scale_bias():
            return np.random.random([16]).astype(np.float32)

        attrs = [{
            "data_format": data_format,
            "dilations": dilations,
            "padding_algorithm": padding_algorithm,
            "groups": groups,
            "paddings": paddings,
            "strides": strides
        }, {
            "axis": axis
        }, {
            'batch_size': batch_size
        }]

        ops_config = [
            {
                "op_type": "conv2d",
                "op_inputs": {
                    "Input": ["input_data"],
                    "Filter": ["conv2d_weight"],
                    "Bias": ["conv2d_bias"],
                },
                "op_outputs": {
                    "Output": ["conv_output"]
                },
                "op_attrs": {
                    "data_format": attrs[0]['data_format'],
                    "dilations": attrs[0]['dilations'],
                    "padding_algorithm": attrs[0]['padding_algorithm'],
                    "groups": attrs[0]['groups'],
                    "paddings": attrs[0]['paddings'],
                    "strides": attrs[0]['strides'],
                    #"output_size": [],
                    #"output_padding": [],
                    "use_cudnn": False,
                    "is_test": True
                }
            },
            {
                "op_type": "elementwise_add",
                "op_inputs": {
                    "X": ["conv_output"],
                    "Y": ["conv2d_bias"]
                },
                "op_outputs": {
                    "Out": ["elementwise_output"]
                },
                "op_attrs": {
                    'axis': attrs[1]['axis']
                },
            }
        ]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, attrs)),
            },
            weights={
                "conv2d_weight":
                TensorConfig(data_gen=partial(generate_weight1)),
                "conv2d_bias":
                TensorConfig(data_gen=partial(generate_scale_bias)),
            },
            outputs=["elementwise_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['conv2d_fusion'], (1e-5, 1e-5)

        # TRT
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=4,
            min_subgraph_size=1,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, ['conv2d_fusion'], (1e-5, 1e-5)

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
            passes=["conv_elementwise_add_fuse_pass"], )


if __name__ == "__main__":
    unittest.main()
