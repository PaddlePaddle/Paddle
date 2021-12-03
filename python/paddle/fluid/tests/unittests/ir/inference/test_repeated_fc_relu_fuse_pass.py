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
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
from functools import reduce


class TestMulLstmFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        axis = draw(st.sampled_from([-1, 1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        dim = draw(st.sampled_from([32, 64, 128]))

        def generate_input(attrs):
            return np.random.random(
                [attrs[2]['batch_size'], attrs[2]['dim']]).astype(np.float32)

        def generate_weight(shape):
            return np.random.random(shape).astype(np.float32)

        attrs = [{
            "x_col": x_col,
            "y_col": y_col
        }, {
            "axis": axis
        }, {
            'batch_size': batch_size,
            'dim': dim
        }]

        ops_config = [{
            "op_type": "mul",
            "op_inputs": {
                "X": ["input_data"],
                "Y": ["mul1_weight"]
            },
            "op_outputs": {
                "Out": ["mul1_output"]
            },
            "op_attrs": {
                "x_num_col_dims": attrs[0]['x_col'],
                "y_num_col_dims": attrs[0]['y_col']
            }
        }, {
            "op_type": "elementwise_add",
            "op_inputs": {
                "X": ["mul1_output"],
                "Y": ["elementwise1_weight"]
            },
            "op_outputs": {
                "Out": ["elementwise1_output"]
            },
            "op_attrs": {
                'axis': attrs[1]['axis']
            },
        }, {
            "op_type": "relu",
            "op_inputs": {
                "X": ["elementwise1_output"]
            },
            "op_outputs": {
                "Out": ["relu1_output"]
            },
            "op_attrs": {}
        }, {
            "op_type": "mul",
            "op_inputs": {
                "X": ["relu1_output"],
                "Y": ["mul2_weight"]
            },
            "op_outputs": {
                "Out": ["mul2_output"]
            },
            "op_attrs": {
                "x_num_col_dims": attrs[0]['x_col'],
                "y_num_col_dims": attrs[0]['y_col']
            }
        }, {
            "op_type": "elementwise_add",
            "op_inputs": {
                "X": ["mul2_output"],
                "Y": ["elementwise2_weight"]
            },
            "op_outputs": {
                "Out": ["elementwise2_output"]
            },
            "op_attrs": {
                'axis': attrs[1]['axis']
            },
        }, {
            "op_type": "relu",
            "op_inputs": {
                "X": ["elementwise2_output"]
            },
            "op_outputs": {
                "Out": ["relu2_output"]
            },
            "op_attrs": {}
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "mul1_weight": TensorConfig(data_gen=partial(
                    generate_weight, [attrs[2]['dim'], 32])),
                "mul2_weight":
                TensorConfig(data_gen=partial(generate_weight, [32, 128])),
                "elementwise1_weight":
                TensorConfig(data_gen=partial(generate_weight, [32])),
                "elementwise2_weight":
                TensorConfig(data_gen=partial(generate_weight, [128]))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, attrs)),
            },
            outputs=["relu2_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["fusion_repeated_fc_relu"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(passes=["repeated_fc_relu_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
