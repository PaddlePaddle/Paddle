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


class TestReshapeTransposeMatmulMkldnnFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        def generate_input():
            return np.random.random([128, 768]).astype(np.float32)

        ops_config = [{
            "op_type": "reshape2",
            "op_inputs": {
                "X": ["input_data"]
            },
            "op_outputs": {
                "Out": ["reshape21_output"],
                "XShape": ["reshape21_xshape"]
            },
            "op_attrs": {
                'shape': [1, 128, 12, 64]
            },
        }, {
            "op_type": "transpose2",
            "op_inputs": {
                "X": ["reshape21_output"]
            },
            "op_outputs": {
                "Out": ["transpose21_output"],
                "XShape": ["transpose21_xshape"]
            },
            "op_attrs": {
                'axis': [0, 2, 3, 1]
            },
        }, {
            "op_type": "reshape2",
            "op_inputs": {
                "X": ["input_data"]
            },
            "op_outputs": {
                "Out": ["reshape22_output"],
                "XShape": ["reshape22_xshape"]
            },
            "op_attrs": {
                'shape': [1, 128, 12, 64]
            },
        }, {
            "op_type": "transpose2",
            "op_inputs": {
                "X": ["reshape22_output"]
            },
            "op_outputs": {
                "Out": ["transpose22_output"],
                "XShape": ["transpose22_xshape"]
            },
            "op_attrs": {
                'axis': [0, 2, 1, 3]
            },
        }, {
            "op_type": "matmul",
            "op_inputs": {
                "X": ["transpose22_output"],
                "Y": ["transpose21_output"]
            },
            "op_outputs": {
                "Out": ["matmul_output"]
            },
            "op_attrs": {
                'transpose_X': False,
                'transpose_Y': False,
                'alpha': 1.0,
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": []
            }
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["matmul_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["matmul"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["reshape_transpose_matmul_mkldnn_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
