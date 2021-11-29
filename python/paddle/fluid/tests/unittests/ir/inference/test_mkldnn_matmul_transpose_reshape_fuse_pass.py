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


class TestMatmulTransposeReshapeMkldnnFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        alpha = draw(st.floats(min_value=0.01, max_value=2))
        axis = draw(st.sampled_from([[0, 2, 1, 3]]))
        shape = draw(st.sampled_from([[0, -1, 128], [-1, 1, 64]]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        channel = draw(st.integers(min_value=1, max_value=64))
        input_dim = draw(st.sampled_from([32, 64]))

        def generate_input(attrs, type):
            if attrs[0]['transpose_X'] and attrs[0]['transpose_Y']:
                shape_x = [
                    attrs[3]['batch_size'], attrs[3]['channel'],
                    attrs[3]['input_dim'], 32
                ]
                shape_y = [
                    attrs[3]['batch_size'], attrs[3]['channel'], 64,
                    attrs[3]['input_dim']
                ]
            elif attrs[0]['transpose_X']:
                shape_x = [
                    attrs[3]['batch_size'], attrs[3]['channel'],
                    attrs[3]['input_dim'], 32
                ]
                shape_y = [
                    attrs[3]['batch_size'], attrs[3]['channel'],
                    attrs[3]['input_dim'], 64
                ]
            elif attrs[0]['transpose_Y']:
                shape_x = [
                    attrs[3]['batch_size'], attrs[3]['channel'], 32,
                    attrs[3]['input_dim']
                ]
                shape_y = [
                    attrs[3]['batch_size'], attrs[3]['channel'], 8,
                    attrs[3]['input_dim']
                ]
            else:
                shape_x = [
                    attrs[3]['batch_size'], attrs[3]['channel'], 32,
                    attrs[3]['input_dim']
                ]
                shape_y = [
                    attrs[3]['batch_size'], attrs[3]['channel'],
                    attrs[3]['input_dim'], 16
                ]

            if type == "x":
                return np.random.random(shape_x).astype(np.float32)
            else:
                return np.random.random(shape_y).astype(np.float32)

        attrs = [{
            "transpose_X": transpose_X,
            "transpose_Y": transpose_Y,
            "alpha": alpha
        }, {
            "axis": axis
        }, {
            "shape": shape
        }, {
            'batch_size': batch_size,
            'channel': channel,
            'input_dim': input_dim
        }]

        ops_config = [{
            "op_type": "matmul",
            "op_inputs": {
                "X": ["input_data1"],
                "Y": ["input_data2"]
            },
            "op_outputs": {
                "Out": ["matmul_output"]
            },
            "op_attrs": {
                "transpose_X": attrs[0]["transpose_X"],
                "transpose_Y": attrs[0]["transpose_Y"],
                "alpha": attrs[0]["alpha"],
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": []
            }
        }, {
            "op_type": "transpose2",
            "op_inputs": {
                "X": ["matmul_output"]
            },
            "op_outputs": {
                "Out": ["transpose2_output"],
                "XShape": ["transpose2_xshape"]
            },
            "op_attrs": {
                'axis': attrs[1]['axis']
            },
        }, {
            "op_type": "reshape2",
            "op_inputs": {
                "X": ["transpose2_output"]
            },
            "op_outputs": {
                "Out": ["reshape2_output"],
                "XShape": ["reshape2_xshape"]
            },
            "op_attrs": {
                'shape': attrs[2]['shape']
            },
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data1":
                TensorConfig(data_gen=partial(generate_input, attrs, "x")),
                "input_data2":
                TensorConfig(data_gen=partial(generate_input, attrs, "y"))
            },
            outputs=["reshape2_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["matmul"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["matmul_transpose_reshape_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
