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


class TestFcGruFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        axis = draw(st.sampled_from([-1]))
        activation = draw(st.sampled_from(['sigmoid', 'tanh']))
        is_reverse = draw(st.booleans())
        has_origin_mode = draw(st.booleans())
        origin_mode = False
        gate_activation = draw(st.sampled_from(['sigmoid', 'tanh']))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input(attrs):
            shape = [attrs[3]['batch_size'], 128, 6, 120]
            return np.full(shape, 0.001).astype(np.float32)

        def generate_weight(shape):
            return np.full(shape, 0.0001).astype(np.float32)

        attrs = [{
            'x_col': x_col,
            'y_col': y_col
        }, {
            'axis': axis
        }, {
            'activation': activation,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'origin_mode': origin_mode
        }, {
            'batch_size': batch_size
        }]

        ops_config = [{
            "op_type": "im2sequence",
            "op_inputs": {
                "X": ["input_data"]
            },
            "op_outputs": {
                "Out": ["seq_out"]
            },
            "op_attrs": {
                "kernels": [6, 1],
                "out_stride": [1, 1],
                "paddings": [0, 0, 0, 0],
                "strides": [1, 1]
            }
        }, {
            "op_type": "mul",
            "op_inputs": {
                "X": ["seq_out"],
                "Y": ["mul_weight"]
            },
            "op_outputs": {
                "Out": ["mul_out"]
            },
            "op_attrs": {
                "x_num_col_dims": attrs[0]['x_col'],
                "y_num_col_dims": attrs[0]['y_col']
            }
        }, {
            "op_type": "elementwise_add",
            "op_inputs": {
                "X": ["mul_out"],
                "Y": ["elt_weight"]
            },
            "op_outputs": {
                "Out": ["elt_out"]
            },
            "op_attrs": {
                "axis": attrs[1]['axis'],
            }
        }, {
            "op_type": "gru",
            "op_inputs": {
                "Input": ["elt_out"],
                "Weight": ["gru_weight"],
                "Bias": ["gru_bias"]
            },
            "op_outputs": {
                "BatchGate": ["batch_gate"],
                "BatchHidden": ["batch_hidden"],
                "BatchResetHiddenPrev": ["batch_reset"],
                "Hidden": ["hidden"]
            },
            "op_attrs": {
                'activation': attrs[2]['activation'],
                'is_reverse': attrs[2]['is_reverse'],
                'gate_activation': attrs[2]['gate_activation'],
                'is_test': True,
            }
        }]

        if has_origin_mode:
            ops_config[3]["op_attrs"]['origin_mode'] = origin_mode

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "mul_weight":
                TensorConfig(data_gen=partial(generate_weight, [768, 600])),
                "elt_weight":
                TensorConfig(data_gen=partial(generate_weight, [600])),
                "gru_weight":
                TensorConfig(data_gen=partial(generate_weight, [200, 600])),
                "gru_bias":
                TensorConfig(data_gen=partial(generate_weight, [1, 600]))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, attrs))
            },
            outputs=["hidden"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["im2sequence", "fusion_gru"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(quant=False, passes=["fc_gru_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
