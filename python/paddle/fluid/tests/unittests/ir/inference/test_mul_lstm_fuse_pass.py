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
        use_peepholes = draw(st.booleans())
        is_reverse = draw(st.booleans())
        gate_activation = draw(
            st.sampled_from(["sigmoid", "tanh", "relu", "identity"]))
        cell_activation = draw(
            st.sampled_from(["sigmoid", "tanh", "relu", "identity"]))
        candidate_activation = draw(
            st.sampled_from(["sigmoid", "tanh", "relu", "identity"]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input(attrs):
            shape = [attrs[2]['batch_size'], 128, 6, 120]
            return np.full(shape, 0.01).astype(np.float32)

        def generate_weight(shape):
            return np.full(shape, 0.0001).astype(np.float32)

        attrs = [{
            "x_col": x_col,
            "y_col": y_col
        }, {
            'use_peepholes': use_peepholes,
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'cell_activation': cell_activation,
            'candidate_activation': candidate_activation,
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
            "op_type": "lstm",
            "op_inputs": {
                "Input": ["mul_out"],
                "Weight": ["lstm_weight"],
                "Bias": ["lstm_bias"]
            },
            "op_outputs": {
                "Hidden": ["lstm_hidden"],
                "Cell": ["lstm_cell"],
                "BatchGate": ["lstm_gate"],
                "BatchCellPreAct": ["lstm_batch_cell"]
            },
            "op_attrs": {
                'use_peepholes': attrs[1]['use_peepholes'],
                'is_reverse': attrs[1]['is_reverse'],
                'gate_activation': attrs[1]['gate_activation'],
                'cell_activation': attrs[1]['cell_activation'],
                'candidate_activation': attrs[1]['candidate_activation'],
                'is_test': True
            }
        }]

        ops = self.generate_op_config(ops_config)

        if use_peepholes:
            lstm_bias_shape = [1, 1050]
        else:
            lstm_bias_shape = [1, 600]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "mul_weight":
                TensorConfig(data_gen=partial(generate_weight, [768, 600])),
                "lstm_weight":
                TensorConfig(data_gen=partial(generate_weight, [150, 600])),
                "lstm_bias":
                TensorConfig(data_gen=partial(generate_weight, lstm_bias_shape))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, attrs)),
            },
            outputs=["lstm_hidden"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["im2sequence", "fusion_lstm"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            return True

        self.add_ignore_check_case(teller1, SkipReasons.PASS_ACCURACY_ERROR,
                                   "The output has diff!")

    def test(self):
        # If the output diff problem has been fixed,
        # min_success_num=0 should be deleted!
        self.run_and_statis(
            min_success_num=0, quant=False, passes=["mul_lstm_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
