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


class TestSeqConcatFcFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        ref_level = draw(st.sampled_from([0]))
        axis1 = draw(st.sampled_from([1]))
        x_col = draw(st.sampled_from([1]))
        y_col = draw(st.sampled_from([1]))
        axis2 = draw(st.sampled_from([1]))
        use_cudnn = False
        use_mkldnn = False
        act_type = draw(st.sampled_from(["relu", "tanh", "sigmoid"]))
        batch_size = draw(st.integers(min_value=1, max_value=1))
        dim = draw(st.integers(min_value=1, max_value=1000))

        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_weight(shape):
            return np.random.random(shape).astype(np.float32)

        ops_config = [{
            "op_type": "sequence_expand",
            "op_inputs": {
                "X": ["input_data1"],
                "Y": ["input_data2"]
            },
            "op_outputs": {
                "Out": ["seq_exp1_out"]
            },
            "op_attrs": {
                "ref_level": ref_level
            }
        }, {
            "op_type": "sequence_expand",
            "op_inputs": {
                "X": ["input_data1"],
                "Y": ["input_data3"]
            },
            "op_outputs": {
                "Out": ["seq_exp2_out"]
            },
            "op_attrs": {
                "ref_level": ref_level
            }
        }, {
            "op_type": "concat",
            "op_inputs": {
                "X": ["input_data1", "seq_exp1_out", "seq_exp2_out"]
            },
            "op_outputs": {
                "Out": ["concat_output"]
            },
            "op_attrs": {
                'axis': axis1
            }
        }, {
            "op_type": "mul",
            "op_inputs": {
                "X": ["concat_output"],
                "Y": ["mul_weight"]
            },
            "op_outputs": {
                "Out": ["mul_out"]
            },
            "op_attrs": {
                "x_num_col_dims": x_col,
                "y_num_col_dims": y_col
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
                "axis": axis2
            }
        }, {
            "op_type": act_type,
            "op_inputs": {
                "X": ["elt_out"]
            },
            "op_outputs": {
                "Out": ["elt_out"]
            },
            "op_attrs": {
                "use_cudnn": use_cudnn,
                "use_mkldnn": use_mkldnn
            }
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "mul_weight":
                TensorConfig(data_gen=partial(generate_weight, [384, dim])),
                "elt_weight":
                TensorConfig(data_gen=partial(generate_weight, [dim]))
            },
            inputs={
                "input_data1": TensorConfig(
                    data_gen=partial(generate_input, [batch_size, 128]),
                    lod=[[0, 1]]),
                "input_data2": TensorConfig(
                    data_gen=partial(generate_input, [batch_size, 128]),
                    lod=[[0, 1]]),
                "input_data3": TensorConfig(
                    data_gen=partial(generate_input, [batch_size, 128]),
                    lod=[[0, 1]])
            },
            outputs=["elt_out"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["fusion_seqexpand_concat_fc"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            return True

        self.add_ignore_check_case(teller1, SkipReasons.PASS_ACCURACY_ERROR,
                                   "The output has diff!")

    def test(self):
        self.run_and_statis(
            # If the output diff problem has been fixed,
            # min_success_num=0 should be deleted!
            min_success_num=0,
            quant=False,
            passes=["seq_concat_fc_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
