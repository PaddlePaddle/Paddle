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
        act_type = draw(st.sampled_from(["tanh", "sigmoid", "relu"]))
        batch_size = draw(st.integers(min_value=1, max_value=1))
        dim = draw(st.integers(min_value=1, max_value=1000))

        def generate_input(shape):
            return np.random.random(shape).astype(np.float32)

        def generate_weight(shape):
            return np.random.random(shape).astype(np.float32)

        sequence_expand_op1 = OpConfig(
            type="sequence_expand",
            inputs={"X": ["input_data1"],
                    "Y": ["input_data2"]},
            outputs={"Out": ["seq_exp1_out"]},
            attrs={"ref_level": ref_level})

        sequence_expand_op2 = OpConfig(
            type="sequence_expand",
            inputs={"X": ["input_data1"],
                    "Y": ["input_data3"]},
            outputs={"Out": ["seq_exp2_out"]},
            attrs={"ref_level": ref_level})

        concat_op = OpConfig(
            type="concat",
            inputs={"X": ["input_data1", "seq_exp1_out", "seq_exp2_out"]},
            outputs={"Out": ["concat_output"]},
            attrs={'axis': axis1})

        mul_op = OpConfig(
            type="mul",
            inputs={"X": ["concat_output"],
                    "Y": ["mul_weight"]},
            outputs={"Out": ["mul_out"]},
            attrs={"x_num_col_dims": x_col,
                   "y_num_col_dims": y_col})

        elt_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["mul_out"],
                    "Y": ["elt_weight"]},
            outputs={"Out": ["elt_out"]},
            attrs={"axis": axis2})

        act_op = OpConfig(
            type=act_type,
            inputs={"X": ["elt_out"]},
            outputs={"Out": ["act_out"]},
            attrs={"use_cudnn": use_cudnn,
                   "use_mkldnn": use_mkldnn})

        model_net = [
            sequence_expand_op1, sequence_expand_op2, concat_op, mul_op, elt_op,
            act_op
        ]

        program_config = ProgramConfig(
            ops=model_net,
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
            outputs=["act_out"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["fusion_seqexpand_concat_fc"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if program_config.ops[-1].type == "relu":
                return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PASS_ACCURACY_ERROR,
            "The pass output has diff in a specific case. We need to fix it as soon as possible."
        )

    def test(self):
        self.run_and_statis(quant=False, passes=["seq_concat_fc_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
