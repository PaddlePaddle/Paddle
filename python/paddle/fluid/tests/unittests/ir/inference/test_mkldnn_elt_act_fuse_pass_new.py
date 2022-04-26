# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from auto_scan_test import PassAutoScanTest
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestElementWiseAddReluFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input():
            return np.random.random(
                [batch_size, 3, 100, 100]).astype(np.float32)

        ops_config = [{
            "op_type": "elementwise_add",
            "op_inputs": {
                "X": ["A"],
                "Y": ["B"]
            },
            "op_outputs": {
                "Out": ["add_output"]
            },
            "op_attrs": {}
        }, {
            "op_type": "relu",
            "op_inputs": {
                "X": ["add_output"]
            },
            "op_outputs": {
                "Out": ["relu_output"]
            },
            "op_attrs": {}
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "A": TensorConfig(data_gen=partial(generate_input)),
                "B": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["relu_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["elementwise_add"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["elt_act_mkldnn_fuse_pass"], min_success_num=4)


if __name__ == "__main__":
    unittest.main()
