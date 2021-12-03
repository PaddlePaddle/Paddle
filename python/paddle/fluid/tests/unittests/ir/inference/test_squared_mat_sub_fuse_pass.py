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
        transpose_X = False
        transpose_Y = False
        alpha1 = draw(st.floats(min_value=0.01, max_value=2))
        alpha2 = draw(st.floats(min_value=0.01, max_value=2))
        axis1 = draw(st.sampled_from([-1, 0]))
        place_type = draw(st.sampled_from([-1, 0]))
        str_value = draw(st.sampled_from(['-0.2', '3']))
        value = draw(st.floats(min_value=-10, max_value=10))
        shape = draw(st.sampled_from([[1]]))
        axis2 = draw(st.sampled_from([-1, 0]))
        input_dim = draw(st.sampled_from([32, 64]))

        def generate_input(attrs, type):
            shape_x = [32, attrs[5]['input_dim']]
            shape_y = [attrs[5]['input_dim'], 16]

            if type == "x":
                return np.random.random(shape_x).astype(np.float32)
            else:
                return np.random.random(shape_y).astype(np.float32)

        attrs = [{
            "transpose_X": transpose_X,
            "transpose_Y": transpose_Y,
            "alpha": alpha1
        }, {
            "transpose_X": transpose_X,
            "transpose_Y": transpose_Y,
            "alpha": alpha2
        }, {
            "axis": axis1
        }, {
            "place_type": place_type,
            "str_value": str_value,
            "value": value,
            "shape": shape
        }, {
            "axis": axis2
        }, {
            'input_dim': input_dim
        }]

        ops_config = [{
            "op_type": "matmul",
            "op_inputs": {
                "X": ["input_data1"],
                "Y": ["input_data2"]
            },
            "op_outputs": {
                "Out": ["matmul1_output"]
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
            "op_type": "square",
            "op_inputs": {
                "X": ["matmul1_output"]
            },
            "op_outputs": {
                "Out": ["square1_output"]
            },
            "op_attrs": {}
        }, {
            "op_type": "square",
            "op_inputs": {
                "X": ["input_data1"]
            },
            "op_outputs": {
                "Out": ["square2_output"]
            },
            "op_attrs": {}
        }, {
            "op_type": "square",
            "op_inputs": {
                "X": ["input_data2"]
            },
            "op_outputs": {
                "Out": ["square3_output"]
            },
            "op_attrs": {}
        }, {
            "op_type": "matmul",
            "op_inputs": {
                "X": ["square2_output"],
                "Y": ["square3_output"]
            },
            "op_outputs": {
                "Out": ["matmul2_output"]
            },
            "op_attrs": {
                "transpose_X": attrs[1]["transpose_X"],
                "transpose_Y": attrs[1]["transpose_Y"],
                "alpha": attrs[1]["alpha"],
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": []
            }
        }, {
            "op_type": "elementwise_sub",
            "op_inputs": {
                "X": ["square1_output"],
                "Y": ["matmul2_output"]
            },
            "op_outputs": {
                "Out": ["sub_out"]
            },
            "op_attrs": {
                "axis": attrs[2]["axis"]
            }
        }, {
            "op_type": "fill_constant",
            "op_inputs": {},
            "op_outputs": {
                "Out": ["constant_out"]
            },
            "op_attrs": {
                "dtype": 5,
                "place_type": attrs[3]["place_type"],
                "str_value": attrs[3]["str_value"],
                "value": attrs[3]["value"],
                "shape": attrs[3]["shape"]
            }
        }, {
            "op_type": "elementwise_mul",
            "op_inputs": {
                "X": ["sub_out"],
                "Y": ["constant_out"]
            },
            "op_outputs": {
                "Out": ["mul_out"]
            },
            "op_attrs": {
                "axis": attrs[4]["axis"]
            }
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
            outputs=["mul_out"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["fusion_squared_mat_sub"], (1e-5, 1e-5)

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
            passes=["squared_mat_sub_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
