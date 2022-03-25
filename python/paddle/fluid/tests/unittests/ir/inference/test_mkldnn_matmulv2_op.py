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

from auto_scan_test import MkldnnAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestMkldnnMatmulv2Op(MkldnnAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if len(program_config.inputs["input_data2"].shape) == 4:
            if program_config.inputs["input_data1"].shape[
                    -4] != 1 and program_config.inputs["input_data2"].shape[
                        -4] != 1:
                if program_config.inputs["input_data1"].shape[
                        -4] != program_config.inputs["input_data2"].shape[-4]:
                    return False

        if program_config.inputs["input_data1"].shape[
                -3] != 1 and program_config.inputs["input_data2"].shape[
                    -3] != 1:
            if program_config.inputs["input_data1"].shape[
                    -3] != program_config.inputs["input_data2"].shape[-3]:
                return False
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(type, *args, **kwargs):
            transpose_X = kwargs["transpose_X"]
            transpose_Y = kwargs["transpose_Y"]
            batch_size1 = kwargs["batch_size1"]
            batch_size2 = kwargs["batch_size2"]
            channel1 = kwargs["channel1"]
            channel2 = kwargs["channel2"]
            input_dim = kwargs["input_dim"]
            y_dim_len = kwargs["y_dim_len"]
            if transpose_X and transpose_Y:
                shape_x = [batch_size1, channel1, input_dim, 32]
                if y_dim_len == 4:
                    shape_y = [batch_size2, channel2, 64, input_dim]
                elif y_dim_len == 3:
                    shape_y = [channel2, 64, input_dim]
            elif transpose_X:
                shape_x = [batch_size1, channel1, input_dim, 32]
                if y_dim_len == 4:
                    shape_y = [batch_size2, channel2, input_dim, 64]
                elif y_dim_len == 3:
                    shape_y = [channel2, input_dim, 64]
            elif transpose_Y:
                shape_x = [batch_size1, channel1, 32, input_dim]
                if y_dim_len == 4:
                    shape_y = [batch_size2, channel2, 8, input_dim]
                elif y_dim_len == 3:
                    shape_y = [channel2, 8, input_dim]
            else:
                shape_x = [batch_size1, channel1, 32, input_dim]
                if y_dim_len == 4:
                    shape_y = [batch_size2, channel2, input_dim, 16]
                elif y_dim_len == 3:
                    shape_y = [channel2, input_dim, 16]

            if type == "x":
                return np.random.random(shape_x).astype(np.float32)
            else:
                return np.random.random(shape_y).astype(np.float32)

        matmul_op = OpConfig(
            type="matmul_v2",
            inputs={"X": ["input_data1"],
                    "Y": ["input_data2"]},
            outputs={"Out": ["matmul_output"]},
            attrs={
                "trans_x": kwargs["transpose_X"],
                "trans_y": kwargs["transpose_Y"],
                "fused_reshape_X": [],
                "fused_reshape_Y": [],
                "fused_transpose_X": [],
                "fused_transpose_Y": [],
                "fused_reshape_Out": [],
                "fused_transpose_Out": []
            })

        program_config = ProgramConfig(
            ops=[matmul_op],
            weights={},
            inputs={
                "input_data1": TensorConfig(data_gen=partial(
                    generate_input, "x", *args, **kwargs)),
                "input_data2": TensorConfig(data_gen=partial(
                    generate_input, "y", *args, **kwargs))
            },
            outputs=["matmul_output"])

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, (1e-5, 1e-5)

    @given(
        transpose_X=st.booleans(),
        transpose_Y=st.booleans(),
        y_dim_len=st.sampled_from([3, 4]),
        batch_size1=st.integers(
            min_value=1, max_value=4),
        batch_size2=st.integers(
            min_value=1, max_value=4),
        channel1=st.sampled_from([1, 16, 32, 64]),
        channel2=st.sampled_from([1, 16, 32, 64]),
        input_dim=st.sampled_from([16, 32, 64]))
    def test(self, *args, **kwargs):
        self.run_test(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
