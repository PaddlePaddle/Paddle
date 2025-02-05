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

import math
import sys
import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

import paddle.inference as paddle_infer


class ReverseRollPass(PassAutoScanTest):
    """
       |
    reshape2
       |
    reshape2
       |
    transpose2
       |
    reshape2
       |
      roll
       |
    reshape2
       |
    """

    def sample_predictor_configs(self, program_config):
        # trt with dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {
                "input0": [64, 9, 96],
            },
            {
                "input0": [512, 144, 768],
            },
            {
                "input0": [64, 49, 96],
            },
        )

        yield config, ['reverse_roll'], (1e-5, 1e-5)

        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {
                "input0": [64, 9, 96],
            },
            {
                "input0": [512, 144, 768],
            },
            {
                "input0": [64, 49, 96],
            },
        )

        yield config, ['reverse_roll'], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))
        window_size = draw(st.sampled_from([3, 5, 7, 12]))
        dim = draw(st.sampled_from([96, 192, 384, 768]))
        window_number = 64

        def generate_input(attrs):
            return np.random.random(
                [
                    attrs[0]["batch_size"] * attrs[1]["window_number"],
                    attrs[1]["window_size"] * attrs[1]["window_size"],
                    attrs[1]["dim"],
                ]
            ).astype(np.float32)

        attrs = [
            {"batch_size": batch_size},
            {
                "window_number": window_number,
                "window_size": window_size,
                "dim": dim,
            },
        ]
        reshape2_00 = OpConfig(
            type="reshape2",
            inputs={"X": ["input0"]},
            outputs={
                "Out": ["reshape2_00_out"],
                "XShape": ["reshape2_00_outXshape"],
            },
            attrs={"shape": [-1, window_size, window_size, dim]},
        )
        reshape2_10 = OpConfig(
            type="reshape2",
            inputs={"X": ["reshape2_00_out"]},
            outputs={
                "Out": ["reshape2_10_out"],
                "XShape": ["reshape2_10_outXshape"],
            },
            attrs={
                "shape": [
                    -1,
                    int(math.sqrt(window_number)),
                    int(math.sqrt(window_number)),
                    window_size,
                    window_size,
                    dim,
                ]
            },
        )
        transpose2_20 = OpConfig(
            type="transpose2",
            inputs={"X": ["reshape2_10_out"]},
            outputs={
                "Out": ["transpose2_20_out"],
                "XShape": ["transpose2_20_outXshape"],
            },
            attrs={"axis": [0, 1, 3, 2, 4, 5]},
        )
        reshape2_30 = OpConfig(
            type="reshape2",
            inputs={"X": ["transpose2_20_out"]},
            outputs={
                "Out": ["reshape2_30_out"],
                "XShape": ["reshape2_30_outXshape"],
            },
            attrs={
                "shape": [
                    -1,
                    int(math.sqrt(window_number)) * window_size,
                    int(math.sqrt(window_number)) * window_size,
                    dim,
                ]
            },
        )
        roll_30_1 = OpConfig(
            type="roll",
            inputs={"X": ["reshape2_30_out"]},
            outputs={"Out": ["roll_30_1_out"]},
            attrs={
                "axis": [1, 2],
                "shifts": [
                    math.floor(window_size // 2),
                    math.floor(window_size // 2),
                ],
            },
        )
        reshape2_40 = OpConfig(
            type="reshape2",
            inputs={"X": ["roll_30_1_out"]},
            outputs={
                "Out": ["reshape2_40_out"],
                "XShape": ["reshape2_40_outXshape"],
            },
            attrs={
                "shape": [-1, window_number * window_size * window_size, dim]
            },
        )

        program_config = ProgramConfig(
            ops=[
                reshape2_00,
                reshape2_10,
                transpose2_20,
                reshape2_30,
                roll_30_1,
                reshape2_40,
            ],
            weights={},
            inputs={
                "input0": TensorConfig(data_gen=partial(generate_input, attrs)),
            },
            outputs=["reshape2_40_out"],
        )

        return program_config

    def test(self):
        max_examples = 50
        min_success_num = 50
        if sys.platform == "win32":
            max_examples = 5
            min_success_num = 5
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["reverse_roll_fuse_pass"],
            max_duration=250,
            min_success_num=min_success_num,
        )


class ReverseRoll2Pass(PassAutoScanTest):
    """
       |
    reshape2
       |
    reshape2
       |
    transpose2
       |
    reshape2
       |
    reshape2
       |
    """

    def sample_predictor_configs(self, program_config):
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {
                "input0": [64, 9, 96],
            },
            {
                "input0": [512, 144, 768],
            },
            {
                "input0": [64, 49, 96],
            },
        )

        yield config, ['reverse_roll'], (1e-5, 1e-5)

        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {
                "input0": [64, 9, 96],
            },
            {
                "input0": [512, 144, 768],
            },
            {
                "input0": [64, 49, 96],
            },
        )

        yield config, ['reverse_roll'], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))
        window_size = draw(st.sampled_from([3, 5, 7, 12]))
        dim = draw(st.sampled_from([96, 192, 384, 768]))
        window_number = 64

        def generate_input(attrs):
            return np.random.random(
                [
                    attrs[0]["batch_size"] * attrs[1]["window_number"],
                    attrs[1]["window_size"] * attrs[1]["window_size"],
                    attrs[1]["dim"],
                ]
            ).astype(np.float32)

        attrs = [
            {"batch_size": batch_size},
            {
                "window_number": window_number,
                "window_size": window_size,
                "dim": dim,
            },
        ]
        reshape2_00 = OpConfig(
            type="reshape2",
            inputs={"X": ["input0"]},
            outputs={
                "Out": ["reshape2_00_out"],
                "XShape": ["reshape2_00_outXshape"],
            },
            attrs={"shape": [-1, window_size, window_size, dim]},
        )
        reshape2_10 = OpConfig(
            type="reshape2",
            inputs={"X": ["reshape2_00_out"]},
            outputs={
                "Out": ["reshape2_10_out"],
                "XShape": ["reshape2_10_outXshape"],
            },
            attrs={
                "shape": [
                    -1,
                    int(math.sqrt(window_number)),
                    int(math.sqrt(window_number)),
                    window_size,
                    window_size,
                    dim,
                ]
            },
        )
        transpose2_20 = OpConfig(
            type="transpose2",
            inputs={"X": ["reshape2_10_out"]},
            outputs={
                "Out": ["transpose2_20_out"],
                "XShape": ["transpose2_20_outXshape"],
            },
            attrs={"axis": [0, 1, 3, 2, 4, 5]},
        )
        reshape2_30 = OpConfig(
            type="reshape2",
            inputs={"X": ["transpose2_20_out"]},
            outputs={
                "Out": ["reshape2_30_out"],
                "XShape": ["reshape2_30_outXshape"],
            },
            attrs={
                "shape": [
                    -1,
                    int(math.sqrt(window_number)) * window_size,
                    int(math.sqrt(window_number)) * window_size,
                    dim,
                ]
            },
        )
        reshape2_40 = OpConfig(
            type="reshape2",
            inputs={"X": ["reshape2_30_out"]},
            outputs={
                "Out": ["reshape2_40_out"],
                "XShape": ["reshape2_40_outXshape"],
            },
            attrs={
                "shape": [-1, window_number * window_size * window_size, dim]
            },
        )

        program_config = ProgramConfig(
            ops=[
                reshape2_00,
                reshape2_10,
                transpose2_20,
                reshape2_30,
                reshape2_40,
            ],
            weights={},
            inputs={
                "input0": TensorConfig(data_gen=partial(generate_input, attrs)),
            },
            outputs=["reshape2_40_out"],
        )

        return program_config

    def test(self):
        max_examples = 50
        min_success_num = 50
        if sys.platform == "win32":
            max_examples = 5
            min_success_num = 5
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["reverse_roll_fuse_pass"],
            max_duration=250,
            min_success_num=min_success_num,
        )


if __name__ == "__main__":
    unittest.main()
