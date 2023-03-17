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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

import paddle.inference as paddle_infer


class TestSplitLayernormToMathOpsPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {
                "input_data": [1, 6, 16],
            },
            {
                "input_data": [4, 6, 16],
            },
            {
                "input_data": [1, 6, 16],
            },
        )
        yield config, [
            'reduce_mean',
            'elementwise_sub',
            'elementwise_pow',
            'reduce_mean',
            'elementwise_add',
            'sqrt',
            'elementwise_div',
            'elementwise_mul',
            'elementwise_add',
        ], (1e-5, 1e-5)

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
                "input_data": [1, 6, 16],
            },
            {
                "input_data": [4, 6, 16],
            },
            {
                "input_data": [1, 6, 16],
            },
        )
        yield config, [
            'reduce_mean',
            'elementwise_sub',
            'elementwise_pow',
            'reduce_mean',
            'elementwise_add',
            'sqrt',
            'elementwise_div',
            'elementwise_mul',
            'elementwise_add',
        ], (1e-2, 1e-2)

        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        yield config, [
            'reduce_mean',
            'elementwise_sub',
            'elementwise_pow',
            'reduce_mean',
            'elementwise_add',
            'sqrt',
            'elementwise_div',
            'elementwise_mul',
            'elementwise_add',
        ], (1e-5, 1e-5)

        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
        yield config, [
            'reduce_mean',
            'elementwise_sub',
            'elementwise_pow',
            'reduce_mean',
            'elementwise_add',
            'sqrt',
            'elementwise_div',
            'elementwise_mul',
            'elementwise_add',
        ], (1e-2, 1e-2)

    def sample_program_config(self, draw):
        epsilon = draw(st.floats(min_value=0.0000001, max_value=0.001))

        begin_norm_axis = draw(st.sampled_from([2, 1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))
        dim0 = 6
        dim1 = 16
        weight_len = dim1
        if begin_norm_axis == 1:
            weight_len *= dim0

        def generate_input(attrs):
            return np.random.random(
                [attrs[1]["batch_size"], *attrs[1]["input_dim"]]
            ).astype(np.float32)

        def generate_weight(attrs):
            return np.random.random(weight_len).astype(np.float32)

        attrs = [
            {
                'begin_norm_axis': begin_norm_axis,
                'epsilon': epsilon,
            },
            {
                'batch_size': batch_size,
                'input_dim': [dim0, dim1],
            },
        ]

        layer_norm_op = OpConfig(
            type="layer_norm",
            inputs={
                "X": ["input_data"],
                "Bias": ["layer_norm_bias"],
                "Scale": ["layer_norm_scale"],
            },
            outputs={
                "Y": ["layer_norm_output1"],
                "Mean": ["layer_norm_output2"],
                "Variance": ["layer_norm_output3"],
            },
            attrs={
                "begin_norm_axis": attrs[0]["begin_norm_axis"],
                "epsilon": attrs[0]["epsilon"],
            },
        )

        program_config = ProgramConfig(
            ops=[
                layer_norm_op,
            ],
            weights={
                "layer_norm_bias": TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
                "layer_norm_scale": TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
            },
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input, attrs)
                ),
            },
            outputs=["layer_norm_output1"],
        )

        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=20,
            passes=["split_layernorm_to_math_ops_pass"],
            max_duration=250,
            min_success_num=20,
        )


if __name__ == "__main__":
    unittest.main()
