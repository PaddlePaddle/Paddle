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


class TestElementGNActPass(PassAutoScanTest):
    #
    #     |           |                            |            |
    # other_op1     other_op2                  other_op1    other_op2
    #     |           |              fuse           \          /
    #    elementwise_add      ->                  skip_groupnorm_act
    #             |                                      |
    #          groupnorm
    #             |
    #            silu

    def sample_predictor_configs(self, program_config):
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
                "input_data_x": [1, 160, 1, 1],
                "input_data_y": [1, 160, 1, 1],
            },
            {
                "input_data_x": [4, 1280, 64, 64],
                "input_data_y": [4, 1280, 1, 1],
            },
            {
                "input_data_x": [1, 320, 1, 1],
                "input_data_y": [1, 320, 1, 1],
            },
        )
        yield config, ['skip_groupnorm_act'], (3e-3, 1e-3)

    def sample_program_config(self, draw):
        axis = draw(st.sampled_from([0, -1]))
        epsilon = draw(st.floats(min_value=0.0000001, max_value=0.001))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        groups = draw(st.sampled_from([4, 8, 16, 32]))
        hw = draw(st.sampled_from([1, 8, 16, 32]))
        channel = draw(st.sampled_from([320, 1280]))

        def generate_input_x(attrs):
            return np.random.random(
                [attrs[1]["batch_size"], *attrs[1]["input_dim_x"]]
            ).astype(np.float32)

        def generate_input_y(attrs):
            return np.random.random(
                [attrs[1]["batch_size"], *attrs[1]["input_dim_y"]]
            ).astype(np.float32)

        def generate_weight(attrs):
            return np.random.random(attrs[1]['input_dim_x'][0]).astype(
                np.float32
            )

        attrs = [
            {
                'axis': axis,
                'epsilon': epsilon,
                'groups': groups,
            },
            {
                'batch_size': batch_size,
                'input_dim_x': [channel, hw, hw],
                'input_dim_y': [channel, 1, 1],
            },
        ]

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["input_data_x"], "Y": ["input_data_y"]},
            outputs={"Out": ["ele_out"]},
            attrs={"axis": attrs[0]['axis']},
        )
        group_norm_op = OpConfig(
            type="group_norm",
            inputs={
                "X": ["ele_out"],
                "Bias": ["group_norm_bias"],
                "Scale": ["group_norm_scale"],
            },
            outputs={
                "Y": ["group_norm_output1"],
                "Mean": ["group_norm_output2"],
                "Variance": ["group_norm_output3"],
            },
            attrs={
                "data_layout": "NCHW",
                "groups": attrs[0]["groups"],
                "epsilon": attrs[0]["epsilon"],
            },
        )
        silu_op = OpConfig(
            type="silu",
            inputs={
                "X": ["group_norm_output1"],
            },
            outputs={
                "Out": ["silu_output"],
            },
        )

        program_config = ProgramConfig(
            ops=[
                elementwise_add_op,
                group_norm_op,
                silu_op,
            ],
            weights={
                "group_norm_bias": TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
                "group_norm_scale": TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
            },
            inputs={
                "input_data_x": TensorConfig(
                    data_gen=partial(generate_input_x, attrs)
                ),
                "input_data_y": TensorConfig(
                    data_gen=partial(generate_input_y, attrs)
                ),
            },
            outputs=["silu_output"],
        )

        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=50,
            passes=["elementwise_groupnorm_act_pass"],
            max_duration=250,
            min_success_num=50,
        )


if __name__ == "__main__":
    unittest.main()
