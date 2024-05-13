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

import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import IgnoreReasons, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

import paddle.inference as paddle_infer


class TestConvEltwiseAddFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        if attrs[0]['data_format'] == "NHWC" and attrs[1]['axis'] != 3:
            return False

        return True

    def sample_program_config(self, draw):
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))
        groups = draw(st.integers(min_value=1, max_value=3))
        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        axis = draw(st.sampled_from([1]))
        filter_channel = draw(st.integers(min_value=1, max_value=16)) * 4
        filter_size = draw(st.integers(min_value=1, max_value=4))
        in_channel = groups * filter_channel
        out_channel_factor = draw(st.integers(min_value=1, max_value=16)) * 4
        out_channel = groups * out_channel_factor
        batch_size = draw(st.integers(min_value=1, max_value=4))
        dilations = draw(
            st.lists(
                st.integers(min_value=1, max_value=2), min_size=2, max_size=2
            )
        )
        paddings = draw(
            st.lists(
                st.integers(min_value=0, max_value=2), min_size=2, max_size=2
            )
        )
        strides = draw(
            st.lists(
                st.integers(min_value=1, max_value=2), min_size=2, max_size=2
            )
        )

        x_shape = (
            [batch_size, in_channel, 64, 64]
            if data_format == "NCHW"
            else [batch_size, 64, 64, in_channel]
        )
        w_shape = [out_channel, filter_channel, filter_size, filter_size]
        scale_shape = [out_channel]
        bias_shape = [out_channel]

        def generate_input():
            return np.random.random(x_shape).astype(np.float32)

        def generate_weight():
            return np.random.random(w_shape).astype(np.float32)

        def generate_bias():
            return np.random.random(bias_shape).astype(np.float32)

        def generate_scale_bias():
            return np.random.random(bias_shape).astype(np.float32)

        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["input_data"],
                "Filter": ["conv2d_weight"],
            },
            outputs={"Output": ["conv_output"]},
            data_format=data_format,
            dilations=dilations,
            padding_algorithm=padding_algorithm,
            groups=groups,
            paddings=paddings,
            strides=strides,
            is_test=True,
        )
        eltwise_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["conv_output"], "Y": ["conv2d_bias"]},
            outputs={"Out": ["elementwise_output"]},
            axis=axis,
        )
        ops = [conv2d_op, eltwise_op]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
            },
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_weight)
                ),
                "conv2d_bias": TensorConfig(
                    data_gen=partial(generate_scale_bias)
                ),
            },
            outputs=["elementwise_output"],
        )
        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=True)
        yield config, ['fused_conv2d_add_act'], (1e-4, 1e-4)

        # # TRT
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=4,
            min_subgraph_size=1,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        yield config, ['fused_conv2d_add_act'], (1e-4, 1e-4)

    def add_ignore_pass_case(self):
        # If the problem has been fixed, the judgment
        # in is_program_valid needs to be deleted!!!
        def teller1(program_config, predictor_config):
            if program_config.ops[0].attrs['data_format'] == "NHWC":
                return True
            return False

        self.add_ignore_check_case(
            teller1,
            IgnoreReasons.PASS_ACCURACY_ERROR,
            "The output format of conv2d is wrong when data_format attribute is NHWC, \
            it will trigger Broadcast dimension mismatch bug \
            when data_format attribute is NHWC and axis of eltwise op is 1 for this pass.",
        )

    def test(self):
        self.run_and_statis(
            quant=False,
            passes=["conv_elementwise_add_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
