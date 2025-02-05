# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class TestOneDNNConvBnFusePass(PassAutoScanTest):
    def sample_program_config(self, draw):
        use_mkldnn = True
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
        epsilon = draw(st.floats(min_value=0.0, max_value=0.001))

        x_shape = (
            [batch_size, in_channel, 64, 64]
            if data_format == "NCHW"
            else [batch_size, 64, 64, in_channel]
        )
        w_shape = [out_channel, filter_channel, filter_size, filter_size]
        scale_shape = [out_channel]
        bias_shape = [out_channel]
        var_shape = [out_channel]
        mean_shape = [out_channel]

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["conv2d_input"],
                "Filter": ["conv2d_weight"],
            },
            outputs={"Output": ["conv2d_out"]},
            data_format=data_format,
            dilations=dilations,
            padding_algorithm=padding_algorithm,
            groups=groups,
            paddings=paddings,
            strides=strides,
            use_mkldnn=use_mkldnn,
            has_bias=False,
            is_test=True,
        )
        bn_op = OpConfig(
            "batch_norm",
            inputs={
                "X": ["conv2d_out"],
                "Scale": ["batch_norm_Scale"],
                "Bias": ["batch_norm_Bias"],
                "Mean": ["batch_norm_Mean"],
                "Variance": ["batch_norm_Variance"],
            },
            outputs={
                "Y": ["batch_norm_Y"],
                "MeanOut": ["batch_norm_Mean"],
                "VarianceOut": ["batch_norm_Variance"],
                "SavedMean": ["batch_norm_SavedMean"],
                "SavedVariance": ["batch_norm_SavedVariance"],
                "ReserveSpace": ["batch_norm_ReserveSpace"],
            },
            epsilon=epsilon,
            trainable_statistics=False,
            data_layout=data_format,
            is_test=True,
        )
        ops = [conv2d_op, bn_op]

        program_config = ProgramConfig(
            ops=ops,
            inputs={
                "conv2d_input": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            weights={
                "conv2d_weight": TensorConfig(
                    data_gen=partial(generate_data, w_shape)
                ),
                "batch_norm_Scale": TensorConfig(
                    data_gen=partial(generate_data, scale_shape)
                ),
                "batch_norm_Bias": TensorConfig(
                    data_gen=partial(generate_data, bias_shape)
                ),
                "batch_norm_Mean": TensorConfig(
                    data_gen=partial(generate_data, mean_shape)
                ),
                "batch_norm_Variance": TensorConfig(
                    data_gen=partial(generate_data, var_shape)
                ),
            },
            outputs=["batch_norm_Y"],
        )

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ['fused_conv2d'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["conv_bn_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
