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


class TestConvBnOneDNNFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_gpu=False)
        config.enable_mkldnn()
        yield config, ['fused_conv2d'], (1e-4, 1e-2)

    def is_program_valid(self, prog_config):
        paddings = prog_config.ops[0].attrs['paddings']
        groups = prog_config.ops[0].attrs['groups']
        padding_algorithm = prog_config.ops[0].attrs['padding_algorithm']
        dilations = prog_config.ops[0].attrs['dilations']
        data_format = prog_config.ops[0].attrs['data_format']
        filter_shape = prog_config.weights['filter'].shape
        input_shape = prog_config.inputs['input_x'].shape

        height = input_shape[data_format.index('H')]
        width = input_shape[data_format.index('W')]
        if padding_algorithm == 'VALID':
            if (
                height - (dilations[0] * (filter_shape[2] - 1) + 1) <= 0
                or width - (dilations[1] * (filter_shape[3] - 1) + 1) <= 0
            ):
                return False
        if padding_algorithm == 'EXPLICIT':
            if (
                height
                + paddings[0]
                + paddings[1]
                - (dilations[0] * (filter_shape[2] - 1) + 1)
                <= 0
                or width
                + paddings[2]
                + paddings[3]
                - (dilations[1] * (filter_shape[3] - 1) + 1)
                <= 0
            ):
                return False

        if data_format == 'NCHW':
            if input_shape[1] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False
        else:
            if input_shape[3] != filter_shape[1] * groups:
                return False
            if filter_shape[0] % groups != 0:
                return False

        return True

    def sample_program_config(self, draw):
        # 1. Generate shape of input:X of conv2d
        x_shape = draw(
            st.lists(
                st.integers(min_value=5, max_value=100), min_size=4, max_size=4
            )
        )
        x_shape[1] = draw(st.integers(min_value=5, max_value=10))

        # 2. Generate legal attr:data_format of conv2d
        data_format = draw(st.sampled_from(['NCHW', 'NHWC']))

        # 3. Generate legal shape of input:Y of conv2d
        f_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=5), min_size=4, max_size=4
            )
        )
        if data_format == 'NCHW':
            f_shape[1] = x_shape[1]
        else:
            f_shape[1] = x_shape[3]

        # 4. Generate legal attr:strides of conv2d
        strides = draw(
            st.lists(
                st.integers(min_value=1, max_value=5), min_size=2, max_size=2
            )
        )

        # 5. Generate legal attr:padding_algorithm of conv2d
        padding_algorithm = draw(st.sampled_from(['EXPLICIT', 'SAME', 'VALID']))

        # 6. Generate legal attr:padding of conv2d
        padding = draw(
            st.lists(
                st.integers(min_value=1, max_value=5), min_size=4, max_size=4
            )
        )

        # 7. Generate legal attr:groups of conv2d
        groups = draw(st.integers(min_value=1, max_value=3))

        # 8. Generate legal attr:dilations of conv2d
        dilations = draw(
            st.lists(
                st.integers(min_value=1, max_value=5), min_size=2, max_size=2
            )
        )

        # 9. Generate legal input:ResidualData of conv2d
        res_shape = []
        if draw(st.booleans()):
            res_shape = draw(
                st.lists(
                    st.integers(min_value=1, max_value=100),
                    min_size=4,
                    max_size=4,
                )
            )

        # 10. Generate legal shape of inputs for conv and bn: bias of conv2d
        bias_shape = [f_shape[0]]
        scale_shape = [f_shape[0]]
        bias_shape = [f_shape[0]]
        var_shape = [f_shape[0]]
        mean_shape = [f_shape[0]]

        def generate_variance():
            return np.random.random(var_shape).astype(np.float32)

        inputs = {
            'Input': ['input_x'],
            'Filter': ['filter'],
            'ResidualData': ['residualdata'],
        }
        weights = {
            'filter': TensorConfig(shape=f_shape),
            "batch_norm_Scale": TensorConfig(shape=scale_shape),
            "batch_norm_Bias": TensorConfig(shape=bias_shape),
            "batch_norm_Mean": TensorConfig(shape=mean_shape),
            "batch_norm_Variance": TensorConfig(
                data_gen=partial(generate_variance)
            ),
        }

        # mkldnn Output has diff with bias!
        has_bias = False
        if has_bias:
            inputs['Bias'] = ['conv_bias']
            weights['conv_bias'] = TensorConfig(shape=bias_shape)

        # 11. Create conv2d op
        conv2d_op = OpConfig(
            'conv2d',
            inputs=inputs,
            outputs={'Output': ['conv2d_out']},
            strides=strides,
            padding_algorithm=padding_algorithm,
            paddings=padding,
            groups=groups,
            dilations=dilations,
            data_format=data_format,
            use_mkldnn=True,
            has_bias=has_bias,
        )

        # 12. Generate bn op
        epsilon = draw(st.floats(min_value=0.0, max_value=0.001))
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
        self.passes = ["conv_bn_fuse_pass"]

        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs={
                'input_x': TensorConfig(shape=x_shape),
                'residualdata': TensorConfig(shape=res_shape),
            },
            outputs=ops[-1].outputs['Y'],
        )
        return program_config

    def test(self):
        self.run_and_statis(quant=False, max_examples=2000, passes=self.passes)


if __name__ == '__main__':
    unittest.main()
