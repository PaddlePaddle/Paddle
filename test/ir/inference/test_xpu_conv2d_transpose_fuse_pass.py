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

from paddle.base import core


@unittest.skipIf(
    core.get_xpu_device_version(0) == core.XPUVersion.XPU3,
    "Unsupported on XPU3",
)
class TestConvTransposeXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["conv2d_transpose_xpu"], (3e-3, 3e-3)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=4, max_value=16), min_size=4, max_size=4
            )
        )
        oc = draw(st.integers(min_value=2, max_value=16))
        weight_shape = [x_shape[1], oc, 4, 4]
        y_shape = [oc]
        has_bn = draw(st.booleans())
        has_add = draw(st.booleans())
        has_relu = draw(st.booleans())

        def generate_data(shape):
            return 0.1 * np.random.random(shape).astype(np.float32)

        deconv_op = OpConfig(
            "conv2d_transpose",
            inputs={"Input": ["input_x"], "Filter": ["weight_x"]},
            outputs={"Output": ["output_x"]},
            data_format="NCHW",
            dilations=[1, 1],
            groups=1,
            paddings=[0, 0],
            padding_algorithm="EXPLICIT",
            strides=[4, 4],
            fuse_relu=False,
        )
        input_name_op = "output_x"
        ops = [deconv_op]

        if has_add:
            add_op = OpConfig(
                "elementwise_add",
                inputs={"X": [input_name_op], "Y": ["bias"]},
                outputs={"Out": ["add_out"]},
                axis=1,
            )
            input_name_op = "add_out"
            ops.append(add_op)

        if has_bn:
            bn_op = OpConfig(
                "batch_norm",
                inputs={
                    "X": [input_name_op],
                    "Bias": ["bn_bias"],
                    "Mean": ["bn_mean"],
                    "Scale": ["bn_scale"],
                    "Variance": ["bn_var"],
                },
                outputs={
                    "Y": ["bn_y"],
                    "MeanOut": ["bn_mean"],
                    "SavedMean": ["bn_mean_save"],
                    "SavedVariance": ["bn_save_var"],
                    "VarianceOut": ["bn_var"],
                },
                data_layout="NCHW",
                epsilon=0.000009999999747378752,
                momentum=0.89999,
                is_test=True,
                use_global_stats=True,
            )
            input_name_op = "bn_y"
            ops.append(bn_op)

        if has_relu:
            relu_op = OpConfig(
                "relu",
                inputs={"X": [input_name_op]},
                outputs={"Out": ["relu_out"]},
            )
            input_name_op = "relu_out"
            ops.append(relu_op)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "weight_x": TensorConfig(
                    data_gen=partial(generate_data, weight_shape)
                ),
                "bias": TensorConfig(data_gen=partial(generate_data, y_shape)),
                "bn_bias": TensorConfig(
                    data_gen=partial(generate_data, y_shape)
                ),
                "bn_mean": TensorConfig(
                    data_gen=partial(generate_data, y_shape)
                ),
                "bn_scale": TensorConfig(
                    data_gen=partial(generate_data, y_shape)
                ),
                "bn_var": TensorConfig(
                    data_gen=partial(generate_data, y_shape)
                ),
            },
            inputs={
                "input_x": TensorConfig(
                    data_gen=partial(generate_data, x_shape)
                ),
            },
            outputs=[input_name_op],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["conv2d_transpose_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
