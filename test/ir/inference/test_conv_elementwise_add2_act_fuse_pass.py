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

import os
import unittest

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


class TestConvElementwiseAdd2ActPass(PassAutoScanTest):
    r"""
     x_var   f_var(persistable)
         \       /
             conv2d
             |
             conv2d_var    y_var(persistable)
                 \          /
             elementwise_add
                     |
    x1_var  elementwise_add_out_var
      \              /
      elementwise_add
             |
            act
             |
           act_var
    """

    def sample_predictor_configs(self, program_config):
        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["fused_conv2d_add_act"], (1e-4, 1e-5)

    def is_program_valid(self, prog_config):
        paddings = prog_config.ops[0].attrs["paddings"]
        strides = prog_config.ops[0].attrs["strides"]
        groups = prog_config.ops[0].attrs["groups"]
        padding_algorithm = prog_config.ops[0].attrs["padding_algorithm"]
        dilations = prog_config.ops[0].attrs["dilations"]
        data_format = prog_config.ops[0].attrs["data_format"]
        filter_shape = prog_config.weights["filter"].shape
        input_shape = prog_config.inputs["input_x"].shape
        if data_format != "NCHW":
            return False
        if padding_algorithm == "VALID":
            if (
                int(
                    (
                        input_shape[2]
                        - (dilations[0] * (filter_shape[2] - 1) + 1)
                    )
                    / strides[0]
                    + 1
                )
                <= 0
                or int(
                    (
                        input_shape[3]
                        - (dilations[1] * (filter_shape[3] - 1) + 1)
                    )
                    / strides[1]
                    + 1
                )
                <= 0
            ):
                return False
        if padding_algorithm == "EXPLICIT":
            if (
                int(
                    (
                        input_shape[2]
                        + paddings[0]
                        + paddings[1]
                        - (dilations[0] * (filter_shape[2] - 1) + 1)
                    )
                    / strides[0]
                    + 1
                )
                <= 0
                or int(
                    (
                        input_shape[3]
                        + paddings[2]
                        + paddings[3]
                        - (dilations[1] * (filter_shape[3] - 1) + 1)
                    )
                    / strides[1]
                    + 1
                )
                <= 0
            ):
                return False
        if padding_algorithm == "SAME":
            if (
                int((input_shape[2] + strides[0] - 1) / strides[0]) <= 0
                or int((input_shape[3] + strides[1] - 1) / strides[1]) <= 0
            ):
                return False
        if data_format == "NCHW":
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
        is_not_valid = True
        program_config = None
        while is_not_valid:
            # 1. Generate shape of input:X of conv2d
            x_shape = draw(
                st.lists(
                    st.integers(min_value=1, max_value=100),
                    min_size=4,
                    max_size=4,
                )
            )
            x_shape[1] = draw(st.integers(min_value=1, max_value=10))

            # 2. Generate legal attr:data_format of conv2d
            data_format = draw(st.sampled_from(["NCHW", "NHWC"]))

            # 3. Generate legal shape of input:Y of conv2d
            f_shape = draw(
                st.lists(
                    st.integers(min_value=1, max_value=7),
                    min_size=4,
                    max_size=4,
                )
            )
            if data_format == "NCHW":
                f_shape[1] = x_shape[1]
            else:
                f_shape[1] = x_shape[3]

            # 4. Generate legal attr:strides of conv2d
            strides = draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=2,
                    max_size=2,
                )
            )

            # 5. Generate legal attr:padding_algorithm of conv2d
            padding_algorithm = draw(
                st.sampled_from(["EXPLICIT", "SAME", "VALID"])
            )

            # 6. Generate legal attr:padding of conv2d
            padding = draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=4,
                    max_size=4,
                )
            )

            # 7. Generate legal attr:groups of conv2d
            groups = draw(st.integers(min_value=1, max_value=3))

            # 8. Generate legal attr:dilations of conv2d
            dilations = draw(
                st.lists(
                    st.integers(min_value=1, max_value=5),
                    min_size=2,
                    max_size=2,
                )
            )

            # 9. Generate legal elementwise_add: X of conv2d
            bias_2_dict = {}
            bias_2_dict[1] = [
                x_shape[0],
                f_shape[0],
                int(
                    (
                        x_shape[2]
                        + padding[0]
                        + padding[1]
                        - (dilations[0] * (f_shape[2] - 1) + 1)
                    )
                    / strides[0]
                    + 1
                ),
                int(
                    (
                        x_shape[3]
                        + padding[2]
                        + padding[3]
                        - (dilations[1] * (f_shape[3] - 1) + 1)
                    )
                    / strides[1]
                    + 1
                ),
            ]

            bias_2_dict[2] = [
                x_shape[0],
                f_shape[0],
                int((x_shape[2] + strides[0] - 1) / strides[0]),
                int((x_shape[3] + strides[1] - 1) / strides[1]),
            ]

            bias_2_dict[3] = [
                x_shape[0],
                f_shape[0],
                int(
                    (x_shape[2] - (dilations[0] * (f_shape[2] - 1) + 1))
                    / strides[0]
                    + 1
                ),
                int(
                    (x_shape[3] - (dilations[1] * (f_shape[3] - 1) + 1))
                    / strides[1]
                    + 1
                ),
            ]
            bias_index = 1
            if padding_algorithm == "SAME":
                bias_index = 2
            if padding_algorithm == "VALID":
                bias_index = 3
            bias_2_shape = bias_2_dict[bias_index]

            if np.sum(np.array(bias_2_shape) <= 0) == 0:
                is_not_valid = False
            else:
                continue

            # 10. Generate legal shape of input:bias of elementwise_add
            bias_shape = [f_shape[0]]

            # 11. Generate legal attr:axis of elementwise_add_1
            axis_1 = 1

            # 12. Generate legal attr:axis of elementwise_add_2
            axis_2 = -1

            conv2d_op = OpConfig(
                "conv2d",
                inputs={"Input": ["input_x"], "Filter": ["filter"]},
                outputs={"Output": ["conv2d_out"]},
                strides=strides,
                padding_algorithm=padding_algorithm,
                paddings=padding,
                groups=groups,
                dilations=dilations,
                data_format=data_format,
            )
            add_1_op = OpConfig(
                "elementwise_add",
                inputs={"X": ["conv2d_out"], "Y": ["bias_1"]},
                outputs={"Out": ["add_1_out"]},
                axis=axis_1,
            )

            add_2_op = OpConfig(
                "elementwise_add",
                inputs={"X": ["bias_2"], "Y": ["add_1_out"]},
                outputs={"Out": ["add_out"]},
                axis=axis_2,
            )

            relu_op = OpConfig(
                "relu", inputs={"X": ["add_out"]}, outputs={"Out": ["relu_out"]}
            )

            ops = [conv2d_op, add_1_op, add_2_op, relu_op]

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "filter": TensorConfig(shape=f_shape),
                    "bias_1": TensorConfig(shape=bias_shape),
                },
                inputs={
                    "input_x": TensorConfig(shape=x_shape),
                    "bias_2": TensorConfig(shape=bias_2_shape),
                },
                outputs=ops[-1].outputs["Out"],
            )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["conv_elementwise_add2_act_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
