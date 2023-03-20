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

import hypothesis.strategies as st
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig


class TestConvXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["conv_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        # 1. Generate shape of input:X of conv2d
        x_shape = draw(
            st.lists(
                st.integers(min_value=10, max_value=100), min_size=4, max_size=4
            )
        )
        x_shape[1] = draw(st.integers(min_value=1, max_value=10))

        # 2. Generate legal attr:data_format of conv2d
        data_format = draw(st.sampled_from(["NCHW"]))
        # 3. Generate legal shape of input:Y of conv2d
        f_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=7), min_size=4, max_size=4
            )
        )
        if data_format == "NCHW":
            f_shape[1] = x_shape[1]

        # 4. Generate legal attr:strides of conv2d
        strides = draw(
            st.lists(
                st.integers(min_value=1, max_value=5), min_size=2, max_size=2
            )
        )

        # 5. Generate legal attr:padding_algorithm of conv2d
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VALID"]))

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

        # 9. Generate legal shape of input:bias of elementwise_add
        ew_bias_shape = [f_shape[0]]

        # 10. Generate legal input:Scale of batch_norm
        bn_scale_shape = [f_shape[0]]

        # 11. Generate legal input:Mean of batch_norm
        bn_mean_shape = [f_shape[0]]

        # 12. Generate legal input:Bias of batch_norm
        bn_bias_shape = [f_shape[0]]

        # 13. Generate legal attr:epsilon of batch_norm
        epsilon = draw(st.floats(min_value=0.00001, max_value=0.001))

        # 14. Generate legal input:Variance of batch_norm
        bn_variance_shape = [f_shape[0]]

        def generate_batch_variance():
            return (
                0.1 + (1.0 - 0.1) * np.random.random(bn_variance_shape)
            ).astype(np.float32)

        # 15. ew_branch_add : Random choose if add a relu operator
        has_branch = draw(st.booleans())

        # 16. ew_branch_add : Random choose if add a relu operator
        # has_act = draw(st.booleans())

        # Here we will compose a program
        # Still has some risks that the program is invalid or cause bug while running
        # Use function `is_program_valid` to filter the invalid programs before running
        # Use function `add_skip_pass_case` to ignore the programs even if they cause bug while runing
        conv2d_op = OpConfig(
            "conv2d",
            inputs={
                "Input": ["input_x"],
                "Filter": ["filter"],
            },
            outputs={"Output": ["conv2d_out"]},
            strides=strides,
            padding_algorithm=padding_algorithm,
            paddings=padding,
            groups=groups,
            dilations=dilations,
            data_format=data_format,
        )
        ew_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["conv2d_out"], "Y": ["bias"]},
            outputs={"Out": ["add_out"]},
        )

        bn_op = OpConfig(
            "batch_norm",
            inputs={
                "X": ["add_out"],
                "Scale": ["scale_in"],
                "Bias": ["bn_bias"],
                "Mean": ["mean_in"],
                "Variance": ["variance_in"],
            },
            outputs={
                "Y": ["y_out"],
                "MeanOut": ["mean_out"],
                "VarianceOut": ["variance_out"],
                "SavedMean": ["SavedMean_out"],
                "SavedVariance": ["SavedVariance_out"],
            },
            epsilon=epsilon,
        )

        ops = [conv2d_op, ew_add_op, bn_op]
        if has_branch:
            ew_branch_op = OpConfig(
                "elementwise_add",
                inputs={"X": ["y_out"], "Y": ["branch_in"]},
                outputs={"Out": ["branch_out"]},
            )
            relu_op = OpConfig(
                "relu",
                inputs={"X": ["branch_out"]},
                outputs={"Out": ["relu_out"]},
            )
            ops.append(ew_branch_op)
            ops.append(relu_op)
            outputs_t = ops[-1].outputs["Out"],
        else
            outputs_t = ops[-1].outputs["Y"]

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "filter": TensorConfig(shape=f_shape),
                "ew_bias": TensorConfig(shape=ew_bias_shape),
                "scale_in": TensorConfig(shape=bn_scale_shape),
                "bn_bias": TensorConfig(shape=bn_bias_shape),
                "mean_in": TensorConfig(shape=bn_mean_shape),
                "variance_in": TensorConfig(data_gen=generate_batch_variance),
            },
            inputs={
                "input_x": TensorConfig(shape=x_shape),
            },
            outputs=outputs_t,
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["conv_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
