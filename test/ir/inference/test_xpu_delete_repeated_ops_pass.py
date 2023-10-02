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


class TestDeleteRepeatedShapeCastPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['shape', 'cast', 'relu', 'relu', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )

        shape_op0 = OpConfig(
            "shape",
            inputs={
                "Input": ["shape_x"],
            },
            outputs={"Out": ["shape0_out"]},
        )
        cast_op0 = OpConfig(
            "cast",
            inputs={
                "X": ["shape0_out"],
            },
            in_dtype=2,
            out_dtype=5,
            outputs={"Out": ["cast0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["cast0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        shape_op1 = OpConfig(
            "shape",
            inputs={
                "Input": ["shape_x"],
            },
            outputs={"Out": ["shape1_out"]},
        )
        cast_op1 = OpConfig(
            "cast",
            inputs={
                "X": ["shape1_out"],
            },
            in_dtype=2,
            out_dtype=5,
            outputs={"Out": ["cast1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["cast1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        shape_op2 = OpConfig(
            "shape",
            inputs={
                "Input": ["shape_x"],
            },
            outputs={"Out": ["shape2_out"]},
        )
        cast_op2 = OpConfig(
            "cast",
            inputs={
                "X": ["shape2_out"],
            },
            in_dtype=2,
            out_dtype=5,
            outputs={"Out": ["cast2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["cast2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )
        ops = [
            shape_op0,
            cast_op0,
            relu_op0,
            shape_op1,
            cast_op1,
            relu_op1,
            shape_op2,
            cast_op2,
            relu_op2,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "shape_x": TensorConfig(shape=x_shape),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


class TestDeleteRepeatedSlicePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['slice', 'relu', 'relu', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        slice_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )

        slice_op0 = OpConfig(
            "slice",
            inputs={
                "Input": ["slice_x"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["slice0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        slice_op1 = OpConfig(
            "slice",
            inputs={
                "Input": ["slice_x"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["slice1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        slice_op2 = OpConfig(
            "slice",
            inputs={
                "Input": ["slice_x"],
            },
            starts=[0],
            ends=[1],
            axes=[0],
            decrease_axis=[0],
            outputs={"Out": ["slice2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["slice2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )
        ops = [slice_op0, relu_op0, slice_op1, relu_op1, slice_op2, relu_op2]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "slice_x": TensorConfig(shape=slice_x),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


class TestDeleteRepeatedAddPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['elementwise_add', 'relu', 'relu', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        add_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )

        add_op0 = OpConfig(
            "elementwise_add",
            inputs={
                "X": ["add_x"],
                "Y": ["add_y"],
            },
            axis=-1,
            outputs={"Out": ["add0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["add0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        add_op1 = OpConfig(
            "elementwise_add",
            inputs={
                "X": ["add_x"],
                "Y": ["add_y"],
            },
            axis=-1,
            outputs={"Out": ["add1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["add1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        add_op2 = OpConfig(
            "elementwise_add",
            inputs={
                "X": ["add_x"],
                "Y": ["add_y"],
            },
            axis=-1,
            outputs={"Out": ["add2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["add2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )
        ops = [add_op0, relu_op0, add_op1, relu_op1, add_op2, relu_op2]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "add_x": TensorConfig(shape=add_x),
                "add_y": TensorConfig(shape=add_x),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


class TestDeleteRepeatedScalePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['scale', 'relu', 'relu', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        scale_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )

        scale_op0 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["scale0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        scale_op1 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["scale1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["scale2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )
        ops = [scale_op0, relu_op0, scale_op1, relu_op1, scale_op2, relu_op2]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "scale_x": TensorConfig(shape=scale_x),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


class TestDeleteRepeatedSqueezePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['scale', 'squeeze2', 'relu', 'relu', 'relu'], (
            1e-5,
            1e-5,
        )

    def sample_program_config(self, draw):
        scale_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )
        scale_x[0] = 1
        axis = 0
        scale_op0 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale0_out"]},
        )
        squeeze_op0 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["scale0_out"],
            },
            axes=[axis],
            outputs={"Out": ["squeeze0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["squeeze0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        scale_op1 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale1_out"]},
        )
        squeeze_op1 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["scale1_out"],
            },
            axes=[axis],
            outputs={"Out": ["squeeze1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["squeeze1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale2_out"]},
        )
        squeeze_op2 = OpConfig(
            "squeeze2",
            inputs={
                "X": ["scale2_out"],
            },
            axes=[axis],
            outputs={"Out": ["squeeze2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["squeeze2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )
        ops = [
            scale_op0,
            squeeze_op0,
            relu_op0,
            scale_op1,
            squeeze_op1,
            relu_op1,
            scale_op2,
            squeeze_op2,
            relu_op2,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "scale_x": TensorConfig(shape=scale_x),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config


class TestDeleteRepeatedUnSqueezePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['scale', 'unsqueeze2', 'relu', 'relu', 'relu'], (
            1e-5,
            1e-5,
        )

    def sample_program_config(self, draw):
        scale_x = draw(
            st.lists(
                st.integers(min_value=1, max_value=20), min_size=2, max_size=4
            )
        )
        axis = 0
        scale_op0 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale0_out"]},
        )
        unsqueeze_op0 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["scale0_out"],
            },
            axes=[axis],
            outputs={"Out": ["unsqueeze0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["unsqueeze0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        scale_op1 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale1_out"]},
        )
        unsqueeze_op1 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["scale1_out"],
            },
            axes=[axis],
            outputs={"Out": ["unsqueeze1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["unsqueeze1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale2_out"]},
        )
        unsqueeze_op2 = OpConfig(
            "unsqueeze2",
            inputs={
                "X": ["scale2_out"],
            },
            axes=[axis],
            outputs={"Out": ["unsqueeze2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["unsqueeze2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )
        ops = [
            scale_op0,
            unsqueeze_op0,
            relu_op0,
            scale_op1,
            unsqueeze_op1,
            relu_op1,
            scale_op2,
            unsqueeze_op2,
            relu_op2,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "scale_x": TensorConfig(shape=scale_x),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config


class TestDeleteRepeatedGatherPass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['scale', 'gather', 'relu', 'relu', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        scale_x = draw(
            st.lists(
                st.integers(min_value=3, max_value=20), min_size=2, max_size=4
            )
        )
        axis = 0

        def generate_index(*args, **kwargs):
            return np.array([0]).astype(np.int64)

        gather_index = np.array([0]).astype(np.int64)
        scale_op0 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale0_out"]},
        )
        gather_op0 = OpConfig(
            "gather",
            inputs={"X": ["scale0_out"], "Index": ["gather_index"]},
            axis=axis,
            outputs={"Out": ["gather0_out"]},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["gather0_out"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        scale_op1 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale1_out"]},
        )
        gather_op1 = OpConfig(
            "gather",
            inputs={"X": ["scale1_out"], "Index": ["gather_index"]},
            axis=axis,
            outputs={"Out": ["gather1_out"]},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["gather1_out"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        scale_op2 = OpConfig(
            "scale",
            inputs={
                "X": ["scale_x"],
            },
            scale=2.0,
            bias=1.0,
            bias_after_scale=True,
            outputs={"Out": ["scale2_out"]},
        )
        gather_op2 = OpConfig(
            "gather",
            inputs={"X": ["scale2_out"], "Index": ["gather_index"]},
            axis=axis,
            outputs={"Out": ["gather2_out"]},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["gather2_out"],
            },
            outputs={"Out": ["relu2_out"]},
        )

        ops = [
            scale_op0,
            gather_op0,
            relu_op0,
            scale_op1,
            gather_op1,
            relu_op1,
            scale_op2,
            gather_op2,
            relu_op2,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "scale_x": TensorConfig(shape=scale_x),
                "gather_index": TensorConfig(data_gen=partial(generate_index)),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


class TestDeleteRepeatedTransposePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ['transpose2', 'relu', 'relu', 'relu'], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))
        H = draw(st.integers(min_value=1, max_value=64))
        W = draw(st.integers(min_value=1, max_value=64))
        in_shape = [batch_size, H, W]
        axis = [0, 2, 1]

        transpose_op0 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["transpose_x"],
            },
            outputs={"Out": ["transpose_output0"]},
            attrs={"axis": axis},
        )
        relu_op0 = OpConfig(
            "relu",
            inputs={
                "X": ["transpose_output0"],
            },
            outputs={"Out": ["relu0_out"]},
        )
        transpose_op1 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["transpose_x"],
            },
            outputs={"Out": ["transpose_output1"]},
            attrs={"axis": axis},
        )
        relu_op1 = OpConfig(
            "relu",
            inputs={
                "X": ["transpose_output1"],
            },
            outputs={"Out": ["relu1_out"]},
        )
        transpose_op2 = OpConfig(
            type='transpose2',
            inputs={
                "X": ["transpose_x"],
            },
            outputs={"Out": ["transpose_output2"]},
            attrs={"axis": axis},
        )
        relu_op2 = OpConfig(
            "relu",
            inputs={
                "X": ["transpose_output2"],
            },
            outputs={"Out": ["relu2_out"]},
        )

        ops = [
            transpose_op0,
            relu_op0,
            transpose_op1,
            relu_op1,
            transpose_op2,
            relu_op2,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "transpose_x": TensorConfig(shape=in_shape),
            },
            outputs=["relu0_out", "relu1_out", "relu2_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["delete_repeated_ops_pass"],
        )


if __name__ == "__main__":
    unittest.main()
