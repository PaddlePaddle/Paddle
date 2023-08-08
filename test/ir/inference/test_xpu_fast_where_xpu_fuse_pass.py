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


class TestFastWhereXPUFusePassOneCase0(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        cast_op = OpConfig(
            "cast",
            inputs={"X": ["condition"]},
            outputs={"Out": ["cast_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["x"], "Y": ["scale_out"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["y"], "Y": ["cast_out"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )

        ops = [cast_op, scale_op, mul0_op, mul1_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition": TensorConfig(data_gen=partial(generate_condition)),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassOneCase1(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        cast_op = OpConfig(
            "cast",
            inputs={"X": ["condition"]},
            outputs={"Out": ["cast_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["x"], "Y": ["cast_out"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["y"], "Y": ["scale_out"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )

        ops = [cast_op, mul0_op, scale_op, mul1_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition": TensorConfig(data_gen=partial(generate_condition)),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassOneCase2(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        cast_op = OpConfig(
            "cast",
            inputs={"X": ["condition"]},
            outputs={"Out": ["cast_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale_out"], "Y": ["x"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast_out"], "Y": ["y"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )

        ops = [cast_op, scale_op, mul0_op, mul1_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition": TensorConfig(data_gen=partial(generate_condition)),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassOneCase3(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        cast_op = OpConfig(
            "cast",
            inputs={"X": ["condition"]},
            outputs={"Out": ["cast_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast_out"], "Y": ["x"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale_out"], "Y": ["y"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )

        ops = [cast_op, mul0_op, scale_op, mul1_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition": TensorConfig(data_gen=partial(generate_condition)),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassOneCase4(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        cast_op = OpConfig(
            "cast",
            inputs={"X": ["condition"]},
            outputs={"Out": ["cast_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale_out"], "Y": ["x"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["y"], "Y": ["cast_out"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )

        ops = [cast_op, scale_op, mul0_op, mul1_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition": TensorConfig(data_gen=partial(generate_condition)),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassOneCase5(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        cast_op = OpConfig(
            "cast",
            inputs={"X": ["condition"]},
            outputs={"Out": ["cast_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast_out"], "Y": ["x"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        scale_op = OpConfig(
            "scale",
            inputs={"X": ["cast_out"]},
            outputs={"Out": ["scale_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["y"], "Y": ["scale_out"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )

        ops = [cast_op, mul0_op, scale_op, mul1_op, add_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition": TensorConfig(data_gen=partial(generate_condition)),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassCascadeCase0(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["logical_or", "fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        # fast_where_xpu0
        cast0_op = OpConfig(
            "cast",
            inputs={"X": ["condition0"]},
            outputs={"Out": ["cast0_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast0_out"], "Y": ["x"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        scale0_op = OpConfig(
            "scale",
            inputs={"X": ["cast0_out"]},
            outputs={"Out": ["scale0_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale0_out"], "Y": ["y"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add0_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )
        # fast_where_xpu1
        cast1_op = OpConfig(
            "cast",
            inputs={"X": ["condition1"]},
            outputs={"Out": ["cast1_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul2_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast1_out"], "Y": ["x"]},
            outputs={"Out": ["mul2_out"]},
            axis=-1,
        )
        scale1_op = OpConfig(
            "scale",
            inputs={"X": ["cast1_out"]},
            outputs={"Out": ["scale1_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul3_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale1_out"], "Y": ["add0_out"]},
            outputs={"Out": ["mul3_out"]},
            axis=-1,
        )
        add1_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul2_out"], "Y": ["mul3_out"]},
            outputs={"Out": ["add1_out"]},
            axis=-1,
        )

        ops = [
            cast0_op,
            mul0_op,
            scale0_op,
            mul1_op,
            add0_op,
            cast1_op,
            mul2_op,
            scale1_op,
            mul3_op,
            add1_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition0": TensorConfig(
                    data_gen=partial(generate_condition)
                ),
                "condition1": TensorConfig(
                    data_gen=partial(generate_condition)
                ),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


class TestFastWhereXPUFusePassCascadeCase1(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["logical_and", "fast_where_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        value_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        condition_shape = value_shape
        condition_shape[-1] = 1

        def generate_condition():
            return np.random.random(condition_shape).astype(bool)

        def generate_value():
            return np.random.random(value_shape).astype(np.float32)

        # fast_where_xpu0
        cast0_op = OpConfig(
            "cast",
            inputs={"X": ["condition0"]},
            outputs={"Out": ["cast0_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul0_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast0_out"], "Y": ["x"]},
            outputs={"Out": ["mul0_out"]},
            axis=-1,
        )
        scale0_op = OpConfig(
            "scale",
            inputs={"X": ["cast0_out"]},
            outputs={"Out": ["scale0_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul1_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale0_out"], "Y": ["y"]},
            outputs={"Out": ["mul1_out"]},
            axis=-1,
        )
        add0_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul0_out"], "Y": ["mul1_out"]},
            outputs={"Out": ["add0_out"]},
            axis=-1,
        )
        # fast_where_xpu1
        cast1_op = OpConfig(
            "cast",
            inputs={"X": ["condition1"]},
            outputs={"Out": ["cast1_out"]},
            in_dtype=0,
            out_dtype=5,
        )
        mul2_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["cast1_out"], "Y": ["add0_out"]},
            outputs={"Out": ["mul2_out"]},
            axis=-1,
        )
        scale1_op = OpConfig(
            "scale",
            inputs={"X": ["cast1_out"]},
            outputs={"Out": ["scale1_out"]},
            scale=-1,
            bias=1,
            base_after_scale=True,
        )
        mul3_op = OpConfig(
            "elementwise_mul",
            inputs={"X": ["scale1_out"], "Y": ["y"]},
            outputs={"Out": ["mul3_out"]},
            axis=-1,
        )
        add1_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["mul2_out"], "Y": ["mul3_out"]},
            outputs={"Out": ["add1_out"]},
            axis=-1,
        )

        ops = [
            cast0_op,
            mul0_op,
            scale0_op,
            mul1_op,
            add0_op,
            cast1_op,
            mul2_op,
            scale1_op,
            mul3_op,
            add1_op,
        ]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "condition0": TensorConfig(
                    data_gen=partial(generate_condition)
                ),
                "condition1": TensorConfig(
                    data_gen=partial(generate_condition)
                ),
                "x": TensorConfig(data_gen=partial(generate_value)),
                "y": TensorConfig(data_gen=partial(generate_value)),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["fast_where_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    np.random.seed(200)
    unittest.main()
