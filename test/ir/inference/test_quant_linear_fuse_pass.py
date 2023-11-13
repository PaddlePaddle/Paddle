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
from auto_scan_test import IgnoreReasons, PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

from paddle.base import core


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "QuantLinear only supports cuda kernel.",
)
class TestQuantLinearFusePass(PassAutoScanTest):
    r"""
        x_var                             y_var(persistable)
          |                                 |
    quantize_linear                 dequantize_linear
          |                                 |
    quantize_linear_out_var       dequantize_linear_out_var
          |                                 /
    dequantize_linear                      /
          |                               /
    dequantize_linear_out_var            /
                \                       /
                 \                     /
                  \                   /
                   \                 /
                    \               /
                     \             /
                      \           /
                       \         /
                        matmul_v2
                            |
                     matmul_v2_out_var  bias_var(persistable)
                                \         /
                              elementwise_add
    """

    def sample_predictor_configs(self, program_config):
        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["quant_linear"], (0.15, 0.15)

    def add_ignore_pass_case(self):
        # Here we put some skip rules to avoid known bugs
        def teller1(program_config, predictor_config):
            # shape of bias should be [1, mul_y_shape[-1]] or [mul_y_shape[-1]]
            y_shape = list(program_config.weights["input_weight"].shape)
            bias_shape = list(program_config.weights["bias"].shape)

            if predictor_config.tensorrt_engine_enabled():
                # TensorRT cann't handle all the situation of elementwise_add
                # disable it until this problem fixed
                predictor_config.exp_disable_tensorrt_ops(["elementwise_add"])

            if bias_shape != [y_shape[-1]] and bias_shape != [1, y_shape[-1]]:
                return True
            return False

        def teller2(program_config, predictor_config):
            # TODO fuse has bug while axis != -1
            axis = program_config.ops[4].attrs["axis"]
            input_num_col_dims = len(program_config.inputs["input_x"].shape) - 1
            if axis != -1 and axis != input_num_col_dims:
                return True
            return False

        self.add_ignore_check_case(
            teller1,
            IgnoreReasons.PASS_ACCURACY_ERROR,
            "The pass output has diff while shape of bias is not [out_size] or [1, out_size].",
        )

        self.add_ignore_check_case(
            teller2,
            IgnoreReasons.PASS_ACCURACY_ERROR,
            "The pass output has diff while axis of elementwise_add is not -1.",
        )

    def is_program_valid(self, prog_config):
        input_num_col_dims = len(prog_config.inputs["input_x"].shape) - 1
        add_x_rank = input_num_col_dims + 1
        add_y_rank = len(prog_config.weights["bias"].shape)
        axis = prog_config.ops[4].attrs["axis"]
        if add_x_rank == add_y_rank:
            if axis != -1 or axis != 0:
                return False
        return True

    def sample_program_config(self, draw):
        def generate_scale():
            return np.zeros(input_shape[-1]).astype(np.float32) + 0.2521234002

        def generate_zeropoint():
            return np.zeros(input_shape[-1]).astype(np.float32)

        # 1. Generate shape of input:X of matmul_v2
        input_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        # 2. align with the behavior of the input_num_col_dims attr of quant_linear
        input_num_col_dims = len(input_shape) - 1

        # 3. Generate legal shape of input:Y of matmul_v2
        weight_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=8), min_size=2, max_size=2
            )
        )
        weight_shape[0] = int(np.prod(input_shape[input_num_col_dims:]))
        # 4. Generate shape of Output of matmul_v2
        mul_out_shape = input_shape[:input_num_col_dims] + weight_shape[1:]

        bias_shape = [mul_out_shape[-1]]

        has_relu = draw(st.booleans())

        quantize_linear_op = OpConfig(
            "quantize_linear",
            inputs={
                "X": ["input_x"],
                "Scale": ["quant_scale"],
                "ZeroPoint": ["quant_zero_point"],
            },
            outputs={"Y": ["quantize_linear_op_out"]},
            attrs={"quant_axis": -1},
        )

        dequantize_linear_op = OpConfig(
            "dequantize_linear",
            inputs={
                "X": ["quantize_linear_op_out"],
                "Scale": ["dequant_scale"],
                "ZeroPoint": ["dequant_zero_point"],
            },
            outputs={"Y": ["dequantize_linear_op_out"]},
            attrs={"quant_axis": -1},
        )

        weight_dequantize_linear_op = OpConfig(
            "dequantize_linear",
            inputs={
                "X": ["input_weight"],
                "Scale": ["weight_dequant_scale"],
                "ZeroPoint": ["weight_dequant_zero_point"],
            },
            outputs={"Y": ["weight_dequantize_linear_op_out"]},
            attrs={"weight_dequant_axis": 0},
        )

        matmul_v2_op = OpConfig(
            "matmul_v2",
            inputs={
                "X": ["dequantize_linear_op_out"],
                "Y": ["weight_dequantize_linear_op_out"],
            },
            outputs={"Out": ["matmul_v2_op_out"]},
        )

        elementwise_add_op = OpConfig(
            "elementwise_add",
            inputs={"X": ["matmul_v2_op_out"], "Y": ["bias"]},
            outputs={"Out": ["elementwise_add_op_out"]},
            axis=-1,
        )

        ops = [
            quantize_linear_op,
            dequantize_linear_op,
            weight_dequantize_linear_op,
            matmul_v2_op,
            elementwise_add_op,
        ]

        if has_relu:
            relu_op = OpConfig(
                "relu",
                inputs={"X": ["elementwise_add_op_out"]},
                outputs={"Out": ["relu_out"]},
            )
            ops.append(relu_op)
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "input_weight": TensorConfig(shape=weight_shape),
                "bias": TensorConfig(shape=bias_shape),
                "quant_scale": TensorConfig(data_gen=partial(generate_scale)),
                "dequant_scale": TensorConfig(data_gen=partial(generate_scale)),
                "weight_dequant_scale": TensorConfig(
                    data_gen=partial(generate_scale)
                ),
                "quant_zero_point": TensorConfig(
                    data_gen=partial(generate_zeropoint)
                ),
                "dequant_zero_point": TensorConfig(
                    data_gen=partial(generate_zeropoint)
                ),
                "weight_dequant_zero_point": TensorConfig(
                    data_gen=partial(generate_zeropoint)
                ),
            },
            inputs={
                "input_x": TensorConfig(shape=input_shape),
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False, max_examples=500, passes=["quant_linear_fuse_pass"]
        )


if __name__ == "__main__":
    unittest.main()
