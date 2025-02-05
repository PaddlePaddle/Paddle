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
        config = self.create_inference_config(
            use_gpu=True, passes=["quant_linear_fuse_pass"]
        )
        yield config, ["quant_linear"], (0.4, 0.3)

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
        # 1. Generate input:X of matmul_v2
        input_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=4
            )
        )
        input_x = np.random.random(input_shape).astype(np.float32)

        def generate_input_x():
            return input_x

        # 2. Genearate quant dequant scale and zeropoint
        def generate_input_scale():
            scale = 1.0 / np.max(input_x)
            return np.array(scale).astype(np.float32)

        def generate_dequant_scale():
            dequant_scale = np.max(input_x)
            return np.array(dequant_scale).astype(np.float32)

        def generate_quant_dequant_zeropoint():
            return np.array(0.0).astype(np.float32)

        def generate_weight_dequant_zeropoint():
            return np.zeros(weight_shape[-1]).astype(np.float32)

        # 3. Generate shape of input:Y of matmul_v2
        weight_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=4), min_size=2, max_size=2
            )
        )
        # follow the behavior of the input_num_col_dims attr of quant_linear
        input_num_col_dims = len(input_shape) - 1
        weight_shape[0] = int(np.prod(input_shape[input_num_col_dims:]))

        def round_array_with_ties_to_even(x):
            xLower = np.floor(x)
            xUpper = np.ceil(x)
            dLower = x - xLower
            dUpper = xUpper - x
            x[(dLower == dUpper) & (xLower % 2 == 0)] = xLower[
                (dLower == dUpper) & (xLower % 2 == 0)
            ]
            x[(dLower == dUpper) & (xLower % 2 != 0)] = xUpper[
                (dLower == dUpper) & (xLower % 2 != 0)
            ]
            x[dLower < dUpper] = xLower[dLower < dUpper]
            x[dLower > dUpper] = xUpper[dLower > dUpper]

        def round_array(x):
            x[x > 0] = np.ceil(x[x > 0])
            x[x <= 0] = np.floor(x[x <= 0])

        weights = np.random.random(weight_shape).astype("float32")

        # 4. Generate the  weight_dequant_scale
        def generate_weight_dequant_scale():
            return np.max(weights, axis=0)

        # 5. Generate the weight which is float type but stores int8 value(align with the behavior of PaddleSlim)
        def generate_input_weights(
            quant_round_type=0, quant_max_bound=127, quant_min_bound=-127
        ):
            # scale_weights = 1.0 / np.max(weights, axis=0)
            scale_weights = 1.0 / generate_weight_dequant_scale()
            quant_weights = quant_max_bound * scale_weights * weights
            if quant_round_type == 0:
                round_array_with_ties_to_even(quant_weights)
            else:
                round_array(quant_weights)
            quant_weights[quant_weights > quant_max_bound] = quant_max_bound
            quant_weights[quant_weights < quant_min_bound] = quant_min_bound
            return quant_weights

        # 6. Generate shape of Output of matmul_v2
        mul_out_shape = input_shape[:input_num_col_dims] + weight_shape[1:]

        # 7. Generate the bias shape
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
            attrs={"quant_axis": -1, "bit_length": 8, "round_type": 0},
        )

        dequantize_linear_op = OpConfig(
            "dequantize_linear",
            inputs={
                "X": ["quantize_linear_op_out"],
                "Scale": ["dequant_scale"],
                "ZeroPoint": ["dequant_zero_point"],
            },
            outputs={"Y": ["dequantize_linear_op_out"]},
            attrs={"quant_axis": -1, "bit_length": 8, "round_type": 0},
        )

        weight_dequantize_linear_op = OpConfig(
            "dequantize_linear",
            inputs={
                "X": ["input_weight"],
                "Scale": ["weight_dequant_scale"],
                "ZeroPoint": ["weight_dequant_zero_point"],
            },
            outputs={"Y": ["weight_dequantize_linear_op_out"]},
            attrs={"quant_axis": 1, "bit_length": 8, "round_type": 0},
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
                "input_weight": TensorConfig(
                    data_gen=partial(generate_input_weights)
                ),
                "bias": TensorConfig(shape=bias_shape),
                "quant_scale": TensorConfig(
                    data_gen=partial(generate_input_scale)
                ),
                "dequant_scale": TensorConfig(
                    data_gen=partial(generate_dequant_scale)
                ),
                "weight_dequant_scale": TensorConfig(
                    data_gen=partial(generate_weight_dequant_scale)
                ),
                "quant_zero_point": TensorConfig(
                    data_gen=partial(generate_quant_dequant_zeropoint)
                ),
                "dequant_zero_point": TensorConfig(
                    data_gen=partial(generate_quant_dequant_zeropoint)
                ),
                "weight_dequant_zero_point": TensorConfig(
                    data_gen=partial(generate_weight_dequant_zeropoint)
                ),
            },
            inputs={
                "input_x": TensorConfig(data_gen=partial(generate_input_x))
            },
            outputs=ops[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=30,
            passes=["quant_linear_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
