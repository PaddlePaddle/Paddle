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

from auto_scan_test import PassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestFcFusePass(PassAutoScanTest):
    '''
        x_var   y_var(persistable)
          \       /
             mul     bias_var(persistable)
              |
          mul_out_var  bias_var(persistable)
                \        / 
              elementwise_add
    '''
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # fuse only support axis>=1 or axis==-1
        if program_config.ops[1].attrs["axis"] == 0:
            return False
        x_shape = program_config.inputs["mul_x"].shape
        y_shape = program_config.weights["mul_y"].shape
        x_num_col_dims = program_config.ops[0].attrs["x_num_col_dims"]
        if y_shape[0] != int(np.prod(x_shape[x_num_col_dims:])):
            return False
        return True

    def sample_program_configs(self, *args, **kwargs):
        x_shape = kwargs["x_shape"]
        y_shape = kwargs["y_shape"]
        x_num_col_dims = kwargs["x_num_col_dims"]
        axis = kwargs["axis"]
        bias_rank = kwargs["bias_rank"]
        use_broadcast = kwargs["use_broadcast"]
        has_relu = kwargs["has_relu"]

        # make y_shape be legal
        y_shape[0] = int(np.prod(x_shape[x_num_col_dims:]))
        mul_out_shape = x_shape[:x_num_col_dims] + y_shape[1:]
        if axis > 0:
            bias_shape = mul_out_shape[axis:][:bias_rank]
        else:
            bias_shape = mul_out_shape[-1 * bias_rank:]
        if use_broadcast:
            bias_shape[0] = 1

        def generate_data(shape):
            return np.random.random(shape).astype(np.float32)

        mul_op = OpConfig(type="mul",
                          inputs={
                              "X": ["mul_x"],
                              "Y": ["mul_y"],
                          },
                          outputs={"Out": ["mul_out"]},
                          attrs={
                              "x_num_col_dims": x_num_col_dims,
                              "y_num_col_dims": 1,
                          })
        add_op = OpConfig(
            type="elementwise_add",
            inputs={
                "X": ["mul_out"],
                "Y": ["bias"],
            },
            outputs={"Out": ["add_out"]},
            attrs={"axis": axis},
        )
        ops = [mul_op, add_op]
        if has_relu:
            relu_op = OpConfig(
                type="relu",
                inputs={"X": ["add_out"]},
                outputs={"Out": ["relu_out"]},
                attrs={},
            )
            ops.append(relu_op)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "mul_y": TensorConfig(data_gen=partial(generate_data, y_shape)),
                "bias":
                TensorConfig(data_gen=partial(generate_data, bias_shape))
            },
            inputs={
                "mul_x": TensorConfig(data_gen=partial(generate_data, x_shape)),
            },
            outputs=ops[-1].outputs["Out"])

        yield program_config

    def sample_predictor_configs(self, program_config):
        # pass for cpu
        before_num_ops = len(program_config.ops) + 2
        config = self.create_inference_config(passes=["fc_fuse_pass"],
                                              use_gpu=False)
        yield config, (before_num_ops, 3), (1e-5, 1e-5)

        # pass for gpu
        config = self.create_inference_config(passes=["fc_fuse_pass"],
                                              use_gpu=True)
        yield config, (before_num_ops, 3), (1e-5, 1e-5)

        # pass for trt static_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=8,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, (before_num_ops, 3), (1e-5, 1e-5)

    def add_skip_pass_case(self):
        def teller1(program_config, predictor_config):
            # TODO fuse has bug while axis != -1
            if program_config.ops[1].attrs["axis"] != -1:
                return True
            return False

        def teller2(program_config, predictor_config):
            # shape of bias should be [1, mul_y_shape[-1]] or [mul_y_shape[-1]]
            x_shape = program_config.inputs["mul_x"].shape
            y_shape = program_config.weights["mul_y"].shape
            bias_shape = program_config.weights["bias"].shape
            if bias_shape != (y_shape[-1]) and bias_shape != (1, y_shape[-1]):
                return True
            return False

        self.add_skip_case(
            teller1, SkipReasons.PASS_ACCURACY_ERROR,
            "The pass output has diff while axis of elementwise_add is not -1.")
        self.add_skip_case(
            teller2, SkipReasons.PASS_ACCURACY_ERROR,
            "The pass output has diff while shape of bias is not [1, out_size] or [out_size]."
        )

    @given(x_shape=st.lists(st.integers(min_value=1, max_value=8),
                            min_size=2,
                            max_size=4),
           y_shape=st.lists(st.integers(min_value=1, max_value=8),
                            min_size=2,
                            max_size=2),
           x_num_col_dims=st.integers(min_value=1, max_value=8),
           y_num_col_dims=st.integers(min_value=1, max_value=8),
           axis=st.integers(min_value=-1, max_value=8),
           bias_rank=st.integers(min_value=1, max_value=8),
           use_broadcast=st.booleans(),
           has_relu=st.booleans())
    def test(self, *args, **kwargs):
        assume(kwargs["axis"] == -1)
        assume(kwargs["bias_rank"] == 2)
        assume(kwargs["use_broadcast"])
        
        # filter illegal parameters
        assume(kwargs["y_num_col_dims"] == 1)
        assume(kwargs["x_num_col_dims"] < len(kwargs["x_shape"]))
        assume(kwargs["axis"] < 0 or kwargs["axis"] <= kwargs["x_num_col_dims"])
        assume(kwargs["bias_rank"] <= kwargs["x_num_col_dims"] + 1)
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
