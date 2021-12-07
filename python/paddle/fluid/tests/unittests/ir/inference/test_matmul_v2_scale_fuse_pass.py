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

from auto_scan_test import PassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestMatmulV2ScaleFusePass(PassAutoScanTest):
    """
     x_var    y_var(persistable)
       \       /
        matmul_v2
           |
         scale  
    """

    def sample_predictor_configs(self, program_config):
        # mkldnn
        config = self.create_inference_config(use_mkldnn=True)
        yield config, ["matmul_v2", ], (1e-5, 1e-5)

        # for gpu
        # config = self.create_inference_config(use_gpu=True)
        # yield config, ["matmul_v2", ], (1e-5, 1e-5)

        # TRT
        # config = self.create_trt_inference_config()
        # config.enable_tensorrt_engine(
        #     max_batch_size=10,
        #     workspace_size=10240,
        #     min_subgraph_size=0,
        #     precision_mode=paddle_infer.PrecisionType.Float32,
        #     use_static=False,
        #     use_calib_mode=False)
        # yield config, ["matmul_v2", ], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        # 1. Generate shape and attr of matmul
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=5))
        y_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=5))
        x_shape_rank = len(x_shape)
        y_shape_rank = len(y_shape)
        y_shape[-2] = x_shape[-1]
        for i in range(y_shape_rank - 3, -1, -1):
            j = x_shape_rank - (y_shape_rank - i)
            if j < 0 or j >= x_shape_rank:
                break
            y_shape[i] = x_shape[j]

        transpose_X = False
        transpose_Y = False
        # scale tensor
        scale_shape = [1]
        scale_value = draw(st.floats(min_value=-5.0, max_value=5.0, width=32))

        def gen_bias():
            return 0.0

        matmul_v2_op = OpConfig(
            "matmul_v2",
            inputs={"X": ["matmul_x"],
                    "Y": ["matmul_y"]},
            outputs={"Out": ["matmul_out"]},
            trans_x=transpose_X,
            trans_y=transpose_Y,
            fused_reshape_Out=[],
            fused_transpose_Out=[], )
        is_scale_tensor = draw(st.booleans())
        if is_scale_tensor:
            scale_op = OpConfig(
                "scale",
                inputs={"X": ["matmul_out"],
                        "ScaleTensor": ["scale_tensor"]},
                outputs={"Out": ["scale_out"]},
                scale=scale_value,
                bias=0.0,
                bias_after_scale=draw(st.booleans()), )
        else:
            scale_op = OpConfig(
                "scale",
                inputs={"X": ["matmul_out"], },
                outputs={"Out": ["scale_out"]},
                scale=scale_value,
                bias=0.0,
                bias_after_scale=draw(st.booleans()), )

        ops = [matmul_v2_op, scale_op]
        weights = {"matmul_y": TensorConfig(shape=y_shape), }
        if is_scale_tensor:
            weights["scale_tensor"] = TensorConfig(shape=scale_shape)
        inputs = {"matmul_x": TensorConfig(shape=x_shape), }
        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=ops[-1].outputs["Out"], )
        # print("x_shape", x_shape)
        # print("y_shape:", y_shape)
        # print(is_scale_tensor, scale_shape)
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["matmul_v2_scale_fuse_pass"],
            # reproduce=reproduce_failure('6.27.1', b'AXicY2RgZAAiIAEDjAwAAFgABg==')
        )


if __name__ == "__main__":
    unittest.main()
