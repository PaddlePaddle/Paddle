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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestAdaptivePool2dConvertGlobalPass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=4, max_size=4))
        pooling_type = draw(st.sampled_from(["max", "avg"]))

        data_format = "NCHW"  #trt support this format only
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))

        paddings = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=2, max_size=2))

        ceil_mode = draw(st.booleans())
        exclusive = draw(st.booleans())
        global_pooling = draw(st.booleans())
        padding_algorithm = draw(st.sampled_from(["EXPLICIT", "SAME", "VAILD"]))

        pool_op = OpConfig(
            "pool2d",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["pool_output"]},
            ksize=[1, 1],
            adaptive=True,
            pooling_type=pooling_type,
            data_format=data_format,
            strides=strides,
            paddings=paddings,
            ceil_mode=ceil_mode,
            global_pooling=global_pooling,
            padding_algorithm=padding_algorithm,
            exclusive=exclusive)
        ops = [pool_op]

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data": TensorConfig(shape=x_shape), },
            outputs=["pool_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, ['pool2d'], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["adaptive_pool2d_convert_global_pass"],
            min_success_num=40)


if __name__ == "__main__":
    unittest.main()
