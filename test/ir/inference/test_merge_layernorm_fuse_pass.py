# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
from functools import partial

import hypothesis.strategies as st
import numpy as np
from auto_scan_test import PassAutoScanTest
from program_config import OpConfig, ProgramConfig, TensorConfig

import paddle.inference as paddle_infer


class TestMergeLayernormFusePass(PassAutoScanTest):
    #       input
    #         | [?x3136x96]
    #       reshape2                                                            input
    #         | [?x56x56x96]                                                      | [?x3136x96]
    #         |--------------|--------------|--------------|                merge_layernorm
    #   strided_slice  strided_slice  strided_slice  strided_slice    ->          | [?x784x384]
    #         | [?x28x28x96] | [?x28x28x96] | [?x28x28x96] |         fused      output
    #         |--------------|--------------|--------------|
    #       concat
    #         | [?x28x28x384]
    #       reshape2
    #         | [?x784x384]
    #       layer_norm
    #         | [?x784x384]
    #        output

    def sample_predictor_configs(self, program_config):
        # trt dynamic_shape fp32
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=1 << 20,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {"input_data": [1, 196, 96]},
            {"input_data": [4, 3136, 384]},
            {"input_data": [1, 3136, 96]},
        )
        yield config, ["merge_layernorm"], (1e-5, 1e-5)
        # trt dynamic_shape fp16
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=1,
            workspace_size=1 << 20,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Half,
            use_static=False,
            use_calib_mode=False,
        )
        config.set_trt_dynamic_shape_info(
            {"input_data": [1, 196, 96]},
            {"input_data": [4, 3136, 384]},
            {"input_data": [1, 3136, 96]},
        )
        yield config, ["merge_layernorm"], (1e-3, 1e-3)

    def sample_program_config(self, draw):
        batch_size = draw(st.integers(min_value=1, max_value=4))
        input_H_W = draw(st.sampled_from([56, 28, 14]))
        input_n = draw(st.sampled_from([96, 192, 384]))
        layernorm_40_begin_norm_axis = 2
        layernorm_40_epsilon = draw(
            st.floats(min_value=0.0000001, max_value=0.001)
        )

        def generate_input(attrs):
            return np.random.random(
                [
                    attrs[3]['batch_size'],
                    attrs[3]['input_H_W'] * attrs[3]['input_H_W'],
                    attrs[3]['input_n'],
                ]
            ).astype(np.float32)

        def generate_weight(attrs):
            return np.random.random([attrs[3]['input_n'] * 4]).astype(
                np.float32
            )

        attrs = [
            {'shape': [-1, input_H_W, input_H_W, input_n]},
            {'shape': [-1, int(input_H_W * input_H_W / 4), int(input_n * 4)]},
            {
                'begin_norm_axis': layernorm_40_begin_norm_axis,
                'epsilon': layernorm_40_epsilon,
            },
            {
                'batch_size': batch_size,
                'input_H_W': input_H_W,
                'input_n': input_n,
            },
        ]
        reshape2_00_op = OpConfig(
            type="reshape2",
            inputs={'X': ['input_data']},
            outputs={
                'Out': ['reshape2_00_out'],
                'XShape': ['reshape2_00_outxshape'],
            },
            attrs={'shape': attrs[0]['shape']},
        )
        strided_slice_10_op = OpConfig(
            type="strided_slice",
            inputs={'Input': ['reshape2_00_out']},
            outputs={'Out': ['strided_slice_10_out']},
            attrs={
                'axes': [1, 2],
                'starts': [0, 0],
                'infer_flags': [1, 1],
                'ends': [attrs[3]['input_H_W'], attrs[3]['input_H_W']],
                'strides': [2, 2],
            },
        )
        strided_slice_11_op = OpConfig(
            type="strided_slice",
            inputs={'Input': ['reshape2_00_out']},
            outputs={'Out': ['strided_slice_11_out']},
            attrs={
                'axes': [1, 2],
                'starts': [1, 0],
                'infer_flags': [1, 1],
                'ends': [attrs[3]['input_H_W'], attrs[3]['input_H_W']],
                'strides': [2, 2],
            },
        )
        strided_slice_12_op = OpConfig(
            type="strided_slice",
            inputs={'Input': ['reshape2_00_out']},
            outputs={'Out': ['strided_slice_12_out']},
            attrs={
                'axes': [1, 2],
                'starts': [0, 1],
                'infer_flags': [1, 1],
                'ends': [attrs[3]['input_H_W'], attrs[3]['input_H_W']],
                'strides': [2, 2],
            },
        )
        strided_slice_13_op = OpConfig(
            type="strided_slice",
            inputs={'Input': ['reshape2_00_out']},
            outputs={'Out': ['strided_slice_13_out']},
            attrs={
                'axes': [1, 2],
                'starts': [1, 1],
                'infer_flags': [1, 1],
                'ends': [attrs[3]['input_H_W'], attrs[3]['input_H_W']],
                'strides': [2, 2],
            },
        )
        concat_20_op = OpConfig(
            type="concat",
            inputs={
                'X': [
                    'strided_slice_10_out',
                    'strided_slice_11_out',
                    'strided_slice_12_out',
                    'strided_slice_13_out',
                ]
            },
            outputs={'Out': ['concat_20_out']},
            attrs={'axis': -1},
        )
        reshape2_30_op = OpConfig(
            type='reshape2',
            inputs={'X': ['concat_20_out']},
            outputs={
                'Out': ['reshape2_30_Out'],
                'XShape': ['reshape2_30_XShape'],
            },
            attrs={'shape': attrs[1]['shape']},
        )
        layernorm_40_op = OpConfig(
            type='layer_norm',
            inputs={
                'X': ['reshape2_30_Out'],
                'Bias': ['layer_norm_bias'],
                'Scale': ['layer_norm_scale'],
            },
            outputs={
                "Y": ["layer_norm_out"],
                "Mean": ["layer_norm_outMean"],
                "Variance": ["layer_norm_outVariance"],
            },
            attrs={
                'begin_norm_axis': attrs[2]['begin_norm_axis'],
                'epsilon': attrs[2]['epsilon'],
            },
        )
        program_config = ProgramConfig(
            ops=[
                reshape2_00_op,
                strided_slice_10_op,
                strided_slice_11_op,
                strided_slice_12_op,
                strided_slice_13_op,
                concat_20_op,
                reshape2_30_op,
                layernorm_40_op,
            ],
            weights={
                'layer_norm_bias': TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
                'layer_norm_scale': TensorConfig(
                    data_gen=partial(generate_weight, attrs)
                ),
            },
            inputs={
                'input_data': TensorConfig(
                    data_gen=partial(generate_input, attrs)
                )
            },
            outputs=['layer_norm_out'],
        )
        return program_config

    def test(self):
        num_examples = 10 if sys.platform == "win32" else 50
        self.run_and_statis(
            quant=False,
            max_examples=num_examples,
            passes=["merge_layernorm_fuse_pass"],
            max_duration=250,
            min_success_num=num_examples,
        )


if __name__ == "__main__":
    unittest.main()
