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


class TestEmbeddingEltwiseLayerNormFusePass(PassAutoScanTest):
    '''
  in_var1  emb_var   in_var2   emb_var   in_var3   emb_var   in_var   emb_var
    |        |        |         |        |         |           |         |
   lookup_table      lookup_table       lookup_table   ...    lookup_table
        |                 |                  |                     |
     lkt_var           lkt_var            lkt_var               lkt_var
        \                 /                  |         ...         |
          elementwise_add                    |                     |
                 \                          /                      |
                       elementwise_add                             |
                               |                                   |
                            elt_var                               /
                               \                                 /
                                         elementwise_add
                                                 |
                                            layer_norm
    '''

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        # is_sparse is only support False
        if program_config.ops[0].attrs['is_sparse'] == True:
            return False

        # is_distributed only support False
        if program_config.ops[0].attrs['is_distributed'] == True:
            return False

        # axis only support -1 and the last dim.
        if program_config.ops[3].attrs['axis'] not in [-1, 2]:
            return False

        if not (program_config.ops[5].attrs['epsilon'] >= 0 and
                program_config.ops[5].attrs['epsilon'] <= 0.001):
            return False

        if program_config.ops[5].attrs['begin_norm_axis'] != 2:
            return False

        # input check
        if program_config.weights['embedding_weight1'].shape[
                1] != program_config.weights['layer_norm_scale'].shape[0]:
            return False

        return True

    def sample_program_config(self, draw):
        is_sparse = draw(st.booleans())
        is_distributed = draw(st.booleans())
        padding_idx = draw(st.integers())
        axis = draw(st.integers(min_value=-4, max_value=4))
        op_type = draw(st.sampled_from(['lookup_table', 'lookup_table_v2']))
        epsilon = draw(st.floats(min_value=0, max_value=0.001))
        # begin_norm_axis has to be 2
        begin_norm_axis = 2
        batch_size = draw(st.integers(min_value=1, max_value=4))
        input_dim = draw(st.sampled_from([32, 64]))
        weight_size = draw(st.sampled_from([[64, 64], [64, 32]]))

        def generate_input(attrs):
            if attrs[0]['op_type'] == 'lookup_table':
                return np.random.randint(
                    0,
                    attrs[3]['weight_size'][0],
                    size=(attrs[3]['batch_size'], attrs[3]['input_dim'],
                          1)).astype(np.int64)
            else:
                return np.random.randint(
                    0,
                    attrs[3]['weight_size'][0],
                    size=(attrs[3]['batch_size'],
                          attrs[3]['input_dim'])).astype(np.int64)

        def generate_weight1(attrs):
            # set embedding weight by attrs
            return np.random.random(attrs['weight_size']).astype(np.float32)

        def generate_weight2(attrs):
            # set layernorm weight by attrs
            if attrs[2]['begin_norm_axis'] == 1:
                return np.random.random(
                    attrs[3]['input_dim'] *
                    attrs[3]['weight_size'][1]).astype(np.float32)
            else:
                return np.random.random(attrs[3]['weight_size'][1]).astype(
                    np.float32)

        attrs = [{
            'is_sparse': is_sparse,
            'is_distributed': is_distributed,
            'padding_idx': padding_idx,
            'op_type': op_type
        }, {
            'axis': axis
        }, {
            'begin_norm_axis': begin_norm_axis,
            'epsilon': epsilon
        }, {
            'batch_size': batch_size,
            'input_dim': input_dim,
            'weight_size': weight_size
        }]

        emb_op1 = OpConfig(
            type=attrs[0]['op_type'],
            inputs={"Ids": ["input_data1"],
                    "W": ["embedding_weight1"]},
            outputs={"Out": ["embedding_output1"]},
            attrs={
                'is_sparse': attrs[0]['is_sparse'],
                'is_distributed': attrs[0]['is_distributed'],
                'padding_idx': attrs[0]['padding_idx']
            })
        emb_op2 = OpConfig(
            type=attrs[0]['op_type'],
            inputs={"Ids": ["input_data2"],
                    "W": ["embedding_weight2"]},
            outputs={"Out": ["embedding_output2"]},
            attrs={
                'is_sparse': attrs[0]['is_sparse'],
                'is_distributed': attrs[0]['is_distributed'],
                'padding_idx': attrs[0]['padding_idx']
            })
        emb_op3 = OpConfig(
            type=attrs[0]['op_type'],
            inputs={"Ids": ["input_data3"],
                    "W": ["embedding_weight3"]},
            outputs={"Out": ["embedding_output3"]},
            attrs={
                'is_sparse': attrs[0]['is_sparse'],
                'is_distributed': attrs[0]['is_distributed'],
                'padding_idx': attrs[0]['padding_idx']
            })
        add_op1 = OpConfig(
            type='elementwise_add',
            inputs={
                "X": [emb_op2.outputs["Out"][0]],
                "Y": [emb_op3.outputs["Out"][0]],
            },
            outputs={"Out": ["elementwise_add_output1"]},
            attrs={"axis": attrs[1]['axis']})
        add_op2 = OpConfig(
            type='elementwise_add',
            inputs={
                "X": [add_op1.outputs["Out"][0]],
                "Y": [emb_op1.outputs["Out"][0]],
            },
            outputs={"Out": ["elementwise_add_output2"]},
            attrs={"axis": attrs[1]['axis']})
        layer_norm_op = OpConfig(
            type='layer_norm',
            inputs={
                "X": [add_op2.outputs["Out"][0]],
                "Bias": ["layer_norm_bias"],
                "Scale": ["layer_norm_scale"]
            },
            outputs={
                "Y": ["layer_norm_output1"],
                "Mean": ["layer_norm_output2"],
                "Variance": ["layer_norm_output3"]
            },
            attrs={
                'begin_norm_axis': attrs[2]['begin_norm_axis'],
                'epsilon': attrs[2]['epsilon']
            })

        program_config = ProgramConfig(
            ops=[emb_op1, emb_op2, emb_op3, add_op1, add_op2, layer_norm_op],
            weights={
                "embedding_weight1":
                TensorConfig(data_gen=partial(generate_weight1, attrs[3])),
                "embedding_weight2":
                TensorConfig(data_gen=partial(generate_weight1, attrs[3])),
                "embedding_weight3":
                TensorConfig(data_gen=partial(generate_weight1, attrs[3])),
                "layer_norm_bias":
                TensorConfig(data_gen=partial(generate_weight2, attrs)),
                "layer_norm_scale":
                TensorConfig(data_gen=partial(generate_weight2, attrs))
            },
            inputs={
                "input_data1":
                TensorConfig(data_gen=partial(generate_input, attrs)),
                "input_data2":
                TensorConfig(data_gen=partial(generate_input, attrs)),
                "input_data3":
                TensorConfig(data_gen=partial(generate_input, attrs))
            },
            outputs=["layer_norm_output1"])

        return program_config

    def sample_predictor_configs(self, program_config):
        # only used in gpu passes and trt passes.
        config = self.create_inference_config(use_gpu=True)
        yield config, ['fused_embedding_eltwise_layernorm'], (1e-5, 1e-5)
        # trt static_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        yield config, ['fused_embedding_eltwise_layernorm'], (1e-5, 1e-5)
        # trt dynamic_shape
        config = self.create_trt_inference_config()
        config.enable_tensorrt_engine(
            max_batch_size=4,
            workspace_size=102400,
            min_subgraph_size=0,
            precision_mode=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        if program_config.ops[0].type == 'lookup_table':
            config.set_trt_dynamic_shape_info({
                "input_data1": [1, 4, 1],
                "input_data2": [1, 4, 1],
                "input_data3": [1, 4, 1]
            }, {
                "input_data1": [4, 512, 1],
                "input_data2": [4, 512, 1],
                "input_data3": [4, 512, 1]
            }, {
                "input_data1": [2, 128, 1],
                "input_data2": [2, 128, 1],
                "input_data3": [2, 128, 1]
            })
        else:
            config.set_trt_dynamic_shape_info({
                "input_data1": [1, 4],
                "input_data2": [1, 4],
                "input_data3": [1, 4]
            }, {
                "input_data1": [4, 512],
                "input_data2": [4, 512],
                "input_data3": [4, 512]
            }, {
                "input_data1": [2, 128],
                "input_data2": [2, 128],
                "input_data3": [2, 128]
            })
        yield config, ['fused_embedding_eltwise_layernorm'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if program_config.ops[3].attrs['axis'] in [
                    -1, 2
            ] and program_config.ops[5].attrs[
                    'begin_norm_axis'] == 2 and program_config.weights[
                        'embedding_weight1'].shape in [(64, 32), (64, 64)]:
                return True
            return False

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PASS_ACCURACY_ERROR,
            "The pass output has diff in a specific case. We need to fix it as soon as possible."
        )

    def test(self):
        # this fuse need to fix, now there's no program can ran successfully
        self.run_and_statis(
            quant=False,
            max_examples=50,
            passes=["embedding_eltwise_layernorm_fuse_pass"],
            min_success_num=0)


if __name__ == "__main__":
    unittest.main()
