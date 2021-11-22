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
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestScaleMatmulMkldnnFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(attrs):
            shape = [attrs[2]['input_dim1'], attrs[2]['input_dim2']]
            if attrs[0]['data_layout'] == "NCHW":
                shape.insert(0, attrs[2]['channel'])
                shape.insert(0, attrs[2]['batch_size'])
            else:
                shape.append(attrs[2]['channel'])
                shape.insert(0, attrs[2]['batch_size'])
            return np.random.random(shape).astype(np.float32)

        def generate_weight(attrs):
            return np.random.random([attrs[2]['channel']]).astype(np.float32)

        attrs = [{
            "data_layout": kwargs['data_layout'],
            "epsilon": kwargs['epsilon'],
            "fuse_with_relu": kwargs['fuse_with_relu'],
            "is_test": kwargs['is_test'],
            "momentum": kwargs['momentum'],
            "trainable_statistics": kwargs['trainable_statistics'],
            "use_global_stats": kwargs['use_global_stats'],
            "use_mkldnn": kwargs['use_mkldnn1']
        }, {
            "use_cudnn": kwargs['use_cudnn'],
            "use_mkldnn": kwargs['use_mkldnn2']
        }, {
            'batch_size': kwargs['batch_size'],
            'channel': kwargs['channel'],
            'input_dim1': kwargs['input_dim1'],
            'input_dim2': kwargs['input_dim2']
        }]

        ops_config = [{
            "op_type": "batch_norm",
            "op_inputs": {
                "X": ["input_data"],
                "Bias": ["Bias"],
                "Mean": ["Mean"],
                "Scale": ["Scale"],
                "Variance": ["Variance"]
            },
            "op_outputs": {
                "Y": ["norm_output"],
                "MeanOut": ["Mean"],
                "VarianceOut": ["Variance"],
                "SavedMean": ["SavedMean"],
                "SavedVariance": ["SavedVariance"]
            },
            "op_attrs": {
                "data_layout": attrs[0]['data_layout'],
                "epsilon": attrs[0]['epsilon'],
                "fuse_with_relu": attrs[0]['fuse_with_relu'],
                "is_test": attrs[0]['is_test'],
                "momentum": attrs[0]['momentum'],
                "trainable_statistics": attrs[0]['trainable_statistics'],
                "use_global_stats": attrs[0]['use_global_stats'],
                "use_mkldnn": attrs[0]['use_mkldnn']
            },
        }, {
            "op_type": "relu",
            "op_inputs": {
                "X": ["norm_output"]
            },
            "op_outputs": {
                "Out": ["relu_output"]
            },
            "op_attrs": {
                "use_cudnn": attrs[1]['use_cudnn'],
                "use_mkldnn": attrs[1]['use_mkldnn']
            }
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
                "Bias": TensorConfig(data_gen=partial(generate_weight, attrs)),
                "Mean": TensorConfig(data_gen=partial(generate_weight, attrs)),
                "Scale": TensorConfig(data_gen=partial(generate_weight, attrs)),
                "Variance": TensorConfig(data_gen=partial(generate_weight,
                                                          attrs))
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, attrs))
            },
            outputs=["relu_output"])

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(
            passes=['batch_norm_act_fuse_pass'], use_mkldnn=True)
        if program_config.ops[0].attrs['trainable_statistics']:
            yield config, (4, 4), (1e-5, 1e-5)
        else:
            yield config, (4, 3), (1e-5, 1e-5)

    @given(
        data_layout=st.sampled_from(["NCHW", "NHWC"]),
        epsilon=st.floats(
            min_value=0.0, max_value=0.001),
        fuse_with_relu=st.booleans(),
        is_test=st.sampled_from([True]),
        momentum=st.floats(
            min_value=0.0, max_value=5),
        trainable_statistics=st.booleans(),
        use_global_stats=st.booleans(),
        use_mkldnn1=st.sampled_from([True]),
        use_cudnn=st.booleans(),
        use_mkldnn2=st.sampled_from([True]),
        batch_size=st.integers(
            min_value=1, max_value=4),
        channel=st.integers(
            min_value=1, max_value=64),
        input_dim1=st.integers(
            min_value=1, max_value=512),
        input_dim2=st.integers(
            min_value=1, max_value=512))
    def test(self, *args, **kwargs):
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
