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
from functools import reduce


class TestSeqConcatFcFusePass(PassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_config(self, draw):
        is_test = True
        pooltype = "SUM"
        pad_value1 = draw(st.floats())
        pad_value2 = draw(st.floats())
        pad_value3 = draw(st.floats())
        use_cvm = True
        axis = draw(st.sampled_from([1]))
        batch_size = draw(st.integers(min_value=1, max_value=4))

        def generate_input1(attrs):
            shape = [attrs[3]['batch_size'], 128, 6, 120]
            return np.random.random(shape).astype(np.float32)

        def generate_input2(attrs):
            shape = [attrs[3]['batch_size'], 2]
            return np.random.random(shape).astype(np.float32)

        def generate_input3():
            return np.random.random([1, 768]).astype(np.float32)

        attrs = [{
            'is_test': is_test,
            'pooltype': pooltype
        }, {
            'use_cvm': use_cvm
        }, {
            'axis': axis
        }, {
            'batch_size': batch_size
        }]

        ops_config = [{
            "op_type": "im2sequence",
            "op_inputs": {
                "X": ["input_data1"]
            },
            "op_outputs": {
                "Out": ["seq_out"]
            },
            "op_attrs": {
                "kernels": [6, 1],
                "out_stride": [1, 1],
                "paddings": [0, 0, 0, 0],
                "strides": [1, 1]
            }
        }, {
            "op_type": "sequence_pool",
            "op_inputs": {
                "X": ["seq_out"]
            },
            "op_outputs": {
                "Out": ["seq_pool1_out"],
                "MaxIndex": ["index1_out"]
            },
            "op_attrs": {
                "is_test": attrs[0]['is_test'],
                "pooltype": attrs[0]['pooltype'],
                "pad_value": pad_value1
            }
        }, {
            "op_type": "sequence_pool",
            "op_inputs": {
                "X": ["seq_out"]
            },
            "op_outputs": {
                "Out": ["seq_pool2_out"],
                "MaxIndex": ["index2_out"]
            },
            "op_attrs": {
                "is_test": attrs[0]['is_test'],
                "pooltype": attrs[0]['pooltype'],
                "pad_value": pad_value2
            }
        }, {
            "op_type": "sequence_pool",
            "op_inputs": {
                "X": ["seq_out"]
            },
            "op_outputs": {
                "Out": ["seq_pool3_out"],
                "MaxIndex": ["index3_out"]
            },
            "op_attrs": {
                "is_test": attrs[0]['is_test'],
                "pooltype": attrs[0]['pooltype'],
                "pad_value": pad_value3
            }
        }, {
            "op_type": "cvm",
            "op_inputs": {
                "X": ["seq_pool1_out"],
                "CVM": ["input_data2"]
            },
            "op_outputs": {
                "Y": ["cvm1_out"]
            },
            "op_attrs": {
                "use_cvm": attrs[1]['use_cvm']
            }
        }, {
            "op_type": "cvm",
            "op_inputs": {
                "X": ["seq_pool2_out"],
                "CVM": ["input_data2"]
            },
            "op_outputs": {
                "Y": ["cvm2_out"]
            },
            "op_attrs": {
                "use_cvm": attrs[1]['use_cvm']
            }
        }, {
            "op_type": "cvm",
            "op_inputs": {
                "X": ["seq_pool3_out"],
                "CVM": ["input_data2"]
            },
            "op_outputs": {
                "Y": ["cvm3_out"]
            },
            "op_attrs": {
                "use_cvm": attrs[1]['use_cvm']
            }
        }, {
            "op_type": "concat",
            "op_inputs": {
                "X": ["cvm1_out", "cvm2_out", "cvm3_out"]
            },
            "op_outputs": {
                "Out": ["concat_output"]
            },
            "op_attrs": {
                'axis': attrs[2]['axis']
            }
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data1":
                TensorConfig(data_gen=partial(generate_input1, attrs)),
                "input_data2":
                TensorConfig(data_gen=partial(generate_input2, attrs)),
                "input_data3": TensorConfig(data_gen=partial(generate_input3))
            },
            outputs=["concat_output"])

        return program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config()
        yield config, ["im2sequence", "fusion_seqpool_cvm_concat"], (1e-5, 1e-5)

    def test(self):
        self.run_and_statis(
            quant=False, passes=["seqpool_cvm_concat_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
