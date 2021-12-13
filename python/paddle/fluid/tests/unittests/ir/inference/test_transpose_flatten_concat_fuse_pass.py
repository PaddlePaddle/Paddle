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
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestTransposeFlattenConcatFusePass(PassAutoScanTest):
    """
        x_1_var              x_2_var
          |                     |
      transpose2            transpose2
          |                     | 
       flatten2              flatten2
          \                     /
    flatten2_out_var    flatten2_out_var
              \              /
                   concat 
    """

    def sample_predictor_configs(self, program_config):
        # TRT  
        # after tensorrt_subgraph_pass ，The pass needs to be deleted on TRT

        # for gpu
        config = self.create_inference_config(use_gpu=True)
        yield config, ["fusion_transpose_flatten_concat", ], (1e-5, 1e-5)

    def is_program_valid(self, prog_config):
        concat_axis = prog_config.ops[-1].attrs["axis"]
        ops_num = len(prog_config.ops) - 1
        if ops_num % 2 != 0:
            return False
        input_num = ops_num // 2
        flatten_shape = 0
        x_trans_axis = prog_config.ops[0].attrs["axis"]
        x_flatten_axis = prog_config.ops[1].attrs["axis"]
        for i in range(input_num):
            input_name = "transpose2_x" + str(i)
            input_shape = prog_config.inputs[input_name].shape
            trans_axis = prog_config.ops[i * 2].attrs["axis"]
            if x_trans_axis != trans_axis:
                return False
            #  calculate shape after transpose
            input_shape = [input_shape[j] for j in trans_axis]
            #  calculate shape after flateen
            flatten_axis = prog_config.ops[i * 2 + 1].attrs["axis"]
            if x_flatten_axis != flatten_axis:
                return False
            flatten_shape1 = flatten_shape2 = 1
            for j in range(len(input_shape)):
                if j < flatten_axis:
                    flatten_shape1 *= input_shape[j]
                else:
                    flatten_shape2 *= input_shape[j]
            if concat_axis == 0:
                if i == 0:
                    flatten_shape = flatten_shape2
                elif flatten_shape != flatten_shape2:
                    return False
            else:
                if i == 0:
                    flatten_shape = flatten_shape1
                elif flatten_shape != flatten_shape1:
                    return False
        return True

    def sample_program_config(self, draw):
        times = draw(st.integers(min_value=1, max_value=6))
        concat_axis = draw(st.integers(min_value=0, max_value=1))
        ops = []
        concat_input = []
        inputs = {}
        x_shape_rank = draw(st.integers(min_value=2, max_value=5))
        #  Generate axis of transpose
        trans_axis = [j for j in range(x_shape_rank)]
        for j in range(x_shape_rank - 1):
            if draw(st.booleans()):
                trans_axis[j], trans_axis[-1] = trans_axis[-1], trans_axis[j]
        #  Generate axis of flatten
        flatten_axis = draw(
            st.integers(
                min_value=0, max_value=x_shape_rank - 1))
        for i in range(times):
            #  Generate x_shape of transpose
            x_shape = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=10),
                    min_size=x_shape_rank,
                    max_size=x_shape_rank))

            str_i = str(i)
            transpose_op = OpConfig(
                "transpose2",
                inputs={"X": ["transpose2_x" + str_i], },
                axis=trans_axis,
                outputs={
                    "Out": ["trans_out" + str_i],
                    "XShape": ["trans_shape" + str_i]
                }, )
            ops.append(transpose_op)
            flatten_op = OpConfig(
                "flatten2",
                inputs={"X": ["trans_out" + str_i], },
                axis=flatten_axis,
                outputs={
                    "Out": ["flatten2_out" + str_i],
                    "XShape": ["xshape" + str_i]
                }, )
            concat_input.append("flatten2_out" + str_i)
            ops.append(flatten_op)
            inputs["transpose2_x" + str_i] = TensorConfig(shape=x_shape)

        concat_op = OpConfig(
            "concat",
            inputs={
                "X": concat_input,
                "AxisTensor": [],
            },
            outputs={"Out": ["concat_out"]},
            axis=concat_axis, )

        ops.append(concat_op)

        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs=inputs,
            outputs=ops[-1].outputs["Out"], )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=300,
            passes=["transpose_flatten_concat_fuse_pass"])


if __name__ == "__main__":
    unittest.main()
