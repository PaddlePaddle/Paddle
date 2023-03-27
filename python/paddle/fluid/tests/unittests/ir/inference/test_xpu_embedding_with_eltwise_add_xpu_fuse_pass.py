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


class TestEmbeddingWithEltwiseAddXPUFusePass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["embedding_with_eltwise_add_xpu"], (1e-3, 1e-3)

    def sample_program_config(self, draw):

        # lookup_table_v2
        lookup_table_num = draw(st.sampled_from([2, 3, 4]))
        print("lookup_table_num: ", lookup_table_num)
        ids_shape = draw(st.sampled_from([[1, 32]]))
        w_shape = draw(st.sampled_from([[1000, 32]]))
        padding_idx = draw(st.sampled_from([-1]))
        axis = draw(st.sampled_from([-1]))

        def gen_lookup_table_ops():
            lookup_table_op_config_list = []
            lookup_table_op_0 = OpConfig(
                "lookup_table_v2",
                inputs={
                    "Ids": ["lookup_table_ids_0"],
                    "W": ["lookup_table_w_0"],
                },
                outputs={"Out": ["lookup_table_out_0"]},
                padding_idx=padding_idx,
            )
            lookup_table_op_1 = OpConfig(
                "lookup_table_v2",
                inputs={
                    "Ids": ["lookup_table_ids_1"],
                    "W": ["lookup_table_w_1"],
                },
                outputs={"Out": ["lookup_table_out_1"]},
                padding_idx=padding_idx,
            )
            lookup_table_ops_list = [lookup_table_op_0, lookup_table_op_1]
            if lookup_table_num >= 3:
                lookup_table_op_2 = OpConfig(
                    "lookup_table_v2",
                    inputs={
                        "Ids": ["lookup_table_ids_2"],
                        "W": ["lookup_table_w_2"],
                    },
                    outputs={"Out": ["lookup_table_out_2"]},
                    padding_idx=padding_idx,
                )
                lookup_table_ops_list.append(lookup_table_op_2)
            if lookup_table_num >= 4:
                lookup_table_op_3 = OpConfig(
                    "lookup_table_v2",
                    inputs={
                        "Ids": ["lookup_table_ids_3"],
                        "W": ["lookup_table_w_3"],
                    },
                    outputs={"Out": ["lookup_table_out_3"]},
                    padding_idx=padding_idx,
                )
                lookup_table_ops_list.append(lookup_table_op_3)
            return lookup_table_ops_list

        add_op_num = lookup_table_num - 1

        def gen_eltwise_add_ops():
            add_op_0 = OpConfig(
                "elementwise_add",
                inputs={
                    "X": ["lookup_table_out_0"],
                    "Y": ["lookup_table_out_1"],
                },
                outputs={"Out": ["add_op_0_out"]},
                axis=axis,
            )
            add_op_list = [add_op_0]
            if add_op_num >= 2:
                add_op_1 = OpConfig(
                    "elementwise_add",
                    inputs={"X": ["add_op_0_out"], "Y": ["lookup_table_out_2"]},
                    outputs={"Out": ["add_op_1_out"]},
                    axis=axis,
                )
                add_op_list.append(add_op_1)

            if add_op_num >= 3:
                add_op_2 = OpConfig(
                    "elementwise_add",
                    inputs={"X": ["add_op_1_out"], "Y": ["lookup_table_out_3"]},
                    outputs={"Out": ["add_op_2_out"]},
                    axis=axis,
                )
                add_op_list.append(add_op_2)
            return add_op_list

        lookup_table_op_list = gen_lookup_table_ops()
        add_op_list = gen_eltwise_add_ops()

        # ops
        ops = []
        ops.extend(lookup_table_op_list)
        ops.extend(add_op_list)

        # inputs
        def generate_input(*args, **kwargs):
            return np.random.randint(0, w_shape[0], ids_shape).astype(np.int64)

        def gen_lookup_table_inputs_data(*args, **kwargs):
            inputs = {}
            for i in range(lookup_table_num):
                input_name = "lookup_table_ids_{}".format(i)
                inputs[input_name] = TensorConfig(
                    data_gen=partial(generate_input)
                )
            return inputs

        inputs = gen_lookup_table_inputs_data()

        # weights
        def gen_lookup_table_weights_data():
            weights = {}
            for i in range(lookup_table_num):
                w_name = "lookup_table_w_{}".format(i)
                weights[w_name] = TensorConfig(shape=w_shape)
            return weights

        weights = gen_lookup_table_weights_data()

        program_config = ProgramConfig(
            ops=ops,
            weights=weights,
            inputs=inputs,
            outputs=add_op_list[-1].outputs["Out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=3,
            min_success_num=3,
            passes=["embedding_with_eltwise_add_xpu_fuse_pass"],
        )


if __name__ == "__main__":
    unittest.main()
