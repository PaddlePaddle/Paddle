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


class TestXpuCastEmbeddingTransIdsToInt32Pass(PassAutoScanTest):
    def sample_predictor_configs(self, program_config):
        config = self.create_inference_config(use_xpu=True)
        yield config, ["cast", "lookup_table_v2"], (1e-5, 1e-5)

    def sample_program_config(self, draw):
        ids_shape = draw(st.integers(min_value=1, max_value=128))
        w_shape = draw(
            st.sampled_from([[20, 64], [32, 32], [23, 15], [24, 33]])
        )
        padding_idx = draw(st.sampled_from([-1]))

        cast_op = OpConfig(
            "cast",
            inputs={
                "X": ["cast_input"],
            },
            outputs={"Out": ["cast_out"]},
            in_dtype=5,
            out_dtype=3,
        )
        lookup_table_op = OpConfig(
            "lookup_table_v2",
            inputs={
                "Ids": ["cast_out"],
                "W": ["lookup_table_w"],
            },
            outputs={"Out": ["lookup_table_out"]},
            padding_idx=padding_idx,
        )

        def gen_lookup_table_weights_data():
            weights = {}
            w_name = "lookup_table_w"
            weights[w_name] = TensorConfig(shape=w_shape)
            return weights

        def generate_cast_input(*args, **kwargs):
            return np.random.randint(0, w_shape[0], ids_shape).astype(
                np.float32
            )

        def gen_input_data(*args, **kwargs):
            inputs = {}
            input_name = "cast_input"
            inputs[input_name] = TensorConfig(
                data_gen=partial(generate_cast_input)
            )
            return inputs

        inputs = gen_input_data()
        weights = gen_lookup_table_weights_data()

        program_config = ProgramConfig(
            ops=[cast_op, lookup_table_op],
            weights=weights,
            inputs=inputs,
            outputs=["lookup_table_out"],
        )
        return program_config

    def test(self):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["cast_embedding_trans_ids_to_int32_pass"],
        )


if __name__ == "__main__":
    unittest.main()
