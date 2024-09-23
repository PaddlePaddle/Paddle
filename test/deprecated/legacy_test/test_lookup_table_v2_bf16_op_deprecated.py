#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import test_lookup_table_bf16_op
from op_test import convert_uint16_to_float

import paddle
from paddle import base


class TestEmbeddingLayerBF16ConstantInitializer(unittest.TestCase):
    """
    Test embedding layer from input api and results for bfloat16
    """

    def set_initializer(self):
        self.initializer = paddle.nn.initializer.Constant(value=self.value)

    def setUp(self):
        self.op_type = "lookup_table_v2"
        self.python_api = paddle.nn.functional.embedding
        self.ids_shape = [4]
        self.w_shape = [10, 64]
        self.ids = np.random.randint(low=0, high=9, size=self.ids_shape).astype(
            "int64"
        )
        self.flat_ids = self.ids.flatten()
        self.value = 3.0
        self.w_fp32 = np.full(self.w_shape, self.value)
        self.place = base.CPUPlace()
        self.prog = base.Program()
        self.startup_prog = base.Program()
        self.set_initializer()

        paddle.enable_static()
        with base.program_guard(self.prog, self.startup_prog):
            x = paddle.static.data(
                name='x', shape=[-1, *self.ids_shape], dtype='int64'
            )
            self.emb = paddle.static.nn.embedding(
                input=x,
                size=self.w_shape,
                param_attr=base.ParamAttr(
                    name="emb_weight", initializer=self.initializer
                ),
                is_sparse=False,
                dtype="uint16",
            )  # bfloat16
        exe = base.Executor(self.place)
        exe.run(self.startup_prog)
        self.result = exe.run(
            self.prog, feed={'x': self.ids}, fetch_list=['emb_weight', self.emb]
        )

    def test_embedding_weights(self):
        result = convert_uint16_to_float(self.result[0])
        np.testing.assert_array_equal(self.w_fp32, result)

    def test_lookup_results(self):
        lookup_result = convert_uint16_to_float(self.result[1])
        lookup_ref = test_lookup_table_bf16_op._lookup(
            self.w_fp32, self.ids, self.flat_ids, self.op_type
        )
        np.testing.assert_array_equal(lookup_result, lookup_ref)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
