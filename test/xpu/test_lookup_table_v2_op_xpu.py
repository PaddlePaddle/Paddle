#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base

paddle.enable_static()


class XPUTestLookupTableOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'lookup_table_v2'
        self.use_dynamic_create_class = False

    class TestLookupTableOPBase(XPUOpTest):
        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.op_type = 'lookup_table_v2'
            self.init_config()
            self.set_case()

        def set_case(self):
            table = np.random.random(self.input_shape).astype(self.dtype)
            ids = np.random.randint(0, self.id_range, self.id_count).astype(
                self.id_dtype
            )
            self.inputs = {'W': table, 'Ids': ids}
            self.outputs = {'Out': table[ids]}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(
                self.place, ['W'], 'Out', no_grad_set=set('Ids')
            )

        def init_config(self):
            self.input_shape = (17, 31)
            self.id_range = 17
            self.id_count = 4
            self.id_dtype = "int32"

    class XPUTestLookupTable1(TestLookupTableOPBase):
        def init_config(self):
            self.input_shape = (25, 52)
            self.id_range = 25
            self.id_count = 14
            self.id_dtype = "int64"

    class TestLookupTableOpWithTensorIds(TestLookupTableOPBase):
        def set_case(self):
            table = np.random.random((17, 31)).astype(self.dtype)
            ids = np.random.randint(low=0, high=17, size=(2, 4, 5)).astype(
                self.id_dtype
            )
            self.inputs = {'W': table, 'Ids': ids}
            self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    class TestLookupTableOpWithPadding(TestLookupTableOPBase):
        def test_check_output(self):
            ids = np.squeeze(self.inputs['Ids'])
            padding_idx = np.random.choice(ids, 1)[0]
            self.outputs['Out'][ids == padding_idx] = np.zeros(31)
            self.attrs = {'padding_idx': int(padding_idx)}
            self.check_output_with_place(self.place)

    class TestLookupTableOpWithTensorIdsAndPadding(
        TestLookupTableOpWithTensorIds
    ):
        def test_check_output(self):
            ids = self.inputs['Ids']
            flatten_idx = ids.flatten()
            padding_idx = np.random.choice(flatten_idx, 1)[0]
            self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
            self.attrs = {'padding_idx': padding_idx}
            self.check_output_with_place(self.place)


class TestEmbeddingInvalidInput(unittest.TestCase):
    def test_empty_weight_with_nonempty_ids(self):
        with base.dygraph.guard(paddle.XPUPlace(0)):
            weight = paddle.empty([0, 8], dtype='float32')
            for dtype in ['int32', 'int64']:
                with self.subTest(dtype=dtype):
                    ids = paddle.zeros([1], dtype=dtype)
                    with self.assertRaisesRegex(
                        Exception,
                        r'Input\(Weight\) in OP\(embedding\).*Input\(Ids\) is not empty',
                    ):
                        paddle.nn.functional.embedding(ids, weight)

    def test_empty_weight_with_empty_ids(self):
        with base.dygraph.guard(paddle.XPUPlace(0)):
            weight = paddle.empty([0, 8], dtype='float32')
            ids = paddle.empty([0], dtype='int64')
            out = paddle.nn.functional.embedding(ids, weight)
            self.assertEqual(out.shape, [0, 8])


support_types = get_xpu_op_support_types('lookup_table_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestLookupTableOP, stype)

if __name__ == "__main__":
    unittest.main()
