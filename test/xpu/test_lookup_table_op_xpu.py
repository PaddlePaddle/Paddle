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

paddle.enable_static()


class XPUTestLookupTableOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "lookup_table"
        self.use_dynamic_create_class = False

    class TestLookupTableOp(XPUOpTest):
        def setUp(self):
            self.op_type = "lookup_table"
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.init_config()
            self.init_test_case()

        def init_test_case(self):
            table = np.random.random((17, 31)).astype(self.dtype)
            ids = np.random.randint(0, 17, 4).astype(self.id_dtype)
            ids_expand = np.expand_dims(ids, axis=1)
            self.inputs = {'W': table, 'Ids': ids_expand}
            self.outputs = {'Out': table[ids]}

        def init_config(self):
            self.id_dtype = "int64"

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestLookupTableOpWithTensorIds(TestLookupTableOp):
        def init_test_case(self):
            table = np.random.random((17, 31)).astype(self.dtype)
            ids = np.random.randint(low=0, high=17, size=(2, 4, 5, 1)).astype(
                self.id_dtype
            )
            self.inputs = {'W': table, 'Ids': ids}
            self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    class TestLookupTableOpWithPadding(TestLookupTableOp):
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


support_types = get_xpu_op_support_types('lookup_table')
for stype in support_types:
    create_test_class(globals(), XPUTestLookupTableOp, stype)

if __name__ == "__main__":
    unittest.main()
