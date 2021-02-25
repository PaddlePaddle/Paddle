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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import (OpTest,
                                                  convert_float_to_uint16)
import paddle.fluid.core as core
from paddle import enable_static


def _get_grad(weights, ids):
    w_shape = weights.shape
    w_grad = np.zeros((w_shape), dtype=np.float32)
    out_grad = weights[ids.flatten()]
    for i, idx in enumerate(ids):
        w_grad[idx] += out_grad[i]
    return w_grad


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16Op(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        self.dtype = np.uint16

        table = np.random.random((17, 31)).astype("float32")
        self.ids = np.random.randint(0, 17, (4, 1)).astype("int64")

        self.mkldnn_data_type = "bfloat16"
        self.w_bf16 = convert_float_to_uint16(table)
        self.out = self.w_bf16[self.ids.flatten()]
        self.w_grad_bf16 = convert_float_to_uint16(_get_grad(table, self.ids))

        self.inputs = {'W': self.w_bf16, 'Ids': self.ids}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_dygraph=False,
            user_defined_grads=[self.w_grad_bf16],
            user_defined_grad_outputs=[self.out])


if __name__ == "__main__":
    enable_static()
    unittest.main()
