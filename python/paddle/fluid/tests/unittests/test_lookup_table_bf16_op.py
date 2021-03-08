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
    out_grad_shape = (np.prod(ids.shape[:-1]), w_shape[-1])
    out_grad = weights[ids.flatten()].reshape(out_grad_shape)
    for i, idx in enumerate(ids.flatten()):
        w_grad[idx, :] += out_grad[i]
    return w_grad


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16Op(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        self.dtype = np.uint16

        table = np.random.random((17, 31)).astype("float32")
        self.ids = np.random.randint(0, 17, (4, 1)).astype("int64")

        self.w_bf16 = convert_float_to_uint16(table)
        self.out_bf16 = self.w_bf16[self.ids.flatten()]
        self.out_fp32 = table[self.ids.flatten()]
        self.w_grad_fp32 = _get_grad(table, self.ids)

        self.inputs = {'W': self.w_bf16, 'Ids': self.ids}
        self.outputs = {'Out': self.out_fp32}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_dygraph=False,
            max_relative_error=1.5e-2,
            user_defined_grads=[self.w_grad_fp32],
            user_defined_grad_outputs=[self.out_bf16])


class TestLookupTableBF16OpIds4D(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        self.dtype = np.uint16

        table = np.random.random((17, 31)).astype("float32")
        self.ids = np.random.randint(0, 17, (2, 4, 5, 1)).astype("int64")

        self.w_bf16 = convert_float_to_uint16(table)
        self.out_bf16 = self.w_bf16[self.ids.flatten()].reshape((2, 4, 5, 31))
        self.out_fp32 = table[self.ids.flatten()].reshape((2, 4, 5, 31))
        self.w_grad_fp32 = _get_grad(table, self.ids)

        self.inputs = {'W': self.w_bf16, 'Ids': self.ids}
        self.outputs = {'Out': self.out_fp32}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_dygraph=False,
            max_relative_error=1.5e-2,
            user_defined_grads=[self.w_grad_fp32],
            user_defined_grad_outputs=[self.out_bf16])


if __name__ == "__main__":
    enable_static()
    unittest.main()
