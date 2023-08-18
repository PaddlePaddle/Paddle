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


def _get_softmax(x, mask):
    masked_x = (x + mask).astype("float32")
    max_value = np.max(masked_x, axis=-1, keepdims=True)
    before_exp = masked_x - max_value
    exp = np.exp(before_exp)
    exp_sum = np.sum(exp, axis=-1, keepdims=True)
    rst = exp / exp_sum
    return rst


class XPUTestFusedSoftmaxMaskOp(XPUOpTestWrapper):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def __init__(self):
        self.op_name = "fused_softmax_mask"
        self.use_dynamic_create_class = False

    class TestFusedSoftmaxMaskOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "fused_softmax_mask"
            self.init_dtype()
            self.set_output()

        def set_output(self):
            x = np.random.random((1, 4, 4096, 4096)).astype("float32")
            mask_input = np.random.random((1, 1, 4096, 4096)).astype("float32")
            self.inputs = {'X': x, 'Mask': mask_input}
            rst = _get_softmax(x, mask_input)
            self.outputs = {'Out': rst}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.place = paddle.XPUPlace(0)

        def init_dtype(self):
            self.dtype = self.in_type


support_types = get_xpu_op_support_types('fused_softmax_mask')
for stype in support_types:
    create_test_class(globals(), XPUTestFusedSoftmaxMaskOp, stype)

if __name__ == '__main__':
    unittest.main()
