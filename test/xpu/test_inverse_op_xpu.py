#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from numpy.linalg import inv
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def inverse_reference(x):
    origin_shape = x.shape
    x = x.reshape((-1, origin_shape[-2], origin_shape[-1]))
    batch = x.shape[0]
    result = inv(x[0])
    for i in range(1, batch):
        result = np.concatenate((result, inv(x[i])))
    result = result.reshape(origin_shape)
    return {'Output': result}


class XPUTestInverseOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'inverse'
        self.use_dynamic_create_class = False

    class TestInverseOpNoBatchBase(XPUOpTest):
        def setUp(self):
            self.op_type = 'inverse'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)

            self.set_inputs()

            self.inputs = {'Input': self.x}
            self.outputs = inverse_reference(self.x)

        def set_inputs(self):
            self.shape = (23, 23)
            self.x = np.random.random(self.shape).astype(self.dtype)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['Input'], 'Output')

    class TestInverseOpBatchBase(XPUOpTest):
        def setUp(self):
            self.op_type = 'inverse'
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)

            self.set_inputs()

            self.inputs = {'Input': self.x}
            self.outputs = inverse_reference(self.x)

        def set_inputs(self):
            self.shape = (2, 3, 23, 23)
            self.x = np.random.random(self.shape).astype(self.dtype)

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, {'x'}, ['out'])


support_types = get_xpu_op_support_types('inverse')
for stype in support_types:
    create_test_class(globals(), XPUTestInverseOp, stype)


if __name__ == '__main__':
    unittest.main()
