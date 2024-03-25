# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_pir_only,
)

import paddle


def fn_with_inplace_op(inplace_op, x):
    y = inplace_op(x)
    z = inplace_op(x)
    return y + z


class TestDealInplace(Dy2StTestBase):
    def run_test(self, dygraph_fn, *inputs):
        dygraph_out = dygraph_fn(*inputs)
        static_fn = paddle.jit.to_static(dygraph_fn)
        static_out = static_fn(*inputs)
        np.testing.assert_allclose(dygraph_out.numpy(), static_out.numpy())

    @test_pir_only
    def test_deal_view(self):
        bn_layer = paddle.nn.BatchNorm2D(10)
        x = paddle.to_tensor(np.random.random((2, 10, 3, 3)).astype('float32'))
        self.run_test(fn_with_inplace_op, bn_layer, x)

    @test_pir_only
    def test_deal_inplace(self):
        sigmoid_layer = paddle.nn.Sigmoid()
        x = paddle.to_tensor(np.random.random((2, 10, 3, 3)).astype('float32'))
        self.run_test(fn_with_inplace_op, sigmoid_layer, x)


if __name__ == '__main__':
    unittest.main()
