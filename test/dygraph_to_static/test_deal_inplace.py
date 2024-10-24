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

import copy
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


class ParamInplaceNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[10], dtype='float32')

    def forward(self, x):
        if paddle.in_dynamic_mode():
            # In dynamic mode, it is invalid if a leaf tensor is inplaced.
            return paddle.assign(self.weight)
        else:
            return paddle._C_ops.assign_(self.weight)


class ParamDirectlyReturnNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[10], dtype='float32')

    def forward(self, x):
        return self.weight


class ParamReturnAfterAssignNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.weight = self.create_parameter(shape=[10], dtype='float32')

    def forward(self, x):
        return paddle.assign(self.weight)


class InputDirectlyReturnNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TestDealInplace(Dy2StTestBase):
    def copy_inputs(self, inputs):
        # Make a copy for inputs to avoid inplace effect.
        return [
            (
                paddle.assign(input)
                if isinstance(input, paddle.Tensor)
                else copy.copy(input)
            )
            for input in inputs
        ]

    def run_test(self, dygraph_fn, *inputs, static_n_times=1):
        dygraph_out = dygraph_fn(*self.copy_inputs(inputs))
        static_fn = paddle.jit.to_static(dygraph_fn)
        for i in range(static_n_times):
            static_out = static_fn(*self.copy_inputs(inputs))
            np.testing.assert_allclose(
                dygraph_out.numpy(),
                static_out.numpy(),
                err_msg=f"Run {i}-th check failed.",
            )

    @test_pir_only
    def test_deal_view(self):
        bn_layer = paddle.nn.BatchNorm2D(10)
        x = paddle.to_tensor(np.random.random((2, 10, 3, 3)).astype('float32'))
        x.stop_gradient = False
        self.run_test(fn_with_inplace_op, bn_layer, x, static_n_times=2)

    @test_pir_only
    def test_deal_inplace(self):
        sigmoid_layer = paddle.nn.Sigmoid()
        x = paddle.to_tensor(np.random.random((2, 10, 3, 3)).astype('float32'))
        x.stop_gradient = False
        self.run_test(fn_with_inplace_op, sigmoid_layer, x, static_n_times=2)

    @test_pir_only
    def test_param_inplace(self):
        net = ParamInplaceNet()
        x = paddle.to_tensor(np.random.random(10).astype('float32'))
        x.stop_gradient = False
        self.run_test(net, x, static_n_times=2)

    @test_pir_only
    def test_param_directly_return(self):
        net = ParamDirectlyReturnNet()
        x = paddle.to_tensor(np.random.random(10).astype('float32'))
        x.stop_gradient = False
        self.run_test(net, x, static_n_times=2)

    @test_pir_only
    def test_param_return_after_assign(self):
        net = ParamReturnAfterAssignNet()
        x = paddle.to_tensor(np.random.random(10).astype('float32'))
        x.stop_gradient = False
        self.run_test(net, x, static_n_times=2)

    @test_pir_only
    def test_input_directly_return(self):
        net = InputDirectlyReturnNet()
        x = paddle.to_tensor(np.random.random(10).astype('float32'))
        x.stop_gradient = False
        self.run_test(net, x, static_n_times=2)


if __name__ == '__main__':
    unittest.main()
