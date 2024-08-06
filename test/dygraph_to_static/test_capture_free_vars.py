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
    test_legacy_and_pt_and_pir,
)

import paddle


def create_simple_closure():
    y = 1

    def simple_closure(x):
        return x + y

    return simple_closure


class BaseLayer(paddle.nn.Layer):
    def add_one(self, x):
        y = x + 1
        return y


class SuperCallWithoutArgumentInForward(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Actually, the super will use the free var `__class__` to
        # call the `add_one` method.
        return super().add_one(x)


class SuperCallWithoutArgumentInControlFlow(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x > 0:
            return super().add_one(x)
        else:
            return x


class TestCaptureFreeVars(Dy2StTestBase):
    def check_fn(self, fn, *inputs):
        dyres = fn(*inputs)
        stres = paddle.jit.to_static(fn)(*inputs)
        np.testing.assert_allclose(dyres.numpy(), stres.numpy())

    @test_legacy_and_pt_and_pir
    def test_simple_closure(self):
        simple_closure = create_simple_closure()
        x = paddle.to_tensor(1.0)
        self.check_fn(simple_closure, x)

    @test_legacy_and_pt_and_pir
    def test_super_call_without_argument_in_forward(self):
        model = SuperCallWithoutArgumentInForward()
        x = paddle.to_tensor(1.0)
        self.check_fn(model, x)

    @test_legacy_and_pt_and_pir
    def test_super_call_without_argument_in_control_flow(self):
        model = SuperCallWithoutArgumentInControlFlow()
        x = paddle.to_tensor(1.0)
        self.check_fn(model, x)


if __name__ == '__main__':
    unittest.main()
