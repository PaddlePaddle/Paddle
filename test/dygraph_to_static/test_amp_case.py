#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
    test_ast_only,
    test_pir_only,
)

import paddle

np.random.seed(1)


def func(x):
    y = x[0:3].astype("float32")
    return y


class TestAmp64Case(Dy2StTestBase):
    def _run_static(self):
        static_func = paddle.jit.to_static(func)
        x = paddle.randn((10, 10)).astype("float64")
        with paddle.amp.auto_cast(True, level="O2"):
            dy_out = func(x)
            st_out = static_func(x)
        np.testing.assert_allclose(dy_out.numpy(), st_out.numpy())

    def test_ast_to_func(self):
        self._run_static()


class Net(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.linear = paddle.nn.Linear(5, 5)

    def forward(self, x):
        out = self.linear(x)
        with paddle.amp.auto_cast(level='O2'):
            out = self.linear(out)
        return out


class TestPartialAutoCast(Dy2StTestBase):
    @test_ast_only
    @test_pir_only
    def test_run(self):
        if not paddle.base.core.is_compiled_with_cuda():
            return
        x = paddle.randn([5, 5], 'float32')
        net = Net()
        net = paddle.jit.to_static(net)
        out = net(x)
        main = net.forward.main_program
        cast_op_count = 0
        for op in main.global_block().ops:
            if op.name() == 'pd_op.cast':
                cast_op_count += 1
        np.testing.assert_equal(cast_op_count, 3)


if __name__ == '__main__':
    unittest.main()
