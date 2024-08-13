# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

os.environ["FLAGS_prim_vjp_skip_default_ops"] = "False"

import unittest

import paddle
from paddle import jit, nn

paddle.core._set_prim_all_enabled(True)

x = paddle.randn([4, 1])
y = paddle.randn([4, 1])

x.stop_gradient = False
y.stop_gradient = False

model = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
model2 = nn.Sequential(
    nn.Linear(1, 1),
)


class TestPaddleSciencemodel(unittest.TestCase):
    def test_concat(self):
        @jit.to_static
        def concat(x, y):
            """abc"""
            z = paddle.concat([x, y], 0)
            out = model(z)
            out0, out1 = paddle.split(out, 2, axis=0)
            g0 = paddle.grad(out0, x)[0]
            g1 = paddle.grad(out1, y)[0]
            return g0, g1

        g0, g1 = concat(x, y)
        loss = g0.sum() + g1.sum()
        loss.backward()


class TestEularBeam(unittest.TestCase):
    def test_eular_beam(self):
        @jit.to_static
        def eular_beam(x):
            """abc"""
            z_ = model(x)
            out = model2(z_)
            g0 = paddle.grad(out, x)[0]
            g1 = paddle.grad(g0, x)[0]
            g2 = paddle.grad(g1, x)[0]
            g3 = paddle.grad(g2, x)[0]
            return g3

        g3 = eular_beam(x)
        loss = g3.sum()
        loss.backward()


if __name__ == '__main__':
    unittest.main()
