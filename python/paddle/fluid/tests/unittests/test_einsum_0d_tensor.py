# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from numpy.testing import assert_allclose

import paddle

os.environ['NVIDIA_TF32_OVERRIDE'] = "0"


class Test0DCase0(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_func(self):
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.rand([])
        y.stop_gradient = False
        z = paddle.einsum("...,...->...", x, y)
        assert_allclose(
            z.numpy(),
            np.einsum('...,...->...', x.numpy(), y.numpy()),
            atol=1e-6,
        )
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == []
        assert y.grad.shape == []


class Test0DCase1(Test0DCase0):
    def test_func(self):
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum("...,ij->...", x, y)
        assert_allclose(
            z.numpy(), np.einsum('...,ij->...', x.numpy(), y.numpy()), atol=1e-6
        )
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == []
        assert y.grad.shape == [2, 2]


class Test0DCase2(Test0DCase0):
    def test_func(self):
        x = paddle.rand([2, 2])
        x.stop_gradient = False
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum("ij,ij->", x, y)
        assert_allclose(
            z.numpy(), np.einsum('ij,ij->', x.numpy(), y.numpy()), atol=1e-6
        )
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == [2, 2]


class Test0DCase3(Test0DCase0):
    def test_func(self):
        x = paddle.rand([2, 2])
        x.stop_gradient = True
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum("ij,ij->", x, y)
        assert_allclose(
            z.numpy(), np.einsum('ij,ij->', x.numpy(), y.numpy()), atol=1e-6
        )
        z.mean().backward()
        assert z.shape == []
        assert x.grad is None
        assert y.grad.shape == [2, 2]


class Test0DCase4(Test0DCase0):
    def test_func(self):
        x = paddle.rand([])
        x.stop_gradient = False
        z = paddle.einsum("...->...", x)
        assert_allclose(z.numpy(), np.einsum('...->...', x.numpy()), atol=1e-6)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == []
        assert x.grad.numpy() == 1.0


class Test0DCase5(Test0DCase0):
    def test_func(self):
        x = paddle.rand([2, 2])
        x.stop_gradient = False
        y = paddle.rand([2, 2])
        y.stop_gradient = False
        z = paddle.einsum("i...j, i...j->...", x, y)
        assert_allclose(
            z.numpy(),
            np.einsum('i...j, i...j->...', x.numpy(), y.numpy()),
            atol=1e-6,
        )
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == [2, 2]


class Test0DCase6(Test0DCase0):
    def test_func(self):
        x = paddle.rand([2, 2])
        x.stop_gradient = False
        z = paddle.einsum("ij->", x)
        assert_allclose(z.numpy(), np.einsum('ij->', x.numpy()), atol=1e-6)
        z.mean().backward()
        assert z.shape == []
        assert x.grad.shape == [2, 2]


class Test0DCase7(Test0DCase0):
    def test_func(self):
        """
        3 operands.
        """
        x = paddle.rand([2, 2])
        y = paddle.rand([])
        z = paddle.rand([])
        x.stop_gradient = False
        y.stop_gradient = False
        z.stop_gradient = False
        o = paddle.einsum("ij...,...,...->...", x, y, z)
        assert_allclose(
            o.numpy(),
            np.einsum("ij...,...,...->...", x.numpy(), y.numpy(), z.numpy()),
            atol=1e-6,
        )
        o.mean().backward()
        assert o.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == []
        assert z.grad.shape == []


class Test0DCase8(Test0DCase0):
    def test_func(self):
        """
        3 operands.
        """
        x = paddle.rand([2, 2])
        y = paddle.rand([])
        z = paddle.rand([])
        e = paddle.rand([3, 1])
        x.stop_gradient = False
        y.stop_gradient = False
        z.stop_gradient = False
        e.stop_gradient = False
        o = paddle.einsum("ij...,...,..., km->...", x, y, z, e)
        assert_allclose(
            o.numpy(),
            np.einsum(
                "ij...,...,...,km->...",
                x.numpy(),
                y.numpy(),
                z.numpy(),
                e.numpy(),
            ),
            atol=1e-6,
        )
        o.mean().backward()
        assert o.shape == []
        assert x.grad.shape == [2, 2]
        assert y.grad.shape == []
        assert z.grad.shape == []
        assert e.grad.shape == [3, 1]


if __name__ == "__main__":
    unittest.main()
