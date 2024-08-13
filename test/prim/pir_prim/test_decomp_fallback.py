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

import paddle


class TestFallBackBase(unittest.TestCase):
    def setUp(self):
        self.func_api = None
        self.dtype = np.float32
        self.tol = 1e-6


def custom_dropout(x, p):
    return paddle.nn.functional.dropout(x, p) + 2.0


class TestDropOutFallBack(TestFallBackBase):
    def setUp(self):
        super().setUp()
        self.func_api = custom_dropout
        self.x = paddle.to_tensor([[1.0, -2], [3.0, 4]], dtype=self.dtype)
        self.p = paddle.to_tensor(0.0, dtype=self.dtype)

    def test_fallback(self):
        static_func = paddle.jit.to_static(self.func_api, full_graph=True)
        dynamic_func = self.func_api

        out = static_func(self.x, self.p)
        ref_out = dynamic_func(self.x, self.p)

        for ref, actual in zip(ref_out, out):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )


def custom_full(shape, value):
    return paddle.full_like(shape, value) + 2.0


class TestFullLikeFallBack(TestFallBackBase):
    def setUp(self):
        super().setUp()
        self.func_api = custom_full
        self.x = paddle.to_tensor([[1.0, -2], [3.0, 4]], dtype=self.dtype)
        self.value = paddle.to_tensor(2, dtype=self.dtype)

    def test_fallback(self):
        static_func = paddle.jit.to_static(self.func_api, full_graph=True)
        dynamic_func = self.func_api

        out = static_func(self.x, self.value)
        ref_out = dynamic_func(self.x, self.value)

        for ref, actual in zip(ref_out, out):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )


def custom_squeeze(x, axis):
    return paddle.squeeze(x, axis) + 2.0


class TestSqueezeFallBack(TestFallBackBase):
    def setUp(self):
        super().setUp()
        self.func_api = custom_squeeze
        self.x = paddle.rand([5, 1, 10], dtype=self.dtype)
        self.axis = paddle.to_tensor(1, dtype=paddle.int64)

    def test_fallback(self):
        static_func = paddle.jit.to_static(self.func_api, full_graph=True)
        dynamic_func = self.func_api

        out = static_func(self.x, self.axis)
        ref_out = dynamic_func(self.x, self.axis)

        for ref, actual in zip(ref_out, out):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )


def custom_unsqueeze(x, axis):
    return paddle.unsqueeze(x, axis) + 2.0


class TestUnsqueezeFallBack(TestFallBackBase):
    def setUp(self):
        super().setUp()
        self.func_api = custom_unsqueeze
        self.x = paddle.rand([5, 10], dtype=self.dtype)
        self.axis = paddle.to_tensor([0, 2], dtype=paddle.int64)

    def test_fallback(self):
        static_func = paddle.jit.to_static(self.func_api, full_graph=True)
        dynamic_func = self.func_api

        out = static_func(self.x, self.axis)
        ref_out = dynamic_func(self.x, self.axis)

        for ref, actual in zip(ref_out, out):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )


def custom_any(x, axis):
    return paddle.any(x, axis)


class TestAnyFallBack(TestFallBackBase):
    def setUp(self):
        super().setUp()
        self.func_api = custom_any
        self.x = paddle.to_tensor([[1, 0], [1, 1]], dtype='int32').cast('bool')
        # Axis cannot accept a list of tensors,
        # the framework will check the argument type before decomposition.
        self.axis = [0]

    def test_fallback(self):
        static_func = paddle.jit.to_static(self.func_api, full_graph=True)
        dynamic_func = self.func_api

        out = static_func(self.x, self.axis)
        ref_out = dynamic_func(self.x, self.axis)

        for ref, actual in zip(ref_out, out):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )


if __name__ == '__main__':
    unittest.main()
