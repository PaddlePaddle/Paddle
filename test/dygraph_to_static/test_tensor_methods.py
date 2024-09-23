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

import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_ast_only,
    test_pir_only,
)

import paddle


def tensor_clone(x):
    x = paddle.to_tensor(x)
    y = x.clone()
    return y


class TestTensorClone(Dy2StTestBase):
    def _run(self):
        x = paddle.ones([1, 2, 3])
        return paddle.jit.to_static(tensor_clone)(x).numpy()

    def test_tensor_clone(self):
        with enable_to_static_guard(False):
            dygraph_res = self._run()

        static_res = self._run()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-05)


def tensor_numpy(x):
    x = paddle.to_tensor(x)
    x.clear_gradient()
    return x


class TestTensorDygraphOnlyMethodError(Dy2StTestBase):
    def _run(self):
        x = paddle.zeros([2, 2])
        y = paddle.jit.to_static(tensor_numpy)(x)
        return y.numpy()

    @test_ast_only
    def test_to_static_numpy_report_error(self):
        with enable_to_static_guard(False):
            dygraph_res = self._run()

        with self.assertRaises(AssertionError):
            static_res = self._run()


def tensor_item(x):
    x = paddle.to_tensor(x)
    y = x.clone()
    return y.item()


class TestTensorItem(Dy2StTestBase):
    def _run(self):
        x = paddle.ones([1])
        return paddle.jit.to_static(tensor_item)(x)

    def test_tensor_clone(self):
        with enable_to_static_guard(False):
            dygraph_res = self._run()

        static_res = self._run()
        np.testing.assert_allclose(dygraph_res, static_res)


def tensor_size(x):
    x = paddle.to_tensor(x)
    x = paddle.reshape(x, paddle.shape(x))  # dynamic shape
    y = x.size
    return y


class TestTensorSize(Dy2StTestBase):
    def _run(self, to_static):
        x = paddle.ones([1, 2, 3])
        if not to_static:
            return tensor_size(x)
        ret = paddle.jit.to_static(tensor_size)(x)
        if hasattr(ret, 'numpy'):
            ret = ret.numpy()
        return ret

    @test_pir_only
    def test_tensor_size(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-5)


def true_div(x, y):
    z = x / y
    return z


class TestTrueDiv(Dy2StTestBase):
    def _run(self):
        x = paddle.to_tensor([3], dtype='int64')
        y = paddle.to_tensor([4], dtype='int64')
        return paddle.jit.to_static(true_div)(x, y).numpy()

    def test_true_div(self):
        with enable_to_static_guard(False):
            dygraph_res = self._run()
        static_res = self._run()
        np.testing.assert_allclose(dygraph_res, static_res, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
