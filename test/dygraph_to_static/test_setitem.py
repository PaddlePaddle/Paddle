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

import unittest

import numpy as np

import paddle


class TestSetItemBase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def init_data(self):
        paddle.seed(2023)
        x = paddle.randn([4, 8, 16, 32])
        x.stop_gradient = False
        return x

    def init_func(self):
        def foo(x):
            y = x + 1
            y[:, 2] = x[:, 2] + 99
            return y

        return foo

    def test_case(self):
        func = self.init_func()
        dy_res = self.run_dygrah(func)
        st_res = self.run_to_static(func)

        for dy_out, st_out in zip(dy_res, st_res):
            np.testing.assert_allclose(dy_out.numpy(), st_out.numpy())

    def run_dygrah(self, func):
        x = self.init_data()
        y = func(x)
        x_grad = paddle.grad(y, x)[0]
        return y, x_grad

    def run_to_static(self, func):
        func = paddle.jit.to_static(func)
        return self.run_dygrah(func)


class TestCase1(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[2] = x[2] + 99  # (2, )
            return y

        return foo


class TestCase2(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[:] = x[:] + 99  # slice(None,None,None)
            return y

        return foo


class TestCase3(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[1::2] = x[1::2] + 99  # slice(1,None,2)
            return y

        return foo


class TestCase4(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[1, 2] = x[1, 2] + 99  # (1, 2)
            return y

        return foo


class TestCase5(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[[1, 2], [2, 3]] = x[[1, 2], [2, 3]] + 99  # ([1,2],[2,3])
            return y

        return foo


class TestCase6(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[1, :, 3] = x[1, :, 3] + 99  # slice(None,None,None),3)
            return y

        return foo


class TestCase7(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[1, ..., 2] = x[1, ..., 2] + 99  # (1, ..., 2)
            return y

        return foo


class TestCase8(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            index = paddle.to_tensor([1, 2], dtype="int64")
            y[index] = x[index] + 99  # Tensor([1,2])
            return y

        return foo


class TestCase9(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            one = paddle.to_tensor(1, dtype="int64")
            two = paddle.to_tensor(2, dtype="int64")
            y[one, :, :, 2] = x[1, :, :, two] + 100  # Tensor(1), Tensor(2)
            return y

        return foo


class TestCase10(TestSetItemBase):
    def init_func(self):
        def foo(x):
            y = x + 1
            y[..., 4:6] = y[..., 4:6] * 10000
            return y

        return foo


class TestCase11(TestSetItemBase):
    # Test gradient of value tensor
    def init_func(self):
        def foo(x, value):
            y = x + 1
            y[2, 4] = value
            return y

        return foo

    def run_dygrah(self, func):
        x = self.init_data()
        value = paddle.ones((16, 32))
        value.stop_gradient = False
        y = func(x, value)
        x_grad, value_grad = paddle.grad(y, [x, value])
        return y, x_grad, value_grad


if __name__ == '__main__':
    unittest.main()
