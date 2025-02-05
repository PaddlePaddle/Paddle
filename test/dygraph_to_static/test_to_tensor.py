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

import numpy
import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle
from paddle.base import core


def case0(x):
    a = paddle.to_tensor([1.0, 2.0, 3.0], dtype="int64")

    return a


def case1(x):
    paddle.set_default_dtype("float64")
    a = paddle.to_tensor([1, 2, 3], stop_gradient=False, dtype='float32')

    return a


def case2(x):
    if core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    elif core.is_compiled_with_xpu():
        place = paddle.XPUPlace(0)
    else:
        place = paddle.CPUPlace()
    a = paddle.to_tensor(
        [1.0, 2.0, 3.0], place=place, dtype="int64", stop_gradient=False
    )

    return a


def case3(x):
    paddle.set_default_dtype("float64")
    if core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    elif core.is_compiled_with_xpu():
        place = paddle.XPUPlace(0)
    else:
        place = paddle.CPUPlace()
    a = paddle.to_tensor([1.0, 2.0, 3.0], place=place)

    return a


def case4(x):
    paddle.set_default_dtype("float64")
    if core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    elif core.is_compiled_with_xpu():
        place = paddle.XPUPlace(0)
    else:
        place = paddle.CPUPlace()
    a = paddle.to_tensor([1], place=place)
    b = paddle.to_tensor([2.1], place=place, stop_gradient=False, dtype="int64")
    c = paddle.to_tensor([a, b, [1]], dtype="float32")

    return c


def case5(x):
    paddle.set_default_dtype("float64")
    a = paddle.to_tensor([1, 2])

    return a


def case6(x):
    na = numpy.array([1, 2], dtype='int32')
    a = paddle.to_tensor(na)

    return a


def case7(x):
    a = paddle.to_tensor(10.0)

    return a


def case8(x):
    a = paddle.to_tensor({1: 1})
    return a


def case_to_tensor_default_dtype():
    return paddle.to_tensor(1)


class TestToTensorReturnVal(Dy2StTestBase):
    def test_to_tensor_badreturn(self):
        paddle.disable_static()
        x = paddle.to_tensor([3])

        a = paddle.jit.to_static(case0)(x)
        b = case0(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case1)(x)
        b = case1(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case2)(x)
        b = case2(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case3)(x)
        b = case3(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case4)(x)
        b = case4(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case5)(x)
        b = case5(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case6)(x)
        b = case6(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

        a = paddle.jit.to_static(case7)(x)
        b = case7(x)
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

    def test_to_tensor_default_dtype(self):
        a = paddle.jit.to_static(case_to_tensor_default_dtype)()
        b = case_to_tensor_default_dtype()
        self.assertTrue(a.dtype == b.dtype)
        self.assertTrue(a.stop_gradient == b.stop_gradient)
        self.assertTrue(a.place._equals(b.place))

    def test_to_tensor_err_log(self):
        paddle.disable_static()
        x = paddle.to_tensor([3])
        try:
            a = paddle.jit.to_static(case8)(x)
        except Exception as e:
            self.assertTrue(
                "Do not support transform type `<class 'dict'>` to tensor"
                in str(e)
            )


class TestStatic(Dy2StTestBase):
    def test_static(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            elif core.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
            else:
                place = paddle.CPUPlace()

            paddle.set_default_dtype("float64")
            x = paddle.to_tensor(
                paddle.randn([5, 2]),
                dtype='float64',
                stop_gradient=False,
                place=place,
            )

            fc_net = paddle.nn.Linear(2, 1)
            out = fc_net(x)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))

            exe = paddle.static.Executor()
            exe.run(startup_prog)
            res = exe.run(fetch_list=[x, out])


class TestInt16(Dy2StTestBase):
    def test_static(self):
        paddle.enable_static()
        data = np.array([1, 2], dtype="int16")
        x = paddle.to_tensor(data)
        self.assertEqual(x.dtype, paddle.int16)

        y = paddle.to_tensor([1, 2], dtype="int16")
        self.assertEqual(y.dtype, paddle.int16)


class TestNestedListWithTensor(Dy2StTestBase):
    def test_nested_list_with_tensor(self):
        paddle.enable_static()
        x = paddle.to_tensor(1)
        y = paddle.to_tensor([[x]])
        if paddle.framework.use_pir_api():
            self.assertEqual(y.shape, [1, 1])
        else:
            self.assertEqual(y.shape, (1, 1))
        self.assertEqual(y.dtype, paddle.int64)


if __name__ == '__main__':
    unittest.main()
