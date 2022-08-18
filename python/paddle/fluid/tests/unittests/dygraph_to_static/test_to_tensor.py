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

from __future__ import print_function

import paddle
import unittest
from paddle.fluid.framework import program_guard, Program
from paddle.fluid import core


def case0(x):
    a = paddle.to_tensor([1.0, 2.0, 3.0], dtype="int64")

    return a


def case1(x):
    paddle.set_default_dtype("float64")
    a = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)

    return a


def case2(x):
    if core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    a = paddle.to_tensor([1.0, 2.0, 3.0],
                         place=place,
                         dtype="int64",
                         stop_gradient=False)

    return a


def case3(x):
    paddle.set_default_dtype("float64")
    if core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    else:
        place = paddle.CPUPlace()
    a = paddle.to_tensor([1.0, 2.0, 3.0], place=place)

    return a


class TestToTensorReturnVal(unittest.TestCase):

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


class TestStatic(unittest.TestCase):

    def test_static(self):
        paddle.enable_static()
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            else:
                place = paddle.CPUPlace()

            x = paddle.to_tensor(paddle.randn([5, 2]),
                                 dtype='float64',
                                 stop_gradient=False,
                                 place=place)

            out = paddle.static.nn.fc(x, 1)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[x, out])


if __name__ == '__main__':
    unittest.main()
