#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.contrib.mixed_precision.amp_nn import check_finite_and_unscale

paddle.enable_static()


class TestCheckFiniteAndUnscale(unittest.TestCase):
    def get_prog(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with program_guard(main_program):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            scale = paddle.static.data(name="scale", shape=[1], dtype='float32')
            c = paddle.fluid.layers.elementwise_div(a, b)
            out, found_inf = check_finite_and_unscale([c], scale)

        return main_program, out, found_inf

    def run_prog(self, a, b, scale):
        main_program, out, found_inf = self.get_prog()
        place = fluid.NPUPlace(0)
        exe = fluid.Executor(place)
        out_, founf_inf_ = exe.run(main_program,
                                   feed={"a": a,
                                         "b": b,
                                         "scale": scale},
                                   fetch_list=[out, found_inf])
        return out_, founf_inf_

    def test_contains_nan(self):
        a = np.zeros((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')
        scale = np.array([2.0]).astype('float32')

        out, found_inf = self.run_prog(a, b, scale)
        print(out, found_inf)

        self.assertTrue(found_inf[0])

    def test_contains_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')
        scale = np.array([2.0]).astype('float32')

        out, found_inf = self.run_prog(a, b, scale)
        print(out, found_inf)

        self.assertTrue(found_inf[0])

    def test_not_contains_nan_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.ones((32, 32)).astype('float32')
        scale = np.array([2.0]).astype('float32')

        out, found_inf = self.run_prog(a, b, scale)
        print(out, found_inf)

        self.assertTrue(np.allclose(out, (a / b) / scale[0]))
        self.assertFalse(found_inf[0])


class TestCheckFiniteAndUnscaleClearFloatStatus(unittest.TestCase):
    def get_prog(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        with program_guard(main_program):
            a = paddle.static.data(name="a", shape=[32, 32], dtype='float32')
            b = paddle.static.data(name="b", shape=[32, 32], dtype='float32')
            scale = paddle.static.data(name="scale", shape=[1], dtype='float32')
            c = paddle.fluid.layers.elementwise_div(a, b)
            out, found_inf = check_finite_and_unscale([c], scale)
            d = paddle.fluid.layers.elementwise_add(a, b)
            out, found_inf = check_finite_and_unscale([d], scale)

        return main_program, out, found_inf

    def run_prog(self, a, b, scale):
        main_program, out, found_inf = self.get_prog()
        place = fluid.NPUPlace(0)
        exe = fluid.Executor(place)
        out_, founf_inf_ = exe.run(main_program,
                                   feed={"a": a,
                                         "b": b,
                                         "scale": scale},
                                   fetch_list=[out, found_inf])
        return out_, founf_inf_

    def test_not_contains_nan_inf(self):
        a = np.ones((32, 32)).astype('float32')
        b = np.zeros((32, 32)).astype('float32')
        scale = np.array([2.0]).astype('float32')

        out, found_inf = self.run_prog(a, b, scale)
        print(out, found_inf)

        self.assertTrue(np.allclose(out, (a + b) / scale[0]))
        self.assertFalse(found_inf[0])


if __name__ == '__main__':
    unittest.main()
