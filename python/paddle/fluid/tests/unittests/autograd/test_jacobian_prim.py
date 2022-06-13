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

import typing
import unittest

import numpy as np
import paddle

import config
import utils

paddle.enable_static()


@utils.place(config.DEVICES)
@utils.parameterize((utils.TEST_CASE_NAME, 'fun', 'args', 'dtype'), (
    ('unary_float32', paddle.tanh, (np.random.rand(2, 3), ), 'float32'),
    ('binary_float32', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float32'),
    ('unary_float64', paddle.tanh, (np.random.rand(2, 3), ), 'float64'),
    ('binary_float64', paddle.matmul,
     (np.random.rand(2, 3), np.random.rand(3, 2)), 'float64'),
))
class TestJacobianPrim(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.args = [arg.astype(self.dtype) for arg in self.args]
        self.rtol = config.TOLERANCE.get(
            self.dtype).get('first_order_grad').get('rtol')
        self.atol = config.TOLERANCE.get(
            self.dtype).get('first_order_grad').get('atol')

    def test_jacobian_prim(self):

        def wrapper(fun, args):
            mp = paddle.static.Program()
            sp = paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                static_args = [
                    paddle.static.data(f'arg{i}', arg.shape, self.dtype)
                    for i, arg in enumerate(args)
                ]
                for arg in static_args:
                    arg.stop_gradient = False
                jac = paddle.incubate.autograd.Jacobian(fun, static_args)[:]
                if paddle.incubate.autograd.prim_enabled():
                    paddle.incubate.autograd.prim2orig(mp.block(0))
            exe = paddle.static.Executor()
            exe.run(sp)
            [jac] = exe.run(mp,
                            feed={f'arg{i}': arg
                                  for i, arg in enumerate(args)},
                            fetch_list=[jac])
            return jac

        paddle.incubate.autograd.enable_prim()
        prim_jac = wrapper(self.fun, self.args)
        paddle.incubate.autograd.disable_prim()
        orig_jac = wrapper(self.fun, self.args)

        np.testing.assert_allclose(orig_jac,
                                   prim_jac,
                                   rtol=self.rtol,
                                   atol=self.atol)


if __name__ == "__main__":
    unittest.main()
