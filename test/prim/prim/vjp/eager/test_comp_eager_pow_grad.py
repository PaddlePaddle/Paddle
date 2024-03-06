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

import sys
import unittest

import numpy as np
import parameterized as param

import paddle
from paddle.base import core

sys.path.append('../../../../legacy_test/')
import gradient_checker

from paddle import base


@param.parameterized_class(
    ('primal', 'cotangent', 'dtype'),
    [
        (np.random.rand(10, 10), np.random.rand(10, 10), np.float32),
    ],
)
class TestPowGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.dtype)
        if cls.cotangent is not None:
            cls.cotangent = cls.cotangent.astype(cls.dtype)

    def test_pow_grad_comp(self):
        def actual(primal):
            paddle.disable_static()
            core.set_prim_eager_enabled(True)
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.pow(x, 6.9)
            x_cotangent = paddle.grad(
                y, x, create_graph=True, retain_graph=True
            )
            return x_cotangent[0]

        def desired(primal):
            paddle.disable_static()
            core.set_prim_eager_enabled(False)
            x = paddle.to_tensor(primal, dtype='float32', stop_gradient=False)
            x.stop_gradient = False
            y = paddle.pow(x, 6.9)
            x_cotangent = paddle.grad(
                y, x, create_graph=True, retain_graph=True
            )
            return x_cotangent[0]

        np.testing.assert_allclose(
            actual=actual(self.primal),
            desired=desired(self.primal),
            rtol=1e-6,
            atol=0,
        )
        core.set_prim_eager_enabled(False)


class TestPowGradCheck(unittest.TestCase):
    def pow_wrapper(self, x, e):
        return paddle.pow(x, e)

    def func(self, place):
        core.set_prim_eager_enabled(True)
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data1 = paddle.static.data('data1', [10, 10], dtype)
        data1.persistable = True
        data1.stop_gradient = False
        e = 6.9
        out = paddle.pow(data1, e)
        data1_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)
        gradient_checker.double_grad_check(
            [data1],
            out,
            x_init=[data1_arr],
            place=place,
            eps=eps,
        )
        core.set_prim_eager_enabled(False)

    def test_grad(self):
        paddle.enable_static()
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == '__main__':
    unittest.main()
