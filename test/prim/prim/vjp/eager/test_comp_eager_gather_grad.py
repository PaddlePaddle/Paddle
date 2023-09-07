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

import unittest

import numpy as np
import parameterized as param

import paddle
from paddle.base import core


@param.parameterized_class(
    ('primal0', 'index', 'axis', 'x_dtype', 'index_dtype', 'v'),
    [
        (
            np.random.rand(100),
            np.array([1, 3, 5]),
            0,
            np.float32,
            np.int32,
            np.random.rand(3),
        ),
        (
            np.random.rand(10, 20),
            np.array([1, 3, 5]),
            0,
            np.float64,
            np.int64,
            np.random.rand(3, 20),
        ),
        (
            np.random.rand(10, 20),
            np.array([1, 1, 3]),
            0,
            np.float32,
            np.int32,
            np.random.rand(3, 20),
        ),
        (
            np.random.rand(3, 88, 30),
            np.array([1, 3, 5]),
            1,
            np.float32,
            np.int32,
            np.random.rand(3, 3, 30),
        ),
        (
            np.random.rand(10, 88, 10),
            np.array([1, 3, 5]),
            0,
            np.float32,
            np.int32,
            np.random.rand(3, 88, 10),
        ),
    ],
)
class TestGatherGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal0 = cls.primal0.astype(cls.x_dtype)
        cls.index = cls.index.astype(cls.index_dtype)
        cls.v = cls.v.astype(cls.x_dtype)

    @classmethod
    def tearDownClass(cls):
        core.set_prim_eager_enabled(False)

    def test_exp_grad_comp(self):
        def actual(primal0, index, axis):
            core.set_prim_eager_enabled(True)
            paddle.disable_static()
            x = paddle.to_tensor(
                primal0, dtype=primal0.dtype, stop_gradient=False
            )
            index = paddle.to_tensor(index, dtype=index.dtype)
            x.stop_gradient = False
            index.stop_gradient = True
            out = paddle.gather(x, index, axis)
            res = paddle.grad(out, [x], create_graph=False, retain_graph=True)
            return res[0].numpy()

        def desired(primal0, index, axis):
            core.set_prim_eager_enabled(False)
            paddle.disable_static()
            x = paddle.to_tensor(
                primal0, dtype=primal0.dtype, stop_gradient=False
            )
            index = paddle.to_tensor(index, dtype=index.dtype)
            x.stop_gradient = False
            index.stop_gradient = True
            out = paddle.gather(x, index, axis)
            res = paddle.grad(out, [x], create_graph=False, retain_graph=True)
            return res[0].numpy()

        np.testing.assert_allclose(
            actual=actual(self.primal0, self.index, self.axis),
            desired=desired(self.primal0, self.index, self.axis),
            rtol=1e-6,
            atol=0,
        )


if __name__ == '__main__':
    unittest.main()
