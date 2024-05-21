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
    ('primal', 'cotangent', 'src_dtype', 'dst_type'),
    [
        (
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            np.float32,
            np.float64,
        ),
        (
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            np.float64,
            np.float32,
        ),
        (
            np.random.rand(10, 10),
            np.random.rand(10, 10),
            np.float32,
            np.float32,
        ),
    ],
)
class TestCastGradComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primal = cls.primal.astype(cls.src_dtype)
        cls.cotangent = cls.cotangent.astype(cls.src_dtype)

    def test_cast_grad_comp(self):
        core.set_prim_eager_enabled(True)

        def actual(primal, cotangent):
            x = paddle.to_tensor(primal)
            x.stop_gradient = False
            v = paddle.to_tensor(cotangent)
            y = paddle.cast(x, self.dst_type)
            x_cotangent = paddle.grad(y, x, v)
            return x_cotangent

        def desired(primal, cotangent):
            return (cotangent * np.ones_like(primal)).astype(primal.dtype)

        actual = actual(self.primal, self.cotangent)
        desired = desired(self.primal, self.cotangent)
        from paddle.base.data_feeder import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE

        if actual[0].dtype in _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE.keys():
            TO_NUMPY_DTYPE = _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE
        else:
            from paddle.base.data_feeder import _PADDLE_DTYPE_2_NUMPY_DTYPE

            TO_NUMPY_DTYPE = _PADDLE_DTYPE_2_NUMPY_DTYPE

        self.assertEqual(TO_NUMPY_DTYPE[actual[0].dtype], desired.dtype)
        np.testing.assert_allclose(
            actual=actual[0],
            desired=desired,
            rtol=1e-6,
            atol=0,
        )
        core.set_prim_eager_enabled(False)


if __name__ == '__main__':
    unittest.main()
