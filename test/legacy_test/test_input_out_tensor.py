# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core


class OutTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

    def test_complex_api(self):
        def run_complex(test_type):
            x = paddle.arange(2, dtype=paddle.float32).unsqueeze(-1)
            y = paddle.arange(3, dtype=paddle.float32)
            x.stop_gradient = False
            y.stop_gradient = False
            z = paddle.ones([100])
            z.stop_gradient = False

            print(x)
            print(y)

            a = x + x
            b = y + y
            c = z + z

            if test_type == 1:
                c = paddle.complex(a, b)
            elif test_type == 2:
                paddle.complex(a, b, out=c)
            else:
                c = paddle.complex(a, b, out=c)

            d = c + c
            print(d)

            d.mean().backward()

            print(x.grad)
            print(y.grad)
            print(z.grad)
            return x.grad, y.grad, z.grad

        x1, y1, z1 = run_complex(1)
        x2, y2, z2 = run_complex(2)
        x3, y3, z3 = run_complex(3)

        np.testing.assert_allclose(
            x1.numpy(),
            x2,
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            x1.numpy(),
            x3,
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            y1.numpy(),
            y2,
            1e-20,
            1e-20,
        )
        np.testing.assert_allclose(
            y1.numpy(),
            y3,
            1e-20,
            1e-20,
        )
        np.testing.assert_equal(z1, None)
        np.testing.assert_equal(z2, None)
        np.testing.assert_equal(z3, None)


if __name__ == '__main__':
    unittest.main()
