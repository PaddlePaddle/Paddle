# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


def _run_ldexp(mode, x, y, device='cpu'):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        # Set device
        paddle.set_device(device)
        # y is scalar
        if isinstance(y, (int, float)):
            x_ = paddle.to_tensor(x)
            y_ = y
            res = paddle.ldexp(x_, y_)
            return res.numpy()
        # y is tensor
        else:
            x_ = paddle.to_tensor(x)
            y_ = paddle.to_tensor(y)
            res = paddle.ldexp(x_, y_)
            return res.numpy()
    # static graph mode
    elif mode == STATIC:
        paddle.enable_static()
        # y is scalar
        if isinstance(y, (int, float)):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.ldexp(x_, y_)
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x}, fetch_list=[res])
                return outs[0]
        # y is tensor
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
                res = paddle.ldexp(x_, y_)
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]


class TestLdexpAPI(unittest.TestCase):
    """TestLdexpAPI."""

    def setUp(self):
        self.places = ['cpu']
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_ldexp(self):
        """test_ldexp."""
        np.random.seed(1024)
        for place in self.places:
            # test 1-d float tensor ** int scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int32)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)

            # dims = (np.random.randint(200, 300),)
            # x = (np.random.rand(*dims) * 10).astype(np.float64)
            # y = np.random.randint(low=1, high=10, size=(1,)).astype(np.int64)
            # res = _run_ldexp(DYNAMIC, x, y, place)
            # np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)
            # res = _run_ldexp(STATIC, x, y, place)
            # np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)

            # test broadcast
            x = (np.random.rand(2, 3) * 10).astype(np.float64)
            y = np.random.randint(low=1, high=10, size=(1, 3)).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)
            res = _run_ldexp(STATIC, x, y)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
