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
from paddle.fluid import core
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


def _run_ldexp(mode, x, y, device='cpu'):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        # Set device
        paddle.set_device(device)
        # y is tensor
        x_ = paddle.to_tensor(x)
        y_ = paddle.to_tensor(y)
        res = paddle.ldexp(x_, y_)
        return res.numpy()
    # static graph mode
    elif mode == STATIC:
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
            y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
            res = paddle.ldexp(x_, y_)
            place = (
                paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
            )
            exe = paddle.static.Executor(place)
            outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
            return outs[0]


class TestLdexpAPI(unittest.TestCase):
    def setUp(self):
        self.places = ['cpu']
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_ldexp(self):
        np.random.seed(7)
        for place in self.places:
            # test 1-d float tensor and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.randint(-10, 10, dims)).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = (np.random.randint(-10, 10, dims)).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            # test 1-d int tensor and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.randint(-10, 10, dims)).astype(np.int64)
            y = (np.random.randint(-10, 10, dims)).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            dims = (np.random.randint(200, 300),)
            x = (np.random.randint(-10, 10, dims)).astype(np.int32)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            # test 1-d float tensor and 1-d int scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            # test 1-d int tensor and 1-d int scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int32)
            y = int(np.random.rand() * 10)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            # test 1-d float scalar and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = np.random.rand() * 10
            y = (np.random.randint(-10, 10, dims)).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            dims = (np.random.randint(200, 300),)
            x = np.random.rand() * 10
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            # test 1-d int scalar and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = int(np.random.rand() * 10)
            y = (np.random.randint(-10, 10, dims)).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            dims = (np.random.randint(200, 300),)
            x = int(np.random.rand() * 10)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)

            # test broadcast
            dims = (
                np.random.randint(1, 10),
                np.random.randint(5, 10),
                np.random.randint(5, 10),
            )
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.randint(-10, 10, dims[-1])).astype(np.int64)
            res = _run_ldexp(DYNAMIC, x, y)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)
            res = _run_ldexp(STATIC, x, y)
            np.testing.assert_allclose(res, np.ldexp(x, y), rtol=1e-02)


if __name__ == '__main__':
    unittest.main()
