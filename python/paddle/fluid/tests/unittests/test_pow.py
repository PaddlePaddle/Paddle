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
import paddle.tensor as tensor
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


def _run_power(mode, x, y):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        # y is scalar
        if isinstance(y, (int, float)):
            x_ = paddle.to_tensor(x)
            y_ = y
            res = paddle.pow(x_, y_)
            return res.numpy()
        # y is tensor
        else:
            x_ = paddle.to_tensor(x)
            y_ = paddle.to_tensor(y)
            res = paddle.pow(x_, y_)
            return res.numpy()
    # static mode
    elif mode == STATIC:
        paddle.enable_static()
        # y is scalar
        if isinstance(y, (int, float)):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.pow(x_, y_)
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x}, fetch_list=[res])
                return outs[0]
        # y is tensor
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
                res = paddle.pow(x_, y_)
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]


class TestPowerAPI(unittest.TestCase):
    """TestPowerAPI."""

    def test_power(self):
        """test_power."""
        np.random.seed(7)
        # test 1-d float tensor ** float scalar
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = np.random.rand() * 10
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        # test 1-d float tensor ** int scalar
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = int(np.random.rand() * 10)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        x = (np.random.rand(*dims) * 10).astype(np.int64)
        y = int(np.random.rand() * 10)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        # test 1-d float tensor ** 1-d float tensor
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(*dims) * 10).astype(np.float64)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        # test 1-d int tensor ** 1-d int tensor
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.int64)
        y = (np.random.rand(*dims) * 10).astype(np.int64)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        # test 1-d int tensor ** 1-d int tensor
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.int32)
        y = (np.random.rand(*dims) * 10).astype(np.int32)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        # test 1-d int tensor ** 1-d int tensor
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.float32)
        y = (np.random.rand(*dims) * 10).astype(np.float32)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

        # test broadcast
        dims = (np.random.randint(1, 10), np.random.randint(5, 10),
                np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1]) * 10).astype(np.float64)
        res = _run_power(DYNAMIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
        res = _run_power(STATIC, x, y)
        np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)


class TestPowerError(unittest.TestCase):
    """TestPowerError."""

    def test_errors(self):
        """test_errors."""
        np.random.seed(7)

        # test dynamic computation graph: inputs must be broadcastable
        dims = (np.random.randint(1, 10), np.random.randint(5, 10),
                np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.float64)
        self.assertRaises(ValueError, _run_power, DYNAMIC, x, y)
        self.assertRaises(ValueError, _run_power, STATIC, x, y)

        # test dynamic computation graph: inputs must be broadcastable
        dims = (np.random.randint(1, 10), np.random.randint(5, 10),
                np.random.randint(5, 10))
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.int8)
        self.assertRaises(TypeError, paddle.pow, x, y)

        # test 1-d float tensor ** int string
        dims = (np.random.randint(200, 300), )
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = int(np.random.rand() * 10)
        self.assertRaises(TypeError, paddle.pow, x, str(y))


if __name__ == '__main__':
    unittest.main()
