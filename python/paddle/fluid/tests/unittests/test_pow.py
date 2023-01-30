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

<<<<<<< HEAD
=======
from __future__ import print_function
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest

import numpy as np

import paddle
<<<<<<< HEAD
import paddle.fluid.core as core
=======
import paddle.tensor as tensor
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


<<<<<<< HEAD
def _run_power(mode, x, y, device='cpu'):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        # Set device
        paddle.set_device(device)
=======
def _run_power(mode, x, y):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
    # static graph mode
=======
    # static mode
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    elif mode == STATIC:
        paddle.enable_static()
        # y is scalar
        if isinstance(y, (int, float)):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.pow(x_, y_)
<<<<<<< HEAD
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
=======
                place = paddle.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x}, fetch_list=[res])
                return outs[0]
        # y is tensor
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
                res = paddle.pow(x_, y_)
<<<<<<< HEAD
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
=======
                place = paddle.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]


class TestPowerAPI(unittest.TestCase):
    """TestPowerAPI."""

<<<<<<< HEAD
    def setUp(self):
        self.places = ['cpu']
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_power(self):
        """test_power."""
        np.random.seed(7)
        for place in self.places:
            # test 1-d float tensor ** float scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = np.random.rand() * 10
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d float tensor ** int scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = int(np.random.rand() * 10)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = int(np.random.rand() * 10)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d float tensor ** 1-d float tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.rand(*dims) * 10).astype(np.float64)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d int tensor ** 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = (np.random.rand(*dims) * 10).astype(np.int64)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d int tensor ** 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int32)
            y = (np.random.rand(*dims) * 10).astype(np.int32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d int tensor ** 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = (np.random.rand(*dims) * 10).astype(np.float32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test float scalar ** 2-d float tensor
            dims = (np.random.randint(2, 10), np.random.randint(5, 10))
            x = np.random.rand() * 10
            y = (np.random.rand(*dims) * 10).astype(np.float32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 2-d float tensor ** float scalar
            dims = (np.random.randint(2, 10), np.random.randint(5, 10))
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = np.random.rand() * 10
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test broadcast
            dims = (
                np.random.randint(1, 10),
                np.random.randint(5, 10),
                np.random.randint(5, 10),
            )
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.rand(dims[-1]) * 10).astype(np.float64)
            res = _run_power(DYNAMIC, x, y)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TestPowerError(unittest.TestCase):
    """TestPowerError."""

    def test_errors(self):
        """test_errors."""
        np.random.seed(7)

        # test dynamic computation graph: inputs must be broadcastable
<<<<<<< HEAD
        dims = (
            np.random.randint(1, 10),
            np.random.randint(5, 10),
            np.random.randint(5, 10),
        )
=======
        dims = (np.random.randint(1, 10), np.random.randint(5, 10),
                np.random.randint(5, 10))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.float64)
        self.assertRaises(ValueError, _run_power, DYNAMIC, x, y)
        self.assertRaises(ValueError, _run_power, STATIC, x, y)

        # test dynamic computation graph: inputs must be broadcastable
<<<<<<< HEAD
        dims = (
            np.random.randint(1, 10),
            np.random.randint(5, 10),
            np.random.randint(5, 10),
        )
=======
        dims = (np.random.randint(1, 10), np.random.randint(5, 10),
                np.random.randint(5, 10))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.int8)
        self.assertRaises(TypeError, paddle.pow, x, y)

        # test 1-d float tensor ** int string
<<<<<<< HEAD
        dims = (np.random.randint(200, 300),)
=======
        dims = (np.random.randint(200, 300), )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = int(np.random.rand() * 10)
        self.assertRaises(TypeError, paddle.pow, x, str(y))


if __name__ == '__main__':
    unittest.main()
