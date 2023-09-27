# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


def _run_vdot(mode, x, y, device="cpu"):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        paddle.set_device(device)
        x_ = paddle.to_tensor(x)
        y_ = paddle.to_tensor(y)
        res = paddle.vdot(x_, y_)
        return res.numpy()
    # static graph mode
    elif mode == STATIC:
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
            y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
            res = paddle.vdot(x_, y_)
            place = (
                paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
            )
            exe = paddle.static.Executor(place)
            outs = exe.run(feed={"x": x, "y": y}, fetch_list=[res])
            return outs[0]


class TestVdotAPI(unittest.TestCase):
    """TestVdotAPI."""

    def setUp(self):
        self.places = ["cpu"]
        if core.is_compiled_with_cuda():
            self.places.append("gpu")

    def test_vdot(self):
        """test_vdot."""
        np.random.seed(8)
        dim = np.random.randint(200, 300)

        for place in self.places:
            # test 1-d float32 * float32
            x = np.random.rand(dim).astype(np.float32)
            y = np.random.rand(dim).astype(np.float32)
            res = _run_vdot(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)
            res = _run_vdot(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)

            # test 1-d float64 * float64
            x = np.random.rand(dim).astype(np.float64)
            y = np.random.rand(dim).astype(np.float64)
            res = _run_vdot(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)
            res = _run_vdot(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)

            # test 1-d complex64 * complex64
            x = np.random.rand(dim).astype(np.float32) + 1.0j * np.random.rand(
                dim
            ).astype(np.float32)
            y = np.random.rand(dim).astype(np.float32) + 1.0j * np.random.rand(
                dim
            ).astype(np.float32)
            res = _run_vdot(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)
            res = _run_vdot(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)

            # test 1-d complex128 * complex128
            x = np.random.rand(dim).astype(np.float64) + 1.0j * np.random.rand(
                dim
            ).astype(np.float64)
            y = np.random.rand(dim).astype(np.float64) + 1.0j * np.random.rand(
                dim
            ).astype(np.float64)
            res = _run_vdot(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)
            res = _run_vdot(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.vdot(x, y), rtol=1e-05)


class TestVdotError(unittest.TestCase):
    """TestVdotError."""

    def test_errors(self):
        """test_errors."""
        np.random.seed(8)
        dim = np.random.randint(2, 10)

        # test input data type
        x = np.random.randint(10, size=dim)
        y = np.random.rand(dim).astype(np.float32)
        self.assertRaises(Exception, _run_vdot, DYNAMIC, x, y)
        self.assertRaises(Exception, _run_vdot, STATIC, x, y)

        x = np.random.rand(dim).astype(np.float32)
        y = np.random.randint(10, size=dim)
        self.assertRaises(Exception, _run_vdot, DYNAMIC, x, y)
        self.assertRaises(Exception, _run_vdot, STATIC, x, y)

        # test input data dimension
        x = np.random.rand(dim, dim).astype(np.float32)
        y = np.random.rand(dim).astype(np.float32)
        self.assertRaises(Exception, _run_vdot, DYNAMIC, x, y)
        self.assertRaises(Exception, _run_vdot, STATIC, x, y)

        x = np.random.rand(dim).astype(np.float32)
        y = np.random.rand(dim, dim).astype(np.float32)
        self.assertRaises(Exception, _run_vdot, DYNAMIC, x, y)
        self.assertRaises(Exception, _run_vdot, STATIC, x, y)

        x = np.random.rand(dim).astype(np.float32)
        y = np.random.rand(dim + 1).astype(np.float32)
        self.assertRaises(Exception, _run_vdot, DYNAMIC, x, y)
        self.assertRaises(Exception, _run_vdot, STATIC, x, y)


if __name__ == "__main__":
    unittest.main()
