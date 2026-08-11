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


# Test select_scatter compatibility
class TestSelectScatterAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3, 4).astype("float32")
        self.np_values = np.random.rand(2, 4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        values = paddle.to_tensor(self.np_values)

        # 1. Paddle Positional arguments
        out1 = paddle.select_scatter(x, values, 1, 1)
        # 2. Paddle keyword arguments
        out2 = paddle.select_scatter(x=x, values=values, axis=1, index=1)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.select_scatter(input=x, src=values, dim=1, index=1)
        # 4. Mixed arguments
        out4 = paddle.select_scatter(x, values, axis=1, index=1)
        # 5. Tensor method - args
        out5 = x.select_scatter(values, 1, 1)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)
            self.assertEqual(out.shape, (2, 3, 4))

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )
            values = paddle.static.data(
                name="values",
                shape=self.np_values.shape,
                dtype=str(self.np_values.dtype),
            )

            # 1. Paddle Positional arguments
            out1 = paddle.select_scatter(x, values, 1, 1)
            # 2. Paddle keyword arguments
            out2 = paddle.select_scatter(x=x, values=values, axis=1, index=1)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.select_scatter(input=x, src=values, dim=1, index=1)
            # 4. Tensor method - args
            out4 = x.select_scatter(values, 1, 1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "values": self.np_values},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test sgn compatibility
class TestSgnAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([3.0, -2.0, 0.0, -5.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.sgn(x)
        # 2. Paddle keyword arguments
        out2 = paddle.sgn(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.sgn(input=x)
        # 4. Mixed arguments
        out4 = paddle.sgn(x, name=None)
        # 5. Tensor method - args
        out5 = x.sgn()
        # 6. out parameter test
        out6 = paddle.empty_like(out1)
        paddle.sgn(x, out=out6)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.sgn(x)
            # 2. Paddle keyword arguments
            out2 = paddle.sgn(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.sgn(input=x)
            # 4. Tensor method - args
            out4 = x.sgn()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test signbit compatibility
class TestSignbitAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([-0.0, 1.1, -2.1, 0.0, 2.5]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.signbit(x)
        # 2. Paddle keyword arguments
        out2 = paddle.signbit(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.signbit(input=x)
        # 4. Mixed arguments
        out4 = paddle.signbit(x, name=None)
        # 5. Tensor method - args
        out5 = x.signbit()
        # 6. out parameter test
        out6 = paddle.empty_like(out1)
        paddle.signbit(x, out=out6)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.signbit(x)
            # 2. Paddle keyword arguments
            out2 = paddle.signbit(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.signbit(input=x)
            # 4. Tensor method - args
            out4 = x.signbit()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test slice_scatter compatibility
class TestSliceScatterAPI(unittest.TestCase):
    """Test slice_scatter decorator compatibility.

    PyTorch: torch.slice_scatter(input, src, dim=0, start=None, end=None, step=1)
    Paddle: paddle.slice_scatter(x, value, axes, starts, ends, strides)

    The decorator handles:
    1. PyTorch style positional args (dim/start/end/step are int, triggers is_paddle_style=False)
    2. PyTorch keyword aliases (input/src/dim/start/end/step)
    3. end=None handling
    4. Auto-calc ends when not provided
    """

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. PyTorch style positional args (is_paddle_style=False branch)
        # Full positional args: dim=1, start=2, end=6, step=2
        x = paddle.zeros((3, 8))
        value = paddle.ones((3, 2))
        out = paddle.slice_scatter(x, value, 1, 2, 6, 2)
        expected = np.zeros((3, 8))
        expected[:, 2] = 1.0
        expected[:, 4] = 1.0
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        # Only dim, start (step defaults to 1)
        x2 = paddle.zeros((3, 8))
        value2 = paddle.ones((3, 4))
        out2 = paddle.slice_scatter(x2, value2, 1, 2)
        expected2 = np.zeros((3, 8))
        expected2[:, 2:6] = 1.0
        np.testing.assert_allclose(out2.numpy(), expected2, rtol=1e-5)

        # Only dim (start defaults to 0)
        x3 = paddle.zeros((3, 5))
        value3 = paddle.ones((3, 2))
        out3 = paddle.slice_scatter(x3, value3, 1)
        expected3 = np.zeros((3, 5))
        expected3[:, 0:2] = 1.0
        np.testing.assert_allclose(out3.numpy(), expected3, rtol=1e-5)

        # 2. PyTorch keyword aliases
        out4 = paddle.slice_scatter(
            input=x, src=value, dim=1, start=2, end=6, step=2
        )
        np.testing.assert_allclose(out4.numpy(), expected, rtol=1e-5)

        # 3. end=None handling (line 1304 branch)
        out5 = paddle.slice_scatter(
            input=x, src=value, dim=1, start=2, end=None, step=2
        )
        np.testing.assert_allclose(out5.numpy(), expected, rtol=1e-5)

        # Not passing end (line 1359-1368 auto-calc ends)
        out6 = paddle.slice_scatter(input=x, src=value, dim=1, start=2, step=2)
        np.testing.assert_allclose(out6.numpy(), expected, rtol=1e-5)

        # 4. Paddle style positional args (is_paddle_style=True branch)
        out7 = paddle.slice_scatter(x, value, [1], [2], [6], [2])
        np.testing.assert_allclose(out7.numpy(), expected, rtol=1e-5)

        # 5. Paddle keyword args
        out8 = paddle.slice_scatter(
            x=x, value=value, axes=[1], starts=[2], ends=[6], strides=[2]
        )
        np.testing.assert_allclose(out8.numpy(), expected, rtol=1e-5)

        # 6. Tensor method
        out9 = x.slice_scatter(value, dim=1, start=2, end=6, step=2)
        np.testing.assert_allclose(out9.numpy(), expected, rtol=1e-5)

        # 7. Multi-axis with auto-calc ends
        x_multi = paddle.zeros((3, 3, 5))
        value_multi = paddle.ones((2, 3, 3))
        out_multi = paddle.slice_scatter(
            x_multi, value_multi, axes=[0, 2], starts=[1, 0], strides=[1, 2]
        )
        expected_multi = np.zeros((3, 3, 5))
        expected_multi[1:3, :, 0:5:2] = (
            1.0  # axes=[0,2], starts=[1,0], auto ends
        )
        np.testing.assert_allclose(out_multi.numpy(), expected_multi, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()

        np_x = np.zeros((3, 8)).astype("float32")
        np_value = np.ones((3, 2)).astype("float32")

        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=(3, 8), dtype="float32")
            value = paddle.static.data(
                name="value", shape=(3, 2), dtype="float32"
            )

            # Paddle style positional args
            out1 = paddle.slice_scatter(x, value, [1], [2], [6], [2])
            # PyTorch keyword args
            out2 = paddle.slice_scatter(
                input=x, src=value, dim=1, start=2, end=6, step=2
            )
            # Tensor method
            out3 = x.slice_scatter(value, [1], [2], [6], [2])

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": np_x, "value": np_value},
                fetch_list=[out1, out2, out3],
            )

            expected = np.zeros((3, 8))
            expected[:, 2] = 1.0
            expected[:, 4] = 1.0

            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test tensordot compatibility
class TestTensordotAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3).astype("float64")
        self.np_y = np.random.rand(3, 4).astype("float64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.tensordot(x, y, axes=1)
        # 2. Paddle keyword arguments
        out2 = paddle.tensordot(x=x, y=y, axes=1)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.tensordot(a=x, b=y, dims=1)
        # 4. Mixed arguments
        out4 = paddle.tensordot(x, y, axes=1)
        # 5. out parameter test
        out5 = paddle.empty((2, 4), dtype='float64')
        paddle.tensordot(x, y, axes=1, out=out5)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )
            y = paddle.static.data(
                name="y", shape=self.np_y.shape, dtype=str(self.np_y.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.tensordot(x, y, axes=1)
            # 2. Paddle keyword arguments
            out2 = paddle.tensordot(x=x, y=y, axes=1)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.tensordot(a=x, b=y, dims=1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test tril_indices compatibility
class TestTrilIndicesAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle Positional arguments
        out1 = paddle.tril_indices(4, 4, 0)
        # 2. Paddle keyword arguments
        out2 = paddle.tril_indices(row=4, col=4, offset=0)
        # 3. PyTorch keyword arguments (device)
        out3 = paddle.tril_indices(4, 4, 0, device="cpu")
        # 4. Mixed arguments
        out4 = paddle.tril_indices(4, 4, offset=0, device="cpu")

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            # 1. Paddle Positional arguments
            out1 = paddle.tril_indices(4, 4, 0)
            # 2. Paddle keyword arguments
            out2 = paddle.tril_indices(row=4, col=4, offset=0)
            # 3. PyTorch keyword arguments (device)
            out3 = paddle.tril_indices(4, 4, 0, device="cpu")

            exe = paddle.static.Executor()
            fetches = exe.run(main, feed={}, fetch_list=[out1, out2, out3])

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test triu_indices compatibility
class TestTriuIndicesAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle Positional arguments
        out1 = paddle.triu_indices(4, 4, 0)
        # 2. Paddle keyword arguments
        out2 = paddle.triu_indices(row=4, col=4, offset=0)
        # 3. PyTorch keyword arguments (device)
        out3 = paddle.triu_indices(4, 4, 0, device="cpu")
        # 4. Mixed arguments
        out4 = paddle.triu_indices(4, 4, offset=0, device="cpu")

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            # 1. Paddle Positional arguments
            out1 = paddle.triu_indices(4, 4, 0)
            # 2. Paddle keyword arguments
            out2 = paddle.triu_indices(row=4, col=4, offset=0)
            # 3. PyTorch keyword arguments (device)
            out3 = paddle.triu_indices(4, 4, 0, device="cpu")

            exe = paddle.static.Executor()
            fetches = exe.run(main, feed={}, fetch_list=[out1, out2, out3])

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test vander compatibility
class TestVanderAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.0, 2.0, 3.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        if paddle.is_compiled_with_xpu():
            self.skipTest("vander is not supported on XPU")

        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.vander(x, 3)
        # 2. Paddle keyword arguments
        out2 = paddle.vander(x=x, n=3)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.vander(x, N=3)
        # 4. Mixed arguments
        out4 = paddle.vander(x, n=3, increasing=False)

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        if paddle.is_compiled_with_xpu():
            self.skipTest("vander is not supported on XPU")

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.vander(x, 3)
            # 2. Paddle keyword arguments
            out2 = paddle.vander(x=x, n=3)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.vander(x, N=3)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test logaddexp compatibility
class TestLogaddexpAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([-1.0, -2.0, -3.0]).astype("float64")
        self.np_y = np.array([-1.0]).astype("float64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.logaddexp(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.logaddexp(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.logaddexp(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.logaddexp(x, y=y)
        # 5. out parameter test
        out5 = paddle.empty_like(out1)
        paddle.logaddexp(x, y, out=out5)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )
            y = paddle.static.data(
                name="y", shape=self.np_y.shape, dtype=str(self.np_y.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.logaddexp(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.logaddexp(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.logaddexp(input=x, other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test logspace compatibility
class TestLogspaceAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle Positional arguments
        out1 = paddle.logspace(0, 10, 5, 2)
        # 2. Paddle keyword arguments
        out2 = paddle.logspace(start=0, stop=10, num=5, base=2)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.logspace(0, end=10, steps=5, base=2)
        # 4. Mixed arguments
        out4 = paddle.logspace(0, 10, num=5, base=2)
        # 5. requires_grad parameter test
        out5 = paddle.logspace(0, 10, 5, 2, requires_grad=True)
        self.assertTrue(out1.stop_gradient)
        self.assertFalse(out5.stop_gradient)

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_device_param(self):
        """Test device parameter separately"""
        paddle.disable_static()
        # device parameter test
        out = paddle.logspace(0, 10, 5, base=2, device="cpu")
        self.assertEqual(str(out.place), "Place(cpu)")
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            # 1. Paddle Positional arguments
            out1 = paddle.logspace(0, 10, 5, 2)
            # 2. Paddle keyword arguments
            out2 = paddle.logspace(start=0, stop=10, num=5, base=2)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.logspace(0, end=10, steps=5, base=2)

            exe = paddle.static.Executor()
            fetches = exe.run(main, feed={}, fetch_list=[out1, out2, out3])

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test moveaxis compatibility
class TestMoveaxisAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 2, 4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.moveaxis(x, 0, 1)
        # 2. Paddle keyword arguments
        out2 = paddle.moveaxis(x=x, source=0, destination=1)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.moveaxis(input=x, source=0, destination=1)
        # 4. Mixed arguments
        out4 = paddle.moveaxis(x, source=0, destination=1)
        # 5. Tensor method - args
        out5 = x.moveaxis(0, 1)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.moveaxis(x, 0, 1)
            # 2. Paddle keyword arguments
            out2 = paddle.moveaxis(x=x, source=0, destination=1)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.moveaxis(input=x, source=0, destination=1)
            # 4. Tensor method - args
            out4 = x.moveaxis(0, 1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test nan_to_num compatibility
class TestNanToNumAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [float('nan'), 0.3, float('+inf'), float('-inf')]
        ).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.nan_to_num(x)
        # 2. Paddle keyword arguments
        out2 = paddle.nan_to_num(x=x, nan=0.0, posinf=None, neginf=None)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.nan_to_num(input=x, nan=0.0)
        # 4. Tensor method - args
        out4 = x.nan_to_num()

        # Verify all outputs (default nan=0, posinf/neginf use large values)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        # 5. Test with custom nan value separately
        out5 = paddle.nan_to_num(x, nan=1.0)
        expected = np.array(
            [1.0, 0.3, np.finfo(np.float32).max, np.finfo(np.float32).min]
        ).astype("float32")
        np.testing.assert_allclose(out5.numpy(), expected, rtol=1e-5)

        # 6. out parameter test
        out6 = paddle.empty_like(out1)
        paddle.nan_to_num(x, out=out6)

        # Verify all outputs (default nan=0, posinf/neginf use large values)
        for out in [out1, out2, out3, out4, out6]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.nan_to_num(x)
            # 2. Paddle keyword arguments
            out2 = paddle.nan_to_num(x=x, nan=0.0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.nan_to_num(input=x, nan=0.0)
            # 4. Tensor method - args
            out4 = x.nan_to_num()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test nanmean compatibility
class TestNanmeanAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('nan'), 0.7]]
        ).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments (no axis - compute mean of all elements)
        out1 = paddle.nanmean(x)
        # 2. Paddle keyword arguments (no axis)
        out2 = paddle.nanmean(x=x)
        # 3. PyTorch keyword arguments (alias, no axis)
        out3 = paddle.nanmean(input=x)
        # 4. Tensor method - args (no axis)
        out4 = x.nanmean()

        # Verify all outputs (all compute global mean, ignoring nan)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        # 5. Test with axis separately
        out5 = paddle.nanmean(x, axis=0)
        out6 = paddle.nanmean(input=x, dim=0)
        np.testing.assert_allclose(out5.numpy(), out6.numpy(), rtol=1e-5)

        # 6. out parameter test
        out7 = paddle.empty_like(out1)
        paddle.nanmean(x, out=out7)

        # 7. dtype parameter test
        out8 = paddle.nanmean(x, dtype='float64')
        self.assertEqual(out8.dtype, paddle.float64)

        # Verify all outputs (all compute global mean, ignoring nan)
        for out in [out1, out2, out3, out4, out7]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments (no axis)
            out1 = paddle.nanmean(x)
            # 2. Paddle keyword arguments (no axis)
            out2 = paddle.nanmean(x=x)
            # 3. PyTorch keyword arguments (alias, no axis)
            out3 = paddle.nanmean(input=x)
            # 4. Tensor method - args (no axis)
            out4 = x.nanmean()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test nansum compatibility
class TestNansumAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('nan'), 0.7]]
        ).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments (no axis - compute sum of all elements)
        out1 = paddle.nansum(x)
        # 2. Paddle keyword arguments (no axis)
        out2 = paddle.nansum(x=x)
        # 3. PyTorch keyword arguments (alias, no axis)
        out3 = paddle.nansum(input=x)
        # 4. Tensor method - args (no axis)
        out4 = x.nansum()

        # Verify all outputs (all compute global sum, ignoring nan)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        # 5. Test with axis separately
        out5 = paddle.nansum(x, axis=0)
        out6 = paddle.nansum(input=x, dim=0)
        np.testing.assert_allclose(out5.numpy(), out6.numpy(), rtol=1e-5)

        # 6. out parameter test
        out7 = paddle.empty_like(out1)
        paddle.nansum(x, out=out7)

        # 7. dtype parameter test
        out8 = paddle.nansum(x, dtype='float64')
        self.assertEqual(out8.dtype, paddle.float64)

        # Verify all outputs (all compute global sum, ignoring nan)
        for out in [out1, out2, out3, out4, out7]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            # 1. Paddle Positional arguments (no axis)
            out1 = paddle.nansum(x)
            # 2. Paddle keyword arguments (no axis)
            out2 = paddle.nansum(x=x)
            # 3. PyTorch keyword arguments (alias, no axis)
            out3 = paddle.nansum(input=x)
            # 4. Tensor method - args (no axis)
            out4 = x.nansum()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test masked_fill compatibility
class TestMaskedFillAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.ones((3, 3)).astype("float32")
        self.np_mask = np.array([[True, True, False]]).astype("bool")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        mask = paddle.to_tensor(self.np_mask)

        # 1. Paddle Positional arguments
        out1 = paddle.masked_fill(x, mask, 2.0)
        # 2. Paddle keyword arguments
        out2 = paddle.masked_fill(x=x, mask=mask, value=2.0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.masked_fill(input=x, mask=mask, value=2.0)
        # 4. Mixed arguments
        out4 = paddle.masked_fill(x, mask, value=2.0)
        # 5. Tensor method - args
        out5 = x.masked_fill(mask, 2.0)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )
            mask = paddle.static.data(
                name="mask",
                shape=self.np_mask.shape,
                dtype=str(self.np_mask.dtype),
            )

            # 1. Paddle Positional arguments
            out1 = paddle.masked_fill(x, mask, 2.0)
            # 2. Paddle keyword arguments
            out2 = paddle.masked_fill(x=x, mask=mask, value=2.0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.masked_fill(input=x, mask=mask, value=2.0)
            # 4. Tensor method - args
            out4 = x.masked_fill(mask, 2.0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "mask": self.np_mask},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test addmv compatibility
class TestAddmvAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3).astype("float32")
        self.np_mat = np.random.rand(3, 4).astype("float32")
        self.np_vec = np.random.rand(4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        mat = paddle.to_tensor(self.np_mat)
        vec = paddle.to_tensor(self.np_vec)

        # 1. Paddle Positional arguments
        out1 = paddle.addmv(input, mat, vec)
        # 2. Paddle keyword arguments
        out2 = paddle.addmv(input=input, mat=mat, vec=vec)
        # 3. With beta and alpha
        out3 = paddle.addmv(input, mat, vec, beta=0.5, alpha=2.0)
        # 4. Tensor method
        out4 = input.addmv(mat, vec)
        # 5. Tensor method with kwargs
        out5 = input.addmv(mat=mat, vec=vec, beta=0.5, alpha=2.0)
        # 6. out parameter test
        out6 = paddle.empty_like(out1)
        paddle.addmv(input, mat, vec, out=out6)

        # Verify outputs
        expected = 1.0 * self.np_input + 1.0 * np.dot(self.np_mat, self.np_vec)
        for out in [out1, out2, out4, out6]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input",
                shape=self.np_input.shape,
                dtype=str(self.np_input.dtype),
            )
            mat = paddle.static.data(
                name="mat",
                shape=self.np_mat.shape,
                dtype=str(self.np_mat.dtype),
            )
            vec = paddle.static.data(
                name="vec",
                shape=self.np_vec.shape,
                dtype=str(self.np_vec.dtype),
            )

            out1 = paddle.addmv(input, mat, vec)
            out2 = paddle.addmv(input=input, mat=mat, vec=vec)
            out3 = input.addmv(mat, vec)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "input": self.np_input,
                    "mat": self.np_mat,
                    "vec": self.np_vec,
                },
                fetch_list=[out1, out2, out3],
            )

            expected = 1.0 * self.np_input + 1.0 * np.dot(
                self.np_mat, self.np_vec
            )
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test addmv_ compatibility (inplace)
class TestAddmv_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3).astype("float32")
        self.np_mat = np.random.rand(3, 4).astype("float32")
        self.np_vec = np.random.rand(4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input.copy())
        mat = paddle.to_tensor(self.np_mat)
        vec = paddle.to_tensor(self.np_vec)

        # Inplace operation
        input.addmv_(mat, vec)

        expected = 1.0 * self.np_input + 1.0 * np.dot(self.np_mat, self.np_vec)
        np.testing.assert_allclose(input.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


# Test addr compatibility
class TestAddrAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3, 4).astype("float32")
        self.np_vec1 = np.random.rand(3).astype("float32")
        self.np_vec2 = np.random.rand(4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        vec1 = paddle.to_tensor(self.np_vec1)
        vec2 = paddle.to_tensor(self.np_vec2)

        # 1. Paddle Positional arguments
        out1 = paddle.addr(input, vec1, vec2)
        # 2. Paddle keyword arguments
        out2 = paddle.addr(input=input, vec1=vec1, vec2=vec2)
        # 3. With beta and alpha
        out3 = paddle.addr(input, vec1, vec2, beta=0.5, alpha=2.0)
        # 4. Tensor method
        out4 = input.addr(vec1, vec2)
        # 5. out parameter test
        out5 = paddle.empty_like(out1)
        paddle.addr(input, vec1, vec2, out=out5)

        # Verify outputs
        expected = 1.0 * self.np_input + 1.0 * np.outer(
            self.np_vec1, self.np_vec2
        )
        for out in [out1, out2, out4, out5]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input",
                shape=self.np_input.shape,
                dtype=str(self.np_input.dtype),
            )
            vec1 = paddle.static.data(
                name="vec1",
                shape=self.np_vec1.shape,
                dtype=str(self.np_vec1.dtype),
            )
            vec2 = paddle.static.data(
                name="vec2",
                shape=self.np_vec2.shape,
                dtype=str(self.np_vec2.dtype),
            )

            out1 = paddle.addr(input, vec1, vec2)
            out2 = input.addr(vec1, vec2)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "input": self.np_input,
                    "vec1": self.np_vec1,
                    "vec2": self.np_vec2,
                },
                fetch_list=[out1, out2],
            )

            expected = 1.0 * self.np_input + 1.0 * np.outer(
                self.np_vec1, self.np_vec2
            )
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test addr_ compatibility (inplace)
class TestAddr_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3, 4).astype("float32")
        self.np_vec1 = np.random.rand(3).astype("float32")
        self.np_vec2 = np.random.rand(4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input.copy())
        vec1 = paddle.to_tensor(self.np_vec1)
        vec2 = paddle.to_tensor(self.np_vec2)

        # Inplace operation
        input.addr_(vec1, vec2)

        expected = 1.0 * self.np_input + 1.0 * np.outer(
            self.np_vec1, self.np_vec2
        )
        np.testing.assert_allclose(input.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


# Test trunc compatibility (with out parameter)
class TestTruncAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.5, -2.7, 0.3, -0.8]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.trunc(x)
        # 2. Paddle keyword arguments
        out2 = paddle.trunc(input=x)
        # 3. out parameter
        out3 = paddle.empty_like(x)
        paddle.trunc(x, out=out3)

        # Verify outputs
        expected = np.trunc(self.np_x)
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            out1 = paddle.trunc(x)
            out2 = paddle.trunc(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2],
            )

            expected = np.trunc(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test fix compatibility (alias for trunc)
class TestFixAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.5, -2.7, 0.3, -0.8]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.fix(x)
        # 2. Paddle keyword arguments
        out2 = paddle.fix(input=x)
        # 3. out parameter
        out3 = paddle.empty_like(x)
        paddle.fix(x, out=out3)
        # 4. Tensor method
        out4 = x.fix()

        # Verify outputs (fix is alias for trunc)
        expected = np.trunc(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


# Test fix_ compatibility (inplace alias for trunc_)
class TestFix_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.5, -2.7, 0.3, -0.8]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x.copy())

        # Inplace operation
        x.fix_()

        expected = np.trunc(self.np_x)
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


class RandomDataset(paddle.utils.data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 10 - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class TestDataLoaderAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(255)
        self.batch_num = 4
        self.batch_size = 8
        self.dataset = RandomDataset(self.batch_num * self.batch_size)
        self.batch_sampler = paddle.utils.data.BatchSampler(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def iter_loader_data(self, loader):
        for _ in range(3):
            for image, label in loader():
                relu = paddle.nn.functional.relu(image)
                self.assertEqual(image.shape, [self.batch_size, 784])
                self.assertEqual(label.shape, [self.batch_size, 1])
                self.assertEqual(relu.shape, [self.batch_size, 784])

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        # case 1
        loader = paddle.utils.data.DataLoader(
            self.dataset,
            self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        self.iter_loader_data(loader)
        # case 2
        loader = paddle.utils.data.dataloader.DataLoader(
            dataset=self.dataset,
            batch_sampler=self.batch_sampler,
        )
        self.iter_loader_data(loader)
        # case 3
        loader = paddle.utils.data.DataLoader(
            dataset=self.dataset,
            sampler=self.batch_sampler,
        )
        self.iter_loader_data(loader)
        paddle.enable_static()

    def test_error(self):
        paddle.disable_static()
        with self.assertRaises(ValueError):
            loader = paddle.utils.data.dataloader.DataLoader(
                dataset=self.dataset,
                sampler=self.batch_sampler,
                batch_sampler=self.batch_sampler,
            )
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
