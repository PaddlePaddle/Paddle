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

import sys
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
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.zeros((3, 9)).astype("float32")
        self.np_value = np.ones((3, 2)).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        value = paddle.to_tensor(self.np_value)

        # 1. Paddle Positional arguments (list)
        out1 = paddle.slice_scatter(x, value, [1], [2], [6], [2])
        # 2. Paddle keyword arguments (list)
        out2 = paddle.slice_scatter(x=x, value=value, axes=[1], starts=[2], ends=[6], strides=[2])
        # 3. PyTorch keyword arguments (int - auto convert to list)
        out3 = paddle.slice_scatter(input=x, src=value, dim=1, start=2, end=6, step=2)
        # 4. Mixed arguments
        out4 = paddle.slice_scatter(x, value, axes=[1], starts=[2], ends=[6], strides=[2])
        # 5. Tensor method - args
        out5 = x.slice_scatter(value, [1], [2], [6], [2])

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)
            self.assertEqual(out.shape, (3, 9))

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )
            value = paddle.static.data(
                name="value",
                shape=self.np_value.shape,
                dtype=str(self.np_value.dtype),
            )

            # 1. Paddle Positional arguments
            out1 = paddle.slice_scatter(x, value, [1], [2], [6], [2])
            # 2. Paddle keyword arguments
            out2 = paddle.slice_scatter(x=x, value=value, axes=[1], starts=[2], ends=[6], strides=[2])
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.slice_scatter(input=x, src=value, dim=1, start=2, end=6, step=2)
            # 4. Tensor method - args
            out4 = x.slice_scatter(value, [1], [2], [6], [2])

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "value": self.np_value},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


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

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
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

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
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
        self.np_x = np.array([float('nan'), 0.3, float('+inf'), float('-inf')]).astype("float32")

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
        expected = np.array([1.0, 0.3, np.finfo(np.float32).max, np.finfo(np.float32).min]).astype("float32")
        np.testing.assert_allclose(out5.numpy(), expected, rtol=1e-5)

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
        self.np_x = np.array([[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('nan'), 0.7]]).astype("float32")

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
        self.np_x = np.array([[float('nan'), 0.3, 0.5, 0.9], [0.1, 0.2, float('nan'), 0.7]]).astype("float32")

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
                name="mask", shape=self.np_mask.shape, dtype=str(self.np_mask.dtype)
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


if __name__ == "__main__":
    unittest.main()
