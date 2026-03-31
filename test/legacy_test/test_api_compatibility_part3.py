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


# Test mv compatibility
@unittest.skipIf(
    paddle.is_compiled_with_custom_device('iluvatar_gpu'),
    "skip iluvatar_gpu which not register mv kernel",
)
class TestMvAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 4).astype("float32")
        self.np_vec = np.random.rand(4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        vec = paddle.to_tensor(self.np_vec)

        # 1. Paddle Positional arguments
        out1 = paddle.mv(x, vec)
        # 2. Paddle keyword arguments
        out2 = paddle.mv(x=x, vec=vec)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.mv(input=x, vec=vec)
        # 4. Mixed arguments
        out4 = paddle.mv(x, vec=vec)
        # 5-6. out parameter test
        out5 = paddle.zeros([3], dtype="float32")
        out6 = paddle.mv(x, vec, out=out5)
        # 7. Tensor method - args
        out7 = x.mv(vec)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.mv(vec=vec)

        # Verify all outputs
        ref_out = np.dot(self.np_x, self.np_vec)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)
            self.assertEqual(out.shape, (3,))

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )
            vec = paddle.static.data(
                name="vec",
                shape=self.np_vec.shape,
                dtype=str(self.np_vec.dtype),
            )

            # 1. Paddle Positional arguments
            out1 = paddle.mv(x, vec)
            # 2. Paddle keyword arguments
            out2 = paddle.mv(x=x, vec=vec)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.mv(input=x, vec=vec)
            # 4. Tensor method - args
            out4 = x.mv(vec)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.mv(vec=vec)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "vec": self.np_vec},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            ref_out = np.dot(self.np_x, self.np_vec)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


class TestDiagflatAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([1, 2, 3]).astype('int64')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.diagflat(x)

        # 2. Paddle keyword arguments
        out2 = paddle.diagflat(x=x)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.diagflat(input=x)

        # 4. Mixed arguments
        out4 = paddle.diagflat(x, offset=0)

        # 5. Tensor method - args
        out5 = x.diagflat()

        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.diagflat()

        # Verify all outputs
        expected_diag = np.diag(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(out.numpy(), expected_diag)

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
            out1 = paddle.diagflat(x)
            # 2. Paddle keyword arguments
            out2 = paddle.diagflat(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.diagflat(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

        # Verify all outputs
        expected_diag = np.diag(self.np_x)
        for out in fetches:
            np.testing.assert_array_equal(out, expected_diag)


@unittest.skipIf(
    paddle.is_compiled_with_custom_device('iluvatar_gpu'),
    "skip iluvatar_gpu which not register fill_diagonal_tensor kernel",
)
class TestDiagonalScatterAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.arange(6.0).reshape((2, 3))
        self.np_y = np.ones((2,))

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.diagonal_scatter(x, y)

        # 2. Paddle keyword arguments
        out2 = paddle.diagonal_scatter(x=x, y=y)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.diagonal_scatter(input=x, src=y)

        # 4. Mixed arguments with dim1/dim2 aliases
        out4 = paddle.diagonal_scatter(x, y, dim1=0, dim2=1)

        # 5. Tensor method - args
        out5 = x.diagonal_scatter(y)

        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.diagonal_scatter(src=y)

        # Verify all outputs
        expected = np.array([[1.0, 1.0, 2.0], [3.0, 1.0, 5.0]])
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), expected)

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
            out1 = paddle.diagonal_scatter(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.diagonal_scatter(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.diagonal_scatter(input=x, src=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

        # Verify all outputs
        expected = np.array([[1.0, 1.0, 2.0], [3.0, 1.0, 5.0]])
        for out in fetches:
            np.testing.assert_allclose(out, expected)


class TestLdexpAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([1, 2, 3], dtype='float32')
        self.np_y = np.array([2, 3, 4], dtype='int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.ldexp(x, y)

        # 2. Paddle keyword arguments
        out2 = paddle.ldexp(x=x, y=y)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.ldexp(input=x, other=y)

        # 4. Mixed arguments
        out4 = paddle.ldexp(x, y=y)

        # 5-6. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.ldexp(x, y, out=out5)
        assert out5 is out6

        # 7. Tensor method - args
        out7 = x.ldexp(y)

        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.ldexp(other=y)

        # Verify all outputs
        expected = np.array([4.0, 16.0, 48.0], dtype='float32')
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
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
            y = paddle.static.data(
                name="y", shape=self.np_y.shape, dtype=str(self.np_y.dtype)
            )

            # 1. Paddle Positional arguments
            out1 = paddle.ldexp(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.ldexp(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.ldexp(input=x, other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

        # Verify all outputs
        expected = np.array([4.0, 16.0, 48.0], dtype='float32')
        for out in fetches:
            np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test inner compatibility
class TestInnerAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([1.0, 2.0, 3.0, 4.0])
        self.np_y = np.array([5.0, 6.0, 7.0, 8.0])

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.inner(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.inner(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.inner(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.inner(x, other=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(out1)
        out6 = paddle.inner(x, y, out=out5)
        assert out5 is out6
        # 7. Tensor method - args
        out7 = x.inner(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.inner(other=y)

        # Verify all outputs
        expected = np.dot(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

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
            out1 = paddle.inner(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.inner(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.inner(input=x, other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            expected = np.dot(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-6)


# Test positive compatibility
class TestPositiveAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-1, 0, 1], dtype='int64')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.positive(x)
        # 2. Paddle keyword arguments
        out2 = paddle.positive(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.positive(input=x)

        # Verify all outputs
        expected = self.np_x
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

            # 1. Paddle Positional arguments
            out1 = paddle.positive(x)
            # 2. Paddle keyword arguments
            out2 = paddle.positive(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.positive(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            expected = self.np_x
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test rad2deg compatibility
class TestRad2degAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([3.142, -3.142, 6.283, -6.283], dtype='float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.rad2deg(x)
        # 2. Paddle keyword arguments
        out2 = paddle.rad2deg(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.rad2deg(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(x)
        out5 = paddle.rad2deg(x, out=out4)
        # 6. Tensor method
        out6 = x.rad2deg()

        # Verify all outputs
        expected = 180.0 / np.pi * self.np_x
        for out in [out1, out2, out3, out4, out5, out6]:
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

            # 1. Paddle Positional arguments
            out1 = paddle.rad2deg(x)
            # 2. Paddle keyword arguments
            out2 = paddle.rad2deg(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.rad2deg(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            expected = 180.0 / np.pi * self.np_x
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test rot90 compatibility
class TestRot90API(unittest.TestCase):
    def setUp(self):
        self.np_x = np.arange(4, dtype='float32').reshape(2, 2)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.rot90(x, 1, [0, 1])
        # 2. Paddle keyword arguments
        out2 = paddle.rot90(x=x, k=1, axes=[0, 1])
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.rot90(input=x, k=1, dims=[0, 1])
        # 4. Mixed arguments
        out4 = paddle.rot90(x, k=1, dims=[0, 1])
        # 5. Tensor method - args
        out5 = x.rot90(1, [0, 1])
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.rot90(k=1, dims=[0, 1])

        # Verify all outputs
        expected = np.array([[1.0, 3.0], [0.0, 2.0]], dtype='float32')
        for out in [out1, out2, out3, out4, out5, out6]:
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

            # 1. Paddle Positional arguments
            out1 = paddle.rot90(x, 1, [0, 1])
            # 2. Paddle keyword arguments
            out2 = paddle.rot90(x=x, k=1, axes=[0, 1])
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.rot90(input=x, k=1, dims=[0, 1])

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            expected = np.array([[1.0, 3.0], [0.0, 2.0]], dtype='float32')
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test nanquantile compatibility
class TestNanquantileAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 5).astype("float32")
        self.np_x[0, 0] = np.nan

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.nanquantile(x, 0.5, 0)
        # 2. Paddle keyword arguments
        out2 = paddle.nanquantile(x=x, q=0.5, axis=0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.nanquantile(input=x, q=0.5, dim=0)
        # 4. Mixed arguments
        out4 = paddle.nanquantile(x, 0.5, dim=0)
        # 5-6. out parameter test
        out5 = paddle.empty_like(out1)
        out6 = paddle.nanquantile(x, 0.5, axis=0, out=out5)
        assert out5 is out6
        # 7. Tensor method - args
        out7 = x.nanquantile(0.5, 0)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.nanquantile(q=0.5, dim=0)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
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
            out1 = paddle.nanquantile(x, 0.5, 0)
            # 2. Paddle keyword arguments
            out2 = paddle.nanquantile(x=x, q=0.5, axis=0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.nanquantile(input=x, q=0.5, dim=0)
            # 4. Tensor method - args
            out4 = x.nanquantile(0.5, 0)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.nanquantile(q=0.5, dim=0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            for out in fetches:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test neg compatibility
class TestNegAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.4, -0.2, 0.1, 0.3], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.neg(x)
        # 2. Paddle keyword arguments
        out2 = paddle.neg(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.neg(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.neg(x, out=out4)
        assert out4 is out5
        # 6. Tensor method
        out6 = x.neg()

        # Verify all outputs
        expected = -self.np_x
        for out in [out1, out2, out3, out4, out5, out6]:
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

            # 1. Paddle Positional arguments
            out1 = paddle.neg(x)
            # 2. Paddle keyword arguments
            out2 = paddle.neg(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.neg(input=x)
            # 4. Tensor method
            out4 = x.neg()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            expected = -self.np_x
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)


# Test remainder_ inplace compatibility
class TestRemainderInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 20, [5, 6]).astype("int64")
        self.np_y = np.random.randint(1, 10, [5, 6]).astype("int64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.remainder_(x.clone(), y)
        # 2. Paddle keyword arguments
        out2 = paddle.remainder_(x=x.clone(), y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.remainder_(input=x.clone(), other=y)
        # 4. Mixed arguments
        out4 = paddle.remainder_(x.clone(), y=y)
        # 5. Tensor method - args
        out5 = x.clone().remainder_(y)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.clone().remainder_(other=y)

        # Verify all outputs
        ref_out = np.mod(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test remainder_ inplace compatibility
class TestModInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 20, [5, 6]).astype("int64")
        self.np_y = np.random.randint(1, 10, [5, 6]).astype("int64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.mod_(x.clone(), y)
        # 2. Paddle keyword arguments
        out2 = paddle.mod_(x=x.clone(), y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.floor_mod_(input=x.clone(), other=y)
        # 4. Mixed arguments
        out4 = paddle.floor_mod_(x.clone(), y=y)
        # 5. Tensor method - args
        out5 = x.clone().mod_(y)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.clone().floor_mod_(other=y)

        # Verify all outputs
        ref_out = np.mod(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test squeeze compatibility
class TestSqueezeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(1, 3, 1, 5).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments (axis=None)
        out1 = paddle.squeeze(x)
        # 2. Paddle Positional arguments (axis=int)
        out2 = paddle.squeeze(x, 0)
        # 3. Paddle keyword arguments
        out3 = paddle.squeeze(x=x, axis=0)
        # 4. PyTorch keyword arguments (alias)
        out4 = paddle.squeeze(input=x, dim=0)
        # 5. Mixed arguments
        out5 = paddle.squeeze(x, axis=0)
        # 6. Tensor method - args
        out6 = x.squeeze(0)
        # 7. Tensor method - kwargs (PyTorch alias)
        out7 = x.squeeze(dim=0)

        ref_out_none = np.squeeze(self.np_x)
        np.testing.assert_allclose(out1.numpy(), ref_out_none)

        # Verify all outputs
        ref_out = np.squeeze(self.np_x, axis=0)
        for out in [out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), ref_out)

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
            out1 = paddle.squeeze(x, 0)
            # 2. Paddle keyword arguments
            out2 = paddle.squeeze(x=x, axis=0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.squeeze(input=x, dim=0)
            # 4. Tensor method - args
            out4 = x.squeeze(0)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.squeeze(dim=0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            ref_out = np.squeeze(self.np_x, axis=0)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test squeeze_ inplace compatibility
class TestSqueezeInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(1, 3, 1, 5).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.squeeze_(x.clone(), 0)
        # 2. Paddle keyword arguments
        out2 = paddle.squeeze_(x=x.clone(), axis=0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.squeeze_(input=x.clone(), dim=0)
        # 4. Mixed arguments
        out4 = paddle.squeeze_(x.clone(), axis=0)
        # 5. Tensor method - args
        out5 = x.clone().squeeze_(0)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.clone().squeeze_(dim=0)

        # Verify all outputs
        ref_out = np.squeeze(self.np_x, axis=0)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out)


# Test unsqueeze compatibility
class TestUnsqueezeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(5, 10).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.unsqueeze(x, 0)
        # 2. Paddle keyword arguments
        out2 = paddle.unsqueeze(x=x, axis=0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.unsqueeze(input=x, dim=0)
        # 4. Mixed arguments
        out4 = paddle.unsqueeze(x, axis=0)
        # 5. Tensor method - args
        out5 = x.unsqueeze(0)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.unsqueeze(dim=0)

        # Verify all outputs
        ref_out = np.expand_dims(self.np_x, axis=0)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out)

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
            out1 = paddle.unsqueeze(x, 0)
            # 2. Paddle keyword arguments
            out2 = paddle.unsqueeze(x=x, axis=0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.unsqueeze(input=x, dim=0)
            # 4. Tensor method - args
            out4 = x.unsqueeze(0)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.unsqueeze(dim=0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            ref_out = np.expand_dims(self.np_x, axis=0)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test unsqueeze_ inplace compatibility
class TestUnsqueezeInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(5, 10).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.unsqueeze_(x.clone(), 0)
        # 2. Paddle keyword arguments
        out2 = paddle.unsqueeze_(x=x.clone(), axis=0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.unsqueeze_(input=x.clone(), dim=0)
        # 4. Mixed arguments
        out4 = paddle.unsqueeze_(x.clone(), axis=0)
        # 5. Tensor method - args
        out5 = x.clone().unsqueeze_(0)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.clone().unsqueeze_(dim=0)

        # Verify all outputs
        ref_out = np.expand_dims(self.np_x, axis=0)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out)


# Test pow_ inplace compatibility
class TestPowInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(5, 6).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y_scalar = 2.0

        # 1. Paddle Positional arguments
        out1 = paddle.pow_(x.clone(), y_scalar)
        # 2. Paddle keyword arguments
        out2 = paddle.pow_(x=x.clone(), y=y_scalar)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.pow_(input=x.clone(), exponent=y_scalar)
        # 4. Mixed arguments
        out4 = paddle.pow_(x.clone(), y=y_scalar)
        # 5. Tensor method - args
        out5 = x.clone().pow_(y_scalar)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.clone().pow_(exponent=y_scalar)

        # Verify all outputs
        ref_out = np.power(self.np_x, y_scalar)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)


# Test floor_divide_ inplace compatibility
class TestFloorDivideInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(10, 100, [5, 6]).astype("int64")
        self.np_y = np.random.randint(1, 10, [5, 6]).astype("int64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.floor_divide_(x.clone(), y)
        # 2. Paddle keyword arguments
        out2 = paddle.floor_divide_(x=x.clone(), y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.floor_divide_(input=x.clone(), other=y)
        # 4. Mixed arguments
        out4 = paddle.floor_divide_(x.clone(), y=y)
        # 5. Tensor method - args
        out5 = x.clone().floor_divide_(y)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.clone().floor_divide_(other=y)

        # Verify all outputs
        ref_out = np.floor_divide(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(out.numpy(), ref_out)


# Test isposinf compatibility
class TestIsposinfAPICompatibility(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array(
            [[1.0, np.inf, -np.inf], [0.0, -1.0, np.inf]]
        ).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.isposinf(x)
        # 2. Paddle keyword arguments
        out2 = paddle.isposinf(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.isposinf(input=x)
        # 4-5. out parameter test
        out4 = paddle.zeros_like(out1)
        out5 = paddle.isposinf(x, out=out4)
        assert out4 is out5
        # 6. Tensor method
        out6 = x.isposinf()

        # Verify all outputs
        ref_out = np.isposinf(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())

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
            out1 = paddle.isposinf(x)
            # 2. Paddle keyword arguments
            out2 = paddle.isposinf(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.isposinf(input=x)
            # 4. Tensor method - args
            out4 = x.isposinf()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            ref_out = np.isposinf(self.np_x)
            for out in fetches:
                np.testing.assert_array_equal(ref_out, out)


# Test isneginf compatibility
class TestIsneginfAPICompatibility(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array(
            [[1.0, np.inf, -np.inf], [0.0, -1.0, np.inf]]
        ).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.isneginf(x)
        # 2. Paddle keyword arguments
        out2 = paddle.isneginf(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.isneginf(input=x)
        # 4-5. out parameter test
        out4 = paddle.zeros_like(out1)
        out5 = paddle.isneginf(x, out=out4)
        assert out4 is out5
        # 6. Tensor method
        out6 = x.isneginf()

        # Verify all outputs
        ref_out = np.isneginf(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())

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
            out1 = paddle.isneginf(x)
            # 2. Paddle keyword arguments
            out2 = paddle.isneginf(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.isneginf(input=x)
            # 4. Tensor method - args
            out4 = x.isneginf()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs
            ref_out = np.isneginf(self.np_x)
            for out in fetches:
                np.testing.assert_array_equal(ref_out, out)


# Test isreal compatibility
class TestIsRealAPICompatibility(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array(
            [[1.0 + 0j, 2.0 + 3j], [4.0 + 0j, 5.0 - 6j]]
        ).astype("complex64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.isreal(x)
        # 2. Paddle keyword arguments
        out2 = paddle.isreal(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.isreal(input=x)
        # 4. Tensor method - args
        out4 = x.isreal()

        # Verify all outputs
        ref_out = np.isreal(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_array_equal(ref_out, out.numpy())

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
            out1 = paddle.isreal(x)
            # 2. Paddle keyword arguments
            out2 = paddle.isreal(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.isreal(input=x)
            # 4. Tensor method - args
            out4 = x.isreal()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

        # Verify all outputs
        ref_out = np.isreal(self.np_x)
        for out in fetches:
            np.testing.assert_array_equal(ref_out, out)


# Test select_scatter compatibility
class TestSelectScatterAPICompatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3, 4).astype("float32")
        self.np_values = np.random.rand(2, 4).astype("float32")
        self.axis = 1
        self.index = 1

    def get_ref_out(self):
        ref_out = self.np_x.copy()
        ref_out[:, self.index, :] = self.np_values
        return ref_out

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        values = paddle.to_tensor(self.np_values)

        # 1. Paddle Positional arguments
        out1 = paddle.select_scatter(x, values, self.axis, self.index)
        # 2. Paddle keyword arguments
        out2 = paddle.select_scatter(
            x=x, values=values, axis=self.axis, index=self.index
        )
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.select_scatter(
            input=x, src=values, dim=self.axis, index=self.index
        )
        # 4. Mixed arguments
        out4 = paddle.select_scatter(
            x, src=values, dim=self.axis, index=self.index
        )
        # 5. Tensor method - args
        out5 = x.select_scatter(values, self.axis, self.index)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.select_scatter(src=values, dim=self.axis, index=self.index)

        # Verify all outputs
        ref_out = self.get_ref_out()
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

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
            out1 = paddle.select_scatter(x, values, self.axis, self.index)
            # 2. Paddle keyword arguments
            out2 = paddle.select_scatter(
                x=x, values=values, axis=self.axis, index=self.index
            )
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.select_scatter(
                input=x, src=values, dim=self.axis, index=self.index
            )
            # 4. Tensor method - args
            out4 = x.select_scatter(values, self.axis, self.index)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.select_scatter(src=values, dim=self.axis, index=self.index)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "values": self.np_values},
                fetch_list=[out1, out2, out3, out4, out5],
            )

        # Verify all outputs
        ref_out = self.get_ref_out()
        for out in fetches:
            np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test logit compatibility
class TestLogitAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.np_x = np.random.uniform(0.1, 0.9, self.shape).astype(self.dtype)

    def _ref_logit(self, x, eps=0.0):
        if eps > 0.0:
            x = np.clip(x, eps, 1.0 - eps)
        return np.log(x / (1.0 - x))

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.logit(x)
        # 2. Paddle keyword arguments
        out2 = paddle.logit(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.logit(input=x)
        # 4. Mixed arguments (positional x + keyword eps)
        out4 = paddle.logit(x, eps=1e-6)
        # 5. out parameter test
        out5 = paddle.empty_like(x)
        paddle.logit(x, out=out5)
        # 6. out parameter with alias keyword
        out6 = paddle.empty_like(x)
        paddle.logit(input=x, out=out6)
        # 7. Tensor method - args
        out7 = x.logit()
        # 8. Tensor method - kwargs
        out8 = x.logit(eps=1e-6)
        # 9. paddle.special.logit alias
        out9 = paddle.special.logit(x)
        # 10. paddle.special.logit with alias keyword
        out10 = paddle.special.logit(input=x)

        # Verify outputs without eps
        ref_out = self._ref_logit(self.np_x)
        for out in [out1, out2, out3, out5, out6, out7, out9, out10]:
            np.testing.assert_allclose(
                out.numpy(), ref_out, rtol=1e-5, atol=1e-6
            )

        # Verify outputs with eps
        ref_out_eps = self._ref_logit(self.np_x, 1e-6)
        for out in [out4, out8]:
            np.testing.assert_allclose(
                out.numpy(), ref_out_eps, rtol=1e-5, atol=1e-6
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.logit(x)
            # 2. Paddle keyword arguments
            out2 = paddle.logit(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.logit(input=x)
            # 4. Mixed arguments (positional x + keyword eps)
            out4 = paddle.logit(x, eps=1e-6)
            # 5. Tensor method - args
            out5 = x.logit()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

        # Verify outputs without eps
        ref_out = self._ref_logit(self.np_x)
        for out in [fetches[0], fetches[1], fetches[2], fetches[4]]:
            np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-6)

        # Verify output with eps
        ref_out_eps = self._ref_logit(self.np_x, 1e-6)
        np.testing.assert_allclose(
            fetches[3], ref_out_eps, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
