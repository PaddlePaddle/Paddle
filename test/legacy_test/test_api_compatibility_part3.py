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


# Test randint compatibility
class TestRandintAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        # basic shape
        x = paddle.randint(high=10, shape=[2, 3])
        self.assertEqual(x.shape, [2, 3])
        self.assertTrue(x.stop_gradient)
        # 'size' is an alias for 'shape'
        x = paddle.randint(high=10, size=[3, 4])
        self.assertEqual(x.shape, [3, 4])
        # requires_grad
        x = paddle.randint(high=10, shape=[2, 3], requires_grad=True)
        self.assertFalse(x.stop_gradient)
        x = paddle.randint(high=10, shape=[2, 3], requires_grad=False)
        self.assertTrue(x.stop_gradient)
        # value range
        x = paddle.randint(low=5, high=10, shape=[100])
        arr = x.numpy()
        self.assertTrue(np.all(arr >= 5) and np.all(arr < 10))
        # torch.randint(high, size) style: second positional arg as shape
        x = paddle.randint(10, [3, 4])
        self.assertEqual(x.shape, [3, 4])
        self.assertTrue(np.all(x.numpy() >= 0) and np.all(x.numpy() < 10))
        # dtype string
        x = paddle.randint(high=10, shape=[3], dtype='int32')
        self.assertEqual(x.dtype, paddle.int32)
        # out param
        out = paddle.zeros([2, 3], dtype='int64')
        result = paddle.randint(high=10, shape=[2, 3], out=out)
        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_array_equal(result.numpy(), out.numpy())
        # out with requires_grad
        out = paddle.zeros([2, 3], dtype='int64')
        result = paddle.randint(
            high=10, shape=[2, 3], out=out, requires_grad=True
        )
        self.assertFalse(result.stop_gradient)
        paddle.enable_static()

    def test_static_Compatibility(self):
        # basic shape and stop_gradient
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.randint(high=10, shape=[2, 3])
            self.assertEqual(x.shape, [2, 3])
            self.assertTrue(x.stop_gradient)
            # requires_grad
            x_grad = paddle.randint(high=10, shape=[2, 3], requires_grad=True)
            self.assertFalse(x_grad.stop_gradient)
            x_no_grad = paddle.randint(
                high=10, shape=[2, 3], requires_grad=False
            )
            self.assertTrue(x_no_grad.stop_gradient)
            # size alias
            x_size = paddle.randint(high=10, size=[2, 3])
            self.assertEqual(x_size.shape, [2, 3])
            # dtype string
            x_dtype = paddle.randint(high=10, shape=[3], dtype='int32')
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(startup)
            result = exe.run(main, fetch_list=[x_dtype])
            self.assertEqual(result[0].dtype, np.int32)


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


class TestLayerAndTensorToAPI(unittest.TestCase):
    """Test paddle.nn.Layer.to and paddle.Tensor.to alignment with PyTorch."""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def _make_model(self):
        """Create a model with float params and an int buffer."""

        class Model(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.linear = paddle.nn.Linear(3, 2)
                self.register_buffer(
                    'int_buf', paddle.to_tensor([1, 2, 3], dtype='int32')
                )

            def forward(self, x):
                return self.linear(x)

        return Model()

    # ---- Layer.to: Positional dtype ----

    def test_layer_positional_paddle_dtype(self):
        """Layer.to(paddle.float64)"""
        linear = paddle.nn.Linear(2, 2)
        ret = linear.to(paddle.float64)
        self.assertEqual(linear.weight.dtype, paddle.float64)
        self.assertEqual(linear.bias.dtype, paddle.float64)
        self.assertIs(ret, linear)

    def test_layer_positional_dtype_string(self):
        """Layer.to('float64')"""
        linear = paddle.nn.Linear(2, 2)
        linear.to('float64')
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_positional_dtype_float16(self):
        """Layer.to(paddle.float16)"""
        linear = paddle.nn.Linear(2, 2)
        linear.to(paddle.float16)
        self.assertEqual(linear.weight.dtype, paddle.float16)

    # ---- Layer.to: Positional tensor ----

    def test_layer_positional_tensor(self):
        """Layer.to(tensor) -- match tensor's dtype and device"""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(ref)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    # ---- Layer.to: Positional device ----

    def test_layer_positional_device_string(self):
        """Layer.to('cpu')"""
        linear = paddle.nn.Linear(2, 2)
        linear.to('cpu')
        self.assertTrue(linear.weight.place.is_cpu_place())

    def test_layer_positional_device_and_dtype(self):
        """Layer.to('cpu', 'float64')"""
        linear = paddle.nn.Linear(2, 2)
        linear.to('cpu', 'float64')
        self.assertTrue(linear.weight.place.is_cpu_place())
        self.assertEqual(linear.weight.dtype, paddle.float64)

    # ---- Layer.to: Keyword args ----

    def test_layer_keyword_device(self):
        """Layer.to(device='cpu')"""
        linear = paddle.nn.Linear(2, 2)
        linear.to(device='cpu')
        self.assertTrue(linear.weight.place.is_cpu_place())

    def test_layer_keyword_dtype(self):
        """Layer.to(dtype='float64')"""
        linear = paddle.nn.Linear(2, 2)
        linear.to(dtype='float64')
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_keyword_device_and_dtype(self):
        """Layer.to(device='cpu', dtype='float64')"""
        linear = paddle.nn.Linear(2, 2)
        linear.to(device='cpu', dtype='float64')
        self.assertTrue(linear.weight.place.is_cpu_place())
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_keyword_non_blocking(self):
        """Layer.to(dtype='float64', non_blocking=False)"""
        linear = paddle.nn.Linear(2, 2)
        linear.to(dtype='float64', non_blocking=False)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_keyword_blocking(self):
        """Layer.to(device='cpu', blocking=True)"""
        linear = paddle.nn.Linear(2, 2)
        linear.to(device='cpu', blocking=True)
        self.assertTrue(linear.weight.place.is_cpu_place())

    # ---- Layer.to: No args ----

    def test_layer_no_args(self):
        """Layer.to() -- returns self unchanged"""
        linear = paddle.nn.Linear(2, 2)
        original_dtype = linear.weight.dtype
        ret = linear.to()
        self.assertIs(ret, linear)
        self.assertEqual(linear.weight.dtype, original_dtype)

    # ---- Layer.to: all-dtype casting ----

    def test_layer_cast_all_with_positional_dtype(self):
        """Layer.to(dtype) casts ALL params and buffers, including int buf."""
        model = self._make_model()
        self.assertEqual(model.int_buf.dtype, paddle.int32)
        model.to(paddle.float64)
        self.assertEqual(model.linear.weight.dtype, paddle.float64)
        self.assertEqual(model.int_buf.dtype, paddle.float64)

    def test_layer_cast_all_with_keyword_dtype(self):
        """Layer.to(dtype='float64') casts ALL params and buffers."""
        model = self._make_model()
        model.to(dtype='float64')
        self.assertEqual(model.linear.weight.dtype, paddle.float64)
        self.assertEqual(model.int_buf.dtype, paddle.float64)

    def test_layer_cast_all_with_tensor(self):
        """Layer.to(tensor) casts ALL params and buffers."""
        model = self._make_model()
        ref = paddle.to_tensor([1.0], dtype='float64')
        model.to(ref)
        self.assertEqual(model.linear.weight.dtype, paddle.float64)
        self.assertEqual(model.int_buf.dtype, paddle.float64)

    # ---- Layer.to: sublayers and chaining ----

    def test_layer_sublayers_cast(self):
        """Layer.to() should recurse into sublayers."""
        model = paddle.nn.Sequential(
            paddle.nn.Linear(3, 4), paddle.nn.Linear(4, 2)
        )
        model.to(paddle.float64)
        for sub in model.sublayers():
            if hasattr(sub, 'weight'):
                self.assertEqual(sub.weight.dtype, paddle.float64)

    def test_layer_returns_self(self):
        """Layer.to() should return self for chaining."""
        linear = paddle.nn.Linear(2, 2)
        self.assertIs(linear.to(paddle.float64), linear)

    def test_layer_sequential_to_calls(self):
        """Multiple Layer.to() calls should work correctly."""
        linear = paddle.nn.Linear(2, 2)
        linear.to(paddle.float64)
        self.assertEqual(linear.weight.dtype, paddle.float64)
        linear.to('float32')
        self.assertEqual(linear.weight.dtype, paddle.float32)

    # ---- Tensor.to ----

    def test_tensor_positional_dtype(self):
        """Tensor.to(paddle.float64)"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to(paddle.float64)
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_positional_dtype_string(self):
        """Tensor.to('float64')"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_positional_device(self):
        """Tensor.to('cpu')"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('cpu')
        self.assertTrue(out.place.is_cpu_place())

    def test_tensor_positional_device_and_dtype(self):
        """Tensor.to('cpu', 'float64')"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('cpu', 'float64')
        self.assertTrue(out.place.is_cpu_place())
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_positional_other(self):
        """Tensor.to(other_tensor)"""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(ref)
        self.assertEqual(out.dtype, paddle.int32)

    def test_tensor_keyword_dtype(self):
        """Tensor.to(dtype='float64')"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to(dtype='float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_keyword_device(self):
        """Tensor.to(device='cpu')"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to(device='cpu')
        self.assertTrue(out.place.is_cpu_place())

    def test_tensor_keyword_non_blocking(self):
        """Tensor.to(dtype='float64', non_blocking=False)"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to(dtype='float64', non_blocking=False)
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_no_args(self):
        """Tensor.to() -- returns self"""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to()
        self.assertEqual(out.dtype, t.dtype)

    # ---- blocking / non_blocking conflict ----

    def test_blocking_non_blocking_conflict_raises(self):
        """Setting both blocking and non_blocking raises TypeError."""
        linear = paddle.nn.Linear(2, 2)
        with self.assertRaises(TypeError):
            linear.to(dtype='float64', blocking=True, non_blocking=False)

    def test_tensor_blocking_non_blocking_conflict_raises(self):
        """Tensor: setting both blocking and non_blocking raises TypeError."""
        t = paddle.to_tensor([1.0])
        with self.assertRaises(TypeError):
            t.to(dtype='float64', blocking=True, non_blocking=False)

    # ---- Error handling ----

    def test_too_many_args(self):
        """to() with too many arguments raises TypeError."""
        linear = paddle.nn.Linear(2, 2)
        with self.assertRaises(TypeError):
            linear.to('cpu', 'float64', True, False, 'extra')

    def test_unexpected_keyword(self):
        """to() with unexpected keyword raises TypeError."""
        linear = paddle.nn.Linear(2, 2)
        with self.assertRaises(TypeError):
            linear.to(foo='bar')

    def test_invalid_first_arg(self):
        """to() with invalid first arg raises ValueError."""
        linear = paddle.nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            linear.to(123)

    # ---- PyTorch keyword alias: other / tensor ----

    def test_layer_keyword_other_alias(self):
        """Layer.to(other=tensor) -- PyTorch alias for tensor overload."""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(other=ref)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_keyword_tensor_alias(self):
        """Layer.to(tensor=tensor) -- PyTorch alias for tensor overload."""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(tensor=ref)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_tensor_keyword_other_alias(self):
        """Tensor.to(other=tensor) -- PyTorch alias for tensor overload."""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(other=ref)
        self.assertEqual(out.dtype, paddle.int32)

    def test_tensor_keyword_tensor_alias(self):
        """Tensor.to(tensor=tensor) -- PyTorch alias for tensor overload."""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(tensor=ref)
        self.assertEqual(out.dtype, paddle.int32)

    # ---- copy parameter: Layer.to (tensor overload) ----

    def test_layer_tensor_overload_copy_positional(self):
        """Layer.to(tensor, blocking, copy) -- copy as 3rd positional."""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(ref, True, True)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_tensor_overload_copy_keyword(self):
        """Layer.to(tensor, copy=True) -- copy as keyword."""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(ref, copy=True)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_tensor_overload_copy_mixed(self):
        """Layer.to(tensor, blocking=True, copy=True) -- mixed."""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(ref, blocking=True, copy=True)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_tensor_overload_copy_false_keyword(self):
        """Layer.to(tensor, copy=False) -- explicit copy=False."""
        linear = paddle.nn.Linear(2, 2)
        ref = paddle.to_tensor([1.0], dtype='float64')
        linear.to(ref, copy=False)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    # ---- copy parameter: Layer.to (dtype overload) ----

    def test_layer_dtype_overload_copy_positional(self):
        """Layer.to(dtype, blocking, copy) -- copy as 3rd positional."""
        linear = paddle.nn.Linear(2, 2)
        linear.to('float64', True, True)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_dtype_overload_copy_keyword(self):
        """Layer.to(dtype, copy=True) -- copy as keyword."""
        linear = paddle.nn.Linear(2, 2)
        linear.to('float64', copy=True)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_dtype_overload_copy_mixed(self):
        """Layer.to(dtype, blocking=True, copy=True) -- mixed."""
        linear = paddle.nn.Linear(2, 2)
        linear.to(paddle.float64, blocking=True, copy=True)
        self.assertEqual(linear.weight.dtype, paddle.float64)

    # ---- copy parameter: Layer.to (device overload) ----

    def test_layer_device_overload_copy_positional(self):
        """Layer.to(device, dtype, blocking, copy) -- copy as 4th positional."""
        linear = paddle.nn.Linear(2, 2)
        linear.to('cpu', 'float64', True, True)
        self.assertTrue(linear.weight.place.is_cpu_place())
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_device_overload_copy_keyword(self):
        """Layer.to(device, copy=True) -- copy as keyword only."""
        linear = paddle.nn.Linear(2, 2)
        linear.to('cpu', copy=True)
        self.assertTrue(linear.weight.place.is_cpu_place())

    def test_layer_device_overload_all_kwargs(self):
        """Layer.to(device=, dtype=, blocking=, copy=) -- all keywords."""
        linear = paddle.nn.Linear(2, 2)
        linear.to(device='cpu', dtype='float64', blocking=True, copy=True)
        self.assertTrue(linear.weight.place.is_cpu_place())
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_layer_device_overload_mixed_copy_keyword(self):
        """Layer.to(device, dtype, copy=True) -- 2 positional + copy kwarg."""
        linear = paddle.nn.Linear(2, 2)
        linear.to('cpu', 'float64', copy=True)
        self.assertTrue(linear.weight.place.is_cpu_place())
        self.assertEqual(linear.weight.dtype, paddle.float64)

    # ---- copy parameter: Tensor.to (tensor overload) ----

    def test_tensor_tensor_overload_copy_positional(self):
        """Tensor.to(other, blocking, copy) -- copy as 3rd positional."""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(ref, True, True)
        self.assertEqual(out.dtype, paddle.int32)

    def test_tensor_tensor_overload_copy_keyword(self):
        """Tensor.to(other, copy=True) -- copy as keyword."""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(ref, copy=True)
        self.assertEqual(out.dtype, paddle.int32)

    def test_tensor_tensor_overload_copy_mixed(self):
        """Tensor.to(other, blocking=True, copy=True) -- mixed."""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(ref, blocking=True, copy=True)
        self.assertEqual(out.dtype, paddle.int32)

    def test_tensor_tensor_overload_full_kwargs(self):
        """Tensor.to(other=, blocking=, copy=) -- all keywords."""
        t = paddle.to_tensor([1.0, 2.0])
        ref = paddle.to_tensor([1], dtype='int32')
        out = t.to(other=ref, blocking=True, copy=True)
        self.assertEqual(out.dtype, paddle.int32)

    # ---- copy parameter: Tensor.to (dtype overload) ----

    def test_tensor_dtype_overload_copy_positional(self):
        """Tensor.to(dtype, blocking, copy) -- copy as 3rd positional."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('float64', True, True)
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_dtype_overload_copy_keyword(self):
        """Tensor.to(dtype, copy=True) -- copy as keyword."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('float64', copy=True)
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_dtype_overload_copy_mixed(self):
        """Tensor.to(dtype, blocking=True, copy=True) -- mixed."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to(paddle.float64, blocking=True, copy=True)
        self.assertEqual(out.dtype, paddle.float64)

    # ---- copy parameter: Tensor.to (device overload) ----

    def test_tensor_device_overload_copy_positional(self):
        """Tensor.to(device, dtype, blocking, copy) -- copy as 4th positional."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('cpu', 'float64', True, True)
        self.assertTrue(out.place.is_cpu_place())
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_device_overload_copy_keyword(self):
        """Tensor.to(device, copy=True) -- copy as keyword only."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('cpu', copy=True)
        self.assertTrue(out.place.is_cpu_place())

    def test_tensor_device_overload_all_kwargs(self):
        """Tensor.to(device=, dtype=, blocking=, copy=) -- all keywords."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to(device='cpu', dtype='float64', blocking=True, copy=True)
        self.assertTrue(out.place.is_cpu_place())
        self.assertEqual(out.dtype, paddle.float64)

    def test_tensor_device_overload_mixed_copy_keyword(self):
        """Tensor.to(device, dtype, copy=True) -- 2 positional + copy kwarg."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('cpu', 'float64', copy=True)
        self.assertTrue(out.place.is_cpu_place())
        self.assertEqual(out.dtype, paddle.float64)

    # ---- copy parameter defaults and validation ----

    def test_copy_default_is_false_layer(self):
        """Layer.to without copy should default copy=False (no error)."""
        linear = paddle.nn.Linear(2, 2)
        linear.to('float64')
        self.assertEqual(linear.weight.dtype, paddle.float64)

    def test_copy_default_is_false_tensor(self):
        """Tensor.to without copy should default copy=False (no error)."""
        t = paddle.to_tensor([1.0, 2.0])
        out = t.to('float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_copy_invalid_type_layer(self):
        """Layer.to(dtype, copy='yes') raises TypeError for non-bool."""
        linear = paddle.nn.Linear(2, 2)
        with self.assertRaises(TypeError):
            linear.to('float64', copy='yes')

    def test_copy_invalid_type_tensor(self):
        """Tensor.to(dtype, copy='yes') raises TypeError for non-bool."""
        t = paddle.to_tensor([1.0, 2.0])
        with self.assertRaises(TypeError):
            t.to('float64', copy='yes')


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


# Test tile compatibility
class TestTileAPICompatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3).astype("float32")
        self.repeat_times = [2, 3]
        self.shape = self.np_x.shape
        self.dtype = str(self.np_x.dtype)
        self.np_x_3d = np.random.rand(1, 2, 2).astype("float64")
        self.repeat_times_3d = [2, 1, 3]
        self.shape_3d = self.np_x_3d.shape
        self.dtype_3d = str(self.np_x_3d.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.tile(x, self.repeat_times)
        # 2. Paddle keyword arguments
        out2 = paddle.tile(x=x, repeat_times=self.repeat_times)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.tile(input=x, dims=self.repeat_times)
        # 4. Mixed arguments
        out4 = paddle.tile(x, dims=self.repeat_times)
        # 5. Tensor method - args
        out5 = x.tile(2, 3)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.tile(dims=self.repeat_times)

        # Verify all outputs
        ref_out = np.tile(self.np_x, self.repeat_times)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.tile(x, self.repeat_times)
            # 2. Paddle keyword arguments
            out2 = paddle.tile(x=x, repeat_times=self.repeat_times)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.tile(input=x, dims=self.repeat_times)
            # 4. Tensor method - args
            out4 = x.tile(2, 3)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.tile(dims=self.repeat_times)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

        # Verify all outputs
        ref_out = np.tile(self.np_x, self.repeat_times)
        for out in fetches:
            np.testing.assert_allclose(out, ref_out, rtol=1e-5)

    def test_dygraph_HighDimCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x_3d)

        # 1. Paddle Positional arguments
        out1 = paddle.tile(x, self.repeat_times_3d)
        # 2. Paddle keyword arguments
        out2 = paddle.tile(x=x, repeat_times=self.repeat_times_3d)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.tile(input=x, dims=self.repeat_times_3d)
        # 4. Mixed arguments
        out4 = paddle.tile(x, dims=self.repeat_times_3d)
        # 5. Tensor method - args
        out5 = x.tile(2, 1, 3)
        # 6. Tensor method - kwargs (PyTorch alias)
        out6 = x.tile(dims=self.repeat_times_3d)

        dims = self.repeat_times_3d
        # 7. Tensor method - args with variable expansion
        out7 = x.tile(*dims)

        # Verify all outputs
        ref_out = np.tile(self.np_x_3d, self.repeat_times_3d)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_HighDimCompatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape_3d, dtype=self.dtype_3d
            )

            # 1. Paddle Positional arguments
            out1 = paddle.tile(x, self.repeat_times_3d)
            # 2. Paddle keyword arguments
            out2 = paddle.tile(x=x, repeat_times=self.repeat_times_3d)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.tile(input=x, dims=self.repeat_times_3d)
            # 4. Tensor method - args
            out4 = x.tile(2, 1, 3)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.tile(dims=self.repeat_times_3d)

            dims = self.repeat_times_3d
            # 6. Tensor method - args with variable expansion
            out6 = x.tile(*dims)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x_3d},
                fetch_list=[out1, out2, out3, out4, out5, out6],
            )

        # Verify all outputs
        ref_out = np.tile(self.np_x_3d, self.repeat_times_3d)
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


# Test conv1d_transpose / conv_transpose1d compatibility
@unittest.skipIf(
    sys.platform == 'win32',
    "Conv transpose compatibility tests not supported on Windows-Inference",
)
class TestConv1dTransposeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.dtype = 'float32'
        self.np_x = np.random.rand(1, 2, 4).astype(self.dtype)
        self.np_weight = np.random.rand(2, 2, 3).astype(self.dtype)
        self.np_bias = np.random.rand(2).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        weight = paddle.to_tensor(self.np_weight)
        bias = paddle.to_tensor(self.np_bias)

        # 1. Paddle Positional arguments
        out1 = paddle.nn.functional.conv1d_transpose(x, weight)
        # 2. Paddle keyword arguments
        out2 = paddle.nn.functional.conv1d_transpose(x=x, weight=weight)
        # 3. PyTorch keyword arguments (alias: input)
        out3 = paddle.nn.functional.conv1d_transpose(input=x, weight=weight)
        # 4. PyTorch function name alias
        out4 = paddle.nn.functional.conv_transpose1d(x, weight)
        # 5. PyTorch function name alias + PyTorch keyword
        out5 = paddle.nn.functional.conv_transpose1d(input=x, weight=weight)
        # 6. Mixed arguments (positional + keyword)
        out6 = paddle.nn.functional.conv1d_transpose(
            x, weight, bias=bias, stride=1, padding=0
        )
        # 7. Positional arguments with bias
        out7 = paddle.nn.functional.conv1d_transpose(x, weight, bias)
        # 8. All positional arguments
        out8 = paddle.nn.functional.conv1d_transpose(
            x, weight, bias, 1, 0, 0, 1, 1, None, 'NCL', None
        )
        # 9. All keyword arguments
        out9 = paddle.nn.functional.conv1d_transpose(
            x=x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            output_size=None,
            data_format='NCL',
            name=None,
        )
        # 10. PyTorch alias + all keyword arguments
        out10 = paddle.nn.functional.conv_transpose1d(
            input=x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            output_size=None,
            data_format='NCL',
            name=None,
        )

        # Verify outputs without bias
        ref = out1.numpy()
        for out in [out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

        # Verify outputs with bias
        ref_bias = out6.numpy()
        for out in [out7, out8, out9, out10]:
            np.testing.assert_allclose(out.numpy(), ref_bias, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[1, 2, 4], dtype=self.dtype)
            weight = paddle.static.data(
                name="weight", shape=[2, 2, 3], dtype=self.dtype
            )

            # 1. Paddle Positional arguments
            out1 = paddle.nn.functional.conv1d_transpose(x, weight)
            # 2. Paddle keyword arguments
            out2 = paddle.nn.functional.conv1d_transpose(x=x, weight=weight)
            # 3. PyTorch keyword arguments (alias: input)
            out3 = paddle.nn.functional.conv1d_transpose(input=x, weight=weight)
            # 4. PyTorch function name alias
            out4 = paddle.nn.functional.conv_transpose1d(x, weight)
            # 5. PyTorch function name alias + PyTorch keyword
            out5 = paddle.nn.functional.conv_transpose1d(input=x, weight=weight)
            # 6. All positional arguments
            out6 = paddle.nn.functional.conv1d_transpose(
                x, weight, None, 1, 0, 0, 1, 1, None, 'NCL', None
            )
            # 7. All keyword arguments
            out7 = paddle.nn.functional.conv1d_transpose(
                x=x,
                weight=weight,
                bias=None,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                dilation=1,
                output_size=None,
                data_format='NCL',
                name=None,
            )

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "weight": self.np_weight,
                },
                fetch_list=[out1, out2, out3, out4, out5, out6, out7],
            )

            # Verify all outputs
            for i in range(1, len(fetches)):
                np.testing.assert_allclose(fetches[0], fetches[i], rtol=1e-5)


# Test Conv1DTranspose layer compatibility
@unittest.skipIf(
    sys.platform == 'win32',
    "Conv transpose compatibility tests not supported on Windows-Inference",
)
class TestConv1DTransposeLayerAPI(unittest.TestCase):
    def test_paddle_style_keyword_only(self):
        paddle.disable_static()
        layer = paddle.nn.Conv1DTranspose(2, 2, 3)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_bias_false_disables_bias_attr(self):
        paddle.disable_static()
        layer = paddle.nn.Conv1DTranspose(2, 2, 3, bias=False)
        self.assertIsNone(layer.bias)

    def test_pytorch_style_positional_bias_only(self):
        paddle.disable_static()
        layer = paddle.nn.Conv1DTranspose(2, 2, 3, 1, 0, 0, 1, True)
        self.assertIsNotNone(layer.bias)

    def test_pytorch_style_full_positional(self):
        paddle.disable_static()
        layer = paddle.nn.ConvTranspose1d(
            2, 2, 3, 1, 0, 0, 1, False, 1, 'zeros', None, None
        )
        self.assertIsNone(layer.bias)

    def test_pytorch_style_duplicate_bias_raises(self):
        paddle.disable_static()
        with self.assertRaises(TypeError):
            paddle.nn.Conv1DTranspose(2, 2, 3, 1, 0, 0, 1, True, bias=True)


# Test conv2d_transpose / conv_transpose2d compatibility
@unittest.skipIf(
    sys.platform == 'win32',
    "Conv transpose compatibility tests not supported on Windows-Inference",
)
class TestConv2dTransposeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.dtype = 'float32'
        self.np_x = np.random.rand(1, 2, 4, 4).astype(self.dtype)
        self.np_weight = np.random.rand(2, 2, 3, 3).astype(self.dtype)
        self.np_bias = np.random.rand(2).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        weight = paddle.to_tensor(self.np_weight)
        bias = paddle.to_tensor(self.np_bias)

        # 1. Paddle Positional arguments
        out1 = paddle.nn.functional.conv2d_transpose(x, weight)
        # 2. Paddle keyword arguments
        out2 = paddle.nn.functional.conv2d_transpose(x=x, weight=weight)
        # 3. PyTorch keyword arguments (alias: input)
        out3 = paddle.nn.functional.conv2d_transpose(input=x, weight=weight)
        # 4. PyTorch function name alias
        out4 = paddle.nn.functional.conv_transpose2d(x, weight)
        # 5. PyTorch function name alias + PyTorch keyword
        out5 = paddle.nn.functional.conv_transpose2d(input=x, weight=weight)
        # 6. Mixed arguments (positional + keyword)
        out6 = paddle.nn.functional.conv2d_transpose(
            x, weight, bias=bias, stride=1, padding=0
        )
        # 7. Positional arguments with bias
        out7 = paddle.nn.functional.conv2d_transpose(x, weight, bias)
        # 8. All positional arguments
        out8 = paddle.nn.functional.conv2d_transpose(
            x, weight, bias, 1, 0, 0, 1, 1, None, 'NCHW', None
        )
        # 9. All keyword arguments
        out9 = paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            output_size=None,
            data_format='NCHW',
            name=None,
        )
        # 10. PyTorch alias + all keyword arguments
        out10 = paddle.nn.functional.conv_transpose2d(
            input=x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            output_size=None,
            data_format='NCHW',
            name=None,
        )

        # Verify outputs without bias
        ref = out1.numpy()
        for out in [out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

        # Verify outputs with bias
        ref_bias = out6.numpy()
        for out in [out7, out8, out9, out10]:
            np.testing.assert_allclose(out.numpy(), ref_bias, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=[1, 2, 4, 4], dtype=self.dtype
            )
            weight = paddle.static.data(
                name="weight", shape=[2, 2, 3, 3], dtype=self.dtype
            )

            # 1. Paddle Positional arguments
            out1 = paddle.nn.functional.conv2d_transpose(x, weight)
            # 2. Paddle keyword arguments
            out2 = paddle.nn.functional.conv2d_transpose(x=x, weight=weight)
            # 3. PyTorch keyword arguments (alias: input)
            out3 = paddle.nn.functional.conv2d_transpose(input=x, weight=weight)
            # 4. PyTorch function name alias
            out4 = paddle.nn.functional.conv_transpose2d(x, weight)
            # 5. PyTorch function name alias + PyTorch keyword
            out5 = paddle.nn.functional.conv_transpose2d(input=x, weight=weight)
            # 6. All positional arguments
            out6 = paddle.nn.functional.conv2d_transpose(
                x, weight, None, 1, 0, 0, 1, 1, None, 'NCHW', None
            )
            # 7. All keyword arguments
            out7 = paddle.nn.functional.conv2d_transpose(
                x=x,
                weight=weight,
                bias=None,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                dilation=1,
                output_size=None,
                data_format='NCHW',
                name=None,
            )

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "weight": self.np_weight,
                },
                fetch_list=[out1, out2, out3, out4, out5, out6, out7],
            )

            # Verify all outputs
            for i in range(1, len(fetches)):
                np.testing.assert_allclose(fetches[0], fetches[i], rtol=1e-5)


# Test Conv2DTranspose layer compatibility
@unittest.skipIf(
    sys.platform == 'win32',
    "Conv transpose compatibility tests not supported on Windows-Inference",
)
class TestConv2DTransposeLayerAPI(unittest.TestCase):
    def test_paddle_style_keyword_only(self):
        paddle.disable_static()
        layer = paddle.nn.Conv2DTranspose(2, 2, 3)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_bias_false_disables_bias_attr(self):
        paddle.disable_static()
        layer = paddle.nn.Conv2DTranspose(2, 2, 3, bias=False)
        self.assertIsNone(layer.bias)

    def test_pytorch_style_positional_bias_only(self):
        paddle.disable_static()
        layer = paddle.nn.Conv2DTranspose(2, 2, 3, 1, 0, 0, 1, True)
        self.assertIsNotNone(layer.bias)

    def test_pytorch_style_full_positional(self):
        paddle.disable_static()
        layer = paddle.nn.ConvTranspose2d(
            2, 2, 3, 1, 0, 0, 1, False, 1, 'zeros', None, None
        )
        self.assertIsNone(layer.bias)

    def test_pytorch_style_duplicate_bias_raises(self):
        paddle.disable_static()
        with self.assertRaises(TypeError):
            paddle.nn.Conv2DTranspose(2, 2, 3, 1, 0, 0, 1, True, bias=True)


# Test conv3d_transpose / conv_transpose3d compatibility
@unittest.skipIf(
    sys.platform == 'win32',
    "Conv transpose compatibility tests not supported on Windows-Inference",
)
class TestConv3dTransposeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.dtype = 'float32'
        self.np_x = np.random.rand(1, 2, 4, 4, 4).astype(self.dtype)
        self.np_weight = np.random.rand(2, 2, 3, 3, 3).astype(self.dtype)
        self.np_bias = np.random.rand(2).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        weight = paddle.to_tensor(self.np_weight)
        bias = paddle.to_tensor(self.np_bias)

        # 1. Paddle Positional arguments
        out1 = paddle.nn.functional.conv3d_transpose(x, weight)
        # 2. Paddle keyword arguments
        out2 = paddle.nn.functional.conv3d_transpose(x=x, weight=weight)
        # 3. PyTorch keyword arguments (alias: input)
        out3 = paddle.nn.functional.conv3d_transpose(input=x, weight=weight)
        # 4. PyTorch function name alias
        out4 = paddle.nn.functional.conv_transpose3d(x, weight)
        # 5. PyTorch function name alias + PyTorch keyword
        out5 = paddle.nn.functional.conv_transpose3d(input=x, weight=weight)
        # 6. Mixed arguments (positional + keyword)
        out6 = paddle.nn.functional.conv3d_transpose(
            x, weight, bias=bias, stride=1, padding=0
        )
        # 7. Positional arguments with bias
        out7 = paddle.nn.functional.conv3d_transpose(x, weight, bias)
        # 8. All positional arguments
        out8 = paddle.nn.functional.conv3d_transpose(
            x, weight, bias, 1, 0, 0, 1, 1, None, 'NCDHW', None
        )
        # 9. All keyword arguments
        out9 = paddle.nn.functional.conv3d_transpose(
            x=x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            output_size=None,
            data_format='NCDHW',
            name=None,
        )
        # 10. PyTorch alias + all keyword arguments
        out10 = paddle.nn.functional.conv_transpose3d(
            input=x,
            weight=weight,
            bias=bias,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            dilation=1,
            output_size=None,
            data_format='NCDHW',
            name=None,
        )

        # Verify outputs without bias
        ref = out1.numpy()
        for out in [out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), ref, rtol=1e-5)

        # Verify outputs with bias
        ref_bias = out6.numpy()
        for out in [out7, out8, out9, out10]:
            np.testing.assert_allclose(out.numpy(), ref_bias, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=[1, 2, 4, 4, 4], dtype=self.dtype
            )
            weight = paddle.static.data(
                name="weight", shape=[2, 2, 3, 3, 3], dtype=self.dtype
            )

            # 1. Paddle Positional arguments
            out1 = paddle.nn.functional.conv3d_transpose(x, weight)
            # 2. Paddle keyword arguments
            out2 = paddle.nn.functional.conv3d_transpose(x=x, weight=weight)
            # 3. PyTorch keyword arguments (alias: input)
            out3 = paddle.nn.functional.conv3d_transpose(input=x, weight=weight)
            # 4. PyTorch function name alias
            out4 = paddle.nn.functional.conv_transpose3d(x, weight)
            # 5. PyTorch function name alias + PyTorch keyword
            out5 = paddle.nn.functional.conv_transpose3d(input=x, weight=weight)
            # 6. All positional arguments
            out6 = paddle.nn.functional.conv3d_transpose(
                x, weight, None, 1, 0, 0, 1, 1, None, 'NCDHW', None
            )
            # 7. All keyword arguments
            out7 = paddle.nn.functional.conv3d_transpose(
                x=x,
                weight=weight,
                bias=None,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
                dilation=1,
                output_size=None,
                data_format='NCDHW',
                name=None,
            )

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "weight": self.np_weight,
                },
                fetch_list=[out1, out2, out3, out4, out5, out6, out7],
            )

            # Verify all outputs
            for i in range(1, len(fetches)):
                np.testing.assert_allclose(fetches[0], fetches[i], rtol=1e-5)


# Test Conv3DTranspose layer compatibility
@unittest.skipIf(
    sys.platform == 'win32',
    "Conv transpose compatibility tests not supported on Windows-Inference",
)
class TestConv3DTransposeLayerAPI(unittest.TestCase):
    def test_paddle_style_keyword_only(self):
        paddle.disable_static()
        layer = paddle.nn.Conv3DTranspose(2, 2, 3)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_bias_false_disables_bias_attr(self):
        paddle.disable_static()
        layer = paddle.nn.Conv3DTranspose(2, 2, 3, bias=False)
        self.assertIsNone(layer.bias)

    def test_pytorch_style_positional_bias_only(self):
        paddle.disable_static()
        layer = paddle.nn.Conv3DTranspose(2, 2, 3, 1, 0, 0, 1, True)
        self.assertIsNotNone(layer.bias)

    def test_pytorch_style_full_positional(self):
        paddle.disable_static()
        layer = paddle.nn.ConvTranspose3d(
            2, 2, 3, 1, 0, 0, 1, False, 1, 'zeros', None, None
        )
        self.assertIsNone(layer.bias)

    def test_pytorch_style_duplicate_bias_raises(self):
        paddle.disable_static()
        with self.assertRaises(TypeError):
            paddle.nn.Conv3DTranspose(2, 2, 3, 1, 0, 0, 1, True, bias=True)


def _assert_unary_inplace_result(
    testcase, x, out, ref_out, rtol=1e-6, atol=1e-6
):
    testcase.assertIs(out, x)
    np.testing.assert_allclose(out.numpy(), ref_out, rtol=rtol, atol=atol)
    np.testing.assert_allclose(x.numpy(), ref_out, rtol=rtol, atol=atol)


class TestExpInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.exp_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.exp_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.exp_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().exp_()

        # Verify all outputs
        ref_out = np.exp(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.exp(self.np_x)

        out = paddle.exp_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestSqrtInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([0.25, 1.5, 2.25, 4.0], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.sqrt_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.sqrt_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.sqrt_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().sqrt_()

        # Verify all outputs
        ref_out = np.sqrt(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.sqrt(self.np_x)

        out = paddle.sqrt_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestRsqrtInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([0.25, 1.5, 2.25, 4.0], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.rsqrt_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.rsqrt_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.rsqrt_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().rsqrt_()

        # Verify all outputs
        ref_out = 1.0 / np.sqrt(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = 1.0 / np.sqrt(self.np_x)

        out = paddle.rsqrt_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestCeilInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.ceil_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.ceil_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.ceil_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().ceil_()

        # Verify all outputs
        ref_out = np.ceil(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.ceil(self.np_x)

        out = paddle.ceil_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestFloorInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.floor_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.floor_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.floor_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().floor_()

        # Verify all outputs
        ref_out = np.floor(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.floor(self.np_x)

        out = paddle.floor_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestReciprocalInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-2.0, -0.5, 0.25, 4.0], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.reciprocal_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.reciprocal_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.reciprocal_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().reciprocal_()

        # Verify all outputs
        ref_out = np.reciprocal(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.reciprocal(self.np_x)

        out = paddle.reciprocal_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestSigmoidInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.sigmoid_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.sigmoid_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.sigmoid_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().sigmoid_()

        # Verify all outputs
        ref_out = 1.0 / (1.0 + np.exp(-self.np_x))
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = 1.0 / (1.0 + np.exp(-self.np_x))

        out = paddle.sigmoid_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestSinInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.sin_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.sin_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.sin_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().sin_()

        # Verify all outputs
        ref_out = np.sin(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.sin(self.np_x)

        out = paddle.sin_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestSinhInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.sinh_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.sinh_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.sinh_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().sinh_()

        # Verify all outputs
        ref_out = np.sinh(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.sinh(self.np_x)

        out = paddle.sinh_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestAsinInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.9, -0.25, 0.25, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.asin_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.asin_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.asin_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().asin_()

        # Verify all outputs
        ref_out = np.arcsin(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.arcsin(self.np_x)

        out = paddle.asin_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestAsinhInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.asinh_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.asinh_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.asinh_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().asinh_()

        # Verify all outputs
        ref_out = np.arcsinh(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.arcsinh(self.np_x)

        out = paddle.asinh_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestCosInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.cos_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.cos_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.cos_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().cos_()

        # Verify all outputs
        ref_out = np.cos(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.cos(self.np_x)

        out = paddle.cos_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestCoshInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.cosh_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.cosh_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.cosh_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().cosh_()

        # Verify all outputs
        ref_out = np.cosh(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.cosh(self.np_x)

        out = paddle.cosh_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestAcosInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.9, -0.25, 0.25, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.acos_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.acos_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.acos_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().acos_()

        # Verify all outputs
        ref_out = np.arccos(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.arccos(self.np_x)

        out = paddle.acos_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestAcoshInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([1.0, 1.5, 2.0, 3.5], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.acosh_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.acosh_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.acosh_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().acosh_()

        # Verify all outputs
        ref_out = np.arccosh(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.arccosh(self.np_x)

        out = paddle.acosh_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestTanInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.tan_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.tan_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.tan_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().tan_()

        # Verify all outputs
        ref_out = np.tan(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.tan(self.np_x)

        out = paddle.tan_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestAtanInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.atan_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.atan_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.atan_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().atan_()

        # Verify all outputs
        ref_out = np.arctan(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.arctan(self.np_x)

        out = paddle.atan_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestAtanhInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.9, -0.25, 0.25, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.atanh_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.atanh_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.atanh_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().atanh_()

        # Verify all outputs
        ref_out = np.arctanh(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.arctanh(self.np_x)

        out = paddle.atanh_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestExpm1InplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.expm1_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.expm1_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.expm1_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().expm1_()

        # Verify all outputs
        ref_out = np.expm1(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.expm1(self.np_x)

        out = paddle.expm1_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


class TestSquareInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.7, -0.2, 0.3, 0.9], dtype="float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.square_(x.clone())
        # 2. Paddle keyword arguments
        out2 = paddle.square_(x=x.clone())
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.square_(input=x.clone())
        # 4. Tensor method - args
        out4 = x.clone().square_()

        # Verify all outputs
        ref_out = np.square(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-6)

        paddle.enable_static()

    def test_dygraph_InplaceInput(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.square(self.np_x)

        out = paddle.square_(x)

        _assert_unary_inplace_result(self, x, out, ref_out)

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
