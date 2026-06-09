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


# Test histc compatibility
class TestHistcAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(100).astype("float32") * 10

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.histc(x, bins=10, min=0, max=10)
        # 2. Paddle keyword arguments
        out2 = paddle.histc(input=x, bins=10, min=0, max=10)
        # 3. Tensor method
        out3 = x.histc(bins=10, min=0, max=10)
        # 4. out parameter test
        out4 = paddle.empty_like(out1)
        paddle.histc(x, bins=10, min=0, max=10, out=out4)

        # Verify outputs are float32 (PyTorch compatibility)
        self.assertEqual(out1.dtype, paddle.float32)
        for out in [out1, out2, out3, out4]:
            self.assertEqual(out.dtype, paddle.float32)

        paddle.enable_static()


# Test mvlgamma compatibility (alias for multigammaln)
class TestMvlgammaAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([2.5, 3.5, 4.5]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments (mvlgamma is alias for multigammaln)
        out1 = paddle.mvlgamma(x, p=2)
        # 2. Paddle keyword arguments
        out2 = paddle.mvlgamma(x=x, p=2)
        # 3. Tensor method
        out3 = x.mvlgamma(p=2)

        # Verify outputs
        for out in [out1, out2, out3]:
            self.assertEqual(out.shape, (3,))

        paddle.enable_static()


# Test mvlgamma_ compatibility (inplace)
class TestMvlgamma_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([2.5, 3.5, 4.5]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x.copy())

        # Inplace operation
        x.mvlgamma_(p=2)

        # Verify shape unchanged
        self.assertEqual(x.shape, (3,))

        paddle.enable_static()


# Test negative_ compatibility (alias for neg_)
class TestNegative_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.0, -2.0, 3.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x.copy())

        # Inplace operation (negative_ is alias for neg_)
        x.negative_()

        expected = -self.np_x
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


# Test to_sparse compatibility (alias for to_sparse_coo)
class TestToSparseAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        if paddle.is_compiled_with_xpu():
            self.skipTest("sparse ops are not supported on XPU")

        paddle.disable_static()
        dense_x = paddle.to_tensor(
            [[0, 1, 0, 2], [0, 0, 3, 4]], dtype='float32'
        )

        # to_sparse is alias for to_sparse_coo
        sparse_x = dense_x.to_sparse(sparse_dim=2)

        self.assertTrue(sparse_x.is_sparse_coo())

        paddle.enable_static()


# Test special.round compatibility
class TestSpecialRoundAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([0.5, -0.3, 1.7, -2.4]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # paddle.special.round is alias for paddle.round
        out1 = paddle.special.round(x)
        out2 = paddle.round(x)

        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)

        paddle.enable_static()


# Test autograd.enable_grad compatibility
class TestAutogradEnableGradAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # paddle.autograd.enable_grad should work
        @paddle.autograd.enable_grad()
        def test_func(x):
            return x * 2

        x = paddle.to_tensor([1.0, 2.0])
        with paddle.no_grad():
            y = test_func(x)

        np.testing.assert_allclose(y.numpy(), [2.0, 4.0], rtol=1e-5)

        paddle.enable_static()


# Test col_indices compatibility (alias for cols)
class TestColIndicesAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        if paddle.is_compiled_with_xpu():
            self.skipTest("sparse ops are not supported on XPU")

        paddle.disable_static()

        # Create a sparse CSR tensor
        crows = paddle.to_tensor([0, 2, 3, 5], dtype='int64')
        cols = paddle.to_tensor([1, 3, 2, 0, 1], dtype='int64')
        values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        dense_shape = [3, 4]

        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)

        # col_indices is alias for cols
        result1 = csr.col_indices()
        result2 = csr.cols()

        np.testing.assert_array_equal(result1.numpy(), result2.numpy())

        paddle.enable_static()


# Test crow_indices compatibility (alias for crows)
class TestCrowIndicesAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        if paddle.is_compiled_with_xpu():
            self.skipTest("sparse ops are not supported on XPU")

        paddle.disable_static()

        # Create a sparse CSR tensor
        crows = paddle.to_tensor([0, 2, 3, 5], dtype='int64')
        cols = paddle.to_tensor([1, 3, 2, 0, 1], dtype='int64')
        values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        dense_shape = [3, 4]

        csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, dense_shape)

        # crow_indices is alias for crows
        result1 = csr.crow_indices()
        result2 = csr.crows()

        np.testing.assert_array_equal(result1.numpy(), result2.numpy())

        paddle.enable_static()


# Test take compatibility
class TestTakeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1, 2, 3, 4, 5]).astype("float32")
        self.np_indices = np.array([0, 2, 4]).astype("int64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        indices = paddle.to_tensor(self.np_indices)

        # 1. Paddle positional arguments
        out1 = paddle.take(x, indices)
        # 2. Paddle keyword arguments
        out2 = paddle.take(x=x, index=indices)
        # 3. Tensor method
        out3 = x.take(indices)

        # Verify outputs
        expected = self.np_x[self.np_indices]
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
            indices = paddle.static.data(
                name="indices",
                shape=self.np_indices.shape,
                dtype=str(self.np_indices.dtype),
            )

            out1 = paddle.take(x, indices)
            out2 = paddle.take(x=x, index=indices)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "indices": self.np_indices,
                },
                fetch_list=[out1, out2],
            )

            expected = self.np_x[self.np_indices]
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)

        paddle.disable_static()


# Test matrix_exp compatibility
class TestMatrixExpAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([[1.0, 0.0], [0.0, 1.0]]).astype("float32")

    def test_dygraph_Compatibility(self):
        if paddle.is_compiled_with_rocm():
            self.skipTest("Skip on DCU due to kernel issue")

        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. paddle.linalg.matrix_exp
        out1 = paddle.linalg.matrix_exp(x)
        # 2. Tensor method
        out2 = x.matrix_exp()

        # Verify outputs - matrix_exp of identity is e^1 * identity
        expected = np.exp(1.0) * np.eye(2)
        for out in [out1, out2]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        if paddle.is_compiled_with_rocm():
            self.skipTest("Skip on DCU due to kernel issue")

        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.np_x.shape, dtype=str(self.np_x.dtype)
            )

            out1 = paddle.linalg.matrix_exp(x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1],
            )

            expected = np.exp(1.0) * np.eye(2)
            np.testing.assert_allclose(fetches[0], expected, rtol=1e-5)

        paddle.disable_static()


# Test retain_grad compatibility
class TestRetainGradAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # Test retain_grad on leaf tensor
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        x.retain_grad()

        y = x * 2
        y.sum().backward()

        # Gradient should be retained
        np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0, 2.0], rtol=1e-5)

        # Test retain_grad on non-leaf tensor
        a = paddle.to_tensor([1.0, 2.0])
        a.stop_gradient = False
        b = a * 3  # non-leaf
        b.retain_grad()
        c = b * 2
        c.sum().backward()

        # b's gradient should be retained
        np.testing.assert_allclose(b.grad.numpy(), [2.0, 2.0], rtol=1e-5)

        paddle.enable_static()


# Test sparse_mask compatibility
class TestSparseMaskAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        # Skip on XPU as sparse_mask is not supported
        if paddle.is_compiled_with_xpu():
            self.skipTest("sparse_mask is not supported on XPU")

        paddle.disable_static()

        # Create dense tensor
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])

        # Create sparse COO tensor as mask
        indices = paddle.to_tensor([[0, 1], [0, 1]], dtype='int64')
        values = paddle.to_tensor([1.0, 1.0], dtype='float32')
        mask = paddle.sparse.sparse_coo_tensor(indices, values, [2, 2])

        # Apply sparse_mask
        result = x.sparse_mask(mask)

        # Verify result is sparse and has correct values
        np.testing.assert_allclose(
            result.values().numpy(), [1.0, 4.0], rtol=1e-5
        )

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
