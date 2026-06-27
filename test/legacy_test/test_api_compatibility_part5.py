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

        # 1. Paddle Positional arguments
        out1 = paddle.take(x, indices)

        # 2. Paddle keyword arguments
        out2 = paddle.take(x=x, index=indices)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.take(input=x, index=indices)

        # 4. Mixed arguments
        out4 = paddle.take(x, index=indices)

        # 5. Tensor method - args
        out5 = x.take(indices)

        # Verify all outputs
        expected = self.np_x[self.np_indices]
        for out in [out1, out2, out3, out4, out5]:
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

            # 1. Paddle Positional arguments
            out1 = paddle.take(x, indices)
            # 2. Paddle keyword arguments
            out2 = paddle.take(x=x, index=indices)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.take(input=x, index=indices)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "indices": self.np_indices,
                },
                fetch_list=[out1, out2, out3],
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


# Test ParameterList compatibility (values -> parameters alias)
class TestParameterListAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle keyword arguments (parameters)
        params1 = [
            paddle.create_parameter(shape=[2, 3], dtype='float32')
            for _ in range(3)
        ]
        pl1 = paddle.nn.ParameterList(parameters=params1)

        # 2. PyTorch keyword arguments (values alias)
        params2 = [
            paddle.create_parameter(shape=[2, 3], dtype='float32')
            for _ in range(3)
        ]
        pl2 = paddle.nn.ParameterList(values=params2)

        # 3. PyTorch positional arguments
        params3 = [
            paddle.create_parameter(shape=[2, 3], dtype='float32')
            for _ in range(3)
        ]
        pl3 = paddle.nn.ParameterList(params3)

        # 4. Mixed arguments
        params4 = [
            paddle.create_parameter(shape=[2, 3], dtype='float32')
            for _ in range(3)
        ]
        pl4 = paddle.nn.ParameterList(params4)

        # 5. Test append with value alias
        pl5 = paddle.nn.ParameterList()
        param = paddle.create_parameter(shape=[2, 3], dtype='float32')
        pl5.append(value=param)

        # 6. Test extend with parameters alias
        pl6 = paddle.nn.ParameterList()
        params6 = [
            paddle.create_parameter(shape=[2, 3], dtype='float32')
            for _ in range(2)
        ]
        pl6.extend(parameters=params6)

        # Verify lengths
        self.assertEqual(len(pl1), 3)
        self.assertEqual(len(pl2), 3)
        self.assertEqual(len(pl3), 3)
        self.assertEqual(len(pl4), 3)
        self.assertEqual(len(pl5), 1)
        self.assertEqual(len(pl6), 2)

        paddle.enable_static()


# Test scatter_reduce_ compatibility (inplace)
class TestScatterReduce_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([[10, 20, 30], [40, 50, 60]]).astype("float32")
        self.np_index = np.zeros((2, 3)).astype("int64")
        self.np_src = np.array([[1, 2, 3], [4, 5, 6]]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle scatter_reduce_ positional
        x1 = paddle.to_tensor(self.np_x.copy())
        index = paddle.to_tensor(self.np_index)
        src = paddle.to_tensor(self.np_src)
        out1 = x1.scatter_reduce_(0, index, src, "sum", include_self=True)

        # 2. Paddle keyword arguments
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = x2.scatter_reduce_(
            dim=0, index=index, src=src, reduce="sum", include_self=True
        )

        # 3. PyTorch keyword arguments (alias)
        # Note: src is alias for src in Paddle too, but let's check
        x3 = paddle.to_tensor(self.np_x.copy())
        out3 = x3.scatter_reduce_(dim=0, index=index, src=src, reduce="sum")

        # 4. Mixed arguments
        x4 = paddle.to_tensor(self.np_x.copy())
        out4 = x4.scatter_reduce_(0, index, src=src, reduce="sum")

        # Verify inplace operation returns self
        self.assertIs(out1, x1)
        self.assertIs(out2, x2)
        self.assertIs(out3, x3)
        self.assertIs(out4, x4)

        # Verify results
        self.assertEqual(out1.shape, [2, 3])
        np.testing.assert_allclose(out1.numpy(), out2.numpy())

        paddle.enable_static()


# Test xavier_uniform compatibility (alias for xavier_uniform_)
class TestXavierUniformAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. paddle.nn.init.xavier_uniform_ (with underscore)
        x1 = paddle.empty([3, 4], dtype='float32')
        paddle.nn.init.xavier_uniform_(x1, gain=1.0)

        # 2. paddle.nn.init.xavier_uniform (without underscore, PyTorch deprecated alias)
        x2 = paddle.empty([3, 4], dtype='float32')
        paddle.nn.init.xavier_uniform(x2, gain=1.0)

        # 3. PyTorch keyword arguments (tensor alias for x)
        x3 = paddle.empty([3, 4], dtype='float32')
        paddle.nn.init.xavier_uniform(tensor=x3, gain=1.0)

        # 4. Mixed arguments
        x4 = paddle.empty([3, 4], dtype='float32')
        paddle.nn.init.xavier_uniform(x4, gain=1.0)

        # Both should work the same
        self.assertEqual(x1.shape, x2.shape)
        self.assertEqual(x1.shape, x3.shape)
        self.assertEqual(x1.shape, x4.shape)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x1 = paddle.static.data(name="x1", shape=[3, 4], dtype="float32")
            x2 = paddle.static.data(name="x2", shape=[3, 4], dtype="float32")

            # 1. paddle.nn.init.xavier_uniform_ (with underscore)
            paddle.nn.init.xavier_uniform_(x1)

            # 2. paddle.nn.init.xavier_uniform (without underscore, PyTorch deprecated alias)
            paddle.nn.init.xavier_uniform(x2)

            # Just verify it doesn't crash
            self.assertIsNotNone(x1)
            self.assertIsNotNone(x2)

        paddle.disable_static()


# Test sign_ compatibility (inplace)
class TestSign_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Tensor method - args
        x1 = paddle.to_tensor(self.np_x.copy())
        out1 = x1.sign_()

        # 2. Paddle function - positional
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = paddle.sign_(x2)

        # 3. Paddle function - keyword
        x3 = paddle.to_tensor(self.np_x.copy())
        out3 = paddle.sign_(x=x3)

        # 4. PyTorch function - keyword (input alias)
        x4 = paddle.to_tensor(self.np_x.copy())
        out4 = paddle.sign_(input=x4)

        # Verify all outputs
        expected = np.sign(self.np_x)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_pir_infer_symbolic_shape(self):
        from paddle.base.libpaddle import pir

        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data(name="x", shape=[5], dtype="float32")
                out = paddle.sign_(x)

            pm = pir.PassManager()
            pir.infer_symbolic_shape_pass(pm, main)
            pm.run(main)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x.copy()},
                fetch_list=[out],
            )
            out_ref = np.sign(self.np_x)
            np.testing.assert_allclose(fetches[0], out_ref, rtol=1e-5)


# Test linalg.pinv compatibility (atol, rtol, out parameters)
class TestLinalgPinvAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 5).astype("float32")
        self.shape = [3, 5]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments (using rcond)
        out1 = paddle.linalg.pinv(x, rcond=1e-15)

        # 2. Paddle keyword arguments
        out2 = paddle.linalg.pinv(x=x, rcond=1e-15, hermitian=False)

        # 3. PyTorch keyword arguments (input alias)
        out3 = paddle.linalg.pinv(input=x)

        # 4. PyTorch keyword arguments (rtol alias)
        out4 = paddle.linalg.pinv(x, rtol=1e-15)

        # 5. Mixed arguments (atol, rtol)
        out5 = paddle.linalg.pinv(x, atol=1e-10, rtol=1e-10)

        # 6. out parameter test
        out6 = paddle.empty([5, 3], dtype='float32')
        paddle.linalg.pinv(x, out=out6)

        # 7. Tensor method - args
        out7 = x.pinverse()

        # 8. Alias paddle.pinverse
        out8 = paddle.pinverse(x)

        # 9. hermitian=True with atol (need square matrix for hermitian)
        x_sq = paddle.to_tensor(np.random.rand(3, 3).astype("float32"))
        x_sym = x_sq @ x_sq.T + paddle.eye(3) * 0.5  # full rank
        out9 = paddle.linalg.pinv(x_sym, hermitian=True, atol=1e-4)

        # 10. hermitian=True with rtol
        out10 = paddle.linalg.pinv(x_sym, hermitian=True, rtol=1e-4)

        # 11. hermitian=True with both atol and rtol
        out11 = paddle.linalg.pinv(x_sym, hermitian=True, atol=1e-4, rtol=1e-4)

        # Verify all outputs
        expected = np.linalg.pinv(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        expected_sym = np.linalg.pinv(x_sym.numpy(), hermitian=True)
        for out in [out9, out10, out11]:
            np.testing.assert_allclose(out.numpy(), expected_sym, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.linalg.pinv(x)
            # 2. Paddle keyword arguments
            out2 = paddle.linalg.pinv(x=x)
            # 3. PyTorch keyword arguments (input alias)
            out3 = paddle.linalg.pinv(input=x)
            # 4. Tensor method - args
            out4 = x.pinverse()
            # 5. Alias paddle.pinverse
            out5 = paddle.pinverse(x)

            # 6. atol parameter test (keyword-only)
            out6 = paddle.linalg.pinv(x, atol=1e-10)

            # 7. rtol parameter test (keyword-only)
            out7 = paddle.linalg.pinv(x, rtol=1e-10)

            # 8. atol and rtol combined (keyword-only)
            out8 = paddle.linalg.pinv(x, atol=1e-10, rtol=1e-10)

            # 9. out parameter test
            out9 = paddle.static.data(
                name="out9", shape=[5, 3], dtype="float32"
            )
            paddle.linalg.pinv(x, out=out9)

            # 10. hermitian=True with atol (square matrix)
            x_sym = paddle.static.data(
                name="x_sym", shape=[3, 3], dtype="float32"
            )
            out10 = paddle.linalg.pinv(x_sym, hermitian=True, atol=1e-4)

            # 11. hermitian=True with rtol
            out11 = paddle.linalg.pinv(x_sym, hermitian=True, rtol=1e-4)

            # 12. hermitian=True with both atol and rtol
            out12 = paddle.linalg.pinv(
                x_sym, hermitian=True, atol=1e-4, rtol=1e-4
            )

            exe = paddle.static.Executor()
            np_x_sym = (
                self.np_x @ self.np_x.T
                + np.eye(3, dtype=self.dtype) * 0.5  # full rank
            )
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "out9": np.empty([5, 3], dtype="float32"),
                    "x_sym": np_x_sym,
                },
                fetch_list=[
                    out1,
                    out2,
                    out3,
                    out4,
                    out5,
                    out6,
                    out7,
                    out8,
                    out9,
                    out10,
                    out11,
                    out12,
                ],
            )

            expected = np.linalg.pinv(self.np_x)
            for out in fetches[:9]:
                np.testing.assert_allclose(out, expected, rtol=1e-5)

            expected_sym = np.linalg.pinv(np_x_sym, hermitian=True)
            for out in fetches[9:]:
                np.testing.assert_allclose(out, expected_sym, rtol=1e-5)

        paddle.disable_static()


# Test nll_loss compatibility (target -> label alias)
class TestNllLossAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(5, 3).astype("float32")
        self.np_label = np.array([0, 2, 1, 1, 0], dtype="int64")
        self.shape_input = [5, 3]
        self.shape_label = [5]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        log_softmax = paddle.nn.LogSoftmax(axis=1)
        input = log_softmax(paddle.to_tensor(self.np_input))
        label = paddle.to_tensor(self.np_label)

        # 1. Paddle positional arguments
        out1 = paddle.nn.functional.nll_loss(input, label)

        # 2. Paddle keyword arguments
        out2 = paddle.nn.functional.nll_loss(input=input, label=label)

        # 3. PyTorch keyword arguments (target alias)
        out3 = paddle.nn.functional.nll_loss(input=input, target=label)

        # 4. Mixed arguments
        out4 = paddle.nn.functional.nll_loss(
            input, target=label, reduction='mean'
        )

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=self.shape_input, dtype=self.dtype
            )
            label = paddle.static.data(
                name="label", shape=self.shape_label, dtype="int64"
            )

            # 1. Paddle positional arguments
            out1 = paddle.nn.functional.nll_loss(input, label)
            # 2. Paddle keyword arguments
            out2 = paddle.nn.functional.nll_loss(input=input, label=label)
            # 3. PyTorch keyword arguments (target alias)
            out3 = paddle.nn.functional.nll_loss(input=input, target=label)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "label": self.np_label},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                self.assertEqual(out.shape, ())

        paddle.disable_static()


# Test bernoulli_ compatibility (inplace)
class TestBernoulli_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 4).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Tensor method - args
        x1 = paddle.to_tensor(self.np_x.copy())
        out1 = x1.bernoulli_()

        # 2. Tensor method - kwargs
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = x2.bernoulli_(p=0.3)

        # 3. Paddle function - positional
        x3 = paddle.to_tensor(self.np_x.copy())
        out3 = paddle.bernoulli_(x3, p=0.5)

        # 4. Paddle function - keyword
        x4 = paddle.to_tensor(self.np_x.copy())
        out4 = paddle.bernoulli_(x=x4, p=0.5)

        # Verify inplace operation returns self
        self.assertIs(out1, x1)
        self.assertIs(out2, x2)
        self.assertIs(out3, x3)
        self.assertIs(out4, x4)

        # Verify output contains only 0s and 1s
        for out in [out1, out2, out3, out4]:
            self.assertTrue(paddle.all((out == 0) | (out == 1)).item())

        paddle.enable_static()


# Test kl_div compatibility (target -> label alias)
class TestKlDivAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(5, 10).astype("float32")
        self.np_target = np.random.rand(5, 10).astype("float32")
        self.shape_input = [5, 10]
        self.shape_target = [5, 10]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        target = paddle.to_tensor(self.np_target)

        # 1. Paddle positional arguments
        out1 = paddle.nn.functional.kl_div(input, target)

        # 2. Paddle keyword arguments
        out2 = paddle.nn.functional.kl_div(input=input, label=target)

        # 3. PyTorch keyword arguments (target alias)
        out3 = paddle.nn.functional.kl_div(input=input, target=target)

        # 4. Mixed arguments
        out4 = paddle.nn.functional.kl_div(
            input, target=target, reduction='mean'
        )

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=self.shape_input, dtype=self.dtype
            )
            target = paddle.static.data(
                name="target", shape=self.shape_target, dtype=self.dtype
            )

            # 1. Paddle positional arguments
            out1 = paddle.nn.functional.kl_div(input, target)
            # 2. Paddle keyword arguments
            out2 = paddle.nn.functional.kl_div(input=input, label=target)
            # 3. PyTorch keyword arguments (target alias)
            out3 = paddle.nn.functional.kl_div(input=input, target=target)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "target": self.np_target},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                self.assertEqual(out.shape, ())

        paddle.disable_static()


# Test hann_window compatibility
class TestHannWindowAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle Positional arguments
        out1 = paddle.hann_window(512)

        # 2. Paddle keyword arguments
        out2 = paddle.hann_window(window_length=512, periodic=True)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.hann_window(window_length=512, periodic=False)

        # 4. Mixed arguments
        out4 = paddle.hann_window(512, periodic=True)

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            self.assertEqual(out.shape, [512])

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            # 1. Paddle Positional arguments
            out1 = paddle.hann_window(512)
            # 2. Paddle keyword arguments
            out2 = paddle.hann_window(window_length=512)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.hann_window(window_length=512, periodic=True)

            exe = paddle.static.Executor()
            fetches = exe.run(main, feed={}, fetch_list=[out1, out2, out3])

            for out in fetches:
                self.assertEqual(out.shape, (512,))

        paddle.disable_static()


# Test paddle.float compatibility (dtype alias)
class TestFloatDtypeAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. paddle.float should be float32
        self.assertEqual(paddle.float, paddle.float32)

        # 2. Create tensor with paddle.float dtype
        x = paddle.to_tensor([1.0, 2.0], dtype=paddle.float)
        self.assertEqual(x.dtype, paddle.float32)

        # 3. Use in create_parameter
        param = paddle.create_parameter(shape=[2, 3], dtype=paddle.float)
        self.assertEqual(param.dtype, paddle.float32)

        paddle.enable_static()


# Test fmod_ compatibility (inplace)
class TestFmod_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([5.0, 7.0, 9.0]).astype("float32")
        self.np_y = np.array([2.0, 3.0, 4.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)

        # 1. Tensor method - positional
        x1 = paddle.to_tensor(self.np_x.copy())
        out1 = x1.fmod_(y)

        # 2. Tensor method - keyword
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = x2.fmod_(other=y)

        # 3. paddle function - positional
        x3 = paddle.to_tensor(self.np_x.copy())
        out3 = paddle.fmod_(x3, y)

        # 4. paddle function - keyword (input alias)
        x4 = paddle.to_tensor(self.np_x.copy())
        out4 = paddle.fmod_(input=x4, other=y)

        # 5. Mixed arguments
        x5 = paddle.to_tensor(self.np_x.copy())
        out5 = paddle.fmod_(x5, other=y)

        # Verify inplace operation returns self
        self.assertIs(out1, x1)
        self.assertIs(out2, x2)
        self.assertIs(out3, x3)
        self.assertIs(out4, x4)
        self.assertIs(out5, x5)

        # Verify result
        expected = np.fmod(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


# Test fill_diagonal_ compatibility (inplace)
class TestFillDiagonal_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.ones((4, 3)).astype("float32") * 2

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Tensor method - positional
        x1 = paddle.to_tensor(self.np_x.copy())
        out1 = x1.fill_diagonal_(1.0)

        # 2. Tensor method - keyword
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = x2.fill_diagonal_(fill_value=1.0)

        # 3. Mixed arguments
        x3 = paddle.to_tensor(self.np_x.copy())
        out3 = x3.fill_diagonal_(1.0, wrap=False)

        # Verify inplace operation returns self
        self.assertIs(out1, x1)
        self.assertIs(out2, x2)
        self.assertIs(out3, x3)

        # Verify all outputs
        for out in [out1, out2, out3]:
            self.assertEqual(out[0, 0].item(), 1.0)
            self.assertEqual(out[1, 1].item(), 1.0)
            self.assertEqual(out[2, 2].item(), 1.0)

        paddle.enable_static()


# Test weight_norm compatibility (module -> layer alias)
class TestWeightNormAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # Create a simple layer
        conv = paddle.nn.Conv2D(3, 5, 3)

        # 1. Paddle keyword arguments
        wn1 = paddle.nn.utils.weight_norm(layer=conv)

        # 2. PyTorch keyword arguments (module alias)
        conv2 = paddle.nn.Conv2D(3, 5, 3)
        wn2 = paddle.nn.utils.weight_norm(module=conv2)

        # 3. Paddle Positional arguments
        conv3 = paddle.nn.Conv2D(3, 5, 3)
        wn3 = paddle.nn.utils.weight_norm(conv3)

        # Verify all work correctly
        self.assertIsNotNone(wn1.weight_g)
        self.assertIsNotNone(wn2.weight_g)
        self.assertIsNotNone(wn3.weight_g)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            # 1. Paddle Positional arguments
            conv1 = paddle.nn.Conv2D(3, 5, 3)
            wn1 = paddle.nn.utils.weight_norm(conv1)
            # 2. PyTorch keyword arguments (module alias)
            conv2 = paddle.nn.Conv2D(3, 5, 3)
            wn2 = paddle.nn.utils.weight_norm(module=conv2)

            # Just verify it doesn't crash in static graph definition
            self.assertIsNotNone(wn1)
            self.assertIsNotNone(wn2)

        paddle.disable_static()


# Test resize_ compatibility (variable args support)
class TestResize_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x.copy())

        # 1. Paddle list/tuple argument
        out1 = x.resize_([2, 3])

        # 2. PyTorch variable args
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = x2.resize_(2, 3)

        # Verify both produce same shape
        self.assertEqual(out1.shape, [2, 3])
        self.assertEqual(out2.shape, [2, 3])

        paddle.enable_static()


# Test Flatten compatibility (start_dim/end_dim -> start_axis/stop_axis)
class TestFlattenAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3, 4, 5).astype("float32")
        self.shape = (2, 3, 4, 5)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle keyword arguments
        layer1 = paddle.nn.Flatten(start_axis=1, stop_axis=-1)
        out1 = layer1(x)

        # 2. PyTorch keyword arguments (dim aliases)
        layer2 = paddle.nn.Flatten(start_dim=1, end_dim=-1)
        out2 = layer2(x)

        # 3. PyTorch positional arguments
        layer3 = paddle.nn.Flatten(1, -1)
        out3 = layer3(x)

        # 4. Mixed arguments
        layer4 = paddle.nn.Flatten(start_dim=1, stop_axis=-1)
        out4 = layer4(x)

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            self.assertEqual(out.shape, [2, 60])

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype="float32")

            # 1. Paddle keyword arguments
            layer1 = paddle.nn.Flatten(start_axis=1, stop_axis=-1)
            out1 = layer1(x)

            # 2. PyTorch keyword arguments (dim aliases)
            layer2 = paddle.nn.Flatten(start_dim=1, end_dim=-1)
            out2 = layer2(x)

            # 3. PyTorch positional arguments
            layer3 = paddle.nn.Flatten(1, -1)
            out3 = layer3(x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                self.assertEqual(out.shape, (2, 60))

        paddle.disable_static()


# Test L1Loss compatibility (size_average/reduce parameters)
class TestL1LossAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3, 5).astype("float32")
        self.np_label = np.random.rand(3, 5).astype("float32")
        self.shape = (3, 5)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        label = paddle.to_tensor(self.np_label)

        # 1. Paddle keyword arguments
        loss1 = paddle.nn.L1Loss(reduction='mean')
        out1 = loss1(input, label)

        # 2. PyTorch keyword arguments (size_average, reduce)
        loss2 = paddle.nn.L1Loss(size_average=True, reduce=True)
        out2 = loss2(input, label)

        # 3. PyTorch size_average=False
        loss3 = paddle.nn.L1Loss(size_average=False, reduce=True)
        out3 = loss3(input, label)

        # 4. PyTorch reduce=False
        loss4 = paddle.nn.L1Loss(reduce=False)
        out4 = loss4(input, label)

        # 5. Mixed arguments
        loss5 = paddle.nn.L1Loss(size_average=True, reduction='mean')
        out5 = loss5(input, label)

        # Verify all outputs
        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        self.assertEqual(out3.shape, [])
        self.assertEqual(out4.shape, [3, 5])
        self.assertEqual(out5.shape, [])

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=self.shape, dtype="float32"
            )
            label = paddle.static.data(
                name="label", shape=self.shape, dtype="float32"
            )

            # 1. Paddle keyword arguments
            loss1 = paddle.nn.L1Loss(reduction='mean')
            out1 = loss1(input, label)

            # 2. PyTorch keyword arguments (size_average, reduce)
            loss2 = paddle.nn.L1Loss(size_average=True, reduce=True)
            out2 = loss2(input, label)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "label": self.np_label},
                fetch_list=[out1, out2],
            )

            for out in fetches:
                self.assertEqual(out.shape, ())

        paddle.disable_static()


# Test linalg.inv compatibility (A -> x alias)
class TestLinalgInvAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 2).astype("float32")
        # Make it invertible
        self.np_x = (np.eye(2) + 0.1 * self.np_x).astype("float32")
        self.shape = [2, 2]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.linalg.inv(x)

        # 2. Paddle keyword arguments
        out2 = paddle.linalg.inv(x=x)

        # 3. PyTorch keyword arguments (A alias)
        out3 = paddle.linalg.inv(A=x)

        # 4. Mixed arguments
        out4 = paddle.linalg.inv(x, name=None)

        # 5. out parameter test
        out5 = paddle.empty_like(x)
        paddle.linalg.inv(x, out=out5)

        # 6. Tensor method - args
        out6 = x.inverse()

        # 7. Alias paddle.inverse
        out7 = paddle.inverse(A=x)

        # Verify all outputs
        expected = np.linalg.inv(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.linalg.inv(x)
            # 2. Paddle keyword arguments
            out2 = paddle.linalg.inv(x=x)
            # 3. PyTorch keyword arguments (A alias)
            out3 = paddle.linalg.inv(A=x)
            # 4. Tensor method - args
            out4 = x.inverse()
            # 5. Alias paddle.inverse
            out5 = paddle.inverse(A=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            expected = np.linalg.inv(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)

        paddle.disable_static()


# Test det compatibility (paddle.det alias)
class TestDetAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 2).astype("float32")
        self.shape = [2, 2]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.linalg.det(x)

        # 2. Paddle keyword arguments
        out2 = paddle.linalg.det(x=x)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.linalg.det(input=x)

        # 4. Mixed arguments
        out4 = paddle.linalg.det(x, name=None)

        # 5. Tensor method - args
        out5 = x.det()

        # Verify all outputs
        expected = np.linalg.det(self.np_x)
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.linalg.det(x)
            # 2. Paddle keyword arguments
            out2 = paddle.linalg.det(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.linalg.det(input=x)
            # 4. Tensor method - args
            out4 = x.det()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            expected = np.linalg.det(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, expected, rtol=1e-5)

        paddle.disable_static()


# Test pinverse compatibility (paddle.pinverse alias)
class TestPinverseAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 2).astype("float32")
        self.shape = [3, 2]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. paddle.pinverse positional
        out1 = paddle.pinverse(x)

        # 2. paddle.pinverse keyword
        out2 = paddle.pinverse(x=x)

        # 3. paddle.pinverse with input alias
        out3 = paddle.pinverse(input=x)

        # 4. out parameter test
        out4 = paddle.empty([2, 3], dtype='float32')
        paddle.pinverse(x, out=out4)

        # 5. atol parameter test (keyword-only)
        out5 = paddle.pinverse(x, atol=1e-10)

        # 6. rtol parameter test (keyword-only)
        out6 = paddle.pinverse(x, rtol=1e-10)

        # 7. atol and rtol combined (keyword-only)
        out7 = paddle.pinverse(x, atol=1e-10, rtol=1e-10)

        # 8. Tensor method - args
        out8 = x.pinverse()

        # 9. hermitian=True with atol (need square matrix for hermitian)
        x_sq = paddle.to_tensor(np.random.rand(3, 3).astype("float32"))
        x_sym = x_sq @ x_sq.T + paddle.eye(3) * 0.5  # full rank
        out9 = paddle.pinverse(x_sym, hermitian=True, atol=1e-4)

        # 10. hermitian=True with rtol
        out10 = paddle.pinverse(x_sym, hermitian=True, rtol=1e-4)

        # 11. hermitian=True with both atol and rtol
        out11 = paddle.pinverse(x_sym, hermitian=True, atol=1e-4, rtol=1e-4)

        # Verify all outputs
        expected = np.linalg.pinv(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        expected_sym = np.linalg.pinv(x_sym.numpy(), hermitian=True)
        for out in [out9, out10, out11]:
            np.testing.assert_allclose(out.numpy(), expected_sym, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. paddle.pinverse positional
            out1 = paddle.pinverse(x)
            # 2. paddle.pinverse keyword
            out2 = paddle.pinverse(x=x)
            # 3. paddle.pinverse with input alias
            out3 = paddle.pinverse(input=x)
            # 4. Tensor method
            out4 = x.pinverse()
            # 5. atol parameter test (keyword-only)
            out5 = paddle.pinverse(x, atol=1e-10)
            # 6. rtol parameter test (keyword-only)
            out6 = paddle.pinverse(x, rtol=1e-10)
            # 7. atol and rtol combined (keyword-only)
            out7 = paddle.pinverse(x, atol=1e-10, rtol=1e-10)
            # 8. out parameter test
            out8 = paddle.static.data(
                name="out8", shape=[2, 3], dtype="float32"
            )
            paddle.pinverse(x, out=out8)

            # 9. hermitian=True with atol (square matrix)
            x_sym = paddle.static.data(
                name="x_sym", shape=[3, 3], dtype="float32"
            )
            out9 = paddle.pinverse(x_sym, hermitian=True, atol=1e-4)

            # 10. hermitian=True with rtol
            out10 = paddle.pinverse(x_sym, hermitian=True, rtol=1e-4)

            # 11. hermitian=True with both atol and rtol
            out11 = paddle.pinverse(x_sym, hermitian=True, atol=1e-4, rtol=1e-4)

            exe = paddle.static.Executor()
            np_x_sym = (
                self.np_x @ self.np_x.T
                + np.eye(3, dtype=self.dtype) * 0.5  # full rank
            )
            fetches = exe.run(
                main,
                feed={
                    "x": self.np_x,
                    "out8": np.empty([2, 3], dtype="float32"),
                    "x_sym": np_x_sym,
                },
                fetch_list=[
                    out1,
                    out2,
                    out3,
                    out4,
                    out5,
                    out6,
                    out7,
                    out8,
                    out9,
                    out10,
                    out11,
                ],
            )

            expected = np.linalg.pinv(self.np_x)
            for out in fetches[:8]:
                np.testing.assert_allclose(out, expected, rtol=1e-5)

            expected_sym = np.linalg.pinv(np_x_sym, hermitian=True)
            for out in fetches[8:]:
                np.testing.assert_allclose(out, expected_sym, rtol=1e-5)

        paddle.disable_static()


# Test addcdiv_ compatibility
class TestAddcdiv_InplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1.0, 2.0, 3.0]).astype("float32")
        self.np_t1 = np.array([4.0, 5.0, 6.0]).astype("float32")
        self.np_t2 = np.array([2.0, 2.0, 2.0]).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        t1 = paddle.to_tensor(self.np_t1)
        t2 = paddle.to_tensor(self.np_t2)

        # 1. Paddle Positional arguments
        x1 = paddle.to_tensor(self.np_x.copy())
        out1 = paddle.addcdiv(x1, t1, t2, 1.0)

        # 2. Paddle keyword arguments
        x2 = paddle.to_tensor(self.np_x.copy())
        out2 = paddle.addcdiv(input=x2, tensor1=t1, tensor2=t2, value=1.0)

        # 3. PyTorch keyword arguments (alias)
        x3 = paddle.to_tensor(self.np_x.copy())
        out3 = paddle.addcdiv(input=x3, tensor1=t1, tensor2=t2, value=1.0)

        # 4. Mixed arguments
        x4 = paddle.to_tensor(self.np_x.copy())
        out4 = paddle.addcdiv(x4, t1, tensor2=t2, value=1.0)

        # 5. out parameter test
        x5 = paddle.to_tensor(self.np_x.copy())
        out5 = paddle.empty_like(x5)
        paddle.addcdiv(x5, t1, t2, value=1.0, out=out5)

        # 6. Tensor method - args
        x6 = paddle.to_tensor(self.np_x.copy())
        out6 = x6.addcdiv_(t1, t2, value=1.0)

        # 7. Tensor method - kwargs
        x7 = paddle.to_tensor(self.np_x.copy())
        out7 = x7.addcdiv_(tensor1=t1, tensor2=t2, value=1.0)

        # Verify all outputs
        expected = self.np_x + 1.0 * (self.np_t1 / self.np_t2)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

        paddle.enable_static()


# Test imag compatibility (compat function)
class TestImagAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[1 + 6j, 2 + 5j], [3 + 4j, 5 + 2j]], dtype='complex64'
        )
        self.shape = [2, 2]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. paddle.imag positional
        out1 = paddle.imag(x)

        # 2. paddle.imag keyword
        out2 = paddle.imag(x=x)

        # 3. paddle.imag with input alias
        out3 = paddle.imag(input=x)

        # Verify outputs
        for out in [out1, out2, out3]:
            self.assertEqual(out.dtype, paddle.float32)
            np.testing.assert_allclose(out.numpy(), self.np_x.imag)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape, dtype="complex64"
            )

            # 1. paddle.imag positional
            out1 = paddle.imag(x)
            # 2. paddle.imag keyword
            out2 = paddle.imag(x=x)
            # 3. paddle.imag with input alias
            out3 = paddle.imag(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                np.testing.assert_allclose(out, self.np_x.imag)

        paddle.disable_static()


# Test real compatibility (compat function)
class TestRealAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[1 + 6j, 2 + 5j], [3 + 4j, 5 + 2j]], dtype='complex64'
        )
        self.shape = [2, 2]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. paddle.real positional
        out1 = paddle.real(x)

        # 2. paddle.real keyword
        out2 = paddle.real(x=x)

        # 3. paddle.real with input alias
        out3 = paddle.real(input=x)
        # Verify outputs
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(out.numpy(), self.np_x.real)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape, dtype="complex64"
            )

            # 1. paddle.real positional
            out1 = paddle.real(x)
            # 2. paddle.real keyword
            out2 = paddle.real(x=x)
            # 3. paddle.real with input alias
            out3 = paddle.real(input=x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                np.testing.assert_allclose(out, self.np_x.real)

        paddle.disable_static()


# Test nan_to_num compatibility (PyTorch parameter alias)
class TestNanToNumAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([np.nan, 0.3, np.inf, -np.inf], dtype="float32")
        self.shape = [4]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.nan_to_num(x, 0.0)

        # 2. Paddle keyword arguments
        out2 = paddle.nan_to_num(x=x, nan=0.0)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.nan_to_num(input=x, nan=0.0)

        # 4. Mixed arguments
        out4 = paddle.nan_to_num(x, nan=0.0, posinf=None)

        # 5. out parameter test
        out5 = paddle.empty_like(out1)
        paddle.nan_to_num(x, nan=0.0, out=out5)

        # 6. Tensor method - args
        out6 = x.nan_to_num(0.0)

        # 7. Tensor method - kwargs
        out7 = x.nan_to_num(nan=0.0)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), out1.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.nan_to_num(x, 0.0)
            # 2. Paddle keyword arguments
            out2 = paddle.nan_to_num(x=x, nan=0.0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.nan_to_num(input=x, nan=0.0)
            # 4. Tensor method - args
            out4 = x.nan_to_num(0.0)
            # 5. Tensor method - kwargs
            out5 = x.nan_to_num(nan=0.0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            for out in fetches:
                np.testing.assert_allclose(out, fetches[0])

        paddle.disable_static()


# Test randint_like compatibility (PyTorch parameter alias and new params)
class TestRandintLikeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.zeros(self.shape, dtype=self.dtype)

        # 1. Paddle Positional arguments
        out1 = paddle.randint_like(x, 0, 10)

        # 2. Paddle keyword arguments
        out2 = paddle.randint_like(x=x, low=0, high=10)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.randint_like(input=x, low=0, high=10)

        # 4. Mixed arguments
        out4 = paddle.randint_like(x, high=10)

        # 5. pin_memory parameter (keyword-only)
        out5 = paddle.randint_like(x, 0, 10, pin_memory=False)

        # 6. requires_grad parameter (keyword-only)
        out6 = paddle.randint_like(x, 0, 10, requires_grad=False)

        # 7. Both pin_memory and requires_grad (keyword-only)
        out7 = paddle.randint_like(
            x, 0, 10, pin_memory=False, requires_grad=False
        )

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            self.assertEqual(out.shape, x.shape)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.randint_like(x, 0, 10)
            # 2. Paddle keyword arguments
            out2 = paddle.randint_like(x=x, low=0, high=10)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.randint_like(input=x, low=0, high=10)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": np.zeros(self.shape, dtype=self.dtype)},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                self.assertEqual(out.shape, tuple(self.shape))

        paddle.disable_static()


# Test resize_as_ compatibility
class TestResizeAsAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3).astype("float32")
        self.np_y = np.random.rand(4, 5).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        x1 = paddle.to_tensor(self.np_x)
        out1 = paddle.resize_as_(x1, y)

        # 2. Paddle keyword arguments
        x2 = paddle.to_tensor(self.np_x)
        out2 = paddle.resize_as_(x=x2, y=y)

        # 3. Mixed arguments
        x3 = paddle.to_tensor(self.np_x)
        out3 = paddle.resize_as_(x3, y=y)

        # 4. Tensor method - args
        x4 = paddle.to_tensor(self.np_x)
        out4 = x4.resize_as_(y)

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            self.assertEqual(out.shape, y.shape)

        paddle.enable_static()


# Test huber_loss compatibility (alias for smooth_l1_loss)
class TestHuberLossAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.input = np.random.rand(3, 5).astype("float32")
        self.target = np.random.rand(3, 5).astype("float32")
        self.shape = (3, 5)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.input)
        target = paddle.to_tensor(self.target)

        # 1. Paddle Positional arguments
        out1 = paddle.nn.functional.huber_loss(input, target)

        # 2. Paddle keyword arguments
        out2 = paddle.nn.functional.huber_loss(input=input, label=target)

        # 3. PyTorch keyword arguments (target alias)
        out3 = paddle.nn.functional.huber_loss(input=input, target=target)

        # 4. Mixed arguments
        out4 = paddle.nn.functional.huber_loss(input, target=target, delta=1.0)

        # 5. smooth_l1_loss should be equivalent
        out5 = paddle.nn.functional.smooth_l1_loss(input, target)

        # Verify outputs are equivalent
        np.testing.assert_allclose(out1.numpy(), out2.numpy())
        np.testing.assert_allclose(out1.numpy(), out3.numpy())
        np.testing.assert_allclose(out1.numpy(), out4.numpy())
        np.testing.assert_allclose(out1.numpy(), out5.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=self.shape, dtype="float32"
            )
            target = paddle.static.data(
                name="target", shape=self.shape, dtype="float32"
            )

            # 1. Paddle positional arguments
            out1 = paddle.nn.functional.huber_loss(
                input, target, reduction='none'
            )
            # 2. Paddle keyword arguments
            out2 = paddle.nn.functional.huber_loss(
                input=input, label=target, reduction='none'
            )
            # 3. PyTorch keyword arguments (target alias)
            out3 = paddle.nn.functional.huber_loss(
                input=input, target=target, reduction='none'
            )

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.input, "target": self.target},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                self.assertEqual(out.shape, (3, 5))

        paddle.disable_static()


# Test fmod compatibility (alias for remainder/mod)
class TestFmodAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([5.0, 7.0, 9.0], dtype="float32")
        self.np_y = np.array([2.0, 3.0, 4.0], dtype="float32")
        self.shape = [3]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle Positional arguments
        out1 = paddle.fmod(x, y)

        # 2. Paddle keyword arguments
        out2 = paddle.fmod(x=x, y=y)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.fmod(input=x, other=y)

        # 4. Mixed arguments
        out4 = paddle.fmod(x, other=y)

        # 5. out parameter test
        out5 = paddle.empty_like(x)
        paddle.fmod(x, y, out=out5)

        # 6. Tensor method - args
        out6 = x.fmod(y)

        # 7. Tensor method - kwargs
        out7 = x.fmod(other=y)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), out1.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.fmod(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.fmod(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.fmod(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.fmod(y)
            # 5. Tensor method - kwargs
            out5 = x.fmod(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            for out in fetches:
                np.testing.assert_allclose(out, fetches[0])

        paddle.disable_static()


# Test absolute compatibility (alias for abs)
class TestAbsoluteAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([-1.0, 2.0, -3.0], dtype="float32")
        self.shape = [3]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        out1 = paddle.absolute(x)

        # 2. Paddle keyword arguments
        out2 = paddle.absolute(x=x)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.absolute(input=x)

        # 4. Mixed arguments
        out4 = paddle.absolute(x, name=None)

        # 5. out parameter test
        out5 = paddle.empty_like(x)
        paddle.absolute(x, out=out5)

        # 6. Tensor method - args
        out6 = x.absolute()

        # 7. Alias paddle.abs
        out7 = paddle.abs(x)

        # Verify all outputs
        expected = np.abs(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(out.numpy(), expected)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # 1. Paddle Positional arguments
            out1 = paddle.absolute(x)
            # 2. Paddle keyword arguments
            out2 = paddle.absolute(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.absolute(input=x)
            # 4. Alias paddle.abs
            out4 = paddle.abs(x)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main, feed={"x": self.np_x}, fetch_list=[out1, out2, out3, out4]
            )

            expected = np.abs(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, expected)

        paddle.disable_static()


# Test assert_allclose compatibility
class TestAssertAllcloseAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([1.0, 2.0, 3.0])

        # 1. Should not raise
        paddle.testing.assert_allclose(x, y)

        # 2. With tolerances
        paddle.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-8)

        # 4. Non-Tensor inputs (isinstance branches)
        paddle.testing.assert_allclose(
            paddle.to_tensor([1.0, 2.0, 3.0]).numpy(),  # actual is ndarray
            paddle.to_tensor([1.0, 2.0, 3.0]).numpy(),  # expected is ndarray
        )

        # 5. Non-Tensor actual with list
        paddle.testing.assert_allclose(
            [1.0, 2.0, 3.0], paddle.to_tensor([1.0, 2.0, 3.0])
        )

        # 6. Non-Tensor expected with list
        paddle.testing.assert_allclose(
            paddle.to_tensor([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0]
        )

        # 3. Should raise on mismatch
        z = paddle.to_tensor([1.0, 2.0, 4.0])
        with self.assertRaises(AssertionError):
            paddle.testing.assert_allclose(x, z)

        paddle.enable_static()


# Test GRU compatibility
class TestGRUAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.input_size = 16
        self.hidden_size = 32
        self.num_layers = 2
        self.batch_size = 4
        self.seq_len = 23
        self.shape_x = [self.batch_size, self.seq_len, self.input_size]
        self.shape_h = [self.num_layers, self.batch_size, self.hidden_size]
        self.dtype = "float32"

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        np.random.seed(2025)
        np_x = np.random.randn(*self.shape_x).astype(self.dtype)
        np_h = np.random.randn(*self.shape_h).astype(self.dtype)
        x = paddle.to_tensor(np_x)
        prev_h = paddle.to_tensor(np_h)

        # --- Forward, with bias (num_layers=self.num_layers) ---
        ref = paddle.nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        ref_sd = ref.state_dict()
        ref_y, ref_h = ref(x, prev_h)

        # 1. Paddle positional arguments
        rnn1 = paddle.nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        rnn1.set_state_dict(ref_sd)
        y1, h1 = rnn1(x, prev_h)
        np.testing.assert_allclose(y1.numpy(), ref_y.numpy(), rtol=1e-5)
        np.testing.assert_allclose(h1.numpy(), ref_h.numpy(), rtol=1e-5)

        # 2. Paddle keyword arguments
        rnn2 = paddle.nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        rnn2.set_state_dict(ref_sd)
        y2, h2 = rnn2(x, prev_h)
        np.testing.assert_allclose(y2.numpy(), ref_y.numpy(), rtol=1e-5)
        np.testing.assert_allclose(h2.numpy(), ref_h.numpy(), rtol=1e-5)

        # 3. PyTorch positional arguments (bias=True at 4th position)
        rnn3 = paddle.nn.GRU(
            self.input_size, self.hidden_size, self.num_layers, True
        )
        rnn3.set_state_dict(ref_sd)
        y3, h3 = rnn3(x, prev_h)
        np.testing.assert_allclose(y3.numpy(), ref_y.numpy(), rtol=1e-5)
        np.testing.assert_allclose(h3.numpy(), ref_h.numpy(), rtol=1e-5)

        # 4. PyTorch keyword arguments (bias, batch_first, bidirectional)
        rnn4 = paddle.nn.GRU(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
        )
        rnn4.set_state_dict(ref_sd)
        y4, h4 = rnn4(x, prev_h)
        np.testing.assert_allclose(y4.numpy(), ref_y.numpy(), rtol=1e-5)
        np.testing.assert_allclose(h4.numpy(), ref_h.numpy(), rtol=1e-5)

        # --- bias=False ---
        ref_nb = paddle.nn.GRU(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bias_ih_attr=False,
            bias_hh_attr=False,
        )
        ref_nb_sd = ref_nb.state_dict()
        ref_y_nb, ref_h_nb = ref_nb(x, prev_h)

        # 5. bias parameter test (bias=False)
        rnn5 = paddle.nn.GRU(
            self.input_size, self.hidden_size, self.num_layers, bias=False
        )
        rnn5.set_state_dict(ref_nb_sd)
        y5, h5 = rnn5(x, prev_h)
        np.testing.assert_allclose(y5.numpy(), ref_y_nb.numpy(), rtol=1e-5)
        np.testing.assert_allclose(h5.numpy(), ref_h_nb.numpy(), rtol=1e-5)

        # 6. device parameter test (constructor only)
        paddle.nn.GRU(self.input_size, self.hidden_size, device="cpu")

        # 7. dtype parameter test (constructor only)
        paddle.nn.GRU(self.input_size, self.hidden_size, dtype="float32")

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x",
                shape=self.shape_x,
                dtype=self.dtype,
            )
            prev_h = paddle.static.data(
                name="prev_h",
                shape=self.shape_h,
                dtype=self.dtype,
            )

            # 1. Paddle positional arguments
            rnn1 = paddle.nn.GRU(
                self.input_size, self.hidden_size, self.num_layers
            )
            y1, h1 = rnn1(x, prev_h)

            # 2. PyTorch keyword arguments (bias, batch_first, bidirectional)
            rnn2 = paddle.nn.GRU(
                self.input_size,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=True,
                batch_first=True,
                bidirectional=False,
            )
            y2, h2 = rnn2(x, prev_h)

            exe = paddle.static.Executor()
            exe.run(startup)
            # Just verify it runs in static mode
            fetches = exe.run(
                main,
                feed={
                    "x": np.random.randn(*self.shape_x).astype(self.dtype),
                    "prev_h": np.random.randn(*self.shape_h).astype(self.dtype),
                },
                fetch_list=[y1, h1, y2, h2],
            )

            self.assertEqual(
                fetches[0].shape,
                (self.batch_size, self.seq_len, self.hidden_size),
            )

        paddle.disable_static()


# Test set_default_tensor_type compatibility
class TestSetDefaultTensorTypeAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # Save original dtype
        original_dtype = paddle.get_default_dtype()

        # ========== float32 tests  ==========
        paddle.set_default_tensor_type(paddle.FloatTensor)
        self.assertEqual(paddle.get_default_dtype(), "float32")
        paddle.set_default_tensor_type(paddle.cuda.FloatTensor)
        self.assertEqual(paddle.get_default_dtype(), "float32")

        paddle.set_default_tensor_type("paddle.FloatTensor")
        self.assertEqual(paddle.get_default_dtype(), "float32")
        paddle.set_default_tensor_type("paddle.cuda.FloatTensor")
        self.assertEqual(paddle.get_default_dtype(), "float32")

        paddle.set_default_tensor_type("torch.FloatTensor")
        self.assertEqual(paddle.get_default_dtype(), "float32")
        paddle.set_default_tensor_type("torch.cuda.FloatTensor")
        self.assertEqual(paddle.get_default_dtype(), "float32")

        # ========== float64 tests  ==========
        paddle.set_default_tensor_type(paddle.DoubleTensor)
        self.assertEqual(paddle.get_default_dtype(), "float64")
        paddle.set_default_tensor_type(paddle.cuda.DoubleTensor)
        self.assertEqual(paddle.get_default_dtype(), "float64")

        paddle.set_default_tensor_type("paddle.DoubleTensor")
        self.assertEqual(paddle.get_default_dtype(), "float64")
        paddle.set_default_tensor_type("paddle.cuda.DoubleTensor")
        self.assertEqual(paddle.get_default_dtype(), "float64")

        paddle.set_default_tensor_type("torch.DoubleTensor")
        self.assertEqual(paddle.get_default_dtype(), "float64")
        paddle.set_default_tensor_type("torch.cuda.DoubleTensor")
        self.assertEqual(paddle.get_default_dtype(), "float64")

        # ========== float16 tests (10 formats) ==========
        paddle.set_default_tensor_type(paddle.HalfTensor)
        self.assertEqual(paddle.get_default_dtype(), "float16")
        paddle.set_default_tensor_type(paddle.cuda.HalfTensor)
        self.assertEqual(paddle.get_default_dtype(), "float16")

        paddle.set_default_tensor_type("paddle.HalfTensor")
        self.assertEqual(paddle.get_default_dtype(), "float16")
        paddle.set_default_tensor_type("paddle.cuda.HalfTensor")
        self.assertEqual(paddle.get_default_dtype(), "float16")

        paddle.set_default_tensor_type("torch.HalfTensor")
        self.assertEqual(paddle.get_default_dtype(), "float16")
        paddle.set_default_tensor_type("torch.cuda.HalfTensor")
        self.assertEqual(paddle.get_default_dtype(), "float16")

        # ========== bfloat16 tests (10 formats) ==========
        paddle.set_default_tensor_type(paddle.BFloat16Tensor)
        self.assertEqual(paddle.get_default_dtype(), "bfloat16")
        paddle.set_default_tensor_type(paddle.cuda.BFloat16Tensor)
        self.assertEqual(paddle.get_default_dtype(), "bfloat16")

        paddle.set_default_tensor_type("paddle.BFloat16Tensor")
        self.assertEqual(paddle.get_default_dtype(), "bfloat16")
        paddle.set_default_tensor_type("paddle.cuda.BFloat16Tensor")
        self.assertEqual(paddle.get_default_dtype(), "bfloat16")

        paddle.set_default_tensor_type("torch.BFloat16Tensor")
        self.assertEqual(paddle.get_default_dtype(), "bfloat16")
        paddle.set_default_tensor_type("torch.cuda.BFloat16Tensor")
        self.assertEqual(paddle.get_default_dtype(), "bfloat16")

        # ========== TypeError branches  ==========
        # Invalid tensor type name (not in dtype_map)
        with self.assertRaises(TypeError):
            paddle.set_default_tensor_type("torch.IntTensor")

        # Passing dtype instead of tensor type
        with self.assertRaises(TypeError):
            paddle.set_default_tensor_type(paddle.float32)

        # Restore original dtype
        paddle.set_default_dtype(original_dtype)

        paddle.enable_static()


# Test PackedSequence compatibility
class TestPackedSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.data = np.random.rand(10, 5).astype("float32")
        self.batch_sizes = np.array([3, 3, 2, 1, 1], dtype="int64")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Create PackedSequence with positional arguments
        data_tensor = paddle.to_tensor(self.data)
        batch_sizes_tensor = paddle.to_tensor(self.batch_sizes)
        packed1 = paddle.nn.utils.rnn.PackedSequence(
            data_tensor, batch_sizes_tensor
        )

        # 2. Create PackedSequence with keyword arguments
        packed2 = paddle.nn.utils.rnn.PackedSequence(
            data=data_tensor, batch_sizes=batch_sizes_tensor
        )

        # 3. Create PackedSequence with sorted_indices
        sorted_indices = paddle.to_tensor([2, 0, 1], dtype="int64")
        packed3 = paddle.nn.utils.rnn.PackedSequence(
            data_tensor, batch_sizes_tensor, sorted_indices=sorted_indices
        )

        # 4. Create PackedSequence with all parameters
        unsorted_indices = paddle.to_tensor([1, 2, 0], dtype="int64")
        packed4 = paddle.nn.utils.rnn.PackedSequence(
            data=data_tensor,
            batch_sizes=batch_sizes_tensor,
            sorted_indices=sorted_indices,
            unsorted_indices=unsorted_indices,
        )

        # Verify all instances
        for packed in [packed1, packed2, packed3, packed4]:
            self.assertEqual(packed.data.shape, [10, 5])
            self.assertEqual(packed.batch_sizes.shape, [5])
            np.testing.assert_allclose(packed.data.numpy(), self.data)
            np.testing.assert_array_equal(
                packed.batch_sizes.numpy(), self.batch_sizes
            )

        # Verify sorted_indices and unsorted_indices
        np.testing.assert_array_equal(packed3.sorted_indices.numpy(), [2, 0, 1])
        np.testing.assert_array_equal(packed4.sorted_indices.numpy(), [2, 0, 1])
        np.testing.assert_array_equal(
            packed4.unsorted_indices.numpy(), [1, 2, 0]
        )

        # 5. Test properties
        # Note: is_cuda returns True if data is on GPU, False on CPU
        # Since the test may run on GPU or CPU, we just check it's a boolean
        self.assertIsInstance(packed1.is_cuda, bool)
        self.assertIsInstance(packed1.is_pinned, bool)

        # 6. Test to() method
        # When called with dtype change, data dtype changes but indices stay int64
        packed_dtype = packed4.to(dtype=paddle.float64)
        self.assertEqual(packed_dtype.data.dtype, paddle.float64)
        self.assertEqual(packed_dtype.sorted_indices.dtype, paddle.int64)
        self.assertEqual(packed_dtype.unsorted_indices.dtype, paddle.int64)
        self.assertEqual(packed_dtype.unsorted_indices.dtype, paddle.int64)

        # 7. Test dtype conversion methods
        packed_double = packed1.double()
        self.assertEqual(packed_double.data.dtype, paddle.float64)

        packed_float = packed1.float()
        self.assertEqual(packed_float.data.dtype, paddle.float32)

        packed_half = packed1.half()
        self.assertEqual(packed_half.data.dtype, paddle.float16)

        packed_long = packed1.long()
        self.assertEqual(packed_long.data.dtype, paddle.int64)

        packed_int = packed1.int()
        self.assertEqual(packed_int.data.dtype, paddle.int32)

        packed_short = packed1.short()
        self.assertEqual(packed_short.data.dtype, paddle.int16)

        packed_char = packed1.char()
        self.assertEqual(packed_char.data.dtype, paddle.int8)

        packed_byte = packed1.byte()
        self.assertEqual(packed_byte.data.dtype, paddle.uint8)

        # 8. Test pin_memory
        if paddle.is_compiled_with_cuda():
            packed_pinned = packed1.pin_memory()
            self.assertIsInstance(
                packed_pinned, paddle.nn.utils.rnn.PackedSequence
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            data = paddle.static.data(
                name="data", shape=[10, 5], dtype="float32"
            )
            batch_sizes = paddle.static.data(
                name="batch_sizes", shape=[5], dtype="int64"
            )

            # 1. Create PackedSequence with positional arguments
            packed1 = paddle.nn.utils.rnn.PackedSequence(data, batch_sizes)

            # 2. Create PackedSequence with keyword arguments
            packed2 = paddle.nn.utils.rnn.PackedSequence(
                data=data, batch_sizes=batch_sizes
            )

            # Verify Variables are preserved
            self.assertEqual(packed1.data.name, data.name)
            self.assertEqual(packed1.batch_sizes.name, batch_sizes.name)
            self.assertEqual(packed2.data.name, data.name)
            self.assertEqual(packed2.batch_sizes.name, batch_sizes.name)

        paddle.disable_static()


# Test invert_permutation compatibility
class TestInvertPermutationAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Basic test with positional argument
        perm = paddle.to_tensor([2, 0, 1], dtype="int64")
        inv_perm1 = paddle.nn.utils.rnn.invert_permutation(perm)
        np.testing.assert_array_equal(inv_perm1.numpy(), [1, 2, 0])

        # 2. Test with keyword argument
        inv_perm2 = paddle.nn.utils.rnn.invert_permutation(permutation=perm)
        np.testing.assert_array_equal(inv_perm2.numpy(), [1, 2, 0])

        # 3. Test with None input
        result = paddle.nn.utils.rnn.invert_permutation(None)
        self.assertIsNone(result)

        # 4. Test with different permutation
        perm2 = paddle.to_tensor([0, 1, 2, 3], dtype="int64")
        inv_perm3 = paddle.nn.utils.rnn.invert_permutation(perm2)
        np.testing.assert_array_equal(inv_perm3.numpy(), [0, 1, 2, 3])

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            perm = paddle.static.data(name="perm", shape=[3], dtype="int64")
            inv_perm1 = paddle.nn.utils.rnn.invert_permutation(perm)
            inv_perm2 = paddle.nn.utils.rnn.invert_permutation(permutation=perm)

            exe = paddle.static.Executor()
            exe.run(startup)
            fetches = exe.run(
                main,
                feed={"perm": np.array([2, 0, 1], dtype="int64")},
                fetch_list=[inv_perm1, inv_perm2],
            )

            for out in fetches:
                np.testing.assert_array_equal(out, [1, 2, 0])

        paddle.disable_static()


# Test pack_padded_sequence compatibility
class TestPackPaddedSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_seq = np.random.rand(5, 3, 10).astype("float32")
        self.lengths = [5, 3, 2]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        seq = paddle.to_tensor(self.np_seq)

        # 1. Paddle Positional arguments
        packed1 = paddle.nn.utils.rnn.pack_padded_sequence(
            seq, paddle.to_tensor(self.lengths)
        )

        # 2. Paddle keyword arguments
        packed2 = paddle.nn.utils.rnn.pack_padded_sequence(
            input=seq, lengths=paddle.to_tensor(self.lengths)
        )

        # 3. PyTorch keyword arguments (batch_first=False)
        packed3 = paddle.nn.utils.rnn.pack_padded_sequence(
            input=seq, lengths=self.lengths, batch_first=False
        )

        # 4. With batch_first=True
        seq_batch_first = seq.transpose([1, 0, 2])
        packed4 = paddle.nn.utils.rnn.pack_padded_sequence(
            seq_batch_first, self.lengths, batch_first=True
        )

        # 5. With enforce_sorted=False
        packed5 = paddle.nn.utils.rnn.pack_padded_sequence(
            seq, self.lengths, enforce_sorted=False
        )

        # 6. With enforce_sorted=False (complex case)
        seq_unsorted = paddle.to_tensor(
            [[1, 2, 0], [3, 0, 0], [4, 5, 6]], dtype='float32'
        )
        packed6 = paddle.nn.utils.rnn.pack_padded_sequence(
            seq_unsorted, [2, 1, 3], batch_first=True, enforce_sorted=False
        )

        # 7. Mixed arguments (positional + keyword)
        packed7 = paddle.nn.utils.rnn.pack_padded_sequence(
            seq, enforce_sorted=True, lengths=paddle.to_tensor(self.lengths)
        )

        # Verify all outputs
        for packed in [packed1, packed2, packed3, packed5, packed7]:
            self.assertIsInstance(packed, paddle.nn.utils.rnn.PackedSequence)
            self.assertTrue(hasattr(packed, 'data'))
            self.assertTrue(hasattr(packed, 'batch_sizes'))
            # All these should be identical
            np.testing.assert_allclose(
                packed.data.numpy(), packed1.data.numpy()
            )
            np.testing.assert_array_equal(
                packed.batch_sizes.numpy(), packed1.batch_sizes.numpy()
            )

        # packed4 (batch_first=True) should be same as packed1
        np.testing.assert_allclose(packed4.data.numpy(), packed1.data.numpy())
        np.testing.assert_array_equal(
            packed4.batch_sizes.numpy(), packed1.batch_sizes.numpy()
        )

        # Verify unsorted case
        self.assertIsNotNone(packed6.sorted_indices)
        self.assertIsNotNone(packed6.unsorted_indices)
        # Verify packed6 data (manual check for small unsorted case)
        # seq_unsorted = [[1, 2, 0], [3, 0, 0], [4, 5, 6]], lengths = [2, 1, 3]
        # sorted by lengths: [4, 5, 6] (3), [1, 2, 0] (2), [3, 0, 0] (1)
        # packed: [4, 1, 3, 5, 2, 6]
        expected_data = [4, 1, 3, 5, 2, 6]
        expected_batch_sizes = [3, 2, 1]
        np.testing.assert_array_equal(
            packed6.data.numpy().flatten(), expected_data
        )
        np.testing.assert_array_equal(
            packed6.batch_sizes.numpy(), expected_batch_sizes
        )

        paddle.enable_static()


# Test pad_packed_sequence compatibility
class TestPadPackedSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_seq = np.random.rand(5, 3, 10).astype("float32")
        self.lengths = [5, 3, 2]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        seq = paddle.to_tensor(self.np_seq)
        lengths_tensor = paddle.to_tensor(self.lengths)

        # Create packed sequence
        packed = paddle.nn.utils.rnn.pack_padded_sequence(seq, lengths_tensor)

        # 1. Paddle Positional arguments
        padded1, lengths1 = paddle.nn.utils.rnn.pad_packed_sequence(packed)

        # 2. Paddle keyword arguments
        padded2, lengths2 = paddle.nn.utils.rnn.pad_packed_sequence(
            sequence=packed
        )

        # 3. With batch_first=True
        packed_bf = paddle.nn.utils.rnn.pack_padded_sequence(
            seq.transpose([1, 0, 2]), lengths_tensor, batch_first=True
        )
        padded3, lengths3 = paddle.nn.utils.rnn.pad_packed_sequence(
            packed_bf, batch_first=True
        )

        # 4. With padding_value
        padded4, lengths4 = paddle.nn.utils.rnn.pad_packed_sequence(
            packed, padding_value=1.0
        )

        # 5. With total_length
        padded5, lengths5 = paddle.nn.utils.rnn.pad_packed_sequence(
            packed, total_length=10
        )

        # 6. Mixed arguments (positional + keyword)
        padded6, lengths6 = paddle.nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=False
        )

        # 7. Another padding_value test
        padded7, lengths7 = paddle.nn.utils.rnn.pad_packed_sequence(
            packed, padding_value=-1.0
        )

        # 8. batch_first=True on packed (which was batch_first=False)
        padded8, lengths8 = paddle.nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True
        )

        # 9. TypeError when sequence is not PackedSequence
        with self.assertRaises(TypeError):
            paddle.nn.utils.rnn.pad_packed_sequence("not_a_packed_sequence")

        # Verify outputs numerical correctness
        # For default padding (0.0)
        expected_padded = self.np_seq.copy()
        # For batch 0, length is 5. No padding.
        # For batch 1, length is 3. Elements at time 3, 4 should be 0.
        expected_padded[3:, 1, :] = 0.0
        # For batch 2, length is 2. Elements at time 2, 3, 4 should be 0.
        expected_padded[2:, 2, :] = 0.0

        for padded, lengths in [
            (padded1, lengths1),
            (padded2, lengths2),
            (padded6, lengths6),
        ]:
            self.assertEqual(padded.shape, [5, 3, 10])
            np.testing.assert_array_equal(lengths.numpy(), self.lengths)
            np.testing.assert_allclose(padded.numpy(), expected_padded)

        # For padding_value=1.0 (padded4)
        expected_padded_1 = self.np_seq.copy()
        expected_padded_1[3:, 1, :] = 1.0
        expected_padded_1[2:, 2, :] = 1.0
        np.testing.assert_allclose(padded4.numpy(), expected_padded_1)

        # For padding_value=-1.0 (padded7)
        expected_padded_neg1 = self.np_seq.copy()
        expected_padded_neg1[3:, 1, :] = -1.0
        expected_padded_neg1[2:, 2, :] = -1.0
        np.testing.assert_allclose(padded7.numpy(), expected_padded_neg1)

        # For batch_first=True (padded3)
        expected_padded_bf = self.np_seq.transpose([1, 0, 2]).copy()
        expected_padded_bf[1, 3:, :] = 0.0
        expected_padded_bf[2, 2:, :] = 0.0
        self.assertEqual(padded3.shape, [3, 5, 10])
        np.testing.assert_allclose(padded3.numpy(), expected_padded_bf)

        # For total_length=10 (padded5)
        self.assertEqual(padded5.shape, [10, 3, 10])
        np.testing.assert_allclose(padded5.numpy()[:5], expected_padded)
        np.testing.assert_allclose(padded5.numpy()[5:], 0.0)

        # For padded8 (batch_first=True on packed which was batch_first=False)
        self.assertEqual(padded8.shape, [3, 5, 10])
        np.testing.assert_allclose(padded8.numpy(), expected_padded_bf)

        paddle.enable_static()


# Test pad_sequence compatibility
class TestPadSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.a = np.random.rand(25, 300).astype("float32")
        self.b = np.random.rand(22, 300).astype("float32")
        self.c = np.random.rand(15, 300).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        a = paddle.to_tensor(self.a)
        b = paddle.to_tensor(self.b)
        c = paddle.to_tensor(self.c)
        sequences = [a, b, c]

        # 1. Paddle Positional arguments
        padded1 = paddle.nn.utils.rnn.pad_sequence(sequences)

        # 2. Paddle keyword arguments
        padded2 = paddle.nn.utils.rnn.pad_sequence(sequences=sequences)

        # 3. PyTorch keyword arguments (batch_first=True)
        padded3 = paddle.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        # 4. With padding_value
        padded4 = paddle.nn.utils.rnn.pad_sequence(sequences, padding_value=1.0)

        # 5. With padding_side='left'
        padded5 = paddle.nn.utils.rnn.pad_sequence(
            sequences, padding_side='left'
        )

        # 6. Mixed arguments (positional + keyword)
        padded6 = paddle.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0.0
        )

        # 7. TypeError when sequences is a string (not a valid iterable of Tensors)
        with self.assertRaises(TypeError):
            paddle.nn.utils.rnn.pad_sequence("not_a_valid_input")

        # 8. ValueError for invalid padding_side
        with self.assertRaises(ValueError):
            paddle.nn.utils.rnn.pad_sequence(sequences, padding_side='invalid')

        # 9. PyTorch-style tuple input (should work the same as list)
        padded_tuple = paddle.nn.utils.rnn.pad_sequence(
            (a, b, c), batch_first=True
        )
        self.assertEqual(padded_tuple.shape, [3, 25, 300])
        np.testing.assert_allclose(padded_tuple.numpy(), padded3.numpy())

        # Verify outputs
        self.assertEqual(padded1.shape, [25, 3, 300])
        self.assertEqual(padded2.shape, [25, 3, 300])
        self.assertEqual(padded3.shape, [3, 25, 300])
        self.assertEqual(padded4.shape, [25, 3, 300])
        self.assertEqual(padded5.shape, [25, 3, 300])
        self.assertEqual(padded6.shape, [3, 25, 300])

        # Numerical checks
        np.testing.assert_allclose(padded1.numpy()[:, 0, :], self.a)
        np.testing.assert_allclose(padded1.numpy()[:22, 1, :], self.b)
        np.testing.assert_allclose(padded1.numpy()[22:, 1, :], 0.0)
        np.testing.assert_allclose(padded1.numpy()[:15, 2, :], self.c)
        np.testing.assert_allclose(padded1.numpy()[15:, 2, :], 0.0)

        # padding_value=1.0
        np.testing.assert_allclose(padded4.numpy()[22:, 1, :], 1.0)

        # batch_first=True
        np.testing.assert_allclose(padded3.numpy()[0, :, :], self.a)

        # padding_side='left'
        # padded5 shape [25, 3, 300]
        # a: [25, 300] -> no padding
        # b: [22, 300] -> 3 elements padded at top
        # c: [15, 300] -> 10 elements padded at top
        np.testing.assert_allclose(padded5.numpy()[:, 0, :], self.a)
        np.testing.assert_allclose(padded5.numpy()[3:, 1, :], self.b)
        np.testing.assert_allclose(padded5.numpy()[:3, 1, :], 0.0)
        np.testing.assert_allclose(padded5.numpy()[10:, 2, :], self.c)
        np.testing.assert_allclose(padded5.numpy()[:10, 2, :], 0.0)

        paddle.enable_static()


# Test unpad_sequence compatibility
class TestUnpadSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.a = np.random.rand(25, 300).astype("float32")
        self.b = np.random.rand(22, 300).astype("float32")
        self.c = np.random.rand(15, 300).astype("float32")

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        a = paddle.to_tensor(self.a)
        b = paddle.to_tensor(self.b)
        c = paddle.to_tensor(self.c)
        sequences = [a, b, c]

        # Pad then unpad
        padded = paddle.nn.utils.rnn.pad_sequence(sequences)
        lengths = paddle.to_tensor([v.shape[0] for v in sequences])

        # 1. Paddle Positional arguments
        unpadded1 = paddle.nn.utils.rnn.unpad_sequence(padded, lengths)

        # 2. Paddle keyword arguments
        unpadded2 = paddle.nn.utils.rnn.unpad_sequence(
            padded_sequences=padded, lengths=lengths
        )

        # 3. PyTorch keyword arguments (batch_first=True)
        padded_bf = paddle.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True
        )
        unpadded3 = paddle.nn.utils.rnn.unpad_sequence(
            padded_bf, lengths, batch_first=True
        )

        # 4. Mixed arguments (positional + keyword)
        unpadded4 = paddle.nn.utils.rnn.unpad_sequence(
            padded, batch_first=False, lengths=lengths
        )

        # Verify outputs
        for i, (original, unpadded) in enumerate(zip(sequences, unpadded1)):
            np.testing.assert_allclose(original.numpy(), unpadded.numpy())

        for i, (original, unpadded) in enumerate(zip(sequences, unpadded2)):
            np.testing.assert_allclose(original.numpy(), unpadded.numpy())

        for i, (original, unpadded) in enumerate(zip(sequences, unpadded3)):
            np.testing.assert_allclose(original.numpy(), unpadded.numpy())

        for i, (original, unpadded) in enumerate(zip(sequences, unpadded4)):
            np.testing.assert_allclose(original.numpy(), unpadded.numpy())

        paddle.enable_static()


# Test pack_sequence compatibility
class TestPackSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle Positional arguments (sorted)
        a = paddle.to_tensor([1, 2, 3])
        b = paddle.to_tensor([4, 5])
        c = paddle.to_tensor([6])
        packed1 = paddle.nn.utils.rnn.pack_sequence([a, b, c])

        # 2. Paddle keyword arguments
        packed2 = paddle.nn.utils.rnn.pack_sequence(sequences=[a, b, c])

        # 3. PyTorch keyword arguments (enforce_sorted=False)
        d = paddle.to_tensor([1, 2, 3])
        e = paddle.to_tensor([4])
        f = paddle.to_tensor([5, 6])
        packed3 = paddle.nn.utils.rnn.pack_sequence(
            [d, e, f], enforce_sorted=False
        )

        # 4. Mixed arguments (positional + keyword)
        g = paddle.to_tensor([7, 8, 9])
        h = paddle.to_tensor([10, 11])
        packed4 = paddle.nn.utils.rnn.pack_sequence([g, h], enforce_sorted=True)

        # Verify outputs
        for packed in [packed1, packed2]:
            self.assertIsInstance(packed, paddle.nn.utils.rnn.PackedSequence)
            np.testing.assert_array_equal(
                packed.data.numpy().flatten(), [1, 4, 6, 2, 5, 3]
            )
            np.testing.assert_array_equal(packed.batch_sizes.numpy(), [3, 2, 1])

        # packed3: d=[1,2,3], e=[4], f=[5,6]. enforce_sorted=False
        # sorted by len: [1,2,3] (3), [5,6] (2), [4] (1)
        # data: [1, 5, 4, 2, 6, 3]
        # batch_sizes: [3, 2, 1]
        self.assertIsNotNone(packed3.sorted_indices)
        self.assertIsNotNone(packed3.unsorted_indices)
        np.testing.assert_array_equal(
            packed3.data.numpy().flatten(), [1, 5, 4, 2, 6, 3]
        )
        np.testing.assert_array_equal(packed3.batch_sizes.numpy(), [3, 2, 1])

        # packed4: g=[7,8,9], h=[10,11]. enforce_sorted=True
        # data: [7, 10, 8, 11, 9]
        # batch_sizes: [2, 2, 1]
        self.assertIsInstance(packed4, paddle.nn.utils.rnn.PackedSequence)
        np.testing.assert_array_equal(
            packed4.data.numpy().flatten(), [7, 10, 8, 11, 9]
        )
        np.testing.assert_array_equal(packed4.batch_sizes.numpy(), [2, 2, 1])

        paddle.enable_static()


# Test unpack_sequence compatibility
class TestUnpackSequenceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # Create sequences
        a = paddle.to_tensor([1, 2, 3])
        b = paddle.to_tensor([4, 5])
        c = paddle.to_tensor([6])
        sequences = [a, b, c]

        # Pack and unpack
        packed = paddle.nn.utils.rnn.pack_sequence(sequences)

        # 1. Paddle Positional arguments
        unpacked1 = paddle.nn.utils.rnn.unpack_sequence(packed)

        # 2. Paddle keyword arguments
        unpacked2 = paddle.nn.utils.rnn.unpack_sequence(packed_sequences=packed)

        # 3. Mixed arguments (positional + keyword)
        unpacked3 = paddle.nn.utils.rnn.unpack_sequence(packed_sequences=packed)

        # Verify outputs
        for i, (original, unpacked) in enumerate(zip(sequences, unpacked1)):
            np.testing.assert_array_equal(original.numpy(), unpacked.numpy())

        for i, (original, unpacked) in enumerate(zip(sequences, unpacked2)):
            np.testing.assert_array_equal(original.numpy(), unpacked.numpy())

        for i, (original, unpacked) in enumerate(zip(sequences, unpacked3)):
            np.testing.assert_array_equal(original.numpy(), unpacked.numpy())

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
