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


# Test block_diag compatibility
class TestBlockDiagAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3).astype('float32')
        self.np_y = np.random.rand(3, 4).astype('float32')
        self.np_z = np.random.rand(1, 2).astype('float32')

    def _ref_block_diag(self, *arrays):
        import scipy.linalg

        return scipy.linalg.block_diag(*arrays)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        z = paddle.to_tensor(self.np_z)

        # 1. Paddle positional arguments
        out1 = paddle.block_diag([x, y, z])
        # 2. Paddle keyword arguments
        out2 = paddle.block_diag(inputs=[x, y, z])
        # 3. PyTorch positional arguments
        out3 = paddle.block_diag(x, y, z)

        ref_out = self._ref_block_diag(self.np_x, self.np_y, self.np_z)
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3], dtype='float32')
            y = paddle.static.data(name="y", shape=[3, 4], dtype='float32')
            z = paddle.static.data(name="z", shape=[1, 2], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.block_diag([x, y, z])
            # 2. Paddle keyword arguments
            out2 = paddle.block_diag(inputs=[x, y, z])
            # 3. PyTorch positional arguments
            out3 = paddle.block_diag(x, y, z)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y, "z": self.np_z},
                fetch_list=[out1, out2, out3],
            )

            ref_out = self._ref_block_diag(self.np_x, self.np_y, self.np_z)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test broadcast_tensors compatibility
class TestBroadcastTensorsAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 1).astype('float32')
        self.np_y = np.random.rand(1, 4).astype('float32')
        self.np_z = np.random.rand(3, 4).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        z = paddle.to_tensor(self.np_z)

        # 1. Paddle positional arguments
        outs1 = paddle.broadcast_tensors([x, y, z])
        # 2. Paddle keyword arguments
        outs2 = paddle.broadcast_tensors(input=[x, y, z])
        # 3. PyTorch positional arguments
        outs3 = paddle.broadcast_tensors(x, y, z)

        # Verify all outputs
        ref_x = np.broadcast_to(self.np_x, [3, 4])
        ref_y = np.broadcast_to(self.np_y, [3, 4])
        ref_z = np.broadcast_to(self.np_z, [3, 4])

        refs = [ref_x, ref_y, ref_z]
        for outs in [outs1, outs2, outs3]:
            self.assertEqual(len(outs), 3)
            for i, ref in enumerate(refs):
                np.testing.assert_allclose(ref, outs[i].numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 1], dtype='float32')
            y = paddle.static.data(name="y", shape=[1, 4], dtype='float32')
            z = paddle.static.data(name="z", shape=[3, 4], dtype='float32')

            # 1. Paddle positional arguments
            outs1 = paddle.broadcast_tensors([x, y, z])
            # 2. Paddle keyword arguments
            outs2 = paddle.broadcast_tensors(input=[x, y, z])
            # 3. PyTorch positional arguments
            outs3 = paddle.broadcast_tensors(x, y, z)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y, "z": self.np_z},
                fetch_list=[
                    outs1[0],
                    outs1[1],
                    outs1[2],
                    outs2[0],
                    outs2[1],
                    outs2[2],
                    outs3[0],
                    outs3[1],
                    outs3[2],
                ],
            )

            ref_x = np.broadcast_to(self.np_x, [3, 4])
            ref_y = np.broadcast_to(self.np_y, [3, 4])
            ref_z = np.broadcast_to(self.np_z, [3, 4])

            refs = [ref_x, ref_y, ref_z] * 3
            for i, ref in enumerate(refs):
                np.testing.assert_allclose(fetches[i], ref)


# Test cartesian_prod compatibility
class TestCartesianProdAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1, 2, 3], dtype='int64')
        self.np_y = np.array([4, 5, 6, 7], dtype='int64')

    def compute_ref_output(self):
        # Compute cartesian product
        x_grid, y_grid = np.meshgrid(self.np_x, self.np_y, indexing='ij')
        return np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.cartesian_prod([x, y])
        # 2. Paddle keyword arguments
        out2 = paddle.cartesian_prod(x=[x, y])
        # 3. PyTorch positional arguments
        out3 = paddle.cartesian_prod(x, y)

        # Verify outputs
        ref_out = self.compute_ref_output()
        for out in [out1, out2, out3]:
            np.testing.assert_array_equal(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3], dtype='int64')
            y = paddle.static.data(name="y", shape=[4], dtype='int64')

            # 1. Paddle positional arguments
            out1 = paddle.cartesian_prod([x, y])
            # 2. Paddle keyword arguments
            out2 = paddle.cartesian_prod(x=[x, y])
            # 3. PyTorch positional arguments
            out3 = paddle.cartesian_prod(x, y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            ref_out = self.compute_ref_output()
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test copysign compatibility
class TestCopysignAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(3, 4).astype('float32')
        self.np_y = np.random.randn(3, 4).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.copysign(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.copysign(x=x, y=y)
        # 3. PyTorch keyword arguments
        out3 = paddle.copysign(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.copysign(x, other=y)
        # 5. out parameter
        out5 = paddle.empty_like(x)
        paddle.copysign(x, y, out=out5)
        # 6. Class method positional arguments
        out6 = x.copysign(y)
        # 7. Class method keyword arguments
        out7 = x.copysign(y=y)

        # Verify all outputs
        ref_out = np.copysign(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype='float32')
            y = paddle.static.data(name="y", shape=[3, 4], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.copysign(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.copysign(x=x, y=y)
            # 3. PyTorch keyword arguments
            out3 = paddle.copysign(input=x, other=y)
            # 4. Class method positional arguments
            out4 = x.copysign(y)
            # 5. Class method keyword arguments
            out5 = x.copysign(y=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.copysign(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test Tensor.copysign_ inplace compatibility
class TestTensorCopysignInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(3, 4).astype('float32')
        self.np_y = np.random.randn(3, 4).astype('float32')

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)
        ref_out = np.copysign(self.np_x, self.np_y)

        # 1. Class method positional arguments
        out1 = paddle.to_tensor(self.np_x)
        out1.copysign_(y)
        # 2. Class method keyword arguments
        out2 = paddle.to_tensor(self.np_x)
        out2.copysign_(y=y)
        # 3. PyTorch keyword arguments
        out3 = paddle.to_tensor(self.np_x)
        out3.copysign_(other=y)

        for out in [out1, out2, out3]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()


# Test Tensor.geometric_ inplace compatibility
class TestTensorGeometricInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()

        # 1. Class method positional arguments
        out1 = paddle.empty([10000], dtype='float32')
        out1.geometric_(0.3)
        # 2. Class method keyword arguments
        out2 = paddle.empty([10000], dtype='float32')
        out2.geometric_(p=0.3)
        # 3. PyTorch keyword arguments
        out3 = paddle.empty([10000], dtype='float32')
        out3.geometric_(probs=0.3)

        for out in [out1, out2, out3]:
            self.assertEqual(out.shape, [10000])
            self.assertTrue((out.numpy() > 0).all())

        paddle.enable_static()


# Test Tensor.hypot_ inplace compatibility
class TestTensorHypotInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 4).astype('float32') + 1.0
        self.np_y = np.random.rand(3, 4).astype('float32') + 1.0

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)
        ref_out = np.hypot(self.np_x, self.np_y)

        # 1. Class method positional arguments
        out1 = paddle.to_tensor(self.np_x)
        out1.hypot_(y)
        # 2. Class method keyword arguments
        out2 = paddle.to_tensor(self.np_x)
        out2.hypot_(y=y)
        # 3. PyTorch keyword arguments
        out3 = paddle.to_tensor(self.np_x)
        out3.hypot_(other=y)

        for out in [out1, out2, out3]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()


# Test index_fill compatibility
class TestIndexFillAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(5, 6).astype('float32')
        self.np_index = np.array([1, 3, 4], dtype='int64')

    def compute_ref_output(self):
        ref = self.np_x.copy()
        ref[:, self.np_index] = -1.0
        return ref

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        index = paddle.to_tensor(self.np_index)

        # 1. Paddle positional arguments
        out1 = paddle.index_fill(x, index, 1, -1.0)
        # 2. Paddle keyword arguments
        out2 = paddle.index_fill(x=x, index=index, axis=1, value=-1.0)
        # 3. PyTorch positional arguments
        out3 = paddle.index_fill(x, 1, index, -1.0)
        # 4. PyTorch keyword arguments
        out4 = paddle.index_fill(input=x, dim=1, index=index, value=-1.0)
        # 5. Mixed arguments
        out5 = paddle.index_fill(x, index, axis=1, value=-1.0)
        # 6. Class method positional arguments
        out6 = x.index_fill(index, 1, -1.0)
        # 7. Class method keyword arguments
        out7 = x.index_fill(index=index, axis=1, value=-1.0)

        # Verify all outputs
        ref_out = self.compute_ref_output()
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            index = paddle.static.data(name="index", shape=[3], dtype='int64')

            # 1. Paddle positional arguments
            out1 = paddle.index_fill(x, index, 1, -1.0)
            # 2. Paddle keyword arguments
            out2 = paddle.index_fill(x=x, index=index, axis=1, value=-1.0)
            # 3. PyTorch positional arguments
            out3 = paddle.index_fill(x, 1, index, -1.0)
            # 4. PyTorch keyword arguments
            out4 = paddle.index_fill(input=x, dim=1, index=index, value=-1.0)
            # 5. Class method positional arguments
            out5 = x.index_fill(index, 1, -1.0)
            # 6. Class method keyword arguments
            out6 = x.index_fill(index=index, axis=1, value=-1.0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "index": self.np_index},
                fetch_list=[out1, out2, out3, out4, out5, out6],
            )

            ref_out = self.compute_ref_output()
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


@unittest.skipIf(
    paddle.is_compiled_with_xpu(),
    "skip xpu which not support index_fill_ (which use stride)",
)
# Test Tensor.index_fill_ inplace compatibility
class TestTensorIndexFillInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(5, 6).astype('float32')
        self.np_index = np.array([1, 3, 4], dtype='int64')

    def compute_ref_output(self):
        ref = self.np_x.copy()
        ref[:, self.np_index] = -1.0
        return ref

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        index = paddle.to_tensor(self.np_index)
        ref_out = self.compute_ref_output()

        # 1. Class method positional arguments
        out1 = paddle.to_tensor(self.np_x)
        out1.index_fill_(index, 1, -1.0)
        # 2. Class method keyword arguments
        out2 = paddle.to_tensor(self.np_x)
        out2.index_fill_(index=index, axis=1, value=-1.0)
        # 3. PyTorch positional arguments
        out3 = paddle.to_tensor(self.np_x)
        out3.index_fill_(1, index, -1.0)
        # 4. PyTorch keyword arguments
        out4 = paddle.to_tensor(self.np_x)
        out4.index_fill_(dim=1, index=index, value=-1.0)
        # 5. Mixed arguments
        out5 = paddle.to_tensor(self.np_x)
        out5.index_fill_(index, axis=1, value=-1.0)

        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)


# Test cross compatibility
class TestCrossAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 3, 3).astype('float32')
        self.np_y = np.random.rand(3, 3, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments (all positional: x, y, axis）
        out1 = paddle.cross(x, y, 1)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.cross(x=x, y=y, axis=1)
        # 3. PyTorch keyword arguments (using aliases input, other, dim)
        out3 = paddle.cross(input=x, other=y, dim=1)
        # 4. Mixed arguments
        out4 = paddle.cross(x, y=y, axis=1)
        # 5. Class method positional arguments
        out5 = x.cross(y, 1)
        # 6. Class method keyword arguments
        out6 = x.cross(y=y, axis=1)

        # Verify all outputs
        ref_out = np.cross(self.np_x, self.np_y, axisa=1, axisb=1, axisc=1)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 3, 3], dtype='float32')
            y = paddle.static.data(name="y", shape=[3, 3, 3], dtype='float32')

            # 1. Paddle positional arguments (all positional: x, y, axis）
            out1 = paddle.cross(x, y, 1)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.cross(x=x, y=y, axis=1)
            # 3. PyTorch keyword arguments (using aliases input, other, dim)
            out3 = paddle.cross(input=x, other=y, dim=1)
            # 4. Class method positional arguments
            out4 = x.cross(y, 1)
            # 5. Class method keyword arguments
            out5 = x.cross(y=y, axis=1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            ref_out = np.cross(self.np_x, self.np_y, axisa=1, axisb=1, axisc=1)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test dist compatibility
class TestDistAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 2).astype('float32')
        self.np_y = np.random.rand(2, 2).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments (all positional: x, y, p）
        out1 = paddle.dist(x, y, 2.0)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.dist(x=x, y=y, p=2.0)
        # 3. PyTorch keyword arguments (using aliases input and other)
        out3 = paddle.dist(input=x, other=y, p=2.0)
        # 4. Mixed arguments
        out4 = paddle.dist(x, y, p=2.0)
        # 5. Class method positional arguments
        out5 = x.dist(y, 2.0)
        # 6. Class method keyword arguments
        out6 = x.dist(y=y, p=2.0)

        # Verify all outputs
        ref_out = float(np.linalg.norm((self.np_x - self.np_y).flatten()))
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 2], dtype='float32')
            y = paddle.static.data(name="y", shape=[2, 2], dtype='float32')

            # 1. Paddle positional arguments (all positional: x, y, p）
            out1 = paddle.dist(x, y, 2.0)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.dist(x=x, y=y, p=2.0)
            # 3. PyTorch keyword arguments (using aliases input and other)
            out3 = paddle.dist(input=x, other=y, p=2.0)
            # 4. Class method positional arguments
            out4 = x.dist(y, 2.0)
            # 5. Class method keyword arguments
            out5 = x.dist(y=y, p=2.0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            ref_out = float(np.linalg.norm((self.np_x - self.np_y).flatten()))
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test flip compatibility
class TestFlipAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 2, 2).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.flip(x, [0, 1])
        # 2. Paddle keyword arguments
        out2 = paddle.flip(x=x, axis=[0, 1])
        # 3. PyTorch keyword arguments (using aliases input and dims)
        out3 = paddle.flip(input=x, dims=[0, 1])
        # 4. Mixed arguments
        out4 = paddle.flip(x, axis=[0, 1])
        # 5. Class method positional arguments
        out5 = x.flip([0, 1])
        # 6. Class method keyword arguments
        out6 = x.flip(axis=[0, 1])

        # Verify all outputs
        ref_out = np.flip(self.np_x, axis=[0, 1])
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 2, 2], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.flip(x, [0, 1])
            # 2. Paddle keyword arguments
            out2 = paddle.flip(x=x, axis=[0, 1])
            # 3. PyTorch keyword arguments (using aliases input and dims)
            out3 = paddle.flip(input=x, dims=[0, 1])
            # 4. Class method positional arguments
            out4 = x.flip([0, 1])
            # 5. Class method keyword arguments
            out5 = x.flip(axis=[0, 1])

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            ref_out = np.flip(self.np_x, axis=[0, 1])
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test count_nonzero compatibility
class TestCountNonzeroAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(-1, 2, [3, 4, 5]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_axis = np.count_nonzero(self.np_x, axis=1, keepdims=True)

        # 1. Paddle positional arguments
        out1 = paddle.count_nonzero(x, 1, True)
        # 2. Paddle keyword arguments
        out2 = paddle.count_nonzero(x=x, axis=1, keepdim=True)
        # 3. PyTorch keyword arguments
        out3 = paddle.count_nonzero(input=x, dim=1, keepdim=True)
        # 4. Mixed arguments
        out4 = paddle.count_nonzero(x, axis=1, keepdim=True)
        # 5. Class method positional arguments
        out5 = x.count_nonzero(1, True)
        # 6. Class method keyword arguments
        out6 = x.count_nonzero(dim=1, keepdim=True)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), ref_axis)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4, 5], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.count_nonzero(x, 1, True)
            # 2. Paddle keyword arguments
            out2 = paddle.count_nonzero(x=x, axis=1, keepdim=True)
            # 3. PyTorch keyword arguments
            out3 = paddle.count_nonzero(input=x, dim=1, keepdim=True)
            # 4. Class method positional arguments
            out4 = x.count_nonzero(1, True)
            # 5. Class method keyword arguments
            out5 = x.count_nonzero(dim=1, keepdim=True)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref = np.count_nonzero(self.np_x, axis=1, keepdims=True)
            for out in fetches:
                np.testing.assert_allclose(out, ref)


# Test renorm compatibility
class TestRenormAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 2, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments (all positional: x, p, axis, max_norm）
        out1 = paddle.renorm(x, 1.0, 2, 2.05)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.renorm(x=x, p=1.0, axis=2, max_norm=2.05)
        # 3. PyTorch keyword arguments (using aliases input, dim, maxnorm)
        out3 = paddle.renorm(input=x, p=1.0, dim=2, maxnorm=2.05)
        # 4. Mixed arguments
        out4 = paddle.renorm(x, p=1.0, axis=2, max_norm=2.05)
        # 5. Class method positional arguments
        out5 = x.renorm(1.0, 2, 2.05)
        # 6. Class method keyword arguments
        out6 = x.renorm(p=1.0, axis=2, max_norm=2.05)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 2, 3], dtype='float32')

            # 1. Paddle positional arguments (all positional: x, p, axis, max_norm）
            out1 = paddle.renorm(x, 1.0, 2, 2.05)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.renorm(x=x, p=1.0, axis=2, max_norm=2.05)
            # 3. PyTorch keyword arguments (using aliases input, dim, maxnorm)
            out3 = paddle.renorm(input=x, p=1.0, dim=2, maxnorm=2.05)
            # 4. Class method positional arguments
            out4 = x.renorm(1.0, 2, 2.05)
            # 5. Class method keyword arguments
            out5 = x.renorm(p=1.0, axis=2, max_norm=2.05)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            # Verify all outputs
            for out in fetches[1:]:
                np.testing.assert_allclose(out, fetches[0], rtol=1e-5)


# Test renorm_ inplace compatibility
class TestRenormInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 2, 3).astype('float32')

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        ref_x = self.np_x.copy()

        # 1. Class method positional arguments
        out1 = paddle.to_tensor(ref_x)
        out1.renorm_(1.0, 2, 2.05)
        # 2. Class method keyword arguments
        out2 = paddle.to_tensor(ref_x)
        out2.renorm_(p=1.0, axis=2, max_norm=2.05)
        # 3. PyTorch keyword arguments
        out3 = paddle.to_tensor(ref_x)
        out3.renorm_(p=1.0, dim=2, maxnorm=2.05)

        for out in [out1, out2, out3]:
            np.testing.assert_allclose(out.numpy(), out1.numpy(), rtol=1e-5)

        paddle.enable_static()


# Test unique compatibility
class TestUniqueAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.x_1d = np.array([3, 1, 2, 1, 3]).astype('int64')
        self.x_2d = np.array([[2, 1, 3], [3, 0, 1], [2, 1, 3]]).astype('int64')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_1d)

        # 1. Paddle positional arguments (all positional: x, return_index, return_inverse, return_counts, axis, dtype, sorted）
        out1 = paddle.unique(x, False, False, False, None, 'int64', True)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.unique(
            x=x,
            return_index=False,
            return_inverse=False,
            return_counts=False,
            axis=None,
            dtype='int64',
            sorted=True,
        )
        # 3. PyTorch keyword arguments (using aliases input and dim)
        out3 = paddle.unique(input=x, sorted=True)
        # 4. Mixed arguments (positional + keyword)
        out4 = paddle.unique(x, sorted=False)
        # 5. Class method positional arguments
        out5 = x.unique()
        # 6. Class method keyword arguments
        out6 = x.unique(sorted=True)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(out1.numpy(), out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5], dtype='int64')
            # 1. Paddle positional arguments (all positional arguments)
            out1 = paddle.unique(x, False, False, False, None, 'int64', True)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.unique(
                x=x,
                return_index=False,
                return_inverse=False,
                return_counts=False,
                axis=None,
                dtype='int64',
                sorted=True,
            )
            # 3. PyTorch keyword arguments (using aliases)
            out3 = paddle.unique(input=x, sorted=True)
            # 4. Class method positional arguments
            out4 = x.unique()
            # 5. Class method keyword arguments
            out5 = x.unique(sorted=True)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.x_1d},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            for i in range(1, len(fetches)):
                np.testing.assert_array_equal(fetches[0], fetches[i])


class TestCloneAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 4).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.clone(x)
        # 2. Paddle keyword arguments
        out2 = paddle.clone(x=x)
        # 3. PyTorch keyword arguments
        out3 = paddle.clone(input=x)
        # 4. Mixed arguments
        # clone only has one parameter x, mixed arguments not applicable
        # 5. Class method positional arguments
        out4 = x.clone()
        # 6. Class method keyword arguments
        # clone class method has no parameters, keyword arguments not applicable

        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), self.np_x)
            self.assertIsNot(out, x)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.clone(x)
            # 2. Paddle keyword arguments
            out2 = paddle.clone(x=x)
            # 3. PyTorch keyword arguments
            out3 = paddle.clone(input=x)
            # 4. Class method positional arguments
            out4 = x.clone()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs match input
            for out in fetches:
                np.testing.assert_allclose(out, self.np_x)


class TestHsplitAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x_2d = np.random.rand(7, 8).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x_2d = paddle.to_tensor(self.np_x_2d)

        # 1. Paddle positional arguments
        out1 = paddle.hsplit(x_2d, 2)
        # 2. Paddle keyword arguments
        out2 = paddle.hsplit(x=x_2d, num_or_indices=2)
        # 3. PyTorch keyword arguments
        out3 = paddle.hsplit(input=x_2d, indices=2)
        # 4. Mixed arguments
        out4 = paddle.hsplit(x_2d, num_or_indices=2)
        # 5. Class method positional arguments
        out5 = x_2d.hsplit(2)
        # 6. Class method keyword arguments
        out6 = x_2d.hsplit(num_or_indices=2)

        ref_out = np.array_split(self.np_x_2d, 2, axis=1)
        for out in [out1, out2, out3, out4, out5, out6]:
            self.assertEqual(len(out), 2)
            for ref, out_item in zip(ref_out, out):
                np.testing.assert_allclose(ref, out_item.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x_2d = paddle.static.data(
                name="x_2d", shape=[7, 8], dtype='float32'
            )

            # 1. Paddle positional arguments
            out1 = paddle.hsplit(x_2d, 2)
            # 2. Paddle keyword arguments
            out2 = paddle.hsplit(x=x_2d, num_or_indices=2)
            # 3. PyTorch keyword arguments
            out3 = paddle.hsplit(input=x_2d, indices=2)
            # 4. Class method positional arguments
            out4 = x_2d.hsplit(2)
            # 5. Class method keyword arguments
            out5 = x_2d.hsplit(num_or_indices=2)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x_2d": self.np_x_2d},
                fetch_list=[
                    out1[0],
                    out1[1],
                    out2[0],
                    out2[1],
                    out3[0],
                    out3[1],
                    out4[0],
                    out4[1],
                    out5[0],
                    out5[1],
                ],
            )

            ref_out = np.array_split(self.np_x_2d, 2, axis=1)
            for i in range(0, 10, 2):
                np.testing.assert_allclose(fetches[i], ref_out[0])
                np.testing.assert_allclose(fetches[i + 1], ref_out[1])


class TestDsplitAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x_3d = np.random.rand(7, 6, 8).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x_3d = paddle.to_tensor(self.np_x_3d)

        # 1. Paddle positional arguments
        out1 = paddle.dsplit(x_3d, 2)
        # 2. Paddle keyword arguments
        out2 = paddle.dsplit(x=x_3d, num_or_indices=2)
        # 3. PyTorch keyword arguments
        out3 = paddle.dsplit(input=x_3d, indices=2)
        # 4. Mixed arguments
        out4 = paddle.dsplit(x_3d, num_or_indices=2)
        # 5. Class method positional arguments
        out5 = x_3d.dsplit(2)
        # 6. Class method keyword arguments
        out6 = x_3d.dsplit(num_or_indices=2)

        ref_out = np.array_split(self.np_x_3d, 2, axis=2)
        for out in [out1, out2, out3, out4, out5, out6]:
            self.assertEqual(len(out), 2)
            for ref, out_item in zip(ref_out, out):
                np.testing.assert_allclose(ref, out_item.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x_3d = paddle.static.data(
                name="x_3d", shape=[7, 6, 8], dtype='float32'
            )

            # 1. Paddle positional arguments
            out1 = paddle.dsplit(x_3d, 2)
            # 2. Paddle keyword arguments
            out2 = paddle.dsplit(x=x_3d, num_or_indices=2)
            # 3. PyTorch keyword arguments
            out3 = paddle.dsplit(input=x_3d, indices=2)
            # 4. Class method positional arguments
            out4 = x_3d.dsplit(2)
            # 5. Class method keyword arguments
            out5 = x_3d.dsplit(num_or_indices=2)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x_3d": self.np_x_3d},
                fetch_list=[
                    out1[0],
                    out1[1],
                    out2[0],
                    out2[1],
                    out3[0],
                    out3[1],
                    out4[0],
                    out4[1],
                    out5[0],
                    out5[1],
                ],
            )

            ref_out = np.array_split(self.np_x_3d, 2, axis=2)
            for i in range(0, 10, 2):
                np.testing.assert_allclose(fetches[i], ref_out[0])
                np.testing.assert_allclose(fetches[i + 1], ref_out[1])


class TestVsplitAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x_2d = np.random.rand(8, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x_2d = paddle.to_tensor(self.np_x_2d)

        # 1. Paddle positional arguments
        out1 = paddle.vsplit(x_2d, 2)
        # 2. Paddle keyword arguments
        out2 = paddle.vsplit(x=x_2d, num_or_indices=2)
        # 3. PyTorch keyword arguments
        out3 = paddle.vsplit(input=x_2d, indices=2)
        # 4. Mixed arguments
        out4 = paddle.vsplit(x_2d, num_or_indices=2)
        # 5. Class method positional arguments
        out5 = x_2d.vsplit(2)
        # 6. Class method keyword arguments
        out6 = x_2d.vsplit(num_or_indices=2)

        ref_out = np.array_split(self.np_x_2d, 2, axis=0)
        for out in [out1, out2, out3, out4, out5, out6]:
            self.assertEqual(len(out), 2)
            for ref, out_item in zip(ref_out, out):
                np.testing.assert_allclose(ref, out_item.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x_2d = paddle.static.data(
                name="x_2d", shape=[8, 6], dtype='float32'
            )

            # 1. Paddle positional arguments
            out1 = paddle.vsplit(x_2d, 2)
            # 2. Paddle keyword arguments
            out2 = paddle.vsplit(x=x_2d, num_or_indices=2)
            # 3. PyTorch keyword arguments
            out3 = paddle.vsplit(input=x_2d, indices=2)
            # 4. Class method positional arguments
            out4 = x_2d.vsplit(2)
            # 5. Class method keyword arguments
            out5 = x_2d.vsplit(num_or_indices=2)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x_2d": self.np_x_2d},
                fetch_list=[
                    out1[0],
                    out1[1],
                    out2[0],
                    out2[1],
                    out3[0],
                    out3[1],
                    out4[0],
                    out4[1],
                    out5[0],
                    out5[1],
                ],
            )

            ref_out = np.array_split(self.np_x_2d, 2, axis=0)
            for i in range(0, 10, 2):
                np.testing.assert_allclose(fetches[i], ref_out[0])
                np.testing.assert_allclose(fetches[i + 1], ref_out[1])


# Test hstack compatibility
class TestHstackAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.inputs = [
            np.random.rand(2, 3).astype('float32'),
            np.random.rand(2, 4).astype('float32'),
        ]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        tensors = [paddle.to_tensor(inp) for inp in self.inputs]

        # 1. Paddle positional arguments
        out1 = paddle.hstack(tensors)
        # 2. Paddle keyword arguments
        out2 = paddle.hstack(x=tensors)
        # 3. PyTorch keyword arguments
        out3 = paddle.hstack(tensors=tensors)
        # 4. Mixed arguments (only one parameter, mixed not applicable)

        ref_out = np.hstack(tuple(inp for inp in self.inputs))
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-8
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        shapes = [[2, 3], [2, 4]]
        with paddle.static.program_guard(main, startup):
            static_tensors = []
            feed_dict = {}
            for i, (shape, inp) in enumerate(zip(shapes, self.inputs)):
                static_tensor = paddle.static.data(
                    name=f"x{i}", shape=shape, dtype='float32'
                )
                static_tensors.append(static_tensor)
                feed_dict[f"x{i}"] = inp

            # 1. Paddle positional arguments
            out1 = paddle.hstack(static_tensors)
            # 2. Paddle keyword arguments
            out2 = paddle.hstack(x=static_tensors)
            # 3. PyTorch keyword arguments
            out3 = paddle.hstack(tensors=static_tensors)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main, feed=feed_dict, fetch_list=[out1, out2, out3]
            )
            ref_out = np.hstack(tuple(inp for inp in self.inputs))
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-8)


class TestVstackAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.inputs = [
            np.random.rand(2, 3).astype('float32'),
            np.random.rand(3, 3).astype('float32'),
        ]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        tensors = [paddle.to_tensor(inp) for inp in self.inputs]
        # 1. Paddle positional arguments
        out1 = paddle.vstack(tensors)
        # 2. Paddle keyword arguments
        out2 = paddle.vstack(x=tensors)
        # 3. PyTorch keyword arguments
        out3 = paddle.vstack(tensors=tensors)

        ref_out = np.vstack(tuple(inp for inp in self.inputs))
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-8
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        shapes = [[2, 3], [3, 3]]
        with paddle.static.program_guard(main, startup):
            static_tensors = []
            feed_dict = {}
            for i, (shape, inp) in enumerate(zip(shapes, self.inputs)):
                static_tensor = paddle.static.data(
                    name=f"x{i}", shape=shape, dtype='float32'
                )
                static_tensors.append(static_tensor)
                feed_dict[f"x{i}"] = inp

            # 1. Paddle positional arguments
            out1 = paddle.vstack(static_tensors)
            # 2. Paddle keyword arguments
            out2 = paddle.vstack(x=static_tensors)
            # 3. PyTorch keyword arguments
            out3 = paddle.vstack(tensors=static_tensors)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main, feed=feed_dict, fetch_list=[out1, out2, out3]
            )

            ref_out = np.vstack(tuple(inp for inp in self.inputs))
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-8)


# Test dstack compatibility
class TestDstackAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.inputs = [
            np.random.rand(2, 3, 4).astype('float32'),
            np.random.rand(2, 3, 4).astype('float32'),
        ]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        tensors = [paddle.to_tensor(inp) for inp in self.inputs]
        # 1. Paddle positional arguments
        out1 = paddle.dstack(tensors)
        # 2. Paddle keyword arguments
        out2 = paddle.dstack(x=tensors)
        # 3. PyTorch keyword arguments
        out3 = paddle.dstack(tensors=tensors)

        # Verify all outputs
        ref_out = np.dstack(tuple(inp for inp in self.inputs))
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-8
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        shapes = [[2, 3, 4], [2, 3, 4]]
        with paddle.static.program_guard(main, startup):
            static_tensors = []
            feed_dict = {}
            for i, (shape, inp) in enumerate(zip(shapes, self.inputs)):
                static_tensor = paddle.static.data(
                    name=f"x{i}", shape=shape, dtype='float32'
                )
                static_tensors.append(static_tensor)
                feed_dict[f"x{i}"] = inp

            # 1. Paddle positional arguments
            out1 = paddle.dstack(static_tensors)
            # 2. Paddle keyword arguments
            out2 = paddle.dstack(x=static_tensors)
            # 3. PyTorch keyword arguments
            out3 = paddle.dstack(tensors=static_tensors)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main, feed=feed_dict, fetch_list=[out1, out2, out3]
            )

            ref_out = np.dstack(tuple(inp for inp in self.inputs))
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-8)


# Test column_stack compatibility
class TestColumnStackAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.inputs = [
            np.random.rand(3, 2).astype('float32'),
            np.random.rand(3, 3).astype('float32'),
        ]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        tensors = [paddle.to_tensor(inp) for inp in self.inputs]
        # 1. Paddle positional arguments
        out1 = paddle.column_stack(tensors)
        # 2. Paddle keyword arguments
        out2 = paddle.column_stack(x=tensors)
        # 3. PyTorch keyword arguments
        out3 = paddle.column_stack(tensors=tensors)

        # Verify all outputs
        ref_out = np.column_stack(tuple(inp for inp in self.inputs))
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-8
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        shapes = [[3, 2], [3, 3]]
        with paddle.static.program_guard(main, startup):
            static_tensors = []
            feed_dict = {}
            for i, (shape, inp) in enumerate(zip(shapes, self.inputs)):
                static_tensor = paddle.static.data(
                    name=f"x{i}", shape=shape, dtype='float32'
                )
                static_tensors.append(static_tensor)
                feed_dict[f"x{i}"] = inp

            # 1. Paddle positional arguments
            out1 = paddle.column_stack(static_tensors)
            # 2. Paddle keyword arguments
            out2 = paddle.column_stack(x=static_tensors)
            # 3. PyTorch keyword arguments
            out3 = paddle.column_stack(tensors=static_tensors)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main, feed=feed_dict, fetch_list=[out1, out2, out3]
            )

            ref_out = np.column_stack(tuple(inp for inp in self.inputs))
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-8)


# Test row_stack compatibility
class TestRowStackAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.inputs = [
            np.random.rand(2, 3).astype('float32'),
            np.random.rand(4, 3).astype('float32'),
        ]

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        tensors = [paddle.to_tensor(inp) for inp in self.inputs]
        # 1. Paddle positional arguments
        out1 = paddle.row_stack(tensors)
        # 2. Paddle keyword arguments
        out2 = paddle.row_stack(x=tensors)
        # 3. PyTorch keyword arguments
        out3 = paddle.row_stack(tensors=tensors)

        # Verify all outputs
        ref_out = np.vstack(tuple(inp for inp in self.inputs))
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-8
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        shapes = [[2, 3], [4, 3]]
        with paddle.static.program_guard(main, startup):
            static_tensors = []
            feed_dict = {}
            for i, (shape, inp) in enumerate(zip(shapes, self.inputs)):
                static_tensor = paddle.static.data(
                    name=f"x{i}", shape=shape, dtype='float32'
                )
                static_tensors.append(static_tensor)
                feed_dict[f"x{i}"] = inp

            # 1. Paddle positional arguments
            out1 = paddle.row_stack(static_tensors)
            # 2. Paddle keyword arguments
            out2 = paddle.row_stack(x=static_tensors)
            # 3. PyTorch keyword arguments
            out3 = paddle.row_stack(tensors=static_tensors)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main, feed=feed_dict, fetch_list=[out1, out2, out3]
            )

            ref_out = np.vstack(tuple(inp for inp in self.inputs))
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-8)


# Test bernoulli compatibility
class TestBernoulliAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(2, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        paddle.seed(100)
        out1 = paddle.bernoulli(x)
        # 2. Paddle keyword arguments
        paddle.seed(100)
        out2 = paddle.bernoulli(x=x)
        # 3. PyTorch keyword arguments
        paddle.seed(100)
        out3 = paddle.bernoulli(input=x)
        # 4. Mixed arguments
        paddle.seed(100)
        out4 = paddle.bernoulli(x, p=0.5)
        # 5. out parameter
        paddle.seed(100)
        out5 = paddle.empty_like(x)
        paddle.bernoulli(x, out=out5)
        # 6. Class method positional arguments
        paddle.seed(100)
        out6 = x.bernoulli()
        # 7. Class method keyword arguments
        paddle.seed(100)
        out7 = x.bernoulli(p=0.5)

        # Verify outputs have correct shape
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, x.dtype)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.bernoulli(x)
            # 2. Paddle keyword arguments
            out2 = paddle.bernoulli(x=x)
            # 3. PyTorch keyword arguments
            out3 = paddle.bernoulli(input=x)

            exe = paddle.static.Executor()
            exe.run(startup)
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify outputs have correct shape
            for out in fetches:
                self.assertEqual(out.shape, (2, 3))


# Test combinations compatibility
class TestCombinationsAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([1, 2, 3, 4]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments (all positional: x, r, with_replacement）
        out1 = paddle.combinations(x, 2, False)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.combinations(x=x, r=2, with_replacement=False)
        # 3. PyTorch keyword arguments
        out3 = paddle.combinations(input=x, r=2)
        # 4. Mixed arguments (with with_replacement parameter)
        out4 = paddle.combinations(x, r=3, with_replacement=True)

        # Verify all outputs
        for out in [out1, out2, out3, out4]:
            self.assertIsInstance(out, paddle.Tensor)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[4], dtype='int32')

            # 1. Paddle positional arguments (all positional: x, r, with_replacement）
            out1 = paddle.combinations(x, 2, False)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.combinations(x=x, r=2, with_replacement=False)
            # 3. PyTorch keyword arguments
            out3 = paddle.combinations(input=x, r=2)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                self.assertIsInstance(out, np.ndarray)


# Test trapezoid compatibility
class TestTrapezoidAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_y = np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype='float32')
        self.np_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments (all positional: y, x, dx, axis）
        out1 = paddle.trapezoid(y, None, None, -1)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.trapezoid(y=y, x=None, dx=None, axis=-1)
        # 3. PyTorch keyword arguments (using alias dim)
        out3 = paddle.trapezoid(y, dim=-1)
        # 5. out parameter
        out4 = paddle.empty([])
        paddle.trapezoid(y, out=out4)

        # Verify outputs
        ref_out = np.trapezoid(self.np_y)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            y = paddle.static.data(name="y", shape=[5], dtype='float32')

            # 1. Paddle positional arguments (all positional: y, x, dx, axis）
            out1 = paddle.trapezoid(y, None, None, -1)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.trapezoid(y=y, x=None, dx=None, axis=-1)
            # 3. PyTorch keyword arguments (using alias dim)
            out3 = paddle.trapezoid(y, dim=-1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            ref_out = np.trapezoid(self.np_y)
            for actual in fetches:
                np.testing.assert_allclose(actual, ref_out, rtol=1e-5)


# Test cumulative_trapezoid compatibility
class TestCumulativeTrapezoidAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_y = np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype='float32')
        self.np_x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments (all positional: y, x, dx, axis）
        out1 = paddle.cumulative_trapezoid(y, None, None, -1)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.cumulative_trapezoid(y=y, x=None, dx=None, axis=-1)
        # 3. PyTorch keyword arguments (using alias dim)
        out3 = paddle.cumulative_trapezoid(y, dim=-1)
        # 4. Mixed arguments (with dx parameter)
        out4 = paddle.cumulative_trapezoid(y, dx=2.0)
        # 5. out parameter
        out5 = paddle.empty([4])
        paddle.cumulative_trapezoid(y, out=out5)

        # Verify outputs
        ref_out = np.array([4.5, 10.0, 16.5, 24.0])
        for actual in [out1, out2, out3, out5]:
            np.testing.assert_allclose(actual.numpy(), ref_out, rtol=1e-5)
        # Output with dx=2.0
        ref_out_dx = np.array([9.0, 20.0, 33.0, 48.0])
        np.testing.assert_allclose(out4.numpy(), ref_out_dx, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            y = paddle.static.data(name="y", shape=[5], dtype='float32')

            # 1. Paddle positional arguments (all positional: y, x, dx, axis）
            out1 = paddle.cumulative_trapezoid(y, None, None, -1)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.cumulative_trapezoid(y=y, x=None, dx=None, axis=-1)
            # 3. PyTorch keyword arguments (using alias dim)
            out3 = paddle.cumulative_trapezoid(y, dim=-1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            ref_out = np.array([4.5, 10.0, 16.5, 24.0])
            for actual in fetches:
                np.testing.assert_allclose(actual, ref_out, rtol=1e-5)


# Test frexp compatibility
class TestFrexpAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[10.0, -2.5, 0.0, 3.14], [128.0, 64.0, -32.0, 16.0]],
            dtype='float32',
        )

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        mantissa1, exponent1 = paddle.frexp(x)
        # 2. Paddle keyword arguments
        mantissa2, exponent2 = paddle.frexp(x=x)
        # 3. PyTorch keyword arguments
        mantissa3, exponent3 = paddle.frexp(input=x)
        # 4. Mixed arguments (only one parameter, mixed not applicable)
        # 5. out parameter
        out_mantissa = paddle.empty_like(x)
        out_exponent = paddle.empty_like(x)
        paddle.frexp(input=x, out=[out_mantissa, out_exponent])
        # 6. Tensor method
        mantissa4, exponent4 = x.frexp()

        # Verify all outputs are consistent
        ref_mantissa = mantissa1.numpy()
        ref_exponent = exponent1.numpy()

        for mantissa, exponent in [
            (mantissa1, exponent1),
            (mantissa2, exponent2),
            (mantissa3, exponent3),
            (out_mantissa, out_exponent),
            (mantissa4, exponent4),
        ]:
            np.testing.assert_allclose(
                mantissa.numpy(), ref_mantissa, rtol=1e-5
            )
            np.testing.assert_allclose(
                exponent.numpy(), ref_exponent, rtol=1e-5
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 4], dtype='float32')

            # 1. Paddle positional arguments
            mantissa1, exponent1 = paddle.frexp(x)
            # 2. Paddle keyword arguments
            mantissa2, exponent2 = paddle.frexp(x=x)
            # 3. PyTorch keyword arguments
            mantissa3, exponent3 = paddle.frexp(input=x)
            # 4. Mixed arguments (only one parameter, mixed not applicable)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[
                    mantissa1,
                    exponent1,
                    mantissa2,
                    exponent2,
                    mantissa3,
                    exponent3,
                ],
            )

            # Verify all outputs are consistent
            for i in range(0, len(fetches), 2):
                np.testing.assert_allclose(fetches[i], fetches[0], rtol=1e-5)
                np.testing.assert_allclose(
                    fetches[i + 1], fetches[1], rtol=1e-5
                )


# Test lgamma compatibility
class TestLgammaAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([-0.4, -0.2, 0.1, 0.3]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.lgamma(x)
        # 2. Paddle keyword arguments
        out2 = paddle.lgamma(x=x)
        # 3. PyTorch keyword arguments
        out3 = paddle.lgamma(input=x)
        # 4. Mixed arguments (only one parameter, mixed not applicable)
        # 5. out parameter
        out4 = paddle.empty_like(x)
        paddle.lgamma(x, out=out4)
        # 6. Class method positional arguments
        out5 = x.lgamma()
        # 7. Class method keyword arguments (no parameters, not applicable)

        # Verify all outputs
        ref_out = np.array(
            [1.31452465, 1.76149750, 2.25271273, 1.09579802], dtype=np.float32
        )
        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[4], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.lgamma(x)
            # 2. Paddle keyword arguments
            out2 = paddle.lgamma(x=x)
            # 3. PyTorch keyword arguments
            out3 = paddle.lgamma(input=x)
            # 4. Mixed arguments (only one parameter, mixed not applicable)
            # 5. Class method positional arguments
            out4 = x.lgamma()
            # 6. Class method keyword arguments (no parameters, not applicable)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            ref_out = np.array(
                [1.31452465, 1.76149750, 2.25271273, 1.09579802],
                dtype=np.float32,
            )
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test kron compatibility
class TestKronAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array([[1, 2], [3, 4]], dtype='int64')
        self.np_y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='int64')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.kron(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.kron(x=x, y=y)
        # 3. PyTorch keyword arguments
        out3 = paddle.kron(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.kron(x, other=y)
        # 5. out parameter
        out5 = paddle.empty([6, 6], dtype='int64')
        paddle.kron(x, y, out=out5)
        # 6. Class method positional arguments
        out6 = x.kron(y)
        # 7. Class method keyword arguments
        out7 = x.kron(y=y)

        # Verify all outputs
        ref_out = np.array(
            [
                [1, 2, 3, 2, 4, 6],
                [4, 5, 6, 8, 10, 12],
                [7, 8, 9, 14, 16, 18],
                [3, 6, 9, 4, 8, 12],
                [12, 15, 18, 16, 20, 24],
                [21, 24, 27, 28, 32, 36],
            ],
            dtype=np.int64,
        )
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_array_equal(out.numpy(), ref_out)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 2], dtype='int64')
            y = paddle.static.data(name="y", shape=[3, 3], dtype='int64')

            # 1. Paddle positional arguments
            out1 = paddle.kron(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.kron(x=x, y=y)
            # 3. PyTorch keyword arguments
            out3 = paddle.kron(input=x, other=y)
            # 4. Class method positional arguments
            out4 = x.kron(y)
            # 5. Class method keyword arguments
            out5 = x.kron(y=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.array(
                [
                    [1, 2, 3, 2, 4, 6],
                    [4, 5, 6, 8, 10, 12],
                    [7, 8, 9, 14, 16, 18],
                    [3, 6, 9, 4, 8, 12],
                    [12, 15, 18, 16, 20, 24],
                    [21, 24, 27, 28, 32, 36],
                ],
                dtype=np.int64,
            )
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test kthvalue compatibility
class TestKthvalueAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [
                [
                    [0.11855337, -0.30557564],
                    [-0.09968963, 0.41220093],
                    [1.24004936, 1.50014710],
                ],
                [
                    [0.08612321, -0.92485696],
                    [-0.09276631, 1.15149164],
                    [-1.46587241, 1.22873247],
                ],
            ]
        ).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        k = 2

        # 1. Paddle positional arguments (all positional: x, k, axis, keepdim）
        values1, indices1 = paddle.kthvalue(x, k, 1, False)
        # 2. Paddle keyword arguments (all keyword arguments)
        values2, indices2 = paddle.kthvalue(x=x, k=k, axis=1, keepdim=False)
        # 3. PyTorch keyword arguments
        values3, indices3 = paddle.kthvalue(input=x, k=k, dim=1)
        # 4. Mixed arguments (with keepdim parameter)
        values4, indices4 = paddle.kthvalue(x, k, axis=1, keepdim=True)
        # 5. out parameter
        values_out = paddle.empty([2, 2], dtype='float32')
        indices_out = paddle.empty([2, 2], dtype='int64')
        paddle.kthvalue(x, k, axis=1, out=(values_out, indices_out))
        # 6. Class method positional arguments
        values6, indices6 = x.kthvalue(k, 1)
        # 7. Class method keyword arguments
        values7, indices7 = x.kthvalue(k, axis=1, keepdim=True)

        # Verify outputs
        ref_values = np.array(
            [[[0.11855337, 0.41220093], [-0.09276631, 1.15149164]]],
            dtype=np.float32,
        ).reshape(2, 2)
        ref_indices = np.array([[0, 1], [1, 1]], dtype=np.int64)
        for values, indices in [
            (values1, indices1),
            (values2, indices2),
            (values3, indices3),
            (values6, indices6),
        ]:
            np.testing.assert_allclose(values.numpy(), ref_values, rtol=1e-5)
            np.testing.assert_array_equal(indices.numpy(), ref_indices)
        # Verify keepdim=True
        for values, indices in [(values4, indices4), (values7, indices7)]:
            np.testing.assert_allclose(
                values.numpy(), ref_values.reshape(2, 1, 2), rtol=1e-5
            )
            np.testing.assert_array_equal(
                indices.numpy(), ref_indices.reshape(2, 1, 2)
            )
        # Verify out parameter
        np.testing.assert_allclose(values_out.numpy(), ref_values, rtol=1e-5)
        np.testing.assert_array_equal(indices_out.numpy(), ref_indices)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3, 2], dtype='float32')
            k = 2

            # 1. Paddle positional arguments (all positional: x, k, axis, keepdim）
            values1, indices1 = paddle.kthvalue(x, k, 1, False)
            # 2. Paddle keyword arguments (all keyword arguments)
            values2, indices2 = paddle.kthvalue(x=x, k=k, axis=1, keepdim=False)
            # 3. PyTorch keyword arguments
            values3, indices3 = paddle.kthvalue(input=x, k=k, dim=1)
            # 4. Class method positional arguments
            values4, indices4 = x.kthvalue(k, 1)
            # 5. Class method keyword arguments
            values5, indices5 = x.kthvalue(k, axis=1, keepdim=True)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[
                    values1,
                    indices1,
                    values2,
                    indices2,
                    values3,
                    indices3,
                    values4,
                    indices4,
                    values5,
                    indices5,
                ],
            )

            ref_values = np.array(
                [[0.11855337, 0.41220093], [-0.09276631, 1.15149164]],
                dtype=np.float32,
            )
            ref_indices = np.array([[0, 1], [1, 1]], dtype=np.int64)
            # Verify all values outputs (no keepdim)
            for i in [0, 2, 4, 6]:
                np.testing.assert_allclose(fetches[i], ref_values, rtol=1e-5)
            # Verify keepdim=True values
            np.testing.assert_allclose(
                fetches[8], ref_values.reshape(2, 1, 2), rtol=1e-5
            )
            # Verify all indices outputs (no keepdim)
            for i in [1, 3, 5, 7]:
                np.testing.assert_array_equal(fetches[i], ref_indices)
            # Verify keepdim=True indices
            np.testing.assert_array_equal(
                fetches[9], ref_indices.reshape(2, 1, 2)
            )


# Test logcumsumexp compatibility
class TestLogcumsumexpAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.ref_out_axis0 = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.01814993, 5.01814993, 6.01814993, 7.01814993],
                [8.01847930, 9.01847930, 10.01847930, 11.01847930],
            ]
        )

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments (all positional: x, axis, dtype）
        out1 = paddle.logcumsumexp(x, 0, None)
        # 2. Paddle keyword arguments (all keyword arguments)
        out2 = paddle.logcumsumexp(x=x, axis=0, dtype=None)
        # 3. PyTorch keyword arguments (using alias dim)
        out3 = paddle.logcumsumexp(input=x, dim=0)
        # 4. Mixed arguments (with dtype parameter)
        out4 = paddle.logcumsumexp(x, axis=0, dtype='float32')
        # 5. out parameter
        out5 = paddle.empty([3, 4], dtype='float32')
        paddle.logcumsumexp(x, axis=0, out=out5)
        # 6. Class method positional arguments
        out6 = x.logcumsumexp(0)
        # 7. Class method keyword arguments
        out7 = x.logcumsumexp(axis=0)

        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(
                out.numpy(), self.ref_out_axis0, rtol=1e-5
            )
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype='float32')

            # 1. Paddle positional arguments (all positional: x, axis, dtype）
            out1 = paddle.logcumsumexp(x, 0, None)
            # 2. Paddle keyword arguments (all keyword arguments)
            out2 = paddle.logcumsumexp(x=x, axis=0, dtype=None)
            # 3. PyTorch keyword arguments (using alias dim)
            out3 = paddle.logcumsumexp(input=x, dim=0)
            # 4. Class method positional arguments
            out4 = x.logcumsumexp(0)
            # 5. Class method keyword arguments
            out5 = x.logcumsumexp(axis=0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            for out in fetches:
                np.testing.assert_allclose(out, self.ref_out_axis0, rtol=1e-5)


# Test poisson compatibility
class TestPoissonAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 4).astype('float32') + 0.5

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        paddle.seed(100)
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.poisson(x)
        # 2. Paddle keyword arguments
        out2 = paddle.poisson(x=x)
        # 3. PyTorch keyword arguments
        out3 = paddle.poisson(input=x)
        # 4. Mixed arguments (only one parameter, mixed not applicable)

        # Verify all outputs have same shape
        for out in [out1, out2, out3]:
            self.assertEqual(out.shape, (3, 4))
            self.assertEqual(out.dtype, x.dtype)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.poisson(x)
            # 2. Paddle keyword arguments
            out2 = paddle.poisson(x=x)
            # 3. PyTorch keyword arguments
            out3 = paddle.poisson(input=x)
            # 4. Mixed arguments (only one parameter, mixed not applicable)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs have correct shape
            for out in fetches:
                self.assertEqual(out.shape, (3, 4))


# Test mode compatibility
class TestModeAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        # Use fixed data for precise comparison
        self.np_x = np.array(
            [
                [
                    [0.5, 0.3, 0.7, 0.2],
                    [0.5, 0.8, 0.7, 0.9],
                    [0.1, 0.3, 0.4, 0.2],
                ],
                [
                    [0.6, 0.4, 0.5, 0.3],
                    [0.6, 0.2, 0.5, 0.7],
                    [0.9, 0.4, 0.8, 0.3],
                ],
            ]
        ).astype('float32')

    def compute_mode_along_axis(self):
        """Compute mode along axis=1"""
        # For axis=1, we find the most frequent value along that dimension
        # When there are ties, paddle returns the last occurrence index
        # [0,0,:]: [0.5, 0.5, 0.1] -> mode=0.5, index=1 (last 0.5)
        # [0,1,:]: [0.3, 0.8, 0.3] -> mode=0.3, index=2 (last 0.3)
        # [0,2,:]: [0.7, 0.7, 0.4] -> mode=0.7, index=1 (last 0.7)
        # [0,3,:]: [0.2, 0.9, 0.2] -> mode=0.2, index=2 (last 0.2)
        ref_values = np.array(
            [[0.5, 0.3, 0.7, 0.2], [0.6, 0.4, 0.5, 0.3]], dtype='float32'
        )
        ref_indices = np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype=np.int64)
        return ref_values, ref_indices

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.mode(x, 1, False)
        # 2. Paddle keyword arguments
        out2 = paddle.mode(x=x, axis=1, keepdim=False)
        # 3. PyTorch keyword arguments
        out3 = paddle.mode(input=x, dim=1)
        # 4. Mixed arguments (with keepdim parameter)
        out4 = paddle.mode(x, axis=1, keepdim=True)
        # 5. out parameter (tuple)
        out5 = (
            paddle.empty([2, 4], dtype='float32'),
            paddle.empty([2, 4], dtype='int64'),
        )
        paddle.mode(x, 1, out=out5)
        # 6. out parameter (list)
        out6 = [
            paddle.empty([2, 4], dtype='float32'),
            paddle.empty([2, 4], dtype='int64'),
        ]
        paddle.mode(x, 1, out=out6)
        # 7. Class method positional arguments
        out7 = x.mode(1)
        # 8. Class method keyword arguments
        out8 = x.mode(axis=1, keepdim=True)

        ref_values, ref_indices = self.compute_mode_along_axis()

        # Verify outputs with keepdim=False
        for out in [out1, out2, out3, out7]:
            np.testing.assert_array_equal(out.values.numpy(), ref_values)
            np.testing.assert_array_equal(out.indices.numpy(), ref_indices)
            np.testing.assert_array_equal(out[0].numpy(), ref_values)
            np.testing.assert_array_equal(out[1].numpy(), ref_indices)

        # Verify outputs with out parameter
        for out in [out5, out6]:
            np.testing.assert_array_equal(out[0].numpy(), ref_values)
            np.testing.assert_array_equal(out[1].numpy(), ref_indices)

        # Verify outputs with keepdim=True
        for out in [out4, out8]:
            np.testing.assert_array_equal(
                out[0].numpy(), ref_values.reshape(2, 1, 4)
            )
            np.testing.assert_array_equal(
                out[1].numpy(), ref_indices.reshape(2, 1, 4)
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 3, 4], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.mode(x, 1, False)
            # 2. Paddle keyword arguments
            out2 = paddle.mode(x=x, axis=1, keepdim=False)
            # 3. PyTorch keyword arguments
            out3 = paddle.mode(input=x, dim=1)
            # 4. Class method positional arguments
            out4 = x.mode(1)
            # 5. Class method keyword arguments
            out5 = x.mode(axis=1, keepdim=True)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[
                    out1[0],
                    out1[1],
                    out2[0],
                    out2[1],
                    out3[0],
                    out3[1],
                    out4[0],
                    out4[1],
                    out5[0],
                    out5[1],
                ],
            )

            ref_values, ref_indices = self.compute_mode_along_axis()

            # Verify outputs with keepdim=False: out1, out2, out3, out4
            for i in [0, 2, 4, 6]:
                np.testing.assert_allclose(
                    fetches[i], ref_values, rtol=1e-5, atol=1e-5
                )
                np.testing.assert_array_equal(fetches[i + 1], ref_indices)

            # Verify output with keepdim=True: out5
            np.testing.assert_allclose(
                fetches[8], ref_values.reshape(2, 1, 4), rtol=1e-5, atol=1e-5
            )
            np.testing.assert_array_equal(
                fetches[9], ref_indices.reshape(2, 1, 4)
            )


# Test topk compatibility
class TestTopkAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[0.5, 0.3, 0.9, 0.2], [0.6, 0.8, 0.4, 0.7], [0.1, 0.4, 0.3, 0.5]]
        ).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # Reference: top 2 values along axis=1
        ref_values = np.array(
            [[0.9, 0.5], [0.8, 0.7], [0.5, 0.4]], dtype='float32'
        )
        ref_indices = np.array([[2, 0], [1, 3], [3, 1]], dtype=np.int64)

        # 1. Paddle positional arguments
        out1 = paddle.topk(x, 2, 1)
        # 2. Paddle keyword arguments
        out2 = paddle.topk(x=x, k=2, axis=1)
        # 3. PyTorch keyword arguments
        out3 = paddle.topk(input=x, k=2, dim=1)
        # 4. Mixed arguments
        out4 = paddle.topk(x, k=2, axis=1)
        # 5. out parameter (tuple)
        out5_values = paddle.empty([3, 2], dtype='float32')
        out5_indices = paddle.empty([3, 2], dtype='int64')
        paddle.topk(x, 2, 1, out=(out5_values, out5_indices))
        # 6. out parameter (list)
        out6_values = paddle.empty([3, 2], dtype='float32')
        out6_indices = paddle.empty([3, 2], dtype='int64')
        paddle.topk(x, 2, 1, out=[out6_values, out6_indices])
        # 7. Class method positional arguments
        out7 = x.topk(2, 1)
        # 8. Class method keyword arguments
        out8 = x.topk(k=2, axis=1)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out7, out8]:
            np.testing.assert_array_equal(out.values.numpy(), ref_values)
            np.testing.assert_array_equal(out.indices.numpy(), ref_indices)
            np.testing.assert_array_equal(out[0].numpy(), ref_values)
            np.testing.assert_array_equal(out[1].numpy(), ref_indices)

        # Verify outputs with out parameter
        np.testing.assert_array_equal(out5_values.numpy(), ref_values)
        np.testing.assert_array_equal(out5_indices.numpy(), ref_indices)
        np.testing.assert_array_equal(out6_values.numpy(), ref_values)
        np.testing.assert_array_equal(out6_indices.numpy(), ref_indices)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 4], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.topk(x, 2, 1)
            # 2. Paddle keyword arguments
            out2 = paddle.topk(x=x, k=2, axis=1)
            # 3. PyTorch keyword arguments
            out3 = paddle.topk(input=x, k=2, dim=1)
            # 4. Class method positional arguments
            out4 = x.topk(2, 1)
            # 5. Class method keyword arguments
            out5 = x.topk(k=2, axis=1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[
                    out1[0],
                    out1[1],
                    out2[0],
                    out2[1],
                    out3[0],
                    out3[1],
                    out4[0],
                    out4[1],
                    out5[0],
                    out5[1],
                ],
            )

            ref_values = np.array(
                [[0.9, 0.5], [0.8, 0.7], [0.5, 0.4]], dtype='float32'
            )
            ref_indices = np.array([[2, 0], [1, 3], [3, 1]], dtype=np.int64)

            # Verify all outputs
            for i in range(0, len(fetches), 2):
                np.testing.assert_array_equal(fetches[i], ref_values)
                np.testing.assert_array_equal(fetches[i + 1], ref_indices)


if __name__ == '__main__':
    unittest.main()
