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
    paddle.is_compiled_with_custom_device('ixuca'),
    "skip ixuca which not register mv kernel",
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
        # 5. out parameter test
        out5 = paddle.zeros([3], dtype="float32")
        paddle.mv(x, vec, out=out5)
        # 6. Tensor method - args
        out6 = x.mv(vec)
        # 7. Tensor method - kwargs (PyTorch alias)
        out7 = x.mv(vec=vec)

        # Verify all outputs
        ref_out = np.dot(self.np_x, self.np_vec)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
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
        np.random.seed(2025)
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
        # 4. out parameter test
        out4 = paddle.zeros_like(out1)
        paddle.isposinf(x, out=out4)
        # 5. Tensor method - args
        out5 = x.isposinf()

        # Verify all outputs
        ref_out = np.isposinf(self.np_x)
        for out in [out1, out2, out3, out4, out5]:
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
        np.random.seed(2025)
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
        # 4. out parameter test
        out4 = paddle.zeros_like(out1)
        paddle.isneginf(x, out=out4)
        # 5. Tensor method - args
        out5 = x.isneginf()

        # Verify all outputs
        ref_out = np.isneginf(self.np_x)
        for out in [out1, out2, out3, out4, out5]:
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
        np.random.seed(2025)
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


class TestLogitAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        # Values in (0, 1) for logit
        self.np_x = np.random.uniform(0.1, 0.9, self.shape).astype(self.dtype)

    def _ref_logit(self, x, eps=0.0):
        if eps > 0.0:
            x = np.clip(x, eps, 1.0 - eps)
        return np.log(x / (1.0 - x))

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        paddle_dygraph_out = []

        # Positional args
        out1 = paddle.logit(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.logit(x=x)
        paddle_dygraph_out.append(out2)

        # Alias keyword args (input)
        out3 = paddle.logit(input=x)
        paddle_dygraph_out.append(out3)

        # Test out parameter
        out4 = paddle.empty_like(x)
        ret4 = paddle.logit(x, out=out4)
        paddle_dygraph_out.append(out4)
        self.assertEqual(ret4.data_ptr(), out4.data_ptr())

        # out parameter with alias keyword
        out5 = paddle.empty_like(x)
        ret5 = paddle.logit(input=x, out=out5)
        paddle_dygraph_out.append(out5)
        self.assertEqual(ret5.data_ptr(), out5.data_ptr())

        ref_out = self._ref_logit(self.np_x)
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )
        paddle.enable_static()

    def test_dygraph_with_eps(self):
        paddle.disable_static()
        x_data = np.array([0.0, 0.01, 0.5, 0.99, 1.0]).astype(self.dtype)
        x = paddle.to_tensor(x_data)
        eps = 1e-6
        paddle_dygraph_out = []

        # Positional eps
        out1 = paddle.logit(x, eps)
        paddle_dygraph_out.append(out1)

        # Keyword eps
        out2 = paddle.logit(x, eps=eps)
        paddle_dygraph_out.append(out2)

        # Alias keyword + eps
        out3 = paddle.logit(input=x, eps=eps)
        paddle_dygraph_out.append(out3)

        # out + eps
        out4 = paddle.empty_like(x)
        ret4 = paddle.logit(x, eps, out=out4)
        paddle_dygraph_out.append(out4)
        self.assertEqual(ret4.data_ptr(), out4.data_ptr())

        ref_out = self._ref_logit(x_data, eps)
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )
        paddle.enable_static()

    def test_dygraph_no_eps_boundary(self):
        """Without eps, values outside (0,1) should produce NaN/Inf."""
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([-0.1, 0.0, 0.5, 1.0, 1.1]).astype(self.dtype)
        )
        out = paddle.logit(x)
        result = out.numpy()
        # x=0.5 should give 0
        np.testing.assert_allclose(result[2], 0.0, atol=1e-6)
        # x=0 -> -inf, x=1 -> +inf
        self.assertTrue(np.isinf(result[1]) and result[1] < 0)
        self.assertTrue(np.isinf(result[3]) and result[3] > 0)
        # x < 0 or x > 1 -> NaN
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[4]))
        paddle.enable_static()

    def test_dygraph_special_alias(self):
        """Test paddle.special.logit API alias."""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        out1 = paddle.logit(x)
        out2 = paddle.special.logit(x)
        np.testing.assert_allclose(
            out1.numpy(), out2.numpy(), rtol=1e-5, atol=1e-6
        )

        # With eps via special alias
        out3 = paddle.special.logit(x, eps=1e-6)
        out4 = paddle.logit(x, eps=1e-6)
        np.testing.assert_allclose(
            out3.numpy(), out4.numpy(), rtol=1e-5, atol=1e-6
        )

        # Alias keyword via special
        out5 = paddle.special.logit(input=x)
        np.testing.assert_allclose(
            out1.numpy(), out5.numpy(), rtol=1e-5, atol=1e-6
        )

        # out parameter via special
        out6 = paddle.empty_like(x)
        ret6 = paddle.special.logit(x, out=out6)
        self.assertEqual(ret6.data_ptr(), out6.data_ptr())
        np.testing.assert_allclose(
            out1.numpy(), out6.numpy(), rtol=1e-5, atol=1e-6
        )
        paddle.enable_static()

    def test_dygraph_dtypes(self):
        """Test logit works with float64."""
        paddle.disable_static()
        x_data = np.random.uniform(0.1, 0.9, self.shape).astype('float64')
        x = paddle.to_tensor(x_data)
        out = paddle.logit(x)
        ref_out = self._ref_logit(x_data)
        np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-10, atol=1e-10)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # Positional args
            out1 = paddle.logit(x)
            # Paddle keyword args
            out2 = paddle.logit(x=x)
            # Alias keyword args
            out3 = paddle.logit(input=x)
            # With eps via alias
            out4 = paddle.logit(input=x, eps=1e-6)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = self._ref_logit(self.np_x)
            ref_out_eps = self._ref_logit(self.np_x, 1e-6)
            for out in fetches[:3]:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(
                fetches[3], ref_out_eps, rtol=1e-5, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
