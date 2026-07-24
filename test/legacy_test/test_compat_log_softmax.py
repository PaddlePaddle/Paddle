#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from scipy.special import log_softmax as scipy_log_softmax

import paddle
import paddle.nn.functional as F


class TestLogSoftmaxAPI(unittest.TestCase):
    """Test paddle.nn.functional.log_softmax (API 1/5)."""

    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3, 4]
        self.dtype = 'float32'
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.ref_out = scipy_log_softmax(self.np_x, axis=-1).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = F.log_softmax(x, -1)

        # 2. Paddle keyword arguments
        out2 = F.log_softmax(x=x, axis=-1, dtype=None, name=None)

        # 3. PyTorch positional arguments
        out3 = F.log_softmax(x, -1, None)

        # 4. PyTorch keyword arguments (alias)
        out4 = F.log_softmax(input=x, dim=-1)

        # 5. Mixed arguments (positional + keyword)
        out5 = F.log_softmax(x, dim=-1)

        # 6. out parameter
        out6 = paddle.empty_like(x)
        F.log_softmax(x, -1, out=out6)

        # 7. Tensor method - positional args
        out7 = x.log_softmax(-1)

        # 8. Tensor method - keyword args
        out8 = x.log_softmax(dim=-1)

        # Verify all outputs match reference
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(
                self.ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )

    def test_static_Compatibility(self):
        # 9. Dynamic and static graph modes
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Paddle positional args
            out1 = F.log_softmax(x, -1)
            # Paddle keyword args
            out2 = F.log_softmax(x=x, axis=-1)
            # PyTorch keyword args (alias)
            out3 = F.log_softmax(input=x, dim=-1)

            exe = paddle.base.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            for out in fetches:
                np.testing.assert_allclose(
                    out, self.ref_out, rtol=1e-5, atol=1e-6
                )

        paddle.disable_static()


class TestPaddleLogSoftmaxAPI(unittest.TestCase):
    """Test paddle.log_softmax (API 2/5)."""

    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3, 4]
        self.dtype = 'float32'
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.ref_out = scipy_log_softmax(self.np_x, axis=-1).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Positional arguments
        out1 = paddle.log_softmax(x, -1)

        # 2. Keyword arguments
        out2 = paddle.log_softmax(input=x, dim=-1, dtype=None)

        # 3. Mixed arguments
        out3 = paddle.log_softmax(x, dim=-1)

        # 4. out parameter
        out4 = paddle.empty_like(x)
        paddle.log_softmax(x, -1, out=out4)

        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(
                self.ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )


class TestTensorLogSoftmaxAPI(unittest.TestCase):
    """Test paddle.Tensor.log_softmax (API 3/5)."""

    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3, 4]
        self.dtype = 'float32'
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.ref_out = scipy_log_softmax(self.np_x, axis=-1).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Positional arguments
        out1 = x.log_softmax(-1)

        # 2. Keyword arguments
        out2 = x.log_softmax(dim=-1)

        # 3. with dtype
        out3 = x.log_softmax(-1, dtype='float64')
        self.assertEqual(out3.dtype, paddle.float64)

        for out in [out1, out2]:
            np.testing.assert_allclose(
                self.ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )


class TestSpecialLogSoftmaxAPI(unittest.TestCase):
    """Test paddle.special.log_softmax (API 4/5)."""

    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3, 4]
        self.dtype = 'float32'
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.ref_out = scipy_log_softmax(self.np_x, axis=-1).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Positional arguments
        out1 = paddle.special.log_softmax(x, -1)

        # 2. Keyword arguments
        out2 = paddle.special.log_softmax(input=x, dim=-1)

        # 3. out parameter
        out3 = paddle.empty_like(x)
        paddle.special.log_softmax(x, -1, out=out3)

        for out in [out1, out2, out3]:
            np.testing.assert_allclose(
                self.ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )


class TestCompatLogSoftmaxAPI(unittest.TestCase):
    """Test paddle.compat.nn.functional.log_softmax (API 5/5)."""

    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3, 4]
        self.dtype = 'float32'
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.ref_out = scipy_log_softmax(self.np_x, axis=-1).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        compat_fn = paddle.compat.nn.functional.log_softmax

        # 1. Positional arguments
        out1 = compat_fn(x, -1)

        # 2. Keyword arguments
        out2 = compat_fn(input=x, dim=-1, dtype=None)

        # 3. Mixed arguments (positional + keyword)
        out3 = compat_fn(x, dim=-1)

        # 4. out parameter
        out4 = paddle.empty_like(x)
        compat_fn(x, -1, out=out4)

        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(
                self.ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )


class TestLogSoftmaxAllAliasesConsistent(unittest.TestCase):
    """Test that all five log_softmax entry points produce the same results."""

    def setUp(self):
        np.random.seed(2025)
        self.shape = [2, 3, 4]
        self.dtype = 'float32'
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.ref_out = scipy_log_softmax(self.np_x, axis=-1).astype(self.dtype)

    def test_all_aliases_consistent(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        out1 = F.log_softmax(x, -1)
        out2 = paddle.log_softmax(x, -1)
        out3 = x.log_softmax(-1)
        out4 = paddle.special.log_softmax(x, -1)
        out5 = paddle.compat.nn.functional.log_softmax(x, dim=-1)

        for out in [out1, out2, out3, out4, out5]:
            np.testing.assert_allclose(
                self.ref_out, out.numpy(), rtol=1e-5, atol=1e-6
            )


class TestCompatLogSoftmaxDimNoneDefault(unittest.TestCase):
    """Test PyTorch-compatible dim=None default behavior."""

    def setUp(self):
        paddle.disable_static()

    def test_0d_defaults_to_dim0(self):
        x = paddle.to_tensor(1.0)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_1d_defaults_to_dim0(self):
        x = paddle.randn([4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_2d_defaults_to_dim1(self):
        x = paddle.randn([3, 4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=1)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_3d_defaults_to_dim0(self):
        x = paddle.randn([2, 3, 4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=0)
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_4d_defaults_to_dim1(self):
        x = paddle.randn([2, 3, 4, 5], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(x)
        expected = paddle.compat.nn.functional.log_softmax(x, dim=1)
        np.testing.assert_allclose(out.numpy(), expected.numpy())


class TestCompatLogSoftmaxDtype(unittest.TestCase):
    """Test dtype casting behavior."""

    def setUp(self):
        paddle.disable_static()

    def test_float32_to_float64(self):
        x = paddle.randn([2, 3, 4], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(
            x, dim=-1, dtype='float64'
        )
        self.assertEqual(out.dtype, paddle.float64)
        x64 = x.cast('float64')
        expected = F.log_softmax(x64, axis=-1)
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-10, atol=1e-10
        )

    def test_float64_to_float32(self):
        x = paddle.randn([2, 3], dtype=paddle.float64)
        out = paddle.compat.nn.functional.log_softmax(x, dim=1, dtype='float32')
        self.assertEqual(out.dtype, paddle.float32)

    def test_dtype_none_preserves_input_dtype(self):
        for dtype in [paddle.float32, paddle.float64]:
            x = paddle.randn([3, 4], dtype=dtype)
            out = paddle.compat.nn.functional.log_softmax(x, dim=-1)
            self.assertEqual(out.dtype, dtype)

    def test_dtype_as_paddle_dtype(self):
        x = paddle.randn([2, 3], dtype=paddle.float32)
        out = paddle.compat.nn.functional.log_softmax(
            x, dim=1, dtype=paddle.float64
        )
        self.assertEqual(out.dtype, paddle.float64)


class TestCompatLogSoftmaxStacklevel(unittest.TestCase):
    """Test that _stacklevel is silently ignored (torch compat)."""

    def setUp(self):
        paddle.disable_static()

    def test_stacklevel_ignored(self):
        x = paddle.randn([3, 4], dtype=paddle.float32)
        out1 = paddle.compat.nn.functional.log_softmax(x, dim=-1)
        out2 = paddle.compat.nn.functional.log_softmax(x, dim=-1, _stacklevel=5)
        np.testing.assert_allclose(out1.numpy(), out2.numpy())


class TestCompatLogSoftmaxErrorHandling(unittest.TestCase):
    """Test that paddle-style keyword arguments are rejected by compat API."""

    def setUp(self):
        paddle.disable_static()

    def test_rejects_x_keyword(self):
        x = paddle.randn([3, 4])
        msg = (
            "paddle.compat.nn.functional.log_softmax() received unexpected keyword argument 'x'. "
            "\nDid you mean to use paddle.nn.functional.log_softmax() instead?"
        )
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.functional.log_softmax(x=x, dim=-1)
        self.assertEqual(str(cm.exception), msg)

    def test_rejects_axis_keyword(self):
        x = paddle.randn([3, 4])
        msg = (
            "paddle.compat.nn.functional.log_softmax() received unexpected keyword argument 'axis'. "
            "\nDid you mean to use paddle.nn.functional.log_softmax() instead?"
        )
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.functional.log_softmax(x, axis=-1)
        self.assertEqual(str(cm.exception), msg)

    def test_rejects_name_keyword(self):
        x = paddle.randn([3, 4])
        msg = (
            "paddle.compat.nn.functional.log_softmax() received unexpected keyword argument 'name'. "
            "\nDid you mean to use paddle.nn.functional.log_softmax() instead?"
        )
        with self.assertRaises(TypeError) as cm:
            paddle.compat.nn.functional.log_softmax(x, dim=-1, name='test')
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
