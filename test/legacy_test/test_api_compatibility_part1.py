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

import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


# Edit By AI Agent
# Test nextafter compatibility
@unittest.skipIf(
    paddle.is_compiled_with_custom_device('iluvatar_gpu'),
    "skip iluvatar_gpu which not register nextafter kernel",
)
class TestNextafterAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('float32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.nextafter(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.nextafter(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.nextafter(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.nextafter(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.nextafter(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.nextafter(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.nextafter(other=y)

        ref_out = np.nextafter(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.nextafter(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.nextafter(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.nextafter(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.nextafter(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.nextafter(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.nextafter(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test round_ compatibility (inplace API)
class TestRoundInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([1.5, 2.3, 3.7, -1.2, -2.8]).astype('float32')

    def test_dygraph_InplaceCompatibility(self):
        """Test round_ inplace API in dynamic mode"""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Tensor method - Paddle positional args
        x1 = x.clone()
        out1 = x1.round_(0)
        assert out1 is x1

        # 2. Tensor method - Paddle keyword args
        x2 = x.clone()
        out2 = x2.round_(decimals=1)
        assert out2 is x2

        # 3. Paddle function - positional args
        x3 = x.clone()
        out3 = paddle.round_(x3, 1)
        assert out3 is x3

        # 4. Paddle function - keyword args
        x4 = x.clone()
        out4 = paddle.round_(x4, decimals=-1)
        assert out4 is x4

        # Verify all outputs
        np.testing.assert_allclose(out1.numpy(), np.round(self.np_x), rtol=1e-6)
        np.testing.assert_allclose(
            out2.numpy(), np.around(self.np_x, decimals=1), rtol=1e-6
        )
        np.testing.assert_allclose(
            out3.numpy(), np.around(self.np_x, decimals=1), rtol=1e-6
        )
        np.testing.assert_allclose(
            out4.numpy(), np.around(self.np_x, decimals=-1), rtol=1e-6
        )

        paddle.enable_static()


# Test erf compatibility
class TestErfAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.4, -0.2, 0.0, 0.1, 0.3]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.erf(x)
        # 2. Paddle keyword arguments
        out2 = paddle.erf(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.erf(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(x)
        out5 = paddle.erf(x, out=out4)
        # 6. Tensor method - positional args
        out6 = x.erf()

        # Verify all outputs
        ref_out = np.array(
            [-0.42839241, -0.22270259, 0.0, 0.11246292, 0.32862678]
        ).astype('float32')
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(
                out.numpy(), ref_out, rtol=1e-5, atol=1e-5
            )

        paddle.enable_static()

    def test_static_Compatibility(self):
        """Test erf API in static mode"""
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.erf(x)
            # 2. Paddle keyword arguments
            out2 = paddle.erf(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.erf(input=x)
            # 4. Tensor method - positional args
            out4 = x.erf()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )

            # Verify all outputs match with loop
            ref_out = np.array(
                [-0.42839241, -0.22270259, 0.0, 0.11246292, 0.32862678]
            ).astype('float32')
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test erf_ inplace compatibility
class TestErfInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.np_x = np.array([-0.4, -0.2, 0.0, 0.1, 0.3]).astype('float32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.array(
            [-0.42839241, -0.22270259, 0.0, 0.11246292, 0.32862678]
        ).astype('float32')

        # 1. Tensor method - positional args
        x1 = x.clone()
        out1 = x1.erf_()
        assert out1 is x1

        # 2. Paddle function - positional args
        x2 = x.clone()
        out2 = paddle.erf_(x2)
        assert out2 is x2

        # Verify all outputs with loop
        for out in [out1, out2]:
            np.testing.assert_allclose(out.numpy(), ref_out, rtol=1e-5)

        paddle.enable_static()


# Test iinfo compatibility
class TestIinfoAPI(unittest.TestCase):
    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        # 1. Paddle positional arguments
        out1 = paddle.iinfo(paddle.int32)
        # 2. Paddle keyword arguments
        out2 = paddle.iinfo(dtype=paddle.int32)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.iinfo(type=paddle.int32)

        # Verify all outputs
        for out in [out1, out2, out3]:
            assert out.min == -2147483648
            assert out.max == 2147483647
            assert out.bits == 32
            assert str(out.dtype) == 'int32'

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            # iinfo is a compile-time function, same in static mode
            out1 = paddle.iinfo(paddle.int32)
            out2 = paddle.iinfo(dtype=paddle.int32)
            out3 = paddle.iinfo(type=paddle.int32)

            for out in [out1, out2, out3]:
                assert out.min == -2147483648
                assert out.max == 2147483647


# Test additional paddle.dtype.itemsize compatibility.
class TestDtypeItemsizeExtendedAPI(unittest.TestCase):
    EXPECTED = (
        'float16',
        'bfloat16',
        'float32',
        'float64',
        'complex64',
        'complex128',
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
        'bool',
        'float8_e4m3fn',
        'float8_e5m2',
    )

    def test_dtype_str(self):
        for name in self.EXPECTED:
            with self.subTest(dtype=name):
                self.assertEqual(str(getattr(paddle, name)), f'paddle.{name}')

    def test_int_alias_matches(self):
        self.assertEqual(paddle.int.itemsize, paddle.int32.itemsize)


class TestFloatingDtypeAPI(unittest.TestCase):
    EXPECTED = {
        'float16': (16, 0.0009765625, -65504.0, 65504.0, 6.103515625e-05),
        'bfloat16': (
            16,
            0.0078125,
            -3.3895313892515355e38,
            3.3895313892515355e38,
            1.1754943508222875e-38,
        ),
        'float32': (
            32,
            1.1920928955078125e-07,
            -3.4028234663852886e38,
            3.4028234663852886e38,
            1.1754943508222875e-38,
        ),
        'float64': (
            64,
            2.220446049250313e-16,
            -1.7976931348623157e308,
            1.7976931348623157e308,
            2.2250738585072014e-308,
        ),
        'float8_e4m3fn': (8, 0.125, -448.0, 448.0, 0.015625),
        'float8_e5m2': (8, 0.25, -57344.0, 57344.0, 6.103515625e-05),
    }

    def check_finfo(self, info, name):
        bits, eps, min_value, max_value, tiny = self.EXPECTED[name]
        self.assertEqual(info.bits, bits)
        self.assertEqual(str(info.dtype), name)
        self.assertEqual(info.eps, eps)
        self.assertEqual(info.min, min_value)
        self.assertEqual(info.max, max_value)
        self.assertEqual(info.smallest_normal, tiny)
        self.assertEqual(info.tiny, tiny)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        for name in self.EXPECTED:
            dtype = getattr(paddle, name)
            with self.subTest(dtype=name):
                # 1. Paddle Positional arguments
                out1 = paddle.finfo(dtype)
                # 2. Paddle keyword arguments
                out2 = paddle.finfo(dtype=dtype)
                # 3. PyTorch keyword arguments (alias)
                out3 = paddle.finfo(type=dtype)

                # Verify all outputs
                for out in [out1, out2, out3]:
                    self.check_finfo(out, name)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            for name in self.EXPECTED:
                dtype = getattr(paddle, name)
                with self.subTest(dtype=name):
                    # 1. Paddle Positional arguments
                    out1 = paddle.finfo(dtype)
                    # 2. Paddle keyword arguments
                    out2 = paddle.finfo(dtype=dtype)
                    # 3. PyTorch keyword arguments (alias)
                    out3 = paddle.finfo(type=dtype)

                    # Verify all outputs
                    for out in [out1, out2, out3]:
                        self.check_finfo(out, name)


class TestIntegerDtypeAPI(unittest.TestCase):
    EXPECTED = {
        'uint8': (0, 255, 8),
        'uint16': (0, 65535, 16),
        'uint32': (0, 4294967295, 32),
        'uint64': (0, 18446744073709551615, 64),
        'int8': (-128, 127, 8),
        'int16': (-32768, 32767, 16),
        'int32': (-2147483648, 2147483647, 32),
        'int64': (-9223372036854775808, 9223372036854775807, 64),
    }

    def check_iinfo(self, info, name):
        min_value, max_value, bits = self.EXPECTED[name]
        self.assertEqual(info.min, min_value)
        self.assertEqual(info.max, max_value)
        self.assertEqual(info.bits, bits)
        self.assertEqual(str(info.dtype), name)
        self.assertIn(f'max={max_value}', repr(info))

    def test_dygraph_Compatibility(self):
        paddle.disable_static()

        for name in self.EXPECTED:
            dtype = getattr(paddle, name)
            with self.subTest(dtype=name):
                # 1. Paddle Positional arguments
                out1 = paddle.iinfo(dtype)
                # 2. Paddle keyword arguments
                out2 = paddle.iinfo(dtype=dtype)
                # 3. PyTorch keyword arguments (alias)
                out3 = paddle.iinfo(type=dtype)
                out4 = paddle.iinfo(name)
                out5 = paddle.iinfo(np.dtype(name))
                out6 = paddle.iinfo(getattr(np, name))
                out7 = paddle.iinfo(name.upper())
                out8 = paddle.iinfo(f" {name} ")

                # Verify all outputs
                for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
                    self.check_iinfo(out, name)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            for name in self.EXPECTED:
                dtype = getattr(paddle, name)
                with self.subTest(dtype=name):
                    # 1. Paddle Positional arguments
                    out1 = paddle.iinfo(dtype)
                    # 2. Paddle keyword arguments
                    out2 = paddle.iinfo(dtype=dtype)
                    # 3. PyTorch keyword arguments (alias)
                    out3 = paddle.iinfo(type=dtype)
                    out4 = paddle.iinfo(name)
                    out5 = paddle.iinfo(np.dtype(name))
                    out6 = paddle.iinfo(getattr(np, name))
                    out7 = paddle.iinfo(name.upper())
                    out8 = paddle.iinfo(f" {name} ")

                    # Verify all outputs
                    for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
                        self.check_iinfo(out, name)


class TestLegacyVarTypeDtypeAPI(unittest.TestCase):
    def test_uint_iinfo_with_pir_disabled(self):
        code = textwrap.dedent(
            """
            import numpy as np
            import paddle

            expected = {
                'uint16': (0, 65535, 16),
                'uint32': (0, 4294967295, 32),
                'uint64': (0, 18446744073709551615, 64),
            }
            assert str(paddle.bfloat16) == 'paddle.bfloat16'
            for name, (min_value, max_value, bits) in expected.items():
                dtype = getattr(paddle, name)
                assert str(dtype) == f'paddle.{name}'
                for arg in (
                    dtype,
                    name,
                    np.dtype(name),
                    getattr(np, name),
                    name.upper(),
                    f" {name} ",
                ):
                    info = paddle.iinfo(arg)
                    assert info.min == min_value
                    assert info.max == max_value
                    assert info.bits == bits
                    assert str(info.dtype) == name
            """
        )
        env = os.environ.copy()
        env["FLAGS_enable_pir_api"] = "0"
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )


# Test angle compatibility
class TestAngleAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        np_x_real = np.random.randn(5, 6).astype('float32')
        np_x_imag = np.random.randn(5, 6).astype('float32')
        self.np_x = np_x_real + 1j * np_x_imag

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.angle(x)
        # 2. Paddle keyword arguments
        out2 = paddle.angle(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.angle(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.angle(x, out=out4)
        # 6. Tensor method
        out6 = x.angle()

        # Verify all outputs
        ref_out = np.angle(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-5
            )
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='complex64')

            # 1. Paddle positional arguments
            out1 = paddle.angle(x)
            # 2. Paddle keyword arguments
            out2 = paddle.angle(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.angle(input=x)
            # 4. Tensor method
            out4 = x.angle()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.angle(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)


# Test atan compatibility
class TestAtanAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.atan(x)
        # 2. Paddle keyword arguments
        out2 = paddle.atan(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.atan(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(x)
        out5 = paddle.atan(x, out=out4)
        # 6. Tensor method
        out6 = x.atan()

        ref_out = np.arctan(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')

            out1 = paddle.atan(x)
            out2 = paddle.atan(x=x)
            out3 = paddle.atan(input=x)
            out4 = x.atan()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.arctan(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


class TestAtan2API(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')
        self.np_y = np.random.randn(5, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.atan2(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.atan2(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.atan2(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.atan2(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(out1)
        out6 = paddle.atan2(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.atan2(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.atan2(other=y)

        # Verify all outputs
        ref_out = np.arctan2(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.atan2(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.atan2(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.atan2(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.atan2(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.atan2(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.arctan2(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


class TestHypotAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')
        self.np_y = np.random.randn(5, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.hypot(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.hypot(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.hypot(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.hypot(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.hypot(x, y, out=out5)
        assert out5 is out6
        # 7. Tensor method - positional args
        out7 = x.hypot(y)
        # 8. Tensor method - keyword args (PyTorch alias)
        out8 = x.hypot(other=y)

        ref_out = np.hypot(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), atol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.hypot(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.hypot(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.hypot(input=x, other=y)
            # 4. Tensor method - positional args
            out4 = x.hypot(y)
            # 5. Tensor method - keyword args (PyTorch alias)
            out5 = x.hypot(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.hypot(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, atol=1e-6)


class TestHypotInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')
        self.np_y = np.random.randn(5, 6).astype('float32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.hypot(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().hypot_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().hypot_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().hypot_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.hypot_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.hypot_(x=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.hypot_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), atol=1e-6)


# Test fmax compatibility
class TestFmaxAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')
        self.np_y = np.random.randn(5, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.fmax(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.fmax(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.fmax(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.fmax(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.fmax(x, y, out=out5)
        # 7. Tensor method - positional args
        out7 = x.fmax(y)
        # 8. Tensor method - keyword args (PyTorch alias)
        out8 = x.fmax(other=y)

        ref_out = np.fmax(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.fmax(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.fmax(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.fmax(input=x, other=y)
            # 4. Tensor method - positional args
            out4 = x.fmax(y)
            # 5. Tensor method - keyword args (PyTorch alias)
            out5 = x.fmax(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.fmax(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test fmin compatibility
class TestFminAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')
        self.np_y = np.random.randn(5, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.fmin(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.fmin(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.fmin(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.fmin(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.fmin(x, y, out=out5)
        # 7. Tensor method - positional args
        out7 = x.fmin(y)
        # 8. Tensor method - keyword args (PyTorch alias)
        out8 = x.fmin(other=y)

        ref_out = np.fmin(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.fmin(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.fmin(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.fmin(input=x, other=y)
            # 4. Tensor method - positional args
            out4 = x.fmin(y)
            # 5. Tensor method - keyword args (PyTorch alias)
            out5 = x.fmin(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.fmin(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test bincount compatibility
class TestBincountAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [10]).astype('int64')
        self.np_weights = np.random.random([10]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        weights = paddle.to_tensor(self.np_weights)

        # 1. Paddle positional arguments
        out1 = paddle.bincount(x, weights, 6)
        # 2. Paddle keyword arguments
        out2 = paddle.bincount(x=x, weights=weights, minlength=6)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bincount(input=x, weights=weights, minlength=6)
        # 4. Mixed arguments
        out4 = paddle.bincount(x, weights=weights, minlength=6)

        ref_out = np.bincount(self.np_x, weights=self.np_weights, minlength=6)
        for out in [out1, out2, out3, out4]:
            np.testing.assert_allclose(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[10], dtype='int64')
            weights = paddle.static.data(
                name="weights", shape=[10], dtype='float32'
            )

            # 1. Paddle positional arguments
            out1 = paddle.bincount(x, weights, 6)
            # 2. Paddle keyword arguments
            out2 = paddle.bincount(x=x, weights=weights, minlength=6)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bincount(input=x, weights=weights, minlength=6)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "weights": self.np_weights},
                fetch_list=[out1, out2, out3],
            )

            ref_out = np.bincount(
                self.np_x, weights=self.np_weights, minlength=6
            )
            for out in fetches:
                np.testing.assert_allclose(ref_out, out)


# Test diag compatibility
class TestDiagAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(3, 3).astype('float32')
        self.np_v = np.random.randn(3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        v = paddle.to_tensor(self.np_v)

        # 1. Paddle positional arguments
        out1 = paddle.diag(x, 1)
        # 2. Paddle keyword arguments
        out2 = paddle.diag(x=x, offset=1)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.diag(input=x, diagonal=1)
        # 4. Mixed arguments
        out4 = paddle.diag(x, offset=1)
        # 5-6. out parameter test
        out5 = paddle.empty_like(v)
        out6 = paddle.diag(v, out=out5)
        # 7. Tensor method - positional args
        out7 = x.diag(1)
        # 8. Tensor method - keyword args (PyTorch alias)
        out8 = x.diag(diagonal=1)

        ref_diag_v = np.diag(self.np_v)
        ref_diag_x_offset = np.diag(self.np_x, 1)

        for out in [out1, out2, out3, out4, out7, out8]:
            np.testing.assert_allclose(ref_diag_x_offset, out.numpy())
        np.testing.assert_allclose(ref_diag_v, out5.numpy())
        np.testing.assert_allclose(ref_diag_v, out6.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 3], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.diag(x, 1)
            # 2. Paddle keyword arguments
            out2 = paddle.diag(x=x, offset=1)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.diag(input=x, diagonal=1)
            # 4. Tensor method - positional args
            out4 = x.diag(1)
            # 5. Tensor method - keyword args (PyTorch alias)
            out5 = x.diag(diagonal=1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.diag(self.np_x, 1)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test heaviside compatibility
class TestHeavisideAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(5, 6).astype('float32')
        self.np_y = np.random.randn(5, 6).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.heaviside(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.heaviside(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.heaviside(input=x, values=y)
        # 4. Mixed arguments
        out4 = paddle.heaviside(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(out1)
        out6 = paddle.heaviside(x, y, out=out5)
        # 7. Tensor method - positional args
        out7 = x.heaviside(y)
        # 8. Tensor method - keyword args (PyTorch alias)
        out8 = x.heaviside(values=y)

        # Verify all outputs
        ref_out = np.heaviside(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.heaviside(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.heaviside(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.heaviside(input=x, values=y)
            # 4. Tensor method - args
            out4 = x.heaviside(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.heaviside(values=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.heaviside(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


class TestAsinhAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.asinh(x)
        # 2. Paddle keyword arguments
        out2 = paddle.asinh(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.asinh(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.asinh(x, out=out4)
        # 6. Tensor method
        out6 = x.asinh()

        # Verify all outputs
        ref_out = np.arcsinh(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.asinh(x)
            # 2. Paddle keyword arguments
            out2 = paddle.asinh(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.asinh(input=x)
            # 4. Tensor method
            out4 = x.asinh()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.arcsinh(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


class TestReciprocalAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 8, [5, 6]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.reciprocal(x)
        # 2. Paddle keyword arguments
        out2 = paddle.reciprocal(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.reciprocal(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.reciprocal(x, out=out4)
        # 6. Tensor method
        out6 = x.reciprocal()

        # Verify all outputs
        ref_out = 1.0 / self.np_x
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.reciprocal(x)
            # 2. Paddle keyword arguments
            out2 = paddle.reciprocal(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.reciprocal(input=x)
            # 4. Tensor method
            out4 = x.reciprocal()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = 1.0 / self.np_x
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


class TestSquareAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.square(x)
        # 2. Paddle keyword arguments
        out2 = paddle.square(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.square(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.square(x, out=out4)
        # 6. Tensor method
        out6 = x.square()

        # Verify all outputs
        ref_out = np.square(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.square(x)
            # 2. Paddle keyword arguments
            out2 = paddle.square(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.square(input=x)
            # 4. Tensor method
            out4 = x.square()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.square(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test masked_fill compatibility
class TestMaskedFillAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 10, [3, 3]).astype('float32')
        self.np_mask = np.random.randint(0, 2, [3, 3]).astype(bool)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        mask = paddle.to_tensor(self.np_mask)

        # 1. Paddle positional arguments
        out1 = paddle.masked_fill(x, mask, 0.0)
        # 2. Paddle keyword arguments
        out2 = paddle.masked_fill(x=x, mask=mask, value=0.0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.masked_fill(input=x, mask=mask, value=0.0)
        # 4. Mixed arguments
        out4 = paddle.masked_fill(x, mask=mask, value=0.0)
        # 5. Tensor method - positional args
        out5 = x.masked_fill(mask, 0.0)
        # 6. Tensor method - keyword args (PyTorch alias)
        out6 = x.masked_fill(mask=mask, value=0.0)

        # Verify all outputs are equal
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out1.numpy(), out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 3], dtype='float32')
            mask = paddle.static.data(name="mask", shape=[3, 3], dtype='bool')

            # Position args
            out1 = paddle.masked_fill(x, mask, 0.0)
            # Paddle keyword args
            out2 = paddle.masked_fill(x=x, mask=mask, value=0.0)
            # Torch keyword args (input alias)
            out3 = paddle.masked_fill(input=x, mask=mask, value=0.0)
            # Tensor method
            out4 = x.masked_fill(mask, 0.0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "mask": self.np_mask},
                fetch_list=[out1, out2, out3, out4],
            )

        # Verify all outputs are equal
        for i in range(1, len(fetches)):
            np.testing.assert_allclose(fetches[0], fetches[i])


class TestTanAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.tan(x)
        # 2. Paddle keyword arguments
        out2 = paddle.tan(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.tan(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.tan(x, out=out4)
        # 6. Tensor method
        out6 = x.tan()

        # Verify all outputs
        ref_out = np.tan(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='float32')

            # 1. Paddle positional arguments
            out1 = paddle.tan(x)
            # 2. Paddle keyword arguments
            out2 = paddle.tan(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.tan(input=x)
            # 4. Tensor method
            out4 = x.tan()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.tan(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


# Test bitwise_and compatibility
class TestBitwiseAndAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.bitwise_and(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.bitwise_and(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bitwise_and(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.bitwise_and(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.bitwise_and(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.bitwise_and(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.bitwise_and(other=y)

        ref_out = np.bitwise_and(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_array_equal(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='int32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='int32')

            # 1. Paddle positional arguments
            out1 = paddle.bitwise_and(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.bitwise_and(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bitwise_and(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.bitwise_and(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.bitwise_and(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.bitwise_and(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_or compatibility
class TestBitwiseOrAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.bitwise_or(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.bitwise_or(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bitwise_or(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.bitwise_or(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(out1)
        out6 = paddle.bitwise_or(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.bitwise_or(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.bitwise_or(other=y)

        ref_out = np.bitwise_or(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='int32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='int32')

            # 1. Paddle positional arguments
            out1 = paddle.bitwise_or(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.bitwise_or(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bitwise_or(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.bitwise_or(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.bitwise_or(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.bitwise_or(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_not compatibility
class TestBitwiseNotAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.bitwise_not(x)
        # 2. Paddle keyword arguments
        out2 = paddle.bitwise_not(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bitwise_not(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty_like(out1)
        out5 = paddle.bitwise_not(x, out=out4)
        # 6. Tensor method
        out6 = x.bitwise_not()

        ref_out = np.bitwise_not(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='int32')

            # 1. Paddle positional arguments
            out1 = paddle.bitwise_not(x)
            # 2. Paddle keyword arguments
            out2 = paddle.bitwise_not(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bitwise_not(input=x)
            # 4. Tensor method
            out4 = x.bitwise_not()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_not(self.np_x)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_xor compatibility
class TestBitwiseXorAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.bitwise_xor(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.bitwise_xor(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bitwise_xor(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.bitwise_xor(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty_like(out1)
        out6 = paddle.bitwise_xor(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.bitwise_xor(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.bitwise_xor(other=y)

        ref_out = np.bitwise_xor(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='int32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='int32')

            # 1. Paddle positional arguments
            out1 = paddle.bitwise_xor(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.bitwise_xor(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bitwise_xor(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.bitwise_xor(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.bitwise_xor(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.bitwise_xor(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_and_ inplace compatibility
class TestBitwiseAndInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.bitwise_and(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().bitwise_and_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().bitwise_and_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().bitwise_and_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.bitwise_and_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.bitwise_and_(x=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.bitwise_and_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_or_ inplace compatibility
class TestBitwiseOrInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.bitwise_or(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().bitwise_or_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().bitwise_or_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().bitwise_or_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.bitwise_or_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.bitwise_or_(x=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.bitwise_or_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_xor_ inplace compatibility
class TestBitwiseXorInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')
        self.np_y = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.bitwise_xor(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().bitwise_xor_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().bitwise_xor_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().bitwise_xor_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.bitwise_xor_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.bitwise_xor_(x=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.bitwise_xor_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_not_ inplace compatibility
class TestBitwiseNotInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(0, 8, [5, 6]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        ref_out = np.bitwise_not(self.np_x)

        # 1. Tensor method - positional args
        out1 = x.clone().bitwise_not_()
        # 2. Tensor method - keyword args
        out2 = x.clone().bitwise_not_()
        # 3. Paddle function - positional args
        out3 = paddle.bitwise_not_(x.clone())
        # 4. Paddle function - keyword args (PyTorch alias)
        out4 = paddle.bitwise_not_(input=x.clone())

        for out in [out1, out2, out3, out4]:
            np.testing.assert_array_equal(ref_out, out.numpy())


class TestCdistAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.rand(3, 5, 4).astype('float32')
        self.np_y = np.random.rand(3, 2, 4).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.cdist(x, y)
        out2 = paddle.cdist(x=x, y=y)
        out3 = paddle.cdist(x1=x, x2=y)
        out4 = paddle.cdist(x, y, p=2.0)
        out5 = paddle.cdist(
            x1=x,
            x2=y,
            p=2.0,
            compute_mode='use_mm_for_euclid_dist_if_necessary',
        )
        for out in [out2, out3, out4, out5]:
            np.testing.assert_allclose(out1.numpy(), out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[3, 5, 4], dtype='float32')
            y = paddle.static.data(name="y", shape=[3, 2, 4], dtype='float32')
            out1 = paddle.cdist(x, y)
            out2 = paddle.cdist(x=x, y=y)
            out3 = paddle.cdist(x1=x, x2=y)
            out4 = paddle.cdist(x, y, p=2.0)
            out5 = paddle.cdist(
                x1=x,
                x2=y,
                p=2.0,
                compute_mode='use_mm_for_euclid_dist_if_necessary',
            )
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            for out in fetches:
                np.testing.assert_allclose(fetches[0], out)

    def test_zero_size(self):
        """Test edge cases: r1==0, r2==0, c1==0."""
        paddle.disable_static()
        # r1==0 (3D batched)
        x1 = paddle.to_tensor(np.random.rand(2, 0, 4).astype('float32'))
        y1 = paddle.to_tensor(np.random.rand(2, 3, 4).astype('float32'))
        out1 = paddle.cdist(x1, y1)
        self.assertEqual(out1.shape, [2, 0, 3])
        # r2==0 (2D non-batched)
        x2 = paddle.to_tensor(np.random.rand(3, 4).astype('float32'))
        y2 = paddle.to_tensor(np.random.rand(0, 4).astype('float32'))
        out2 = paddle.cdist(x2, y2)
        self.assertEqual(out2.shape, [3, 0])
        # c1==0 (3D batched, should return zeros)
        x3 = paddle.to_tensor(np.random.rand(2, 3, 0).astype('float32'))
        y3 = paddle.to_tensor(np.random.rand(2, 2, 0).astype('float32'))
        out3 = paddle.cdist(x3, y3)
        self.assertEqual(out3.shape, [2, 3, 2])
        np.testing.assert_allclose(out3.numpy(), 0.0)
        paddle.enable_static()


class TestAddmmAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(2, 3).astype('float32')
        self.np_x = np.random.rand(2, 4).astype('float32')
        self.np_y = np.random.rand(4, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
        # 1. Paddle positional arguments
        out1 = paddle.addmm(input, x, y, 1.0, 1.0)
        # 2. Paddle keyword arguments
        out2 = paddle.addmm(input=input, x=x, y=y, beta=1.0, alpha=1.0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.addmm(beta=1.0, alpha=1.0, input=input, mat1=x, mat2=y)
        # 4. Mixed arguments
        out4 = paddle.addmm(input, x, y, beta=1.0, alpha=1.0)
        # 5-6. out parameter test
        out5 = paddle.empty_like(input)
        out6 = paddle.addmm(input, x, y, out=out5)
        # 7. Tensor method - args
        out7 = input.addmm(x, y, beta=1.0, alpha=1.0)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = input.addmm(mat1=x, mat2=y, beta=1.0, alpha=1.0)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)

        paddle.enable_static()

    def test_error(self):
        """Test invalid input dimensions that should raise ValueError."""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # Test 3D input (invalid)
        input_3d = paddle.to_tensor(np.random.rand(2, 2, 3).astype('float32'))
        with self.assertRaises(ValueError):
            paddle.addmm(input_3d, x, y)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=[2, 3], dtype='float32'
            )
            x = paddle.static.data(name="x", shape=[2, 4], dtype='float32')
            y = paddle.static.data(name="y", shape=[4, 3], dtype='float32')
            # 1. Paddle positional arguments
            out1 = paddle.addmm(input, x, y, 1.0, 1.0)
            # 2. Paddle keyword arguments
            out2 = paddle.addmm(input=input, x=x, y=y, beta=1.0, alpha=1.0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.addmm(beta=1, alpha=1, input=input, mat1=x, mat2=y)
            # 4. Tensor method - args
            out4 = input.addmm(x, y, beta=1.0, alpha=1.0)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = input.addmm(mat1=x, mat2=y, beta=1.0, alpha=1.0)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
            for out in fetches:
                np.testing.assert_allclose(ref_out, out, rtol=1e-6)


class TestAddmmInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(2, 3).astype('float32')
        self.np_x = np.random.rand(2, 4).astype('float32')
        self.np_y = np.random.rand(4, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.addmm_(input.clone(), x, y, beta=1.0, alpha=1.0)
        out2 = paddle.addmm_(input=input.clone(), x=x, y=y, beta=1.0, alpha=1.0)
        out3 = paddle.addmm_(
            input=input.clone(), mat1=x, mat2=y, beta=1.0, alpha=1.0
        )
        out4 = input.clone().addmm_(x, y, beta=1.0, alpha=1.0)
        out5 = input.clone().addmm_(x=x, y=y, beta=1.0, alpha=1.0)
        out6 = input.clone().addmm_(mat1=x, mat2=y, beta=1.0, alpha=1.0)
        ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()


class TestLdexpInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(3, 4).astype('float32')
        self.np_y = np.random.randint(-3, 4, [3, 4]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.ldexp(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().ldexp_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().ldexp_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().ldexp_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.ldexp_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.ldexp_(input=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.ldexp_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()


# Test baddbmm API compatibility (paddle.baddbmm and paddle.Tensor.baddbmm)
class TestBaddbmmAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3, 2, 3).astype('float32')
        self.np_x = np.random.rand(3, 2, 4).astype('float32')
        self.np_y = np.random.rand(3, 4, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        # 1. Paddle positional arguments
        out1 = paddle.baddbmm(input, x, y, 1.0, 1.0)
        # 2. Paddle keyword arguments
        out2 = paddle.baddbmm(input=input, x=x, y=y, beta=1.0, alpha=1.0)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.baddbmm(
            beta=1.0, alpha=1.0, input=input, batch1=x, batch2=y
        )
        # 4. Mixed arguments
        out4 = paddle.baddbmm(input, x, y, beta=1.0, alpha=1.0)
        # 5-6. out parameter test
        out5 = paddle.empty_like(input)
        out6 = paddle.baddbmm(input, x, y, out=out5)
        # 7. Tensor method - args
        out7 = input.baddbmm(x, y, beta=1.0, alpha=1.0)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = input.baddbmm(batch1=x, batch2=y, beta=1.0, alpha=1.0)
        ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)

        # 2D input (1,1) broadcasts to result shape [3, 2, 3]
        input_2d = paddle.to_tensor(np.array([[0.5]]).astype('float32'))
        out8 = paddle.baddbmm(input_2d, x, y)
        ref_out_2d = 0.5 + self.np_x @ self.np_y
        np.testing.assert_allclose(ref_out_2d, out8.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_error(self):
        """Test invalid input dimensions that should raise ValueError."""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # Test 1D input (invalid)
        input_1d = paddle.to_tensor(np.random.rand(3).astype('float32'))
        with self.assertRaises(ValueError):
            paddle.baddbmm(input_1d, x, y)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=[3, 2, 3], dtype='float32'
            )
            x = paddle.static.data(name="x", shape=[3, 2, 4], dtype='float32')
            y = paddle.static.data(name="y", shape=[3, 4, 3], dtype='float32')
            # 1. Paddle positional arguments
            out1 = paddle.baddbmm(input, x, y, 1.0, 1.0)
            # 2. Paddle keyword arguments
            out2 = paddle.baddbmm(input=input, x=x, y=y, beta=1.0, alpha=1.0)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.baddbmm(
                beta=1, alpha=1, input=input, batch1=x, batch2=y
            )
            # 4. Tensor method - args
            out4 = input.baddbmm(x, y, beta=1.0, alpha=1.0)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = input.baddbmm(batch1=x, batch2=y, beta=1.0, alpha=1.0)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
            for out in fetches:
                np.testing.assert_allclose(ref_out, out, rtol=1e-6)


# Test baddbmm_ API compatibility (paddle.baddbmm_ and paddle.Tensor.baddbmm_)
class TestBaddbmmInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_input = np.random.rand(3, 2, 3).astype('float32')
        self.np_x = np.random.rand(3, 2, 4).astype('float32')
        self.np_y = np.random.rand(3, 4, 3).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.baddbmm_(input.clone(), x, y, beta=0.5, alpha=0.7)
        out2 = paddle.baddbmm_(
            input=input.clone(), x=x, y=y, beta=0.5, alpha=0.7
        )
        out3 = paddle.baddbmm_(
            input=input.clone(), batch1=x, batch2=y, beta=0.5, alpha=0.7
        )
        out4 = input.clone().baddbmm_(x, y, beta=0.5, alpha=0.7)
        out5 = input.clone().baddbmm_(x=x, y=y, beta=0.5, alpha=0.7)
        out6 = input.clone().baddbmm_(batch1=x, batch2=y, beta=0.5, alpha=0.7)
        ref_out = 0.5 * self.np_input + 0.7 * self.np_x @ self.np_y
        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()


# Test bitwise_left_shift compatibility
class TestBitwiseLeftShiftAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 10, [5, 6]).astype('int32')
        self.np_y = np.random.randint(1, 5, [5, 6]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.bitwise_left_shift(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.bitwise_left_shift(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bitwise_left_shift(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.bitwise_left_shift(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty([5, 6], dtype='int32')
        out6 = paddle.bitwise_left_shift(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.bitwise_left_shift(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.bitwise_left_shift(other=y)

        ref_out = np.left_shift(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='int32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='int32')

            # 1. Paddle positional arguments
            out1 = paddle.bitwise_left_shift(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.bitwise_left_shift(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bitwise_left_shift(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.bitwise_left_shift(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.bitwise_left_shift(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )
            ref_out = np.left_shift(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_left_shift_ inplace compatibility
class TestBitwiseLeftShiftInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 10, [5, 6]).astype('int32')
        self.np_y = np.random.randint(1, 5, [5, 6]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.left_shift(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().bitwise_left_shift_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().bitwise_left_shift_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().bitwise_left_shift_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.bitwise_left_shift_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.bitwise_left_shift_(x=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.bitwise_left_shift_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_right_shift compatibility
class TestBitwiseRightShiftAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(10, 100, [5, 6]).astype('int32')
        self.np_y = np.random.randint(1, 5, [5, 6]).astype('int32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.bitwise_right_shift(x, y)
        # 2. Paddle keyword arguments
        out2 = paddle.bitwise_right_shift(x=x, y=y)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.bitwise_right_shift(input=x, other=y)
        # 4. Mixed arguments
        out4 = paddle.bitwise_right_shift(x, y=y)
        # 5-6. out parameter test
        out5 = paddle.empty([5, 6], dtype='int32')
        out6 = paddle.bitwise_right_shift(x, y, out=out5)
        # 7. Tensor method - args
        out7 = x.bitwise_right_shift(y)
        # 8. Tensor method - kwargs (PyTorch alias)
        out8 = x.bitwise_right_shift(other=y)

        ref_out = np.right_shift(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_array_equal(ref_out, out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='int32')
            y = paddle.static.data(name="y", shape=[5, 6], dtype='int32')

            # 1. Paddle positional arguments
            out1 = paddle.bitwise_right_shift(x, y)
            # 2. Paddle keyword arguments
            out2 = paddle.bitwise_right_shift(x=x, y=y)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.bitwise_right_shift(input=x, other=y)
            # 4. Tensor method - args
            out4 = x.bitwise_right_shift(y)
            # 5. Tensor method - kwargs (PyTorch alias)
            out5 = x.bitwise_right_shift(other=y)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4, out5],
            )

            ref_out = np.right_shift(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_right_shift_ inplace compatibility
class TestBitwiseRightShiftInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(10, 100, [5, 6]).astype('int32')
        self.np_y = np.random.randint(1, 5, [5, 6]).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.right_shift(self.np_x, self.np_y)

        # 1. Tensor method - positional args
        out1 = x.clone().bitwise_right_shift_(y)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone().bitwise_right_shift_(y=y)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone().bitwise_right_shift_(other=y)
        # 4. Paddle function - positional args
        out4 = paddle.bitwise_right_shift_(x.clone(), y)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.bitwise_right_shift_(x=x.clone(), y=y)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.bitwise_right_shift_(input=x.clone(), other=y)

        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test cauchy_ inplace compatibility
class TestCauchyInplaceAPI(unittest.TestCase):
    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        # 1. Tensor method - positional args
        out1 = paddle.randn([3, 4], dtype='float32')
        out1.cauchy_(1.0, 2.0)
        # 2. Tensor method - Paddle keyword args
        out2 = paddle.randn([3, 4], dtype='float32')
        out2.cauchy_(loc=1.0, scale=2.0)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = paddle.randn([3, 4], dtype='float32')
        out3.cauchy_(median=1.0, sigma=2.0)
        # 4. Paddle function - positional args
        out4 = paddle.randn([3, 4], dtype='float32')
        paddle.cauchy_(out4, 1.0, 2.0)
        # 5. Paddle function - Paddle keyword args
        out5 = paddle.randn([3, 4], dtype='float32')
        paddle.cauchy_(out5, loc=1.0, scale=2.0)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = paddle.randn([3, 4], dtype='float32')
        paddle.cauchy_(out6, median=1.0, sigma=2.0)

        for out in [out1, out2, out3, out4, out5, out6]:
            self.assertEqual(out.shape, [3, 4])


class TestTensorCumsumInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randint(1, 5, size=(3, 4)).astype('int64')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Tensor method - positional args
        out1 = x.clone()
        out1.cumsum_(1)
        # 2. Tensor method - Paddle keyword args
        out2 = x.clone()
        out2.cumsum_(axis=1)
        # 3. Tensor method - PyTorch keyword args (alias)
        out3 = x.clone()
        out3.cumsum_(dim=1)
        # 4. Paddle function - positional args
        out4 = x.clone()
        paddle.cumsum_(out4, 1)
        # 5. Paddle function - Paddle keyword args
        out5 = x.clone()
        paddle.cumsum_(out5, axis=1)
        # 6. Paddle function - PyTorch keyword args (alias)
        out6 = x.clone()
        paddle.cumsum_(out6, dim=1)

        ref = np.cumsum(self.np_x, axis=1)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref, out.numpy())


# Test real compatibility
class TestRealAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        np_x_real = np.random.randn(5, 6).astype('float32')
        np_x_imag = np.random.randn(5, 6).astype('float32')
        self.np_x = np_x_real + 1j * np_x_imag

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = paddle.real(x)
        # 2. Paddle keyword arguments
        out2 = paddle.real(x=x)
        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.real(input=x)
        # 4-5. out parameter test
        out4 = paddle.empty([5, 6], dtype='float32')
        out5 = paddle.real(x, out=out4)
        # 6. Tensor method
        out6 = x.real()

        # Verify all outputs
        ref_out = np.real(self.np_x)
        for out in [out1, out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[5, 6], dtype='complex64')

            # 1. Paddle positional arguments
            out1 = paddle.real(x)
            # 2. Paddle keyword arguments
            out2 = paddle.real(x=x)
            # 3. PyTorch keyword arguments (alias)
            out3 = paddle.real(input=x)
            # 4. Tensor method
            out4 = x.real()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.real(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(ref_out, out, rtol=1e-6)


# Test pixel_shuffle compatibility
class TestPixelShuffleAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.random.randn(2, 9, 4, 4).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle positional arguments
        out1 = F.pixel_shuffle(x, 3)
        # 2. Paddle keyword arguments
        out2 = F.pixel_shuffle(x=x, upscale_factor=3)
        # 3. PyTorch keyword arguments (alias)
        out3 = F.pixel_shuffle(input=x, upscale_factor=3)
        # 4. Mixed arguments
        out4 = F.pixel_shuffle(x, upscale_factor=3)

        # Verify all outputs match
        for out in [out2, out3, out4]:
            np.testing.assert_array_equal(out1.numpy(), out.numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=[2, 9, 4, 4], dtype='float32'
            )

            # 1. Paddle positional arguments
            out1 = F.pixel_shuffle(x, 3)
            # 2. Paddle keyword arguments
            out2 = F.pixel_shuffle(x=x, upscale_factor=3)
            # 3. PyTorch keyword arguments (alias)
            out3 = F.pixel_shuffle(input=x, upscale_factor=3)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )
            for out in fetches[1:]:
                np.testing.assert_array_equal(fetches[0], out)


# Test paddle.set_rng_state compatibility
class TestSetRngStateAPI(unittest.TestCase):
    def test_Compatibility(self):
        states = paddle.get_rng_state()
        # 1. positional argument
        paddle.set_rng_state(states)
        # 2. paddle-style keyword argument
        paddle.set_rng_state(state_list=states)
        # 3. torch-style keyword argument
        paddle.set_rng_state(new_state=states)


# Test DistributedSampler compatibility
class TestDistributedSamplerAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)

        class SimpleDataset:
            def __init__(self, size):
                self.size = size

            def __getitem__(self, idx):
                x = idx
                y = 2 * idx
                return x, y

            def __len__(self):
                return self.size

        self.dataset = SimpleDataset(100)

    def test_dygraph_Compatibility(self):
        """Test DistributedSampler as alias for DistributedBatchSampler"""
        # 1. positional arguments
        sampler1 = paddle.utils.data.DistributedSampler(
            self.dataset, 2, 0, False, 2026, False
        )
        # 2. keyword arguments
        sampler2 = paddle.utils.data.DistributedSampler(
            dataset=self.dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
            seed=2026,
            drop_last=False,
        )
        # Verify both samplers produce same batches
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        self.assertEqual(len(batches1), len(batches2))
        for b1, b2 in zip(batches1, batches2):
            self.assertEqual(b1, b2)

    def test_set_epoch(self):
        """Test set_epoch method"""
        sampler = paddle.utils.data.DistributedSampler(
            self.dataset, num_replicas=2, rank=0, shuffle=True
        )
        # Collect samples from epoch 0
        sampler.set_epoch(0)
        batches0 = list(sampler)
        # Collect samples from epoch 1
        sampler.set_epoch(1)
        batches1 = list(sampler)
        self.assertEqual(len(batches0), len(batches1))


if __name__ == '__main__':
    unittest.main()
