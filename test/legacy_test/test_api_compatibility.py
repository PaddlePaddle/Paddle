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


# Edit By AI Agent
# Test nextafter compatibility
class TestNextafterAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.nextafter(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.nextafter(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.nextafter(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.nextafter(y, out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.nextafter(y=y)
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.nextafter(x, y, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.nextafter(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.nextafter(x, y)
            # Paddle keyword args
            out2 = paddle.nextafter(x=x, y=y)
            # Torch keyword args
            out3 = paddle.nextafter(input=x, other=y)
            # Tensor method
            out4 = x.nextafter(y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.nextafter(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test angle compatibility
class TestAngleAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'complex64'
        self.init_data()

    def init_data(self):
        self.np_x_real = np.random.randn(*self.shape).astype('float32')
        self.np_x_imag = np.random.randn(*self.shape).astype('float32')
        self.np_x = self.np_x_real + 1j * self.np_x_imag

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.angle(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.angle(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.angle(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.angle(out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.angle()
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.angle(x, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.angle(self.np_x)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-5, atol=1e-5
            )
        paddle.enable_static()


# Edit by AI Agent
# Test atan compatibility
class TestAtanAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        paddle_dygraph_out = []

        # Position args
        out1 = paddle.atan(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.atan(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.atan(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.atan(out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.atan()
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.atan(x, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.arctan(self.np_x)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.atan(x)
            # Paddle keyword args
            out2 = paddle.atan(x=x)
            # Torch keyword args
            out3 = paddle.atan(input=x)
            # Tensor method
            out4 = x.atan()

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.arctan(self.np_x)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


class TestAtan2API_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        paddle_dygraph_out = []

        # Position args
        out1 = paddle.atan2(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.atan2(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.atan2(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Test out parameter
        out4 = paddle.empty([])
        paddle.atan2(x, y, out=out4)
        paddle_dygraph_out.append(out4)

        # Numpy reference output
        ref_out = np.arctan2(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.atan2(x, y)
            # Paddle keyword args
            out2 = paddle.atan2(x=x, y=y)
            # Torch keyword args
            out3 = paddle.atan2(input=x, other=y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )
            ref_out = np.arctan2(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


class TestHypotAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        paddle_dygraph_out = []

        # Paddle positional args
        out1 = paddle.hypot(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.hypot(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # PyTorch keyword args
        out3 = paddle.hypot(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Mixed args
        out4 = paddle.hypot(x, y=y)
        paddle_dygraph_out.append(out4)

        # Test out parameter
        out5 = paddle.empty_like(x)
        out6 = paddle.hypot(x, y, out=out5)
        paddle_dygraph_out.append(out5)
        paddle_dygraph_out.append(out6)

        ref_out = np.hypot(self.np_x, self.np_y)
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-6, atol=1e-6
            )
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Paddle positional args
            out1 = paddle.hypot(x, y)
            # Paddle keyword args
            out2 = paddle.hypot(x=x, y=y)
            # PyTorch keyword args
            out3 = paddle.hypot(input=x, other=y)
            # Mixed args
            out4 = paddle.hypot(x, y=y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.hypot(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6, atol=1e-6)


class TestHypotInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.disable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_InplaceCompatibility(self):
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.hypot(self.np_x, self.np_y)

        for out in [
            x.clone().hypot_(y),
            x.clone().hypot_(y=y),
            x.clone().hypot_(other=y),
            paddle.hypot_(x.clone(), y),
            paddle.hypot_(x=x.clone(), y=y),
            paddle.hypot_(input=x.clone(), other=y),
        ]:
            np.testing.assert_allclose(
                ref_out, out.numpy(), rtol=1e-6, atol=1e-6
            )


# Edit by AI Agent
# Test fmax compatibility
class TestFmaxAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        paddle_dygraph_out = []

        # Position args
        out1 = paddle.fmax(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.fmax(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.fmax(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Test out parameter
        out4 = paddle.empty([])
        paddle.fmax(x, y, out=out4)
        paddle_dygraph_out.append(out4)

        # Numpy reference output
        ref_out = np.fmax(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.fmax(x, y)
            # Paddle keyword args
            out2 = paddle.fmax(x=x, y=y)
            # Torch keyword args
            out3 = paddle.fmax(input=x, other=y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )
            ref_out = np.fmax(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Edit by AI Agent
# Test fmin compatibility
class TestFminAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        paddle_dygraph_out = []

        # Position args
        out1 = paddle.fmin(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.fmin(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.fmin(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Test out parameter
        out4 = paddle.empty([])
        paddle.fmin(x, y, out=out4)
        paddle_dygraph_out.append(out4)

        # Numpy reference output
        ref_out = np.fmin(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.fmin(x, y)
            # Paddle keyword args
            out2 = paddle.fmin(x=x, y=y)
            # Torch keyword args
            out3 = paddle.fmin(input=x, other=y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )
            ref_out = np.fmin(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Edit by AI Agent
# Test bincount compatibility
class TestBincountAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [10]
        self.dtype = 'int64'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_weights = np.random.random(self.shape).astype('float32')

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        weights = paddle.to_tensor(self.np_weights)
        paddle_dygraph_out = []

        # Position args
        out1 = paddle.bincount(x)
        paddle_dygraph_out.append(out1)

        # Position args with weights
        out2 = paddle.bincount(x, weights)
        paddle_dygraph_out.append(out2)

        # Position args with weights and minlength
        out3 = paddle.bincount(x, weights, 6)
        paddle_dygraph_out.append(out3)

        # Paddle keyword args
        out4 = paddle.bincount(x=x)
        paddle_dygraph_out.append(out4)

        out5 = paddle.bincount(x=x, weights=weights)
        paddle_dygraph_out.append(out5)

        out6 = paddle.bincount(x=x, weights=weights, minlength=6)
        paddle_dygraph_out.append(out6)

        # Torch keyword args
        out7 = paddle.bincount(input=x)
        paddle_dygraph_out.append(out7)

        out8 = paddle.bincount(input=x, weights=weights)
        paddle_dygraph_out.append(out8)

        out9 = paddle.bincount(input=x, weights=weights, minlength=6)
        paddle_dygraph_out.append(out9)

        # Numpy reference outputs
        ref_out1 = np.bincount(self.np_x)
        ref_out2 = np.bincount(self.np_x, weights=self.np_weights)
        ref_out3 = np.bincount(self.np_x, weights=self.np_weights, minlength=6)

        # Verify each output with corresponding reference
        np.testing.assert_allclose(ref_out1, out1.numpy())
        np.testing.assert_allclose(ref_out2, out2.numpy())
        np.testing.assert_allclose(ref_out3, out3.numpy())
        np.testing.assert_allclose(ref_out1, out4.numpy())
        np.testing.assert_allclose(ref_out2, out5.numpy())
        np.testing.assert_allclose(ref_out3, out6.numpy())
        np.testing.assert_allclose(ref_out1, out7.numpy())
        np.testing.assert_allclose(ref_out2, out8.numpy())
        np.testing.assert_allclose(ref_out3, out9.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            weights = paddle.static.data(
                name="weights", shape=self.shape, dtype='float32'
            )

            # Position args
            out1 = paddle.bincount(x)
            out2 = paddle.bincount(x, weights)
            out3 = paddle.bincount(x, weights, 6)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "weights": self.np_weights},
                fetch_list=[out1, out2, out3],
            )
            # Numpy reference outputs
            ref_out1 = np.bincount(self.np_x)
            ref_out2 = np.bincount(self.np_x, weights=self.np_weights)
            ref_out3 = np.bincount(
                self.np_x, weights=self.np_weights, minlength=6
            )
            np.testing.assert_allclose(ref_out1, fetches[0])
            np.testing.assert_allclose(ref_out2, fetches[1])
            np.testing.assert_allclose(ref_out3, fetches[2])


# Edit by AI Agent
# Test diag compatibility
class TestDiagAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [3, 3]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_v = np.random.randn(3).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        v = paddle.to_tensor(self.np_v)
        paddle_dygraph_out = []

        # 1D tensor input (construct diagonal matrix)
        out1 = paddle.diag(v)
        paddle_dygraph_out.append(out1)

        # 2D tensor input (extract diagonal)
        out2 = paddle.diag(x)
        paddle_dygraph_out.append(out2)

        # 2D tensor with offset
        out3 = paddle.diag(x, 1)
        paddle_dygraph_out.append(out3)

        # Paddle keyword args
        out4 = paddle.diag(x=x)
        paddle_dygraph_out.append(out4)

        out5 = paddle.diag(x=x, offset=1)
        paddle_dygraph_out.append(out5)

        # Torch keyword args
        out6 = paddle.diag(input=x)
        paddle_dygraph_out.append(out6)

        out7 = paddle.diag(input=x, diagonal=1)
        paddle_dygraph_out.append(out7)

        # Test out parameter
        out8 = paddle.empty([])
        paddle.diag(v, out=out8)
        paddle_dygraph_out.append(out8)

        # Verify outputs
        np.testing.assert_allclose(np.diag(self.np_v), out1.numpy())
        np.testing.assert_allclose(np.diag(self.np_x), out2.numpy())
        np.testing.assert_allclose(np.diag(self.np_x, 1), out3.numpy())
        np.testing.assert_allclose(np.diag(self.np_v), out8.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            v = paddle.static.data(name="v", shape=[3], dtype=self.dtype)

            # 1D tensor input
            out1 = paddle.diag(v)
            # 2D tensor input
            out2 = paddle.diag(x)
            out3 = paddle.diag(x, 1)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "v": self.np_v},
                fetch_list=[out1, out2, out3],
            )
            np.testing.assert_allclose(np.diag(self.np_v), fetches[0])
            np.testing.assert_allclose(np.diag(self.np_x), fetches[1])
            np.testing.assert_allclose(np.diag(self.np_x, 1), fetches[2])


# Test heaviside compatibility
class TestHeavisideAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.heaviside(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.heaviside(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.heaviside(input=x, values=y)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.heaviside(y, out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.heaviside(y=y)
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.heaviside(x, y, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.heaviside(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.heaviside(x, y)
            # Paddle keyword args
            out2 = paddle.heaviside(x=x, y=y)
            # Torch keyword args
            out3 = paddle.heaviside(input=x, values=y)
            # Tensor method
            out4 = x.heaviside(y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.heaviside(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


class TestAsinhAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.asinh(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.asinh(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.asinh(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method args
        out4 = paddle.empty([])
        out5 = x.asinh(out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method kwargs
        out6 = x.asinh()
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.asinh(x, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.arcsinh(self.np_input)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.asinh(x)
            # Paddle keyword args
            out2 = paddle.asinh(x=x)
            # Torch keyword args
            out3 = paddle.asinh(input=x)
            # Tensor method
            out4 = x.asinh()

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.arcsinh(self.np_input)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


class TestReciprocalAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(1, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.reciprocal(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.reciprocal(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.reciprocal(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method kwargs
        out4 = x.reciprocal()
        paddle_dygraph_out.append(out4)

        # Test out parameter
        out5 = paddle.empty([])
        paddle.reciprocal(x, out=out5)
        paddle_dygraph_out.append(out5)

        # Numpy reference output
        ref_out = 1.0 / self.np_input

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.reciprocal(x)
            # Paddle keyword args
            out2 = paddle.reciprocal(x=x)
            # Torch keyword args
            out3 = paddle.reciprocal(input=x)
            # Tensor method
            out4 = x.reciprocal()

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = 1.0 / self.np_input
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


class TestSquareAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.square(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.square(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.square(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method args
        out4 = paddle.empty([])
        out5 = x.square(out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method kwargs
        out6 = x.square()
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.square(x, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.square(self.np_input)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.square(x)
            # Paddle keyword args
            out2 = paddle.square(x=x)
            # Torch keyword args
            out3 = paddle.square(input=x)
            # Tensor method
            out4 = x.square()

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.square(self.np_input)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out)


# Test masked_fill compatibility
class TestMaskedFillAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [3, 3]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(1, 10, self.shape).astype(self.dtype)
        self.np_mask = np.random.randint(0, 2, self.shape).astype(bool)

    def test_dygraph_compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        mask = paddle.to_tensor(self.np_mask)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.masked_fill(x, mask, 0.0)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.masked_fill(x=x, mask=mask, value=0.0)
        paddle_dygraph_out.append(out2)

        # Torch keyword args (input alias)
        out3 = paddle.masked_fill(input=x, mask=mask, value=0.0)
        paddle_dygraph_out.append(out3)

        # Verify all outputs are equal
        for i in range(1, len(paddle_dygraph_out)):
            np.testing.assert_allclose(
                paddle_dygraph_out[0].numpy(), paddle_dygraph_out[i].numpy()
            )
        paddle.enable_static()

    def test_static_compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            mask = paddle.static.data(
                name="mask", shape=self.shape, dtype='bool'
            )

            # Position args
            out1 = paddle.masked_fill(x, mask, 0.0)
            # Paddle keyword args
            out2 = paddle.masked_fill(x=x, mask=mask, value=0.0)
            # Torch keyword args (input alias)
            out3 = paddle.masked_fill(input=x, mask=mask, value=0.0)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "mask": self.np_mask},
                fetch_list=[out1, out2, out3],
            )

        # Verify all outputs are equal
        for i in range(1, len(fetches)):
            np.testing.assert_allclose(fetches[0], fetches[i])


class TestTanAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.tan(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args (kwargs)
        out2 = paddle.tan(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.tan(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method args
        out4 = paddle.empty([])
        out5 = x.tan(out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method kwargs
        out6 = x.tan()
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.tan(x, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.tan(self.np_input)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.tan(x)
            # Paddle keyword args
            out2 = paddle.tan(x=x)
            # Torch keyword args
            out3 = paddle.tan(input=x)
            # Tensor method
            out4 = x.tan()

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.tan(self.np_input)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-6)


# Edit by AI Agent
# Test bitwise_and compatibility
class TestBitwiseAndAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.bitwise_and(x, y)
        out2 = paddle.bitwise_and(x=x, y=y)
        out3 = paddle.bitwise_and(input=x, other=y)
        out4 = paddle.empty([])
        out5 = x.bitwise_and(y, out=out4)
        out6 = x.bitwise_and(y=y)
        out7 = paddle.empty([])
        paddle.bitwise_and(x, y, out=out7)
        ref_out = np.bitwise_and(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)
            out1 = paddle.bitwise_and(x, y)
            out2 = paddle.bitwise_and(x=x, y=y)
            out3 = paddle.bitwise_and(input=x, other=y)
            out4 = x.bitwise_and(y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_and(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_or compatibility
class TestBitwiseOrAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.bitwise_or(x, y)
        out2 = paddle.bitwise_or(x=x, y=y)
        out3 = paddle.bitwise_or(input=x, other=y)
        out4 = paddle.empty([])
        out5 = x.bitwise_or(y, out=out4)
        out6 = x.bitwise_or(y=y)
        out7 = paddle.empty([])
        paddle.bitwise_or(x, y, out=out7)
        ref_out = np.bitwise_or(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)
            out1 = paddle.bitwise_or(x, y)
            out2 = paddle.bitwise_or(x=x, y=y)
            out3 = paddle.bitwise_or(input=x, other=y)
            out4 = x.bitwise_or(y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_or(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_not compatibility
class TestBitwiseNotAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        out1 = paddle.bitwise_not(x)
        out2 = paddle.bitwise_not(x=x)
        out3 = paddle.bitwise_not(input=x)
        out4 = paddle.empty([])
        out5 = x.bitwise_not(out=out4)
        out6 = x.bitwise_not()
        out7 = paddle.empty([])
        paddle.bitwise_not(x, out=out7)
        paddle_dygraph_out = [out1, out2, out3, out4, out5, out6, out7]
        ref_out = np.bitwise_not(self.np_x)
        for out in paddle_dygraph_out:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            out1 = paddle.bitwise_not(x)
            out2 = paddle.bitwise_not(x=x)
            out3 = paddle.bitwise_not(input=x)
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
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.bitwise_xor(x, y)
        out2 = paddle.bitwise_xor(x=x, y=y)
        out3 = paddle.bitwise_xor(input=x, other=y)
        out4 = paddle.empty([])
        out5 = x.bitwise_xor(y, out=out4)
        out6 = x.bitwise_xor(y=y)
        out7 = paddle.empty([])
        paddle.bitwise_xor(x, y, out=out7)
        ref_out = np.bitwise_xor(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)
            out1 = paddle.bitwise_xor(x, y)
            out2 = paddle.bitwise_xor(x=x, y=y)
            out3 = paddle.bitwise_xor(input=x, other=y)
            out4 = x.bitwise_xor(y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_xor(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_and_ inplace compatibility
class TestBitwiseAndInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.disable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_InplaceCompatibility(self):
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.bitwise_and(self.np_x, self.np_y)
        # Test all calling patterns: position args, Paddle/Torch keyword args, function call
        for out in [
            x.clone().bitwise_and_(y),
            x.clone().bitwise_and_(y=y),
            x.clone().bitwise_and_(other=y),
            paddle.bitwise_and_(x.clone(), y),
        ]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_or_ inplace compatibility
class TestBitwiseOrInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.disable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_InplaceCompatibility(self):
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.bitwise_or(self.np_x, self.np_y)
        # Test all calling patterns: position args, Paddle/Torch keyword args, function call
        for out in [
            x.clone().bitwise_or_(y),
            x.clone().bitwise_or_(y=y),
            x.clone().bitwise_or_(other=y),
            paddle.bitwise_or_(x.clone(), y),
        ]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_xor_ inplace compatibility
class TestBitwiseXorInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.disable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_InplaceCompatibility(self):
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.bitwise_xor(self.np_x, self.np_y)
        # Test all calling patterns: position args, Paddle/Torch keyword args, function call
        for out in [
            x.clone().bitwise_xor_(y),
            x.clone().bitwise_xor_(y=y),
            x.clone().bitwise_xor_(other=y),
            paddle.bitwise_xor_(x.clone(), y),
        ]:
            np.testing.assert_array_equal(ref_out, out.numpy())


# Test bitwise_not_ inplace compatibility
class TestBitwiseNotInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.disable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(0, 8, self.shape).astype(self.dtype)

    def test_dygraph_InplaceCompatibility(self):
        x = paddle.to_tensor(self.np_x)
        ref_out = np.bitwise_not(self.np_x)
        # Test all calling patterns (Paddle/Torch keyword args are identical)
        for out in [x.clone().bitwise_not_(), paddle.bitwise_not_(x.clone())]:
            np.testing.assert_array_equal(ref_out, out.numpy())


class TestCdistAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape_x = [3, 5, 4]
        self.shape_y = [3, 2, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape_x).astype(self.dtype)
        self.np_y = np.random.rand(*self.shape_y).astype(self.dtype)

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
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape_x, dtype=self.dtype
            )
            y = paddle.static.data(
                name="y", shape=self.shape_y, dtype=self.dtype
            )
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
        x1 = paddle.to_tensor(np.random.rand(2, 0, 4).astype(self.dtype))
        y1 = paddle.to_tensor(np.random.rand(2, 3, 4).astype(self.dtype))
        out1 = paddle.cdist(x1, y1)
        self.assertEqual(out1.shape, [2, 0, 3])
        # r2==0 (2D non-batched)
        x2 = paddle.to_tensor(np.random.rand(3, 4).astype(self.dtype))
        y2 = paddle.to_tensor(np.random.rand(0, 4).astype(self.dtype))
        out2 = paddle.cdist(x2, y2)
        self.assertEqual(out2.shape, [3, 0])
        # c1==0 (3D batched, should return zeros)
        x3 = paddle.to_tensor(np.random.rand(2, 3, 0).astype(self.dtype))
        y3 = paddle.to_tensor(np.random.rand(2, 2, 0).astype(self.dtype))
        out3 = paddle.cdist(x3, y3)
        self.assertEqual(out3.shape, [2, 3, 2])
        np.testing.assert_allclose(out3.numpy(), 0.0)
        paddle.enable_static()


class TestAddmmAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.enable_static()
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.rand(2, 3).astype(self.dtype)
        self.np_x = np.random.rand(2, 4).astype(self.dtype)
        self.np_y = np.random.rand(4, 3).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
        out1 = paddle.addmm(input, x, y)
        out2 = paddle.addmm(input, x, y, 1.0, 1.0)
        out3 = paddle.addmm(input=input, x=x, y=y)
        out4 = paddle.addmm(input=input, x=x, y=y, beta=1.0, alpha=1.0)
        out5 = paddle.addmm(beta=1.0, alpha=1.0, input=input, mat1=x, mat2=y)
        out6 = paddle.empty_like(input)
        paddle.addmm(input, x, y, out=out6)
        out7 = input.addmm(x, y)
        out8 = input.addmm(x=x, y=y, beta=1.0, alpha=1.0)
        for out in [out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)

        input_1d = paddle.to_tensor(np.random.rand(1).astype(self.dtype))
        out9 = paddle.addmm(input_1d, x, y)
        self.assertEqual(out9.shape, [2, 3])
        paddle.enable_static()

    def test_error(self):
        """Test invalid input dimensions that should raise ValueError."""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # Test 3D input (invalid)
        input_3d = paddle.to_tensor(np.random.rand(2, 2, 3).astype(self.dtype))
        with self.assertRaises(ValueError):
            paddle.addmm(input_3d, x, y)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=[2, 3], dtype=self.dtype
            )
            x = paddle.static.data(name="x", shape=[2, 4], dtype=self.dtype)
            y = paddle.static.data(name="y", shape=[4, 3], dtype=self.dtype)
            out1 = paddle.addmm(input, x, y)
            out2 = paddle.addmm(input=input, x=x, y=y)
            out3 = paddle.addmm(beta=1, alpha=1, input=input, mat1=x, mat2=y)
            out4 = input.addmm(x, y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
            for out in fetches:
                np.testing.assert_allclose(ref_out, out, rtol=1e-6)


class TestAddmmInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.disable_static()
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.rand(2, 3).astype(self.dtype)
        self.np_x = np.random.rand(2, 4).astype(self.dtype)
        self.np_y = np.random.rand(4, 3).astype(self.dtype)

    def test_dygraph_Compatibility(self):
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
        # Verify all outputs
        for out in [out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out1.numpy(), out.numpy(), rtol=1e-6)
        paddle.enable_static()


class TestLdexpInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [3, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randint(-3, 4, self.shape).astype('int32')

    def test_dygraph_InplaceCompatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = np.ldexp(self.np_x, self.np_y)

        for out in [
            x.clone().ldexp_(y),
            x.clone().ldexp_(y=y),
            x.clone().ldexp_(other=y),
            paddle.ldexp_(x.clone(), y),
        ]:
            np.testing.assert_allclose(ref_out, out.numpy())
        paddle.enable_static()


# Test baddbmm API compatibility (paddle.baddbmm and paddle.Tensor.baddbmm)
class TestBaddbmmAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.enable_static()
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.rand(3, 2, 3).astype(self.dtype)
        self.np_x = np.random.rand(3, 2, 4).astype(self.dtype)
        self.np_y = np.random.rand(3, 4, 3).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        input = paddle.to_tensor(self.np_input)
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
        out1 = paddle.baddbmm(input, x, y)
        out2 = paddle.baddbmm(input, x, y, 1.0, 1.0)
        out3 = paddle.baddbmm(input=input, x=x, y=y)
        out4 = paddle.baddbmm(input=input, x=x, y=y, beta=1.0, alpha=1.0)
        out5 = paddle.baddbmm(
            beta=1.0, alpha=1.0, input=input, batch1=x, batch2=y
        )
        out6 = paddle.empty_like(input)
        paddle.baddbmm(input, x, y, out=out6)
        out7 = input.baddbmm(x, y)
        out8 = input.baddbmm(x=x, y=y, beta=1.0, alpha=1.0)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-6)

        input_2d = paddle.to_tensor(np.random.rand(1, 1).astype(self.dtype))
        out9 = paddle.baddbmm(input_2d, x, y)
        self.assertEqual(out9.shape, [3, 2, 3])
        paddle.enable_static()

    def test_error(self):
        """Test invalid input dimensions that should raise ValueError."""
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # Test 1D input (invalid)
        input_1d = paddle.to_tensor(np.random.rand(3).astype(self.dtype))
        with self.assertRaises(ValueError):
            paddle.baddbmm(input_1d, x, y)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            input = paddle.static.data(
                name="input", shape=[3, 2, 3], dtype=self.dtype
            )
            x = paddle.static.data(name="x", shape=[3, 2, 4], dtype=self.dtype)
            y = paddle.static.data(name="y", shape=[3, 4, 3], dtype=self.dtype)
            out1 = paddle.baddbmm(input, x, y)
            out2 = paddle.baddbmm(input=input, x=x, y=y)
            out3 = paddle.baddbmm(
                beta=1, alpha=1, input=input, batch1=x, batch2=y
            )
            out4 = input.baddbmm(x, y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"input": self.np_input, "x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = 1.0 * self.np_input + 1.0 * self.np_x @ self.np_y
            for out in fetches:
                np.testing.assert_allclose(ref_out, out, rtol=1e-6)


# Test baddbmm_ API compatibility (paddle.baddbmm_ and paddle.Tensor.baddbmm_)
class TestBaddbmmInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.disable_static()
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.rand(3, 2, 3).astype(self.dtype)
        self.np_x = np.random.rand(3, 2, 4).astype(self.dtype)
        self.np_y = np.random.rand(3, 4, 3).astype(self.dtype)

    def test_dygraph_Compatibility(self):
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
        # Verify all outputs
        for out in [out2, out3, out4, out5, out6]:
            np.testing.assert_allclose(out1.numpy(), out.numpy(), rtol=1e-6)
        paddle.enable_static()


# Test bitwise_left_shift compatibility
class TestBitwiseLeftShiftAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(1, 10, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(1, 5, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.bitwise_left_shift(x, y)
        out2 = paddle.bitwise_left_shift(x=x, y=y)
        out3 = paddle.bitwise_left_shift(input=x, other=y)
        out4 = paddle.bitwise_left_shift(x, y, is_arithmetic=True)
        out5 = paddle.bitwise_left_shift(x, y, is_arithmetic=False)
        out6 = paddle.empty([])
        out7 = x.bitwise_left_shift(y, out=out6)
        out8 = x.bitwise_left_shift(y=y)
        out9 = paddle.empty([])
        paddle.bitwise_left_shift(x, y, out=out9)
        ref_out = np.left_shift(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8, out9]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)
            out1 = paddle.bitwise_left_shift(x, y)
            out2 = paddle.bitwise_left_shift(x=x, y=y)
            out3 = paddle.bitwise_left_shift(input=x, other=y)
            out4 = x.bitwise_left_shift(y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.left_shift(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_left_shift_ inplace compatibility
class TestBitwiseLeftShiftInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(1, 10, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(1, 5, self.shape).astype(self.dtype)

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        x.bitwise_left_shift_(y)
        ref_out = np.left_shift(self.np_x, self.np_y)
        np.testing.assert_array_equal(ref_out, x.numpy())
        paddle.enable_static()


# Test bitwise_right_shift compatibility
class TestBitwiseRightShiftAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(10, 100, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(1, 5, self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        out1 = paddle.bitwise_right_shift(x, y)
        out2 = paddle.bitwise_right_shift(x=x, y=y)
        out3 = paddle.bitwise_right_shift(input=x, other=y)
        out4 = paddle.bitwise_right_shift(x, y, is_arithmetic=True)
        out5 = paddle.bitwise_right_shift(x, y, is_arithmetic=False)
        out6 = paddle.empty([])
        out7 = x.bitwise_right_shift(y, out=out6)
        out8 = x.bitwise_right_shift(y=y)
        out9 = paddle.empty([])
        paddle.bitwise_right_shift(x, y, out=out9)
        ref_out = np.right_shift(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8, out9]:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)
            out1 = paddle.bitwise_right_shift(x, y)
            out2 = paddle.bitwise_right_shift(x=x, y=y)
            out3 = paddle.bitwise_right_shift(input=x, other=y)
            out4 = x.bitwise_right_shift(y)
            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.right_shift(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_right_shift_ inplace compatibility
class TestBitwiseRightShiftInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [5, 6]
        self.dtype = 'int32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randint(10, 100, self.shape).astype(self.dtype)
        self.np_y = np.random.randint(1, 5, self.shape).astype(self.dtype)

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        x.bitwise_right_shift_(y)
        ref_out = np.right_shift(self.np_x, self.np_y)
        np.testing.assert_array_equal(ref_out, x.numpy())
        paddle.enable_static()


# Test cauchy_ inplace compatibility
class TestCauchyInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [3, 4]
        self.dtype = 'float32'

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()

        # Test 1: Paddle positional arguments
        x1 = paddle.randn(self.shape, dtype=self.dtype)
        x1.cauchy_(1.0, 2.0)
        self.assertEqual(x1.shape, self.shape)

        # Test 2: Paddle keyword arguments
        x2 = paddle.randn(self.shape, dtype=self.dtype)
        x2.cauchy_(loc=1.0, scale=2.0)
        self.assertEqual(x2.shape, self.shape)

        # Test 3: PyTorch positional arguments
        x3 = paddle.randn(self.shape, dtype=self.dtype)
        x3.cauchy_(1.0, 2.0)
        self.assertEqual(x3.shape, self.shape)

        # Test 4: PyTorch keyword arguments (alias)
        x4 = paddle.randn(self.shape, dtype=self.dtype)
        x4.cauchy_(median=1.0, sigma=2.0)
        self.assertEqual(x4.shape, self.shape)

        # Test 5: Mixed arguments
        x5 = paddle.randn(self.shape, dtype=self.dtype)
        x5.cauchy_(1.0, scale=2.0)
        self.assertEqual(x5.shape, self.shape)

        # Test 6: Mixed arguments with alias
        x6 = paddle.randn(self.shape, dtype=self.dtype)
        x6.cauchy_(median=1.0, scale=2.0)
        self.assertEqual(x6.shape, self.shape)


class TestTensorCumsumInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.data = np.random.randint(1, 5, size=(3, 4)).astype('int64')

    def test_dygraph_dim_alias(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.data)
        y = x.cumsum_(dim=1)
        np.testing.assert_allclose(np.cumsum(self.data, axis=1), y.numpy())
        paddle.enable_static()

    def test_dygraph_axis(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.data)
        y = x.cumsum_(axis=0)
        np.testing.assert_allclose(np.cumsum(self.data, axis=0), y.numpy())
        paddle.enable_static()


# Edit by AI Agent
# Test block_diag compatibility
class TestBlockDiagAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape1 = [2, 3]
        self.shape2 = [3, 4]
        self.shape3 = [1, 2]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape1).astype(self.dtype)
        self.np_y = np.random.rand(*self.shape2).astype(self.dtype)
        self.np_z = np.random.rand(*self.shape3).astype(self.dtype)

    def compute_ref_output(self):
        import scipy.linalg

        return scipy.linalg.block_diag(self.np_x, self.np_y, self.np_z)

    def test_dygraph_Compatibility(self):
        import scipy.linalg

        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        z = paddle.to_tensor(self.np_z)

        # 1. PyTorch positional arguments (variadic)
        out1 = paddle.block_diag(x, y, z)

        # 2. Paddle style with list
        out2 = paddle.block_diag([x, y, z])

        # 3. Paddle keyword arguments with list
        out3 = paddle.block_diag(inputs=[x, y, z])

        # 4. Mixed: some tensors
        out4 = paddle.block_diag(x, y)

        # Verify all outputs
        ref_out = scipy.linalg.block_diag(self.np_x, self.np_y, self.np_z)
        for out in [out1, out2, out3]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        ref_out_two = scipy.linalg.block_diag(self.np_x, self.np_y)
        np.testing.assert_allclose(ref_out_two, out4.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape1, dtype=self.dtype
            )
            y = paddle.static.data(
                name="y", shape=self.shape2, dtype=self.dtype
            )
            z = paddle.static.data(
                name="z", shape=self.shape3, dtype=self.dtype
            )

            # Test with list
            out1 = paddle.block_diag(x, y, z)
            out2 = paddle.block_diag([x, y, z])
            out3 = paddle.block_diag(inputs=[x, y, z])

            exe = paddle.base.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y, "z": self.np_z},
                fetch_list=[out1, out2, out3],
            )

            ref_out = self.compute_ref_output()
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test broadcast_tensors compatibility
class TestBroadcastTensorsAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape1 = [3, 1]
        self.shape2 = [1, 4]
        self.shape3 = [3, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape1).astype(self.dtype)
        self.np_y = np.random.rand(*self.shape2).astype(self.dtype)
        self.np_z = np.random.rand(*self.shape3).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        z = paddle.to_tensor(self.np_z)

        # 1. PyTorch positional arguments (variadic)
        outs1 = paddle.broadcast_tensors(x, y, z)

        # 2. Paddle style with list
        outs2 = paddle.broadcast_tensors([x, y, z])

        # 3. Paddle keyword arguments with list
        outs3 = paddle.broadcast_tensors(input=[x, y, z])

        # 4. Mixed: two tensors
        outs4 = paddle.broadcast_tensors(x, y)

        # Verify all outputs
        ref_x = np.broadcast_to(self.np_x, [3, 4])
        ref_y = np.broadcast_to(self.np_y, [3, 4])
        ref_z = np.broadcast_to(self.np_z, [3, 4])

        for outs in [outs1, outs2, outs3]:
            self.assertEqual(len(outs), 3)
            np.testing.assert_allclose(ref_x, outs[0].numpy())
            np.testing.assert_allclose(ref_y, outs[1].numpy())
            np.testing.assert_allclose(ref_z, outs[2].numpy())

        self.assertEqual(len(outs4), 2)
        np.testing.assert_allclose(ref_x, outs4[0].numpy())
        np.testing.assert_allclose(ref_y, outs4[1].numpy())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape1, dtype=self.dtype
            )
            y = paddle.static.data(
                name="y", shape=self.shape2, dtype=self.dtype
            )
            z = paddle.static.data(
                name="z", shape=self.shape3, dtype=self.dtype
            )

            # Test with list
            outs1 = paddle.broadcast_tensors(x, y, z)
            outs2 = paddle.broadcast_tensors(input=[x, y, z])

            exe = paddle.base.Executor()
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
                ],
            )

            ref_x = np.broadcast_to(self.np_x, [3, 4])
            ref_y = np.broadcast_to(self.np_y, [3, 4])
            ref_z = np.broadcast_to(self.np_z, [3, 4])

            np.testing.assert_allclose(fetches[0], ref_x)
            np.testing.assert_allclose(fetches[1], ref_y)
            np.testing.assert_allclose(fetches[2], ref_z)
            np.testing.assert_allclose(fetches[3], ref_x)
            np.testing.assert_allclose(fetches[4], ref_y)
            np.testing.assert_allclose(fetches[5], ref_z)


# Test cartesian_prod compatibility
class TestCartesianProdAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape1 = [3]
        self.shape2 = [4]
        self.dtype = 'int64'
        self.init_data()

    def init_data(self):
        self.np_x = np.array([1, 2, 3], dtype=self.dtype)
        self.np_y = np.array([4, 5, 6, 7], dtype=self.dtype)

    def compute_ref_output(self):
        # Compute cartesian product
        x_grid, y_grid = np.meshgrid(self.np_x, self.np_y, indexing='ij')
        return np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. PyTorch positional arguments (variadic)
        out1 = paddle.cartesian_prod(x, y)

        # 2. Paddle style with list
        out2 = paddle.cartesian_prod([x, y])

        # 3. Paddle keyword arguments with list
        out3 = paddle.cartesian_prod(x=[x, y])

        # 4. Single tensor
        out4 = paddle.cartesian_prod(x)

        # Verify outputs
        ref_out = self.compute_ref_output()
        for out in [out1, out2, out3]:
            np.testing.assert_array_equal(ref_out, out.numpy())

        np.testing.assert_array_equal(self.np_x.flatten(), out4.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(
                name="x", shape=self.shape1, dtype=self.dtype
            )
            y = paddle.static.data(
                name="y", shape=self.shape2, dtype=self.dtype
            )

            # Test with list
            out1 = paddle.cartesian_prod(x, y)
            out2 = paddle.cartesian_prod([x, y])
            out3 = paddle.cartesian_prod(x=[x, y])

            exe = paddle.base.Executor()
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
        self.shape = [3, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # 1. Paddle positional arguments
        out1 = paddle.copysign(x, y)

        # 2. Paddle keyword arguments
        out2 = paddle.copysign(x=x, y=y)

        # 3. PyTorch keyword arguments (alias)
        out3 = paddle.copysign(input=x, other=y)

        # 4. Mixed arguments
        out4 = paddle.copysign(x, other=y)

        # 5. out parameter test
        out5 = paddle.empty_like(x)
        out6 = paddle.copysign(x, y, out=out5)

        # 6. Tensor method - positional args
        out7 = x.copysign(y)

        # 7. Tensor method - keyword args
        out8 = x.copysign(y=y)

        # Verify all outputs
        ref_out = np.copysign(self.np_x, self.np_y)
        for out in [out1, out2, out3, out4, out5, out6, out7, out8]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.copysign(x, y)
            # Paddle keyword args
            out2 = paddle.copysign(x=x, y=y)
            # PyTorch keyword args
            out3 = paddle.copysign(input=x, other=y)
            # Tensor method
            out4 = x.copysign(y)

            exe = paddle.base.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )

            ref_out = np.copysign(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


# Test Tensor.copysign_ inplace compatibility
class TestTensorCopysignInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [3, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.randn(*self.shape).astype(self.dtype)
        self.np_y = np.random.randn(*self.shape).astype(self.dtype)

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()

        # Test 1: Paddle positional arguments
        x1 = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)
        x1.copysign_(y)
        ref_out = np.copysign(self.np_x, self.np_y)
        np.testing.assert_allclose(ref_out, x1.numpy(), rtol=1e-5)

        # Test 2: Paddle keyword arguments
        x2 = paddle.to_tensor(self.np_x)
        x2.copysign_(y=y)
        np.testing.assert_allclose(ref_out, x2.numpy(), rtol=1e-5)

        # Test 3: PyTorch keyword arguments (alias)
        x3 = paddle.to_tensor(self.np_x)
        x3.copysign_(other=y)
        np.testing.assert_allclose(ref_out, x3.numpy(), rtol=1e-5)

        paddle.enable_static()


# Test Tensor.geometric_ inplace compatibility
class TestTensorGeometricInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [10000]
        self.dtype = 'float32'
        self.p = 0.3

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()

        # Test 1: Paddle positional arguments
        x1 = paddle.empty(self.shape, dtype=self.dtype)
        x1.geometric_(self.p)
        self.assertEqual(x1.shape, self.shape)
        # Verify values are positive integers
        self.assertTrue((x1.numpy() > 0).all())

        # Test 2: Paddle keyword arguments
        x2 = paddle.empty(self.shape, dtype=self.dtype)
        x2.geometric_(p=self.p)
        self.assertEqual(x2.shape, self.shape)
        self.assertTrue((x2.numpy() > 0).all())

        # Test 3: PyTorch keyword arguments (alias)
        x3 = paddle.empty(self.shape, dtype=self.dtype)
        x3.geometric_(probs=self.p)
        self.assertEqual(x3.shape, self.shape)
        self.assertTrue((x3.numpy() > 0).all())

        # Test 4: Mixed arguments
        x4 = paddle.empty(self.shape, dtype=self.dtype)
        x4.geometric_(probs=self.p)
        self.assertEqual(x4.shape, self.shape)
        self.assertTrue((x4.numpy() > 0).all())

        paddle.enable_static()


# Test Tensor.hypot_ inplace compatibility
class TestTensorHypotInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [3, 4]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype) + 1.0
        self.np_y = np.random.rand(*self.shape).astype(self.dtype) + 1.0

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        y = paddle.to_tensor(self.np_y)

        # Test 1: Paddle positional arguments
        x1 = paddle.to_tensor(self.np_x)
        x1.hypot_(y)
        ref_out = np.hypot(self.np_x, self.np_y)
        np.testing.assert_allclose(ref_out, x1.numpy(), rtol=1e-5)

        # Test 2: Paddle keyword arguments
        x2 = paddle.to_tensor(self.np_x)
        x2.hypot_(y=y)
        np.testing.assert_allclose(ref_out, x2.numpy(), rtol=1e-5)

        # Test 3: PyTorch keyword arguments (alias)
        x3 = paddle.to_tensor(self.np_x)
        x3.hypot_(other=y)
        np.testing.assert_allclose(ref_out, x3.numpy(), rtol=1e-5)

        paddle.enable_static()


# Test index_fill compatibility
class TestIndexFillAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.axis = 1
        self.index_shape = [3]
        self.value = -1.0
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)
        self.np_index = np.array([1, 3, 4], dtype='int64')

    def compute_ref_output(self):
        ref = self.np_x.copy()
        ref[:, self.np_index] = self.value
        return ref

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        index = paddle.to_tensor(self.np_index)

        # 1. Paddle positional arguments
        out1 = paddle.index_fill(x, index, self.axis, self.value)

        # 2. Paddle keyword arguments
        out2 = paddle.index_fill(
            x=x, index=index, axis=self.axis, value=self.value
        )

        # 3. PyTorch positional arguments (different order)
        out3 = paddle.index_fill(x, self.axis, index, self.value)

        # 4. PyTorch keyword arguments (alias)
        out4 = paddle.index_fill(
            input=x, dim=self.axis, index=index, value=self.value
        )

        # 5. Mixed arguments
        out5 = paddle.index_fill(x, index, axis=self.axis, value=self.value)

        # 6. Tensor method - Paddle style
        out6 = x.index_fill(index, self.axis, self.value)

        # 7. Tensor method - PyTorch style
        out7 = x.index_fill(self.axis, index, self.value)

        # Verify all outputs
        ref_out = self.compute_ref_output()
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-5)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            index = paddle.static.data(
                name="index", shape=self.index_shape, dtype='int64'
            )

            # Paddle style
            out1 = paddle.index_fill(x, index, self.axis, self.value)
            # Paddle keyword args
            out2 = paddle.index_fill(
                x=x, index=index, axis=self.axis, value=self.value
            )
            # PyTorch style
            out3 = paddle.index_fill(x, self.axis, index, self.value)
            # Tensor method
            out4 = x.index_fill(index, self.axis, self.value)

            exe = paddle.base.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "index": self.np_index},
                fetch_list=[out1, out2, out3, out4],
            )

            ref_out = self.compute_ref_output()
            for out in fetches:
                np.testing.assert_allclose(out, ref_out, rtol=1e-5)


@unittest.skipIf(
    paddle.device.is_compiled_with_xpu(),
    "skip xpu which not support index_fill_",
)
# Test Tensor.index_fill_ inplace compatibility
class TestTensorIndexFillInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [5, 6]
        self.dtype = 'float32'
        self.axis = 1
        self.index_shape = [3]
        self.value = -1.0
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)
        self.np_index = np.array([1, 3, 4], dtype='int64')

    def compute_ref_output(self):
        ref = self.np_x.copy()
        ref[:, self.np_index] = self.value
        return ref

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        index = paddle.to_tensor(self.np_index)
        ref_out = self.compute_ref_output()

        # Test 1: Paddle positional arguments
        x1 = paddle.to_tensor(self.np_x)
        x1.index_fill_(index, self.axis, self.value)
        np.testing.assert_allclose(ref_out, x1.numpy(), rtol=1e-5)

        # Test 2: Paddle keyword arguments
        x2 = paddle.to_tensor(self.np_x)
        x2.index_fill_(index=index, axis=self.axis, value=self.value)
        np.testing.assert_allclose(ref_out, x2.numpy(), rtol=1e-5)

        # Test 3: PyTorch positional arguments (different order)
        x3 = paddle.to_tensor(self.np_x)
        x3.index_fill_(self.axis, index, self.value)
        np.testing.assert_allclose(ref_out, x3.numpy(), rtol=1e-5)

        # Test 4: PyTorch keyword arguments (alias)
        x4 = paddle.to_tensor(self.np_x)
        x4.index_fill_(dim=self.axis, index=index, value=self.value)
        np.testing.assert_allclose(ref_out, x4.numpy(), rtol=1e-5)

        # Test 5: Mixed arguments
        x5 = paddle.to_tensor(self.np_x)
        x5.index_fill_(index, axis=self.axis, value=self.value)
        np.testing.assert_allclose(ref_out, x5.numpy(), rtol=1e-5)


# Test cross compatibility
class TestCrossAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [3, 3, 3]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)
        self.np_y = np.random.rand(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # Paddle positional arguments
        out1 = paddle.cross(x, y)

        # Paddle keyword arguments
        out2 = paddle.cross(x=x, y=y)

        # PyTorch keyword arguments (alias)
        out3 = paddle.cross(x, y, dim=1)

        # Mixed arguments
        out4 = paddle.cross(x, y=y, axis=1)

        # Tensor method - positional
        out5 = x.cross(y)

        # Tensor method - keyword
        out6 = x.cross(y, axis=1)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, self.np_x.shape)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            out1 = paddle.cross(x, y)
            out2 = paddle.cross(x=x, y=y)
            out3 = paddle.cross(x, y, axis=1)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                self.assertEqual(out.shape, tuple(self.shape))


# Test dist compatibility
class TestDistAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [2, 2]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)
        self.np_y = np.random.rand(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)
        y = paddle.to_tensor(self.np_y)

        # Paddle positional arguments
        out1 = paddle.dist(x, y)

        # Paddle keyword arguments
        out2 = paddle.dist(x=x, y=y)

        # With p parameter
        out3 = paddle.dist(x, y, p=2.0)

        # With different p values
        out4 = paddle.dist(x, y, p=float("inf"))
        out5 = paddle.dist(x, y, p=float("-inf"))

        # Tensor method - positional
        out6 = x.dist(y)

        # Tensor method - keyword
        out7 = x.dist(y=y, p=2.0)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, ())

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            out1 = paddle.dist(x, y)
            out2 = paddle.dist(x=x, y=y)
            out3 = paddle.dist(x, y, p=2.0)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                self.assertEqual(out.shape, ())


# Test flip compatibility
class TestFlipAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [3, 2, 2]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # Paddle positional arguments (list)
        out1 = paddle.flip(x, [0, 1])

        # Paddle keyword arguments
        out2 = paddle.flip(x=x, axis=[0, 1])

        # Single axis as int
        out3 = paddle.flip(x, 1)

        # Single axis as list
        out4 = paddle.flip(x, [1])

        # Negative axis
        out5 = paddle.flip(x, -1)

        # Tensor method - positional
        out6 = x.flip([0, 1])

        # Tensor method - keyword
        out7 = x.flip(axis=[0, 1])

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6, out7]:
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, tuple(self.shape))

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            out1 = paddle.flip(x, [0, 1])
            out2 = paddle.flip(x=x, axis=[0, 1])
            out3 = paddle.flip(x, [1])

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                self.assertEqual(out.shape, tuple(self.shape))


# Test renorm compatibility
class TestRenormAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [2, 2, 3]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # Paddle positional arguments
        out1 = paddle.renorm(x, 1.0, 2, 2.05)

        # Paddle keyword arguments
        out2 = paddle.renorm(x=x, p=1.0, axis=2, max_norm=2.05)

        # Mixed arguments
        out3 = paddle.renorm(x, p=1.0, axis=2, max_norm=2.05)

        # Different parameters
        out4 = paddle.renorm(x, 2.0, 1, 1.5)

        # Tensor method - positional
        out5 = x.renorm(1.0, 2, 2.05)

        # Tensor method - keyword
        out6 = x.renorm(p=1.0, axis=2, max_norm=2.05)

        # Verify all outputs
        for out in [out1, out2, out3, out4, out5, out6]:
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, tuple(self.shape))

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            out1 = paddle.renorm(x, 1.0, 2, 2.05)
            out2 = paddle.renorm(x=x, p=1.0, axis=2, max_norm=2.05)
            out3 = paddle.renorm(x, 2.0, 1, 1.5)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3],
            )

            # Verify all outputs
            for out in fetches:
                self.assertEqual(out.shape, tuple(self.shape))


# Test renorm_ inplace compatibility
class TestRenormInplace(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.shape = [2, 2, 3]
        self.dtype = 'float32'
        self.init_data()

    def init_data(self):
        self.np_x = np.random.rand(*self.shape).astype(self.dtype)

    def test_dygraph_inplace_Compatibility(self):
        paddle.disable_static()
        ref_x = self.np_x.copy()

        # Test 1: Paddle positional arguments
        x1 = paddle.to_tensor(ref_x)
        out1 = x1.renorm_(1.0, 2, 2.05)
        self.assertIsNotNone(out1)
        self.assertEqual(out1.shape, tuple(self.shape))
        # Verify inplace operation
        self.assertIs(out1, x1)

        # Test 2: Paddle keyword arguments
        x2 = paddle.to_tensor(ref_x)
        out2 = x2.renorm_(p=1.0, axis=2, max_norm=2.05)
        self.assertIsNotNone(out2)
        self.assertEqual(out2.shape, tuple(self.shape))
        # Verify inplace operation
        self.assertIs(out2, x2)

        # Test 3: Mixed arguments
        x3 = paddle.to_tensor(ref_x)
        out3 = x3.renorm_(1.0, axis=2, max_norm=2.05)
        self.assertIsNotNone(out3)
        self.assertEqual(out3.shape, tuple(self.shape))
        # Verify inplace operation
        self.assertIs(out3, x3)

        # Test 4: Different parameters
        x4 = paddle.to_tensor(ref_x)
        out4 = x4.renorm_(2.0, 1, 1.5)
        self.assertIsNotNone(out4)
        self.assertEqual(out4.shape, tuple(self.shape))
        # Verify inplace operation
        self.assertIs(out4, x4)

        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
