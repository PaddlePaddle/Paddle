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


if __name__ == '__main__':
    unittest.main()
