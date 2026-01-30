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
class TestNextafterAPI_Compatibility(unittest.TestCase):
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
class TestAngleAPI_Compatibility(unittest.TestCase):
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
class TestAtanAPI_Compatibility(unittest.TestCase):
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


# Edit by AI Agent
# Test fmax compatibility
class TestFmaxAPI_Compatibility(unittest.TestCase):
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
class TestFminAPI_Compatibility(unittest.TestCase):
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
class TestBincountAPI_Compatibility(unittest.TestCase):
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
class TestDiagAPI_Compatibility(unittest.TestCase):
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
class TestHeavisideAPI_Compatibility(unittest.TestCase):
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


class TestAsinhAPI_Compatibility(unittest.TestCase):
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


class TestReciprocalAPI_Compatibility(unittest.TestCase):
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


class TestSquareAPI_Compatibility(unittest.TestCase):
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


class TestTanAPI_Compatibility(unittest.TestCase):
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
class TestBitwiseAndAPI_Compatibility(unittest.TestCase):
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
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.bitwise_and(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.bitwise_and(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.bitwise_and(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.bitwise_and(y, out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.bitwise_and(y=y)
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.bitwise_and(x, y, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.bitwise_and(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.bitwise_and(x, y)
            # Paddle keyword args
            out2 = paddle.bitwise_and(x=x, y=y)
            # Torch keyword args
            out3 = paddle.bitwise_and(input=x, other=y)
            # Tensor method
            out4 = x.bitwise_and(y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_and(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_not compatibility
class TestBitwiseNotAPI_Compatibility(unittest.TestCase):
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
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.bitwise_not(x)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.bitwise_not(x=x)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.bitwise_not(input=x)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.bitwise_not(out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.bitwise_not()
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.bitwise_not(x, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.bitwise_not(self.np_x)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.bitwise_not(x)
            # Paddle keyword args
            out2 = paddle.bitwise_not(x=x)
            # Torch keyword args
            out3 = paddle.bitwise_not(input=x)
            # Tensor method
            out4 = x.bitwise_not()

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_not(self.np_x)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


# Test bitwise_xor compatibility
class TestBitwiseXorAPI_Compatibility(unittest.TestCase):
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
        paddle_dygraph_out = []

        # Position args (args)
        out1 = paddle.bitwise_xor(x, y)
        paddle_dygraph_out.append(out1)

        # Paddle keyword args
        out2 = paddle.bitwise_xor(x=x, y=y)
        paddle_dygraph_out.append(out2)

        # Torch keyword args
        out3 = paddle.bitwise_xor(input=x, other=y)
        paddle_dygraph_out.append(out3)

        # Tensor method - args
        out4 = paddle.empty([])
        out5 = x.bitwise_xor(y, out=out4)
        paddle_dygraph_out.append(out4)
        paddle_dygraph_out.append(out5)

        # Tensor method - kwargs
        out6 = x.bitwise_xor(y=y)
        paddle_dygraph_out.append(out6)

        # Test out parameter
        out7 = paddle.empty([])
        paddle.bitwise_xor(x, y, out=out7)
        paddle_dygraph_out.append(out7)

        # Numpy reference output
        ref_out = np.bitwise_xor(self.np_x, self.np_y)

        # Verify all outputs
        for out in paddle_dygraph_out:
            np.testing.assert_array_equal(ref_out, out.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            y = paddle.static.data(name="y", shape=self.shape, dtype=self.dtype)

            # Position args
            out1 = paddle.bitwise_xor(x, y)
            # Paddle keyword args
            out2 = paddle.bitwise_xor(x=x, y=y)
            # Torch keyword args
            out3 = paddle.bitwise_xor(input=x, other=y)
            # Tensor method
            out4 = x.bitwise_xor(y)

            exe = paddle.base.Executor(paddle.CPUPlace())
            fetches = exe.run(
                main,
                feed={"x": self.np_x, "y": self.np_y},
                fetch_list=[out1, out2, out3, out4],
            )
            ref_out = np.bitwise_xor(self.np_x, self.np_y)
            for out in fetches:
                np.testing.assert_array_equal(out, ref_out)


class TestTensorCumsumInplaceCompatibility(unittest.TestCase):
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
