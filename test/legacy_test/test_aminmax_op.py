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
from op_test import OpTest

import paddle


def ref_aminmax(x, axis=None, keepdim=False):
    min_val = np.amin(x, axis=axis, keepdims=keepdim)
    max_val = np.amax(x, axis=axis, keepdims=keepdim)
    return min_val, max_val


def ref_aminmax_grad(x, axis=None, keepdim=False):
    """Compute expected gradient: amin_grad + amax_grad (evenly distributed)."""
    min_val = np.amin(x, axis=axis, keepdims=True)
    max_val = np.amax(x, axis=axis, keepdims=True)
    min_mask = (x == min_val).astype(x.dtype)
    max_mask = (x == max_val).astype(x.dtype)
    min_count = np.sum(min_mask, axis=axis, keepdims=True)
    max_count = np.sum(max_mask, axis=axis, keepdims=True)
    grad = min_mask / min_count + max_mask / max_count
    return grad


class TestAminmaxOp(OpTest):
    def setUp(self):
        self.op_type = "aminmax"
        self.python_api = paddle.aminmax
        self.public_python_api = paddle.aminmax
        self.init_dtype()
        self.init_shape()
        self.init_args()
        np.random.seed(2025)
        self.input_data = np.random.random(self.shape).astype(self.dtype)
        self.inputs = {'X': self.input_data}
        self.attrs = {'axis': self.axis, 'keepdim': self.keepdim}
        min_val, max_val = ref_aminmax(
            self.input_data, axis=self.axis_for_np, keepdim=self.keepdim
        )
        self.outputs = {'Min': min_val, 'Max': max_val}

    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape = [2, 3, 4]

    def init_args(self):
        self.axis = []
        self.axis_for_np = None
        self.keepdim = False

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['Min', 'Max'],
            check_pir=True,
        )


class TestAminmaxOpAxis0(TestAminmaxOp):
    def init_args(self):
        self.axis = [0]
        self.axis_for_np = 0
        self.keepdim = False


class TestAminmaxOpAxis1(TestAminmaxOp):
    def init_args(self):
        self.axis = [1]
        self.axis_for_np = 1
        self.keepdim = False


class TestAminmaxOpAxisNeg(TestAminmaxOp):
    def init_args(self):
        self.axis = [-1]
        self.axis_for_np = -1
        self.keepdim = False


class TestAminmaxOpKeepdim(TestAminmaxOp):
    def init_args(self):
        self.axis = [1]
        self.axis_for_np = 1
        self.keepdim = True


class TestAminmaxOpFloat32(TestAminmaxOp):
    def init_dtype(self):
        self.dtype = np.float32

    def init_shape(self):
        self.shape = [3, 5]


class TestAminmaxOpZeroDim(TestAminmaxOp):
    def init_shape(self):
        self.shape = []


class TestAminmaxAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.x_np = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        min_val, max_val = paddle.aminmax(x, axis=1, keepdim=True)

        expected_min, expected_max = ref_aminmax(
            self.x_np, axis=1, keepdim=True
        )
        np.testing.assert_allclose(min_val.numpy(), expected_min, rtol=1e-05)
        np.testing.assert_allclose(max_val.numpy(), expected_max, rtol=1e-05)
        paddle.enable_static()

    def test_dygraph_gradient(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        min_val, max_val = paddle.aminmax(x)
        loss = min_val.sum() + max_val.sum()
        loss.backward()

        expected_grad = ref_aminmax_grad(self.x_np)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)
        paddle.enable_static()

    def test_dygraph_gradient_with_axis(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        min_val, max_val = paddle.aminmax(x, axis=1)
        loss = min_val.sum() + max_val.sum()
        loss.backward()

        expected_grad = ref_aminmax_grad(self.x_np, axis=1)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)
        paddle.enable_static()

    def test_dygraph_gradient_duplicate_values(self):
        paddle.disable_static()
        x_np = np.array(
            [[0.9, 0.1, 0.1, 0.9], [0.1, 0.9, 0.5, 0.1]], dtype='float64'
        )
        x = paddle.to_tensor(x_np, stop_gradient=False)
        min_val, max_val = paddle.aminmax(x)
        loss = min_val.sum() + max_val.sum()
        loss.backward()

        expected_grad = ref_aminmax_grad(x_np)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-05)
        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name='x', dtype='float64', shape=[2, 4])
            min_val, max_val = paddle.aminmax(x, axis=1, keepdim=True)

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={'x': self.x_np},
                fetch_list=[min_val, max_val],
            )

            expected_min, expected_max = ref_aminmax(
                self.x_np, axis=1, keepdim=True
            )
            np.testing.assert_allclose(fetches[0], expected_min, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], expected_max, rtol=1e-05)


class TestAminmaxCompatibility(unittest.TestCase):
    """Test paddle/torch keyword alias compatibility."""

    def setUp(self):
        np.random.seed(2025)
        self.np_x = np.array(
            [[0.2, 0.3, 0.5, 0.9], [0.1, 0.2, 0.6, 0.7]], dtype='float64'
        )

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_x)

        # 1. Paddle Positional arguments
        min1, max1 = paddle.aminmax(x)
        # 2. Paddle keyword arguments
        min2, max2 = paddle.aminmax(x=x)
        # 3. PyTorch keyword arguments (alias)
        min3, max3 = paddle.aminmax(input=x)
        # 4. Mixed arguments (positional x + keyword axis)
        min4, max4 = paddle.aminmax(x, axis=0)
        # 5. PyTorch keyword arguments (dim alias)
        min5, max5 = paddle.aminmax(x, dim=0)
        # 6. Tensor method
        min6, max6 = x.aminmax()

        # Verify outputs without axis
        ref_min = np.amin(self.np_x)
        ref_max = np.amax(self.np_x)
        for min_val in [min1, min2, min3, min6]:
            np.testing.assert_allclose(min_val.numpy(), ref_min)
        for max_val in [max1, max2, max3, max6]:
            np.testing.assert_allclose(max_val.numpy(), ref_max)

        # Verify outputs with axis
        ref_min_ax0 = np.amin(self.np_x, axis=0)
        ref_max_ax0 = np.amax(self.np_x, axis=0)
        for min_val in [min4, min5]:
            np.testing.assert_allclose(min_val.numpy(), ref_min_ax0)
        for max_val in [max4, max5]:
            np.testing.assert_allclose(max_val.numpy(), ref_max_ax0)

        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.static.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=[2, 4], dtype='float64')

            # 1. Paddle Positional arguments
            min1, max1 = paddle.aminmax(x)
            # 2. Paddle keyword arguments
            min2, max2 = paddle.aminmax(x=x)
            # 3. PyTorch keyword arguments (alias)
            min3, max3 = paddle.aminmax(input=x)
            # 4. Tensor method
            min4, max4 = x.aminmax()

            exe = paddle.static.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_x},
                fetch_list=[
                    min1,
                    max1,
                    min2,
                    max2,
                    min3,
                    max3,
                    min4,
                    max4,
                ],
            )

            ref_min = np.amin(self.np_x)
            ref_max = np.amax(self.np_x)
            for i in range(0, len(fetches), 2):
                np.testing.assert_allclose(fetches[i], ref_min)
                np.testing.assert_allclose(fetches[i + 1], ref_max)


if __name__ == '__main__':
    unittest.main()
