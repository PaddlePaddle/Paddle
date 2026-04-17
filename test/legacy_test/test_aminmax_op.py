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
from utils import dygraph_guard, static_guard

import paddle
from paddle import base
from paddle.static import InputSpec


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
        self.shape = [4, 5, 6]

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
        self.shape = [10, 10]

    def test_check_grad(self):
        # float32 numerical gradient is unreliable for aminmax:
        # small perturbations can flip which element is min/max,
        # causing the numerical and analytic gradients to diverge.
        pass


class TestAminmaxOpZeroDim(TestAminmaxOp):
    def init_shape(self):
        self.shape = []


class TestAminmaxOpEmptyTensor(unittest.TestCase):
    """Test aminmax with empty tensor to cover the numel==0 early return in grad kernel."""

    def test_empty_tensor_grad(self):
        paddle.disable_static()
        x = paddle.zeros([0, 3], dtype='float64')
        x.stop_gradient = False
        min_val, max_val = paddle.aminmax(x, axis=0)
        loss = min_val.sum() + max_val.sum()
        loss.backward()
        self.assertEqual(list(x.grad.shape), [0, 3])
        paddle.enable_static()


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

    def test_dygraph_out(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        min_out = paddle.empty([])
        max_out = paddle.empty([])
        min_val, max_val = paddle.aminmax(x, out=(min_out, max_out))

        expected_min, expected_max = ref_aminmax(self.x_np)
        np.testing.assert_allclose(min_val.numpy(), expected_min, rtol=1e-05)
        np.testing.assert_allclose(max_val.numpy(), expected_max, rtol=1e-05)
        np.testing.assert_allclose(min_out.numpy(), expected_min, rtol=1e-05)
        np.testing.assert_allclose(max_out.numpy(), expected_max, rtol=1e-05)
        paddle.enable_static()

    def test_dygraph_out_with_axis(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np)
        min_out = paddle.empty([2])
        max_out = paddle.empty([2])
        min_val, max_val = paddle.aminmax(x, axis=1, out=(min_out, max_out))

        expected_min, expected_max = ref_aminmax(self.x_np, axis=1)
        np.testing.assert_allclose(min_val.numpy(), expected_min, rtol=1e-05)
        np.testing.assert_allclose(max_val.numpy(), expected_max, rtol=1e-05)
        np.testing.assert_allclose(min_out.numpy(), expected_min, rtol=1e-05)
        np.testing.assert_allclose(max_out.numpy(), expected_max, rtol=1e-05)
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


class TestAminmaxAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.enable_static()
        self.shape = [5, 6]
        self.dtype = 'float64'
        self.init_data()

    def init_data(self):
        self.np_input = np.random.random(self.shape).astype(self.dtype)

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        paddle_dygraph_min = []
        paddle_dygraph_max = []
        # 1. Paddle Positional arguments
        min1, max1 = paddle.aminmax(x, 1, True)
        paddle_dygraph_min.append(min1)
        paddle_dygraph_max.append(max1)
        # 2. Paddle keyword arguments
        min2, max2 = paddle.aminmax(x=x, axis=1, keepdim=True)
        paddle_dygraph_min.append(min2)
        paddle_dygraph_max.append(max2)
        # 4. PyTorch keyword arguments (alias)
        min3, max3 = paddle.aminmax(input=x, dim=1, keepdim=True)
        paddle_dygraph_min.append(min3)
        paddle_dygraph_max.append(max3)
        # 5. Mixed arguments
        min4, max4 = paddle.aminmax(x, dim=1, keepdim=True)
        paddle_dygraph_min.append(min4)
        paddle_dygraph_max.append(max4)
        # 6. out parameter test
        min_out = paddle.empty([])
        max_out = paddle.empty([])
        paddle.aminmax(x, 1, True, out=(min_out, max_out))
        paddle_dygraph_min.append(min_out)
        paddle_dygraph_max.append(max_out)
        # 7. Tensor method - args
        min5, max5 = x.aminmax(1, True)
        paddle_dygraph_min.append(min5)
        paddle_dygraph_max.append(max5)
        # 8. Tensor method - kwargs
        min6, max6 = x.aminmax(dim=1, keepdim=True)
        paddle_dygraph_min.append(min6)
        paddle_dygraph_max.append(max6)
        # Test default value
        min8, max8 = x.aminmax(1)
        # Numpy reference out
        ref_min = np.amin(self.np_input, 1, keepdims=True)
        ref_max = np.amax(self.np_input, 1, keepdims=True)
        # Check
        for out in paddle_dygraph_min:
            np.testing.assert_allclose(ref_min, out.numpy())
        for out in paddle_dygraph_max:
            np.testing.assert_allclose(ref_max, out.numpy())
        ref_min = np.amin(self.np_input, 1, keepdims=False)
        ref_max = np.amax(self.np_input, 1, keepdims=False)
        np.testing.assert_allclose(ref_min, min8.numpy())
        np.testing.assert_allclose(ref_max, max8.numpy())
        paddle.enable_static()

    def test_static_Compatibility(self):
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            # 1. Paddle Positional arguments
            min1, max1 = paddle.aminmax(x, 1, True)
            # 2. Paddle keyword arguments
            min2, max2 = paddle.aminmax(x=x, axis=1, keepdim=True)
            # 4. PyTorch keyword arguments (alias)
            min3, max3 = paddle.aminmax(input=x, dim=1, keepdim=True)
            # 5. Mixed arguments
            min4, max4 = paddle.aminmax(x, dim=1, keepdim=True)
            # 6. Do not support out in static
            # 7. Tensor method - args
            min5, max5 = x.aminmax(1, True)
            # 8. Tensor method - kwargs
            min6, max6 = x.aminmax(dim=1, keepdim=True)
            # Test default value
            min8, max8 = x.aminmax()
            exe = base.Executor()
            fetches = exe.run(
                main,
                feed={"x": self.np_input},
                fetch_list=[
                    min1,
                    max1,
                    min2,
                    max2,
                    min3,
                    max3,
                    min4,
                    max4,
                    min5,
                    max5,
                    min6,
                    max6,
                    min8,
                    max8,
                ],
            )
            ref_min = np.amin(self.np_input, 1, keepdims=True)
            ref_max = np.amax(self.np_input, 1, keepdims=True)
            for i in range(0, len(fetches) - 2, 2):
                np.testing.assert_allclose(fetches[i], ref_min)
                np.testing.assert_allclose(fetches[i + 1], ref_max)
            ref_min = np.amin(self.np_input)
            ref_max = np.amax(self.np_input)
            np.testing.assert_allclose(fetches[-2], ref_min)
            np.testing.assert_allclose(fetches[-1], ref_max)


def aminmax_net(x):
    return paddle.aminmax(x, axis=1)


class TestAminmaxDynamicShape(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.shape = [4, 5, 6]
        self.dtype = "float32"
        self.init_shape = [None, None, 6]
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.net = aminmax_net
        self.enable_cinn = True
        self.tol = 1e-6

    def base_net(self, flag=None):
        x = paddle.to_tensor(self.x)
        if flag == "static":
            fn = paddle.jit.to_static(
                self.net,
                input_spec=[
                    InputSpec(shape=self.init_shape, dtype=self.dtype),
                ],
                backend="CINN" if self.enable_cinn else None,
                full_graph=True,
            )
            fn.eval()
            res = fn(x)
        elif flag == "dygraph":
            fn = self.net
            res = fn(x)
        return res

    def test_all_dynamic(self):
        with dygraph_guard():
            res_ref = self.base_net("dygraph")
            res = self.base_net("static")
            for ref, actual in zip(res_ref, res):
                np.testing.assert_allclose(ref, actual, rtol=self.tol)


class TestAminmaxInferSymbolicShapePass(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.x_np = np.random.random((2, 4, 6)).astype("float32")

    def test_infer_symbolic_shape_pass(self):
        with static_guard(), paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.static.data(
                    name='x', shape=[None, 4, 6], dtype='float32'
                )
                min_val, max_val = paddle.aminmax(x, axis=1, keepdim=False)

                op_names = [op.name() for op in main.global_block().ops]
                self.assertIn("pd_op.aminmax", op_names)

                pm = paddle.base.libpaddle.pir.PassManager()
                paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, main)
                pm.run(main)

                exe = paddle.static.Executor()
                fetches = exe.run(
                    main,
                    feed={'x': self.x_np},
                    fetch_list=[min_val, max_val],
                )

                ref_min, ref_max = ref_aminmax(self.x_np, axis=1, keepdim=False)
                np.testing.assert_allclose(fetches[0], ref_min, rtol=1e-05)
                np.testing.assert_allclose(fetches[1], ref_max, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
