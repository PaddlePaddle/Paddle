# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce
from operator import mul

import numpy as np
from prim.composite_ops.utils import SUB_TOLERANCE

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.incubate.autograd import primapi

TOLERANCE_NUMPY = {
    "float32": {"rtol": 2e-5, "atol": 2e-5},
    "float64": {"rtol": 1e-11, "atol": 1e-11},
}

TOLERANCE_COMP_GRAD = {
    "float64": {"rtol": 1e-13, "atol": 1e-13},
    "float32": {"rtol": 1e-5, "atol": 1e-5},
    "float16": {"rtol": 1e-3, "atol": 1e-3},  # amp
}


def generate_data(shape1, shape2, shape3, dtype="float32"):
    np.random.seed(12)
    np_data1 = np.random.random(shape1).astype(dtype)
    np_data2 = np.random.random(shape2).astype(dtype)
    np_data3 = np.random.random(shape3).astype(dtype)
    np_data4 = np.ones_like(np_data1).astype(dtype)
    return np_data1, np_data2, np_data3, np_data4


def _reference_layer_norm_naive(
    x, scale, beta, epsilon=1e-5, begin_norm_axis=1
):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [N, D]

    mean = np.mean(x, axis=1)
    difference = x - mean.reshape([N, 1])
    var_tmp1 = np.power(difference, 2.0)
    variance = np.mean(var_tmp1, axis=1)
    var = variance + epsilon
    # var = np.var(x, axis=1) + epsilon
    output = np.divide(
        (x - mean.reshape([N, 1])), (np.sqrt(var)).reshape([N, 1])
    )
    if scale is not None:
        output = scale.reshape([1, D]) * output
    if beta is not None:
        output = output + beta.reshape([1, D])

    x.shape, output.shape = x_shape, x_shape
    return output, mean, var


def _reference_layer_norm_grad(
    x, grad_y, scale, bias, mean, var, begin_norm_axis=1
):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)

    if scale is not None:
        scale_shape = scale.shape
        scale.shape = [1, D]
    x.shape, grad_y.shape = [N, D], [N, D]
    var.shape, mean.shape = [N, 1], [N, 1]

    # d_bias
    if bias is not None:
        d_bias = np.sum(grad_y, axis=0).reshape([1, D])
    else:
        d_bias = None
    # d_scale
    if scale is not None:
        d_scale = np.sum(
            ((x - mean) * np.sqrt(1 / var)) * grad_y, axis=0
        ).reshape([1, D])
    else:
        d_scale = None
    # dx
    if scale is not None:
        dx_end = scale * np.sqrt(1.0 / var) * grad_y
        d_mean_0 = np.sum(-np.sqrt(1.0 / var) * grad_y * scale, axis=1).reshape(
            [N, 1]
        )  # the second part equals to zero.
        d_mean = 1.0 / D * d_mean_0
        d_std = np.sum(
            -(1.0 / var) * (x - mean) * grad_y * scale, axis=1
        ).reshape([N, 1]) * (
            1.0 / D * np.sqrt(1.0 / var).reshape([N, 1]) * (x - mean)
        )
    else:
        dx_end = 1.0 * np.sqrt(1.0 / var) * grad_y
        d_mean_0 = np.sum(-np.sqrt(1.0 / var) * grad_y * 1.0, axis=1).reshape(
            [N, 1]
        )  # the second part equals to zero.
        d_mean = 1.0 / D * d_mean_0
        d_std = np.sum(
            -(1.0 / var) * (x - mean) * grad_y * 1.0, axis=1
        ).reshape([N, 1]) * (
            1.0 / D * np.sqrt(1.0 / var).reshape([N, 1]) * (x - mean)
        )

    grad_x = dx_end + d_mean + d_std

    grad_x.shape, x.shape, grad_y.shape = x_shape, x_shape, x_shape
    var.shape, mean.shape = [N], [N]

    if scale is not None:
        scale.shape = scale_shape

    return grad_x, d_scale, d_bias


class Attr:
    def __init__(self) -> None:
        self.dtype = None
        self.n_shape = None
        self.shape1 = None
        self.shape2 = None
        self.shape3 = None

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype

    def set_shape(self, n_shape, shape1, shape2, shape3) -> None:
        self.n_shape = n_shape
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3

    def get_rtol(self, flag):
        rtol = SUB_TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = SUB_TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x, norm_shape, w, b):
    return F.layer_norm(x, norm_shape, w, b)


def dygraph_fused_backward_withNone(x, norm_shape, w, b, y_g):
    paddle.disable_static()
    x.stop_gradient = False
    res = fn(x, norm_shape, w, b)
    gradients = paddle.grad(res, x, y_g)
    return gradients


def dygraph_fused_backward(x, norm_shape, w, b, y_g):
    paddle.disable_static()
    x.stop_gradient = False
    w.stop_gradient = False
    b.stop_gradient = False
    res = fn(x, norm_shape, w, b)
    gradients = paddle.grad(res, [x, w, b], y_g)
    return gradients[0], gradients[1], gradients[2]


class TestCompositelayer_norm(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32"]
        self.n_shape = [[4], [64, 128], [64]]
        self.shape1s = [[3, 4], [64, 64, 128], [128, 64, 64]]
        self.shape2s = [[4], [64 * 128], [64]]
        self.shape3s = [[4], [64 * 128], [64]]

    def static_comp_forward(self, inputs, norm_shape, weight, bias, y_g):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            w.stop_gradient = False
            b = paddle.static.data('b', shape=bias.shape, dtype=str(bias.dtype))
            b.stop_gradient = False

            y = fn(x, norm_shape, w, b)

            y_grad = paddle.static.data(
                'y_grad', shape=y_g.shape, dtype=str(y_g.dtype)
            )

            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm is splitted into small ops
            self.assertTrue('layer_norm' not in fwd_ops_new)

            z = paddle.static.gradients([y], [x, w, b], y_grad)

            fwd_ops_grad = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm_grad not in grad block
            self.assertTrue('layer_norm_grad' not in fwd_ops_grad)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'w': weight,
                'b': bias,
                'y_grad': y_g,
            },
            fetch_list=z,
        )
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def static_comp_forward_withNone(
        self, inputs, norm_shape, weight, bias, y_g
    ):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y_grad = paddle.static.data(
                'y_grad', shape=y_g.shape, dtype=str(y_g.dtype)
            )
            x.stop_gradient = False
            y = fn(x, norm_shape, weight, bias)

            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm is splitted into small ops
            self.assertTrue('layer_norm' not in fwd_ops_new)

            z = paddle.static.gradients([y], x, y_grad)
            fwd_ops_grad = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm_grad not in grad block
            self.assertTrue('layer_norm_grad' not in fwd_ops_grad)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'y_grad': y_g,
            },
            fetch_list=z,
        )
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    # to_pirm after gradient can call comp_layer_norm_grad
    def static_comp_forward_and_backward(
        self, inputs, norm_shape, weight, bias, y_g
    ):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            w.stop_gradient = False
            b = paddle.static.data('b', shape=bias.shape, dtype=str(bias.dtype))
            b.stop_gradient = False

            y_grad = paddle.static.data(
                'y_grad', shape=y_g.shape, dtype=str(y_g.dtype)
            )

            y = fn(x, norm_shape, w, b)

            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            z = paddle.static.gradients([y], [x, w, b], y_grad)

            primapi.to_prim(blocks)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'w': weight,
                'b': bias,
                'y_grad': y_g,
            },
            fetch_list=z,
        )
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def static_comp_forward_and_backward_withNone(
        self, inputs, norm_shape, weight, bias, y_g
    ):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False

            y_grad = paddle.static.data(
                'y_grad', shape=y_g.shape, dtype=str(y_g.dtype)
            )

            y = fn(x, norm_shape, weight, bias)

            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            z = paddle.static.gradients([y], [x], y_grad)

            primapi.to_prim(blocks)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'y_grad': y_g,
            },
            fetch_list=z,
        )
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def compare_comp_forward(self):
        x, w, b, y_g = generate_data(
            attrs.shape1, attrs.shape2, attrs.shape3, attrs.dtype
        )
        n_shape = attrs.n_shape
        x_p = paddle.to_tensor(x)
        w_p = paddle.to_tensor(w)
        b_p = paddle.to_tensor(b)
        y_g_p = paddle.to_tensor(y_g)

        expect = dygraph_fused_backward(x_p, n_shape, w_p, b_p, y_g_p)
        actual_fwd = self.static_comp_forward(x, n_shape, w, b, y_g)
        actual_all = self.static_comp_forward_and_backward(
            x, n_shape, w, b, y_g
        )

        assert expect[0].numpy().dtype == actual_fwd[0].dtype
        np.testing.assert_allclose(
            expect[0].numpy(),
            actual_fwd[0],
            rtol=attrs.get_rtol("backward"),
            atol=attrs.get_atol("backward"),
        )

        np.testing.assert_allclose(
            actual_fwd[0],
            actual_all[0],
            rtol=TOLERANCE_COMP_GRAD[attrs.dtype]['rtol'],
            atol=TOLERANCE_COMP_GRAD[attrs.dtype]['atol'],
        )

    def compare_comp_forward_withNone(self):
        x, w, b, y_g = generate_data(
            attrs.shape1, attrs.shape2, attrs.shape3, attrs.dtype
        )
        n_shape = attrs.n_shape
        x_p = paddle.to_tensor(x)
        w_p = paddle.to_tensor(w)
        b_p = paddle.to_tensor(b)
        y_g_p = paddle.to_tensor(y_g)

        expect_2 = dygraph_fused_backward_withNone(
            x_p, n_shape, None, None, y_g_p
        )[0].numpy()
        actual_2 = self.static_comp_forward_withNone(
            x, n_shape, None, None, y_g
        )[0]
        actual_all_2 = self.static_comp_forward_and_backward_withNone(
            x, n_shape, None, None, y_g
        )[0]

        assert expect_2.dtype == actual_2.dtype
        np.testing.assert_allclose(
            expect_2,
            actual_2,
            rtol=attrs.get_rtol("backward"),
            atol=attrs.get_atol("backward"),
        )

        np.testing.assert_allclose(
            expect_2,
            actual_all_2,
            rtol=TOLERANCE_COMP_GRAD[attrs.dtype]['rtol'],
            atol=TOLERANCE_COMP_GRAD[attrs.dtype]['atol'],
        )

    def test_backward(self):
        for j in self.dtypes:
            if paddle.device.get_device() == "cpu":
                print("need pass this case")
                continue
            for t in range(0, len(self.shape1s)):
                attrs.set_dtype(j)
                attrs.set_shape(
                    self.n_shape[t],
                    self.shape1s[t],
                    self.shape2s[t],
                    self.shape3s[t],
                )
                self.compare_comp_forward()

    def test_backward_withNone(self):
        for t in range(0, len(self.shape1s)):
            if paddle.device.get_device() == "cpu":
                print("need pass this case")
                continue
            attrs.set_dtype("float32")
            attrs.set_shape(
                self.n_shape[t],
                self.shape1s[t],
                self.shape2s[t],
                self.shape3s[t],
            )
            self.compare_comp_forward_withNone()


class TestCompositelayer_normPrimBackward(unittest.TestCase):
    def setUp(self):
        core._set_prim_backward_enabled(True)
        self.dtypes = ["float32"]
        self.n_shape = [[4], [64, 128], [64]]
        self.shape1s = [[3, 4], [64, 64, 128], [128, 64, 64]]
        self.shape2s = [[4], [64 * 128], [64]]
        self.shape3s = [[4], [64 * 128], [64]]

    def static_comp_forward_and_backward(
        self, inputs, norm_shape, weight, bias
    ):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            b = paddle.static.data('b', shape=bias.shape, dtype=str(bias.dtype))
            y = fn(x, norm_shape, w, b)

            blocks = main_program.blocks
            primapi.to_prim(blocks)
            z = paddle.static.gradients([y], x)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'w': weight,
                'b': bias,
            },
            fetch_list=[z],
        )
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def static_comp_forward_and_backward_withNone(
        self, inputs, norm_shape, weight, bias
    ):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            y = fn(x, norm_shape, weight, bias)

            blocks = main_program.blocks
            primapi.to_prim(blocks)
            z = paddle.static.gradients([y], x)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
            },
            fetch_list=[z],
        )
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res

    def compare_backward(self):
        x, w, b, y_g = generate_data(
            attrs.shape1, attrs.shape2, attrs.shape3, attrs.dtype
        )
        n_shape = attrs.n_shape
        x_p = paddle.to_tensor(x)
        w_p = paddle.to_tensor(w)
        b_p = paddle.to_tensor(b)
        y_g_p = paddle.to_tensor(y_g)

        expect = dygraph_fused_backward(x_p, n_shape, w_p, b_p, y_g_p)[
            0
        ].numpy()
        actual = self.static_comp_forward_and_backward(x, n_shape, w, b)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("prim_backward"),
            atol=attrs.get_rtol("prim_backward"),
        )

        expect_2 = dygraph_fused_backward_withNone(
            x_p, n_shape, None, None, y_g_p
        )[0].numpy()
        actual_2 = self.static_comp_forward_and_backward_withNone(
            x, n_shape, None, None
        )[0]
        assert expect_2.dtype == actual_2.dtype
        np.testing.assert_allclose(
            expect_2,
            actual_2,
            rtol=attrs.get_rtol("prim_backward"),
            atol=attrs.get_atol("prim_backward"),
        )

    def test_prim_backward(self):
        for j in self.dtypes:
            if paddle.device.get_device() == "cpu":
                print("need pass this case")
                continue
            for t in range(0, len(self.shape1s)):
                attrs.set_dtype(j)
                attrs.set_shape(
                    self.n_shape[t],
                    self.shape1s[t],
                    self.shape2s[t],
                    self.shape3s[t],
                )
                self.compare_backward()


class TestCompositeNumpylayer_norm(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float32", "float64"]
        self.n_shape = [
            [4],
            [64, 128],
        ]
        self.shape1s = [
            [3, 4],
            [64, 64, 128],
        ]
        self.shape2s = [
            [4],
            [64 * 128],
        ]
        self.shape3s = [
            [4],
            [64 * 128],
        ]

    def static_comp_forward(self, inputs, norm_shape, weight, bias, y_grad):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            b = paddle.static.data('b', shape=bias.shape, dtype=str(bias.dtype))
            y = fn(x, norm_shape, w, b)
            y_g = paddle.static.data(
                'y_g', shape=y_grad.shape, dtype=str(y_grad.dtype)
            )
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            primapi.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm is splitted into small ops
            self.assertTrue('layer_norm' not in fwd_ops_new)

            z = paddle.static.gradients([y], x, y_g)
            fwd_ops_grad = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm_grad not in grad block

            self.assertTrue('layer_norm_grad' not in fwd_ops_grad)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={
                'x': inputs,
                'w': weight,
                'b': bias,
                'y_g': y_grad,
            },
            fetch_list=[y, z[0]],
        )
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res[0], res[1]

    def static_comp_forward_prim(
        self, inputs, norm_shape, weight, bias, y_grad
    ):
        paddle.enable_static()
        core._set_prim_all_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            x.stop_gradient = False
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            b = paddle.static.data('b', shape=bias.shape, dtype=str(bias.dtype))
            y = fn(x, norm_shape, w, b)
            y_g = paddle.static.data(
                'y_g', shape=y_grad.shape, dtype=str(y_grad.dtype)
            )

            blocks = main_program.blocks
            primapi.to_prim(blocks)
            z = paddle.static.gradients([y], x)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program,
            feed={'x': inputs, 'w': weight, 'b': bias, 'y_g': y_grad},
            fetch_list=[y, z[0]],
        )
        paddle.disable_static()
        core._set_prim_all_enabled(False)
        return res[0], res[1]

    def compare_backward(self):
        x, w, b, y_grad = generate_data(
            attrs.shape1, attrs.shape2, attrs.shape3, attrs.dtype
        )

        n_shape = attrs.n_shape

        composite1, composite2 = self.static_comp_forward(
            x, n_shape, w, b, y_grad
        )
        composite_p1, composite_p2 = self.static_comp_forward_prim(
            x, n_shape, w, b, y_grad
        )

        numpy1, mean, variance = _reference_layer_norm_naive(
            x,
            w,
            b,
        )
        numpy2, _, _ = _reference_layer_norm_grad(
            x,
            y_grad,
            w,
            b,
            mean,
            variance,
        )

        # forward_prim
        np.testing.assert_allclose(
            composite1,
            numpy1,
            rtol=TOLERANCE_NUMPY[attrs.dtype]['rtol'],
            atol=TOLERANCE_NUMPY[attrs.dtype]['atol'],
        )
        # forward_prim + backward
        np.testing.assert_allclose(
            composite2,
            numpy2,
            rtol=TOLERANCE_NUMPY[attrs.dtype]['rtol'],
            atol=TOLERANCE_NUMPY[attrs.dtype]['atol'],
        )
        # forward_prim + backward_prim
        np.testing.assert_allclose(
            composite_p2,
            numpy2,
            rtol=TOLERANCE_NUMPY[attrs.dtype]['rtol'],
            atol=TOLERANCE_NUMPY[attrs.dtype]['atol'],
        )

    def test_backward(self):
        for j in self.dtypes:
            for t in range(0, len(self.shape1s)):
                attrs.set_dtype(j)
                attrs.set_shape(
                    self.n_shape[t],
                    self.shape1s[t],
                    self.shape2s[t],
                    self.shape3s[t],
                )
                self.compare_backward()


if __name__ == '__main__':
    unittest.main()
