#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Note:
# 0D Tensor indicates that the tensor's dimension is 0
# 0D Tensor's shape is always [], numel is 1
# which can be created by paddle.rand([])

import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle
from paddle.framework import in_pir_mode

# Use to test zero-dim of Sundry API, which is unique and can not be classified
# with others. It can be implemented here flexibly.


class TestSundryAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    def assertShapeEqual(self, out, target_tuple):
        if not paddle.framework.in_pir_mode():
            out_shape = list(out.shape)
        else:
            out_shape = out.shape
        self.assertEqual(out_shape, target_tuple)

    @prog_scope()
    def test_polygamma(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.polygamma(x, 2)
        grad_list = paddle.static.append_backward(out, parameter_list=[x])
        x_grad = grad_list[0][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

    @prog_scope()
    def test_frexp(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out1, out2 = paddle.frexp(x)
        grad_list = paddle.static.append_backward(out1, parameter_list=[x])
        x_grad = grad_list[0][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, out2, x_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

    @prog_scope()
    def test_pairwise_distance(self):
        x = paddle.rand([5])
        x.stop_gradient = False
        y = paddle.rand([5])
        y.stop_gradient = False

        out = paddle.nn.functional.pairwise_distance(x, y)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, y])
        x_grad, y_grad = (_grad for _param, _grad in grad_list)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (5,))
        self.assertEqual(res[2].shape, (5,))

    @prog_scope()
    def test_take(self):
        x1 = paddle.rand([4, 5])
        x1.stop_gradient = False
        out1 = paddle.take(x1, paddle.to_tensor(2))
        x1_grad = paddle.static.append_backward(out1, parameter_list=[x1])
        x1_grad = x1_grad[0][1]

        x2 = paddle.rand([])
        x2.stop_gradient = False
        out2 = paddle.take(x2, paddle.to_tensor(0))
        x2_grad = paddle.static.append_backward(out2, parameter_list=[x2])
        x2_grad = x2_grad[0][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, x1_grad, out2, x2_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 5))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        np.testing.assert_allclose(res[3], 1.0)

    @prog_scope()
    def test_trapezoid(self):
        y = paddle.rand([5])
        y.stop_gradient = False
        out = paddle.trapezoid(y, dx=2.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[y])
        y_grad = grad_list[0][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, y_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (5,))

    @prog_scope()
    def test_create_parameter(self):
        if not in_pir_mode():
            zero_dim_param = paddle.create_parameter(shape=[], dtype='float32')
            self.assertShapeEqual(zero_dim_param, [])
            prog = paddle.static.default_startup_program()
            res = self.exe.run(prog, fetch_list=[zero_dim_param])
            self.assertEqual(res[0].shape, ())
            return
        zero_dim_param = paddle.create_parameter(shape=[], dtype='float32')
        self.assertEqual(zero_dim_param.shape, [])
        startup_prog = paddle.static.default_startup_program()
        main_prog = paddle.static.default_main_program()
        self.exe.run(startup_prog)
        (zero_dim_param_res,) = self.exe.run(
            main_prog, fetch_list=[zero_dim_param]
        )
        self.assertEqual(zero_dim_param_res.shape, ())

    @prog_scope()
    def test_getitem(self):
        # case1: When all axis have a scalar indice, output should be a 0-d Tensor;
        x = paddle.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        x.stop_gradient = False
        out = x[1, 2, 3, 4]
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        x_out_grad = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, *x_out_grad])

        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(res[0], np.array(119))
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[2], 1.0)
        self.assertEqual(res[1].shape, (2, 3, 4, 5))
        x_grad_expected = np.zeros((2, 3, 4, 5))
        x_grad_expected[1, 2, 3, 4] = 1.0
        np.testing.assert_allclose(res[1], x_grad_expected)

        # case2: When one axis has a 0-d Tensor indice, the output should be same as int indice.
        x2 = paddle.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        out1 = x2[1, 2]
        out2 = x2[
            paddle.full([], 1, dtype='int32'), paddle.full([], 2, dtype='int32')
        ]
        res = self.exe.run(prog, fetch_list=[out1, out2])
        np.testing.assert_allclose(res[0], res[1])

        # case3: When all axis have a scalar indice (i.e. case1) and has None indice,
        # ndim of output should be same with numbers of None.
        x3 = paddle.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        out3 = x3[1, 2, None, 3, 4]
        out4 = x3[1, None, 2, None, 3, 4]
        res = self.exe.run(prog, fetch_list=[out3, out4])
        self.assertEqual(res[0].shape, (1,))
        np.testing.assert_allclose(res[0], np.array([119]))
        self.assertEqual(res[1].shape, (1, 1))
        np.testing.assert_allclose(res[1], np.array([[119]]))

        # case4: 1-D Tensor will be treated as vector, no axis decrease will happen.
        x4 = paddle.ones((2, 3, 4))
        indice = paddle.ones([1], dtype='int32')
        out5 = x4[indice]
        out6 = x4[indice, indice]
        res = self.exe.run(prog, fetch_list=[out5, out6])

        self.assertEqual(res[0].shape, (1, 3, 4))
        np.testing.assert_allclose(res[0], np.ones((1, 3, 4)))
        self.assertEqual(res[1].shape, (1, 4))
        np.testing.assert_allclose(res[1], np.ones((1, 4)))

    @prog_scope()
    def test_expand(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out = paddle.expand(x, shape=[1])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, (1,))
        self.assertEqual(res[3], 1.0)

        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        out1 = paddle.expand(x1, shape=[])
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        grad_list = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x1, out1, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

        x2 = paddle.full([], 1, 'float32')
        x2.stop_gradient = False
        out2 = paddle.expand(x2, shape=[3, 3])
        grad_list = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )
        grad_list = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x2, out2, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (3, 3))
        self.assertEqual(res[1].any(), 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 9)
        self.assertEqual(res[3].shape, (3, 3))
        self.assertEqual(res[3].any(), 1.0)

    @prog_scope()
    def test_expand_as(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        y = paddle.full([], 1, 'float32')
        y.stop_gradient = False
        out = paddle.expand_as(x, y)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        y1 = paddle.full([1], 1, 'float32')
        y1.stop_gradient = False
        out1 = paddle.expand_as(x1, y1)
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x1, out1, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, (1,))
        self.assertEqual(res[3], 1.0)

        x2 = paddle.full([], 1, 'float32')
        x2.stop_gradient = False
        y2 = paddle.full([3, 3], 1, 'float32')
        y2.stop_gradient = False
        out2 = paddle.expand_as(x2, y2)
        grad_list = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )
        grad_list = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x2, out2, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (3, 3))
        self.assertEqual(res[1].any(), 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 9)
        self.assertEqual(res[3].shape, (3, 3))
        self.assertEqual(res[3].any(), 1.0)

    @prog_scope()
    def test_top_k(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out, indices = paddle.topk(x, k=1, axis=0)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, indices, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 0.0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[4], 1.0)

        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        out1, indices1 = paddle.topk(x1, k=1, axis=-1)
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        grad_list = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x1, out1, indices1, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 0.0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[4], 1.0)

        with self.assertRaises(ValueError):
            tmp = paddle.topk(x1, k=1, axis=2)

    @prog_scope()
    def test_broadcast_to(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out = paddle.broadcast_to(x, shape=[1])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, (1,))
        self.assertEqual(res[3], 1.0)

        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        out1 = paddle.broadcast_to(x1, shape=[])
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x1, out1, *grad_list])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

    @prog_scope()
    def test_argmin(self):
        # 1) x is 0D
        x = paddle.rand([])
        out1 = paddle.argmin(x, 0)
        out2 = paddle.argmin(x, -1)
        out3 = paddle.argmin(x, None)

        # 2) x is ND
        x4 = paddle.rand([3, 5])
        out4 = paddle.argmin(x, None)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                out4,
            ],
        )
        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(res[0], 0.0)
        self.assertEqual(res[1].shape, ())
        np.testing.assert_allclose(res[1], 0.0)
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[2], 0.0)
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_argmax(self):
        # 1) x is 0D
        x = paddle.rand([])
        out1 = paddle.argmax(x, 0)
        out2 = paddle.argmax(x, -1)
        out3 = paddle.argmax(x, None)

        # 2) x is ND
        x4 = paddle.rand([3, 5])
        out4 = paddle.argmax(x, None)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                out4,
            ],
        )
        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(res[0], 0.0)
        self.assertEqual(res[1].shape, ())
        np.testing.assert_allclose(res[1], 0.0)
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[2], 0.0)
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_kthvalue(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out, index = paddle.kthvalue(x, 1)
        grad_list = paddle.static.append_backward(out, parameter_list=[x])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, index, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertTrue(res[1] == res[0])
        self.assertEqual(res[2].shape, ())
        self.assertTrue(res[2] == 0)

        self.assertEqual(res[3].shape, ())
        self.assertTrue(res[3] == 1.0)

        # 2) x is 1D
        x1 = paddle.rand([5])
        x1.stop_gradient = False
        out1, index1 = paddle.kthvalue(x1, 1)
        grad_list = paddle.static.append_backward(out1, parameter_list=[x1])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, index1, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (5,))

    @prog_scope()
    def test_mode(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out, index = paddle.mode(x)
        grad_list = paddle.static.append_backward(out, parameter_list=[x])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, index, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertTrue(res[2] == 1.0)

        # 2) x is 1D
        x1 = paddle.rand([5])
        x1.stop_gradient = False
        out1, index1 = paddle.mode(x1)
        grad_list = paddle.static.append_backward(out1, parameter_list=[x1])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, index1, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (5,))

    @prog_scope()
    def test_is_empty(self):
        # 1) x is 0D
        x1 = paddle.rand([])
        out1 = paddle.is_empty(x1)

        # 2) x is 1D
        x2 = paddle.rand([5])
        out2 = paddle.is_empty(x2)

        # 3) x is ND
        x3 = paddle.rand([3, 5])
        out3 = paddle.is_empty(x3)

        x4 = paddle.rand([3, 0, 5])
        out4 = paddle.is_empty(x4)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[out1, out2, out3, out4],
        )

        self.assertEqual(res[0].shape, ())
        self.assertFalse(bool(res[0]))
        self.assertEqual(res[1].shape, ())
        self.assertFalse(bool(res[1]))
        self.assertEqual(res[2].shape, ())
        self.assertFalse(bool(res[2]))
        self.assertEqual(res[3].shape, ())
        self.assertTrue(bool(res[3]))

    @prog_scope()
    def test_as_complex(self):
        x = paddle.rand([2])
        x.stop_gradient = False
        out = paddle.as_complex(x)
        self.assertShapeEqual(
            x,
            [2],
        )
        self.assertShapeEqual(out, [])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, out, *grad_list],
        )

        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2,))
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_dot(self):
        # 1) x is 1d
        x = paddle.rand([2])
        x.stop_gradient = False
        y = paddle.rand([2])
        y.stop_gradient = False
        out = paddle.dot(x, y)

        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        x_grad = grad_list[0][1]
        out_grad = grad_list[1][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, x_grad, out, out_grad],
        )

        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (2,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

        # 2) x is 2D
        x1 = paddle.rand([2, 2])
        x1.stop_gradient = False
        y1 = paddle.rand([2, 2])
        y1.stop_gradient = False
        out1 = paddle.dot(x1, y1)

        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        x1_grad = grad_list[0][1]
        out1_grad = grad_list[1][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x1, x1_grad, out1, out1_grad],
        )

        self.assertEqual(res[0].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 2))
        self.assertEqual(res[2].shape, (2,))
        self.assertEqual(res[3].shape, (2,))

    @prog_scope()
    def test_inner(self):
        # 1) input is 1D
        x1 = paddle.rand([2])
        x1.stop_gradient = False
        y1 = paddle.rand([2])
        y1.stop_gradient = False
        out1 = paddle.inner(x1, y1)
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        x1_grad = grad_list[0][1]
        out1_grad = grad_list[1][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x1,
                x1_grad,
                out1,
                out1_grad,
            ],
        )
        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (2,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

        # 2) input is 2D
        x = paddle.rand([2, 3])
        x.stop_gradient = False
        y = paddle.rand([2, 3])
        y.stop_gradient = False
        out = paddle.inner(x, y)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        x_grad = grad_list[0][1]
        out_grad = grad_list[1][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                x_grad,
                out,
                out_grad,
            ],
        )

        self.assertEqual(res[0].shape, (2, 3))
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (2, 2))
        self.assertEqual(res[3].shape, (2, 2))

    @prog_scope()
    def test_tensordot(self):
        x = paddle.full(shape=[10], fill_value=0.25, dtype='float64')
        x.stop_gradient = False
        y = paddle.full(shape=[10], fill_value=0.25, dtype='float64')
        y.stop_gradient = False
        out = paddle.tensordot(x, y, axes=1)

        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        x_grad = grad_list[0][1]
        out_grad = grad_list[1][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, x_grad, out, out_grad],
        )

        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

        x = paddle.arange(6, dtype='float64').reshape([2, 3])
        y = paddle.arange(6, dtype='float64').reshape([2, 3])
        x.stop_gradient = False
        out = paddle.tensordot(x, y, axes=2)

        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        x_grad = grad_list[0][1]
        out_grad = grad_list[1][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, x_grad, out, out_grad],
        )

        self.assertEqual(res[0].shape, (2, 3))
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_metric_accuracy(self):
        x = paddle.full(shape=[2, 4], fill_value=0.25)
        y = paddle.full(shape=[2, 1], fill_value=1, dtype="int64")
        out = paddle.metric.accuracy(input=x, label=y, k=1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[out],
        )

        self.assertEqual(res[0].shape, ())

    @prog_scope()
    def test_static_accuracy(self):
        x = paddle.full(shape=[2, 4], fill_value=0.25)
        y = paddle.full(shape=[2, 1], fill_value=1, dtype="int64")
        out = paddle.static.accuracy(input=x, label=y, k=1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[out],
        )

        self.assertEqual(res[0].shape, ())

    @prog_scope()
    def test_std(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out1 = paddle.std(x)
        out2 = paddle.std(x, [])
        grad_list = paddle.static.append_backward(
            out1, parameter_list=[x, out1]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out1,
                out2,
                *grad_list,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())

    @prog_scope()
    def test_var(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out1 = paddle.var(x)
        out2 = paddle.var(x, [])
        grad_list = paddle.static.append_backward(
            out1, parameter_list=[x, out1]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out1,
                out2,
                *grad_list,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())


if __name__ == "__main__":
    unittest.main()
