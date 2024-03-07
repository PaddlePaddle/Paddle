#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.pir_utils import test_with_pir_api

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

    @test_with_pir_api
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

    @test_with_pir_api
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

    @test_with_pir_api
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

    @test_with_pir_api
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

    @test_with_pir_api
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
    def test_create_parameter_var(self):
        zero_dim_param = paddle.create_parameter(shape=[], dtype='float32')
        self.assertShapeEqual(zero_dim_param, [])
        prog = paddle.static.default_startup_program()
        res = self.exe.run(prog, fetch_list=[zero_dim_param])
        self.assertEqual(res[0].shape, ())

        zero_dim_var = paddle.static.create_global_var(
            shape=[], value=0.5, dtype='float32'
        )
        self.assertEqual(zero_dim_var.shape, ())
        prog = paddle.static.default_startup_program()
        res = self.exe.run(prog, fetch_list=[zero_dim_var])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 0.5)

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
        res = self.exe.run(prog, fetch_list=[out] + x_out_grad)

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
    def test_setitem(self):
        # NOTE(zoooo0820): __setitem__ has gradient problem in static graph.
        # To solve this, we may not support __setitem__ in static graph.
        # These unit tests will delete soon.

        # case1: all axis have a scalar indice
        x = paddle.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
        x.stop_gradient = False
        out = x * 2
        out = paddle.static.setitem(out, (1, 2, 3, 4), 10)
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name])

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(res[0][1, 2, 3, 4], np.array(10))
        self.assertEqual(res[1].shape, (2, 3, 4, 5))
        x_grad_expected = np.ones((2, 3, 4, 5)) * 2
        x_grad_expected[1, 2, 3, 4] = 0
        np.testing.assert_allclose(res[1], x_grad_expected)

        # case2: 0-D Tensor indice in some axis
        # NOTE(zoooo0820): Now, int/slice with 0-D Tensor will still be
        # treated as combined indexing, which is not support backward.
        # There should have more test cases such as out[1, indice, :] = 0.5 when this
        # problem is fixed.
        x = paddle.randn((2, 3, 4, 5))
        x.stop_gradient = False
        indice = paddle.full([], 1, dtype='int32')
        out = x * 1
        out = paddle.static.setitem(out, (indice, indice), 0.5)
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name])

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(res[0][1, 1], np.ones((4, 5)) * 0.5)
        x_grad_expected = np.ones((2, 3, 4, 5))
        x_grad_expected[1, 1] = 0
        np.testing.assert_allclose(res[1], x_grad_expected)

        # case3：0-D Tensor indice in some axis, value is a Tensor
        # and there is broadcast
        x = paddle.randn((2, 3, 4, 5))
        x.stop_gradient = False
        v = paddle.ones((4, 5), dtype='float32') * 5
        v.stop_gradient = False
        indice = paddle.full([], 1, dtype='int32')
        out = x * 1
        out = paddle.static.setitem(out, indice, v)
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, v.grad_name])

        self.assertEqual(out.shape, x.shape)
        np.testing.assert_allclose(res[0][1], np.ones((3, 4, 5)) * 5)
        x_grad_expected = np.ones((2, 3, 4, 5))
        x_grad_expected[1] = 0
        np.testing.assert_allclose(res[1], x_grad_expected)

    @test_with_pir_api
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
        res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[x1, out1] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[x2, out2] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (3, 3))
        self.assertEqual(res[1].any(), 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 9)
        self.assertEqual(res[3].shape, (3, 3))
        self.assertEqual(res[3].any(), 1.0)

    @test_with_pir_api
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
        res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[x1, out1] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[x2, out2] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, (3, 3))
        self.assertEqual(res[1].any(), 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 9)
        self.assertEqual(res[3].shape, (3, 3))
        self.assertEqual(res[3].any(), 1.0)

    @test_with_pir_api
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
        res = self.exe.run(prog, fetch_list=[x, out, indices] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[x1, out1, indices1] + grad_list)
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

    @test_with_pir_api
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
        res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[x1, out1] + grad_list)

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

    @test_with_pir_api
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

    @test_with_pir_api
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

    @test_with_pir_api
    @prog_scope()
    def test_kthvalue(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out, index = paddle.kthvalue(x, 1)
        grad_list = paddle.static.append_backward(out, parameter_list=[x])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, index] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[out1, index1] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (5,))

    @test_with_pir_api
    @prog_scope()
    def test_mode(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out, index = paddle.mode(x)
        grad_list = paddle.static.append_backward(out, parameter_list=[x])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, index] + grad_list)
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
        res = self.exe.run(prog, fetch_list=[out1, index1] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (5,))

    @test_with_pir_api
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

    @test_with_pir_api
    @prog_scope()
    def test_as_complex(self):
        x = paddle.rand([2])
        x.stop_gradient = False
        out = paddle.as_complex(x)
        self.assertShapeEqual(
            x,
            [
                2,
            ],
        )
        self.assertShapeEqual(out, [])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, out] + grad_list,
        )

        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2,))
        self.assertEqual(res[3].shape, ())

    @test_with_pir_api
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

    @test_with_pir_api
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

    @test_with_pir_api
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

    @test_with_pir_api
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
    def test_static_auc(self):
        x = paddle.full(shape=[3, 2], fill_value=0.25)
        y = paddle.full(shape=[3], fill_value=1, dtype="int64")
        out = paddle.static.auc(input=x, label=y)[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[out],
        )

        self.assertEqual(res[0].shape, ())

    @test_with_pir_api
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
            ]
            + grad_list,
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())

    @test_with_pir_api
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
            ]
            + grad_list,
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_quantile(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False
        out1 = paddle.quantile(x1, 0.5, axis=None)
        grad_list1 = paddle.static.append_backward(
            out1, parameter_list=[x1, out1]
        )
        grad_list1 = [_grad for _param, _grad in grad_list1]

        x2 = paddle.rand([2, 3])
        x2.stop_gradient = False
        out2 = paddle.quantile(x2, 0.5, axis=None)
        grad_list2 = paddle.static.append_backward(
            out2, parameter_list=[x2, out2]
        )
        grad_list2 = [_grad for _param, _grad in grad_list2]

        out_empty_list = paddle.quantile(x1, 0.5, axis=[])
        self.assertShapeEqual(out_empty_list, [])

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
            ]
            + grad_list1
            + grad_list2,
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

        self.assertEqual(res[4].shape, (2, 3))
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[5], 1.0)

    @test_with_pir_api
    @prog_scope()
    def test_nanquantile(self):
        # 1) x is 0D
        x1 = paddle.rand([])
        x1.stop_gradient = False
        out1 = paddle.nanquantile(x1, 0.5, axis=None)
        grad_list = paddle.static.append_backward(out1, parameter_list=[x1])
        x1_grad = grad_list[0][1]

        # 2) x is ND with 'nan'
        x2 = paddle.to_tensor([[float('nan'), 2.0, 3.0], [0.0, 1.0, 2.0]])
        x2.stop_gradient = False
        out2 = paddle.nanquantile(x2, 0.5, axis=None)
        print(out2)
        grad_list = paddle.static.append_backward(out2, parameter_list=[x2])
        x2_grad = grad_list[0][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                x1_grad,
                out2,
                x2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, (2, 3))

    @test_with_pir_api
    @prog_scope()
    def test_flip(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.flip(x, axis=[])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_equal_scalar(self):
        x = paddle.rand([])
        out = paddle.equal(x, 2.0)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], False)

    @test_with_pir_api
    @prog_scope()
    def test_pow_scalar(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.pow(x, 2.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_cast(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cast(x, 'int32')
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_cumprod(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cumprod(x, 0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

        with self.assertRaises(ValueError):
            tmp = paddle.cumprod(x, 2)

    @test_with_pir_api
    @prog_scope()
    def test_clip(self):
        x = paddle.uniform([], None, -10, 10)
        x.stop_gradient = False
        out = paddle.clip(x, -5, 5)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        x_grad, out_grad = (_grad for _param, _grad in grad_list)

        x1 = paddle.uniform([], None, -10, 10)
        x1.stop_gradient = False
        out1 = paddle.clip(x1, paddle.full([], -5.0), paddle.full([], 5.0))
        grad_list = paddle.static.append_backward(
            out1, parameter_list=[x1, out1]
        )
        x1_grad, out1_grad = (_grad for _param, _grad in grad_list)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out,
                x_grad,
                out_grad,
                x1,
                out1,
                x1_grad,
                out1_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[6].shape, ())
        self.assertEqual(res[7].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_increment(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.increment(x, 1.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])

        prog = paddle.static.default_main_program()
        if paddle.framework.in_pir_mode():
            grad_list = [_grad for _param, _grad in grad_list if _grad]
            res = self.exe.run(prog, fetch_list=[x, out] + grad_list)
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            if len(grad_list) > 0:
                self.assertEqual(res[2].shape, ())
            if len(grad_list) > 1:
                self.assertEqual(res[3].shape, ())
        else:
            res = self.exe.run(
                prog, fetch_list=[x, out, x.grad_name, out.grad_name]
            )
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, ())
            self.assertEqual(res[3].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_bitwise_not(self):
        # have no backward
        x = paddle.randint(-1, 1, [])
        out = paddle.bitwise_not(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_logical_not(self):
        # have no backward
        x = paddle.randint(0, 1, [])
        out = paddle.logical_not(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_searchsorted(self):
        # have no backward
        x = paddle.full([10], 1.0, 'float32')
        y = paddle.full([], 1.0, 'float32')
        out = paddle.searchsorted(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 0)

    @test_with_pir_api
    @prog_scope()
    def test_transpose(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.transpose(x, [])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)

        with self.assertRaises(ValueError):
            x = paddle.transpose(x, [0])

    @test_with_pir_api
    @prog_scope()
    def test_moveaxis(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.moveaxis(x, [], [])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)

        with self.assertRaises(AssertionError):
            x = paddle.moveaxis(x, [0], [1])

    @test_with_pir_api
    @prog_scope()
    def test_gather_1D(self):
        x = paddle.full([10], 1.0, 'float32')
        x.stop_gradient = False
        index = paddle.full([], 2, 'int64')
        out = paddle.gather(x, index)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_gather_XD_axis_0(self):
        x = paddle.full([2, 3], 1.0, 'float32')
        x.stop_gradient = False
        index = paddle.full([], 1, 'int64')
        out = paddle.gather(x, index)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, (3,))
        np.testing.assert_array_equal(res[0], [1.0, 1.0, 1.0])
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (3,))

    @test_with_pir_api
    @prog_scope()
    def test_gather_XD_axis_1(self):
        x = paddle.full([2, 3], 1.0, 'float32')
        x.stop_gradient = False
        index = paddle.full([], 1, 'int64')
        out = paddle.gather(x, index, axis=1)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, (2,))
        np.testing.assert_array_equal(res[0], [1.0, 1.0])
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (2,))

    @test_with_pir_api
    @prog_scope()
    def test_gather_nd(self):
        x1 = paddle.full([10], 1.0, 'float32')
        x1.stop_gradient = False
        x2 = paddle.full([2, 3], 1.0, 'float32')
        x2.stop_gradient = False

        index1 = paddle.full([1], 1, 'int64')
        index2 = paddle.full([2], 1, 'int64')

        out1 = paddle.gather_nd(x1, index1)
        out2 = paddle.gather_nd(x2, index2)
        grad_list1 = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        grad_list2 = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )

        (_, x1_grad), (_, out1_grad) = grad_list1
        (_, x2_grad), (_, out2_grad) = grad_list2

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
                out1_grad,
                out2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        np.testing.assert_array_equal(res[0], 1.0)
        np.testing.assert_array_equal(res[1], 1.0)
        self.assertEqual(res[2].shape, (10,))
        self.assertEqual(res[3].shape, (2, 3))
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[5].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_scatter_1D(self):
        x = paddle.full([10], 1.0, 'float32')
        x.stop_gradient = False
        index = paddle.full([], 2, 'int64')
        updates = paddle.full([], 4, 'float32')
        out = paddle.scatter(x, index, updates)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[0][2], 4.0)
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

    @test_with_pir_api
    @prog_scope()
    def test_scatter_XD(self):
        x = paddle.full([2, 3], 1.0, 'float32')
        x.stop_gradient = False
        index = paddle.full([], 1, 'int64')
        updates = paddle.full([3], 4, 'float32')
        out = paddle.scatter(x, index, updates)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, (2, 3))
        np.testing.assert_array_equal(res[0][1], [4.0, 4.0, 4.0])
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (2, 3))

    @test_with_pir_api
    @prog_scope()
    def test_diagflat(self):
        # have no backward
        x1 = paddle.rand([])
        out1 = paddle.diagflat(x1, 1)

        x2 = paddle.rand([])
        out2 = paddle.diagflat(x2, -1)

        x3 = paddle.rand([])
        out3 = paddle.diagflat(x3)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, out2, out3])
        self.assertEqual(res[0].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 2))
        self.assertEqual(res[2].shape, (1, 1))

    @test_with_pir_api
    @prog_scope()
    def test_scatter__1D(self):
        x = paddle.full([10], 1.0, 'float32')
        index = paddle.full([], 2, 'int64')
        updates = paddle.full([], 4, 'float32')
        out = paddle.scatter_(x, index, updates)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0][2], 4)

    @test_with_pir_api
    @prog_scope()
    def test_scatter__XD(self):
        x = paddle.full([2, 3], 1.0, 'float32')
        index = paddle.full([], 1, 'int64')
        updates = paddle.full([3], 4, 'float32')
        out = paddle.scatter_(x, index, updates)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        np.testing.assert_array_equal(res[0][1], [4.0, 4.0, 4.0])

    @test_with_pir_api
    @prog_scope()
    def test_scatter_nd(self):
        index = paddle.full([1], 3, dtype='int64')
        updates = paddle.full([], 2, 'float32')
        updates.stop_gradient = False
        out = paddle.scatter_nd(index, updates, [5])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[out, updates]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, (5,))
        self.assertEqual(res[0][3], 2)
        self.assertEqual(res[1].shape, (5,))
        self.assertEqual(res[2].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_flatten(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False

        start_axis = 0
        stop_axis = -1

        out = paddle.flatten(x, start_axis=start_axis, stop_axis=stop_axis)
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[x, out]
        )
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out] + grad_list)

        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (1,))

    @test_with_pir_api
    @prog_scope()
    def test_histogram(self):
        x = paddle.full([], 1, 'float32')
        out = paddle.histogram(x, bins=5, min=1, max=5)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out])

        self.assertEqual(res[0].shape, (5,))

    @test_with_pir_api
    @prog_scope()
    def test_scale(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.scale(x, scale=2.0, bias=1.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out] + grad_list)
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_floor_divide(self):
        # 1-d // 0-d
        x = paddle.to_tensor([1, -2, 3], dtype="int64")
        y = paddle.full([], 2, dtype='int64')
        out1_1 = paddle.floor_divide(x, y)
        out1_2 = x // y

        # 0-d // 1-d
        out2_1 = paddle.floor_divide(y, x)
        out2_2 = y // x

        # 0-d // 0-d
        x = paddle.full([], 3, dtype='int64')
        out3_1 = paddle.floor_divide(x, y)
        out3_2 = x // y

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[out1_1, out1_2, out2_1, out2_2, out3_1, out3_2]
        )
        out1_1, out1_2, out2_1, out2_2, out3_1, out3_2 = res

        np.testing.assert_array_equal(out1_1, out1_2)
        np.testing.assert_array_equal(out1_1, np.asarray([0, -1, 1]))
        np.testing.assert_array_equal(out2_1, out2_2)
        np.testing.assert_array_equal(out2_2, np.asarray([2, -1, 0]))
        np.testing.assert_array_equal(out3_1, out3_2)
        np.testing.assert_array_equal(out3_2, np.asarray(1))

    @test_with_pir_api
    @prog_scope()
    def test_cumsum(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False

        out1 = paddle.cumsum(x1)
        out2 = paddle.cumsum(x1, axis=0)
        out3 = paddle.cumsum(x1, axis=-1)

        (_, x1_grad), (_, out1_grad) = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out2_grad) = paddle.static.append_backward(
            out2.sum(), parameter_list=[x1, out2]
        )
        (_, x1_grad), (_, out3_grad) = paddle.static.append_backward(
            out3.sum(), parameter_list=[x1, out3]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                x1_grad,
                out1_grad,
                out2_grad,
                out3_grad,
            ],
        )
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)
        self.assertEqual(res[4].shape, (1,))
        self.assertEqual(res[4], 1.0)
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[5], 1.0)
        self.assertEqual(res[6].shape, ())
        self.assertEqual(res[6], 1.0)
        self.assertShapeEqual(out2, [])
        self.assertShapeEqual(out3, [])

    @test_with_pir_api
    @prog_scope()
    def test_logcumsumexp(self):
        x = paddle.rand([])
        x.stop_gradient = False

        out1 = paddle.logcumsumexp(x)
        out2 = paddle.logcumsumexp(x, axis=0)
        out3 = paddle.logcumsumexp(x, axis=-1)

        grad_list1 = paddle.static.append_backward(out1, parameter_list=[x])
        grad_list2 = paddle.static.append_backward(out2, parameter_list=[x])
        grad_list3 = paddle.static.append_backward(out3, parameter_list=[x])

        x_grad = grad_list3[0][1]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                x_grad,
            ],
        )
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

    @test_with_pir_api
    @prog_scope()
    def test_add_n(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False
        x2 = paddle.rand([])
        x2.stop_gradient = False
        x3 = paddle.rand([])
        x3.stop_gradient = False

        out1 = paddle.add_n(x1)
        out2 = paddle.add_n([x2, x3])

        grad_list1 = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        grad_list23 = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, x3, out2]
        )

        (_, x1_grad), (_, out1_grad) = grad_list1
        (_, x2_grad), (_, x3_grad), (_, out2_grad) = grad_list23

        prog = paddle.static.default_main_program()
        block = prog.global_block()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
                x3_grad,
                out1_grad,
                out2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[4], 1)
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[6].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_reshape_list(self):
        x1 = paddle.rand([])
        x2 = paddle.rand([])
        x3 = paddle.rand([])
        x4 = paddle.rand([])
        x1.stop_gradient = False
        x2.stop_gradient = False
        x3.stop_gradient = False
        x4.stop_gradient = False

        out1 = paddle.reshape(x1, [])
        grad_list1 = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out1_grad) = grad_list1

        out2 = paddle.reshape(x2, [1])
        grad_list2 = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )
        (_, x2_grad), (_, out2_grad) = grad_list2

        out3 = paddle.reshape(x3, [-1])
        grad_list3 = paddle.static.append_backward(
            out3.sum(), parameter_list=[x3, out3]
        )
        (_, x3_grad), (_, out3_grad) = grad_list3

        out4 = paddle.reshape(x4, [-1, 1])
        grad_list4 = paddle.static.append_backward(
            out4.sum(), parameter_list=[x4, out4]
        )
        (_, x4_grad), (_, out4_grad) = grad_list4

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                out4,
                x1_grad,
                x2_grad,
                x3_grad,
                x4_grad,
                out1_grad,
                out2_grad,
                out3_grad,
                out4_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[2].shape, (1,))
        self.assertEqual(res[3].shape, (1, 1))

        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[6].shape, ())
        self.assertEqual(res[7].shape, ())

        self.assertEqual(res[8].shape, ())
        self.assertEqual(res[9].shape, (1,))
        self.assertEqual(res[10].shape, (1,))
        self.assertEqual(res[11].shape, (1, 1))

    @test_with_pir_api
    @prog_scope()
    def test_reshape_tensor(self):
        x1 = paddle.rand([1, 1])
        x1.stop_gradient = False
        new_shape = paddle.full([3], 1, "int32")
        out1 = paddle.reshape(x1, new_shape)
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out1_grad) = grad_list

        x2 = paddle.rand([1, 1])
        x2.stop_gradient = False
        new_shape = paddle.full([1], -1, "int32")
        out2 = paddle.reshape(x2, new_shape)
        grad_list = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )
        (_, x2_grad), (_, out2_grad) = grad_list

        x3 = paddle.rand([1, 1])
        x3.stop_gradient = False
        new_shape = [paddle.full([], -1, "int32"), paddle.full([], 1, "int32")]
        out3 = paddle.reshape(x3, new_shape)
        grad_list = paddle.static.append_backward(
            out3.sum(), parameter_list=[x3, out3]
        )
        (_, x3_grad), (_, out3_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                x1_grad,
                x2_grad,
                x3_grad,
                out1_grad,
                out2_grad,
                out3_grad,
            ],
        )
        self.assertEqual(res[0].shape, (1, 1, 1))
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[2].shape, (1, 1))

        self.assertEqual(res[3].shape, (1, 1))
        self.assertEqual(res[4].shape, (1, 1))
        self.assertEqual(res[5].shape, (1, 1))

        self.assertEqual(res[6].shape, (1, 1, 1))
        self.assertEqual(res[7].shape, (1,))
        self.assertEqual(res[8].shape, (1, 1))

    @test_with_pir_api
    @prog_scope()
    def test_reverse(self):
        x = paddle.rand([])
        x.stop_gradient = False

        out = paddle.reverse(x, axis=[])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        (_, x_grad), (out_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, x_grad, out_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_sort(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False
        out1 = paddle.sort(x1, axis=-1)
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out1_grad) = grad_list

        x2 = paddle.rand([])
        x2.stop_gradient = False
        out2 = paddle.sort(x2, axis=0)
        grad_list = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )
        (_, x2_grad), (_, out2_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out1_grad,
                out2_grad,
                x1_grad,
                x2_grad,
            ],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[4], 1.0)
        self.assertEqual(res[5], 1.0)

    @test_with_pir_api
    @prog_scope()
    def test_argsort(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # have no backward
            x1 = paddle.rand([])
            out1 = paddle.argsort(x1, axis=-1)

            x2 = paddle.rand([])
            x2.stop_gradient = False
            out2 = paddle.argsort(x2, axis=0)

            prog = paddle.static.default_main_program()
            res = self.exe.run(prog, fetch_list=[out1, out2])

            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[0], 0.0)
            self.assertEqual(res[1], 0.0)

    @test_with_pir_api
    @prog_scope()
    def test_lerp(self):
        shapes = [
            [(), (), (), ()],
            [(), (64, 64), (), (64, 64)],
            [(64, 64), (), (), (64, 64)],
            [(64, 64), (), 0.5, (64, 64)],
        ]
        for shape in shapes:
            x = paddle.rand(shape[0])
            y = paddle.rand(shape[1])
            if isinstance(shape[2], float):
                w = shape[2]
            else:
                w = paddle.rand(shape[2])

            x.stop_gradient = False
            y.stop_gradient = False
            out = paddle.lerp(x, y, w)
            grad_list = paddle.static.append_backward(
                out.sum(), parameter_list=[out, y, x]
            )
            (_, out_grad), (_, y_grad), (_, x_grad) = grad_list

            prog = paddle.static.default_main_program()
            res = self.exe.run(prog, fetch_list=[out, out_grad, y_grad, x_grad])
            self.assertEqual(res[0].shape, shape[3])
            self.assertEqual(res[1].shape, shape[3])
            self.assertEqual(res[2].shape, shape[1])
            self.assertEqual(res[3].shape, shape[0])

    @test_with_pir_api
    @prog_scope()
    def test_repeat_interleave(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.repeat_interleave(x1, 2, None)
        grad_list1 = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out1_grad) = grad_list1

        x2 = paddle.full([], 1.0, 'float32')
        x2.stop_gradient = False
        repeats = paddle.to_tensor([3], dtype='int32')
        out2 = paddle.repeat_interleave(x2, repeats, None)
        grad_list2 = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2, out2]
        )
        (_, x2_grad), (_, out2_grad) = grad_list2

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
                out1_grad,
                out2_grad,
            ],
        )
        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (3,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, (2,))
        self.assertEqual(res[5].shape, (3,))

    @test_with_pir_api
    @prog_scope()
    def test_allclose(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        out = paddle.allclose(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

        # 2) x is ND
        x = paddle.full([2, 3], 0.5)
        y = paddle.full([2, 3], 0.6)
        out = paddle.allclose(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

    @test_with_pir_api
    @prog_scope()
    def test_equal_all(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        out = paddle.equal_all(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

        # 2) x is ND
        x = paddle.full([2, 3], 0.5)
        y = paddle.full([2, 3], 0.6)
        out = paddle.equal_all(x, y)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertFalse(res[0])

    @test_with_pir_api
    @prog_scope()
    def test_where(self):
        x1 = paddle.full([], 1, 'float32')
        x2 = paddle.full([], 2, 'float32')
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.where(x1 > x2, x1, x2)
        loss = paddle.mean(out)
        grad_list = paddle.static.append_backward(
            loss, parameter_list=[out, x1, x2]
        )
        (_, out_grad), (_, x1_grad), (_, x2_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={},
            fetch_list=[out, out_grad, x1_grad, x2_grad],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)

    @test_with_pir_api
    @prog_scope()
    def test_atan2(self):
        x1 = paddle.full([], 0, 'float32')
        x2 = paddle.full([], 2, 'float32')
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.atan2(x1, x2)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out])

        self.assertEqual(res[0].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_interpolate(self):
        from paddle.nn.functional import interpolate

        input_x = paddle.rand([2, 3, 6, 6])
        input_x.stop_gradient = False

        output_size = [
            paddle.full([], 12, dtype="int32"),
            paddle.full([], 12, dtype="int32"),
        ]

        out1 = interpolate(
            x=input_x, size=output_size, mode="bilinear", align_corners=False
        )
        _, input_x_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[input_x]
        )[0]
        prog = paddle.static.default_main_program()
        res1 = self.exe.run(prog, feed={}, fetch_list=[out1, input_x_grad])

        scale_1 = paddle.full([], 2)
        out2 = interpolate(
            x=input_x,
            scale_factor=scale_1,
            mode="bilinear",
            align_corners=False,
        )
        _, input_x_grad = paddle.static.append_backward(
            out2.sum(), parameter_list=[input_x]
        )[0]
        prog = paddle.static.default_main_program()
        res2 = self.exe.run(prog, feed={}, fetch_list=[out2, input_x_grad])

        self.assertEqual(res1[0].shape, (2, 3, 12, 12))
        self.assertEqual(res1[1].shape, (2, 3, 6, 6))
        self.assertEqual(res2[0].shape, (2, 3, 12, 12))
        self.assertEqual(res2[1].shape, (2, 3, 6, 6))

    @test_with_pir_api
    @prog_scope()
    def test_upsample(self):
        from paddle.nn.functional import upsample

        input_x = paddle.rand([2, 3, 6, 6])
        input_x.stop_gradient = False

        output_size = [
            paddle.full([], 12, dtype="int32"),
            paddle.full([], 12, dtype="int32"),
        ]

        out1 = upsample(
            x=input_x, size=output_size, mode="bilinear", align_corners=False
        )
        _, input_x_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[input_x]
        )[0]
        prog = paddle.static.default_main_program()
        res1 = self.exe.run(prog, feed={}, fetch_list=[out1, input_x_grad])

        self.assertEqual(res1[0].shape, (2, 3, 12, 12))
        self.assertEqual(res1[1].shape, (2, 3, 6, 6))

    @test_with_pir_api
    @prog_scope()
    def test_unstack(self):
        x1 = paddle.full([1], 0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.unstack(x1, 0)
        out1 = paddle.add_n(out1)
        _, x1_grad = paddle.static.append_backward(out1, parameter_list=[x1])[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out1, x1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))

        x2 = paddle.full([2], 2, 'float32')
        x2.stop_gradient = False
        out2 = paddle.unstack(x2, 0)
        out2_sum = paddle.add_n(out2)
        _, x2_grad = paddle.static.append_backward(
            out2_sum, parameter_list=[x2]
        )[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out2_sum, x2_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2,))

    @test_with_pir_api
    @prog_scope()
    def test_unbind(self):
        x1 = paddle.full([1], 0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.unbind(x1, 0)
        out1 = paddle.add_n(out1)
        _, x1_grad = paddle.static.append_backward(out1, parameter_list=[x1])[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out1, x1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))

        x2 = paddle.full([2], 2, 'float32')
        x2.stop_gradient = False
        out2 = paddle.unbind(x2, 0)
        out2_sum = paddle.add_n(out2)
        _, x2_grad = paddle.static.append_backward(
            out2_sum, parameter_list=[x2]
        )[0]
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out2_sum, x2_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2,))

    @test_with_pir_api
    @prog_scope()
    def test_masked_select(self):
        x = paddle.rand([])
        x.stop_gradient = False
        mask = paddle.full([], True, dtype='bool')
        y = paddle.masked_select(x, mask)
        grad_list = paddle.static.append_backward(
            y.sum(), parameter_list=[y, x]
        )
        (_, y_grad), (_, x_grad) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, y, y_grad, x_grad])
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[1], res[0])
        self.assertEqual(res[2].shape, (1,))
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)

    @test_with_pir_api
    @prog_scope()
    def test_squeeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        out1 = paddle.squeeze(x1, axis=0)
        _, x1_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1]
        )[0]

        x2 = paddle.full([], 3)
        x3 = paddle.full([], 0, dtype='int32')
        x2.stop_gradient = False
        out2 = paddle.squeeze(x2, axis=x3)
        _, x2_grad = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @test_with_pir_api
    @prog_scope()
    def test_unsqueeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        out1 = paddle.unsqueeze(x1, axis=0)
        _, x1_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1]
        )[0]

        x2 = paddle.full([], 3)
        x3 = paddle.full([], 0, dtype='int32')
        x2.stop_gradient = False
        out2 = paddle.unsqueeze(x2, axis=x3)
        _, x2_grad = paddle.static.append_backward(
            out2.sum(), parameter_list=[x2]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
            ],
        )
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_t(self):
        x = paddle.full([], 2.0)
        x.stop_gradient = False
        out = paddle.t(x)
        grad_list = paddle.static.append_backward(out, parameter_list=[out, x])

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, feed={}, fetch_list=[out, out.grad_name, x.grad_name]
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

    @prog_scope()
    def test_sequence_pad(self):
        x = paddle.static.data("x", [-1, 2], dtype=paddle.int64, lod_level=1)
        value = paddle.to_tensor(1000, dtype=paddle.int64).squeeze()
        out = paddle.static.nn.sequence_pad(x, value)

        x_tensor = paddle.base.create_lod_tensor(
            np.arange(20).astype(np.int64).reshape(-1, 2),
            [[3, 3, 4]],
            place=self.exe.place,
        )
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={"x": x_tensor}, fetch_list=[out])
        self.assertEqual(res[0].shape, (3, 4, 2))

    @prog_scope()
    def test_static_data(self):
        x1 = paddle.static.data(name="x1", shape=[])
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={
                "x1": np.array(1.0, dtype='float32'),
            },
            fetch_list=[
                x1.name,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], np.array(1.0))

        x2 = paddle.static.data(name="x2", shape=[])
        x3 = paddle.static.data(name="x3", shape=[])
        y = x2 + x3
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={
                "x2": 100.5,
                "x3": 200.5,
            },
            fetch_list=[
                y.name,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 301.0)

    @test_with_pir_api
    @prog_scope()
    def test_prelu(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        w1 = paddle.to_tensor([0.25], dtype='float32')
        out1 = paddle.nn.functional.prelu(x1, w1)
        (_, out1_grad), (_, x1_grad) = paddle.static.append_backward(
            out1.sum(), parameter_list=[out1, x1]
        )

        x2 = paddle.full([], 1.0, 'float32')
        x2.stop_gradient = False
        w2 = paddle.full([], 0.25, dtype='float32')
        out2 = paddle.nn.functional.prelu(x2, w2)
        (_, out2_grad), (_, x2_grad) = paddle.static.append_backward(
            out2.sum(), parameter_list=[out2, x2]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1_grad,
                x2_grad,
                out1_grad,
                out2_grad,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())
        self.assertEqual(res[5].shape, ())

    @prog_scope()
    def test_static_nn_prelu(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.static.nn.prelu(x1, 'all')
        grad_list = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1, out1]
        )
        (_, x1_grad), (_, out1_grad) = grad_list

        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                x1_grad,
                out1_grad,
            ],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[0], np.array(1))
        np.testing.assert_allclose(res[1], np.array(1))

    @test_with_pir_api
    @prog_scope()
    def test_while_loop(self):
        def cond(i, x):
            return paddle.less_than(i, eleven)

        def body(i, x):
            x = x + i
            i = i + 1
            return [i, x]

        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, paddle.static.Program()):
            i = paddle.static.data(name='i', shape=[], dtype='float32')
            i.stop_gradient = False
            i.persistable = True
            eleven = paddle.full([], 11, 'float32')
            x = paddle.static.data(name='x', shape=[], dtype='float32')
            x.stop_gradient = False
            x.persistable = True
            out_i, out_x = paddle.static.nn.while_loop(cond, body, [i, x])
            grad_list = paddle.static.append_backward(out_x)

        feed = {
            'i': np.array(1.0, dtype='float32'),
            'x': np.array(0.0, dtype='float32'),
        }
        if paddle.framework.in_pir_mode():
            fetch_list = [out_i, out_x]
            for _, g in grad_list:
                fetch_list.append(g)
            res = self.exe.run(
                main_program,
                feed=feed,
                fetch_list=fetch_list,
            )
        else:
            res = self.exe.run(
                main_program,
                feed=feed,
                fetch_list=[out_i.name, out_x.name, i.grad_name, x.grad_name],
            )

        self.assertEqual(res[0].shape, ())
        np.testing.assert_allclose(res[0], np.array(11))
        self.assertEqual(res[1].shape, ())
        np.testing.assert_allclose(res[1], np.array(55))
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[2], np.array(10))
        self.assertEqual(res[3].shape, ())
        np.testing.assert_allclose(res[3], np.array(1.0))

    @test_with_pir_api
    @prog_scope()
    def test_numel(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        out = paddle.numel(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(1))

        # 2) x is ND
        x = paddle.full([3, 5], 0.5)
        out = paddle.numel(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(15))

    @test_with_pir_api
    @prog_scope()
    def test_rank(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        out = paddle.rank(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(0))

        # 1) x is ND
        x = paddle.full([3, 5], 0.5)
        out = paddle.rank(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        np.testing.assert_array_equal(res[0], np.array(2))

    @test_with_pir_api
    @prog_scope()
    def test_shape(self):
        x = paddle.full([], 0.5)
        out = paddle.shape(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        np.testing.assert_array_equal(res[0], np.array([]))
        self.assertEqual(res[0].shape, (0,))

    @test_with_pir_api
    def test_broadcast_tensors(self):
        # 1) x is 0D, y is 0D
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertShapeEqual(out1, [])
        self.assertShapeEqual(out2, [])

        # 2) x is ND , y is 0D
        x1 = paddle.full([2, 3], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertShapeEqual(out1, [2, 3])
        self.assertShapeEqual(out2, [2, 3])

        # 3) x is 0D , y is ND
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([2, 3], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertShapeEqual(out1, [2, 3])
        self.assertShapeEqual(out2, [2, 3])

    @test_with_pir_api
    @prog_scope()
    def test_to_tensor(self):
        out1 = paddle.to_tensor(1)
        out2 = paddle.to_tensor(2.5)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, out2])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 2.5)

    @test_with_pir_api
    @prog_scope()
    def test_matmul(self):
        # 1) no transpose
        x = paddle.randn([10])
        x.stop_gradient = False
        y = paddle.randn([10])
        y.stop_gradient = False
        out = paddle.matmul(x, y)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, y])
        (_, x_grad), (_, y_grad) = grad_list

        self.assertShapeEqual(out, [])

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

        # 2) transpose x and y
        x = paddle.randn([10])
        x.stop_gradient = False
        y = paddle.randn([10])
        y.stop_gradient = False
        out = paddle.matmul(x, y, True, True)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, y])
        (_, x_grad), (_, y_grad) = grad_list

        self.assertShapeEqual(out, [])

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

    @test_with_pir_api
    @prog_scope()
    def test_linalg_slogdet(self):
        # 2-D input
        x = paddle.randn([3, 3])
        x.stop_gradient = False
        out = paddle.linalg.slogdet(x)
        _, x_grad = paddle.static.append_backward(
            out.sum(), parameter_list=[x]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (3, 3))

        # 3-D input
        x1 = paddle.randn([3, 3, 3])
        x1.stop_gradient = False
        out1 = paddle.linalg.slogdet(x1)
        _, x1_grad = paddle.static.append_backward(
            out1.sum(), parameter_list=[x1]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, x1_grad])
        self.assertEqual(res[0].shape, (2, 3))
        self.assertEqual(res[1].shape, (3, 3, 3))

    @test_with_pir_api
    @prog_scope()
    def test_multi_dot(self):
        a = paddle.randn([4])
        a.stop_gradient = False
        b = paddle.randn([4, 5])
        b.stop_gradient = False
        c = paddle.randn([5])
        c.stop_gradient = False

        out = paddle.linalg.multi_dot([a, b, c])
        grad_list = paddle.static.append_backward(
            out.sum(), parameter_list=[a, b, c]
        )
        (_, a_grad), (_, b_grad), (_, c_grad) = grad_list
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, a_grad, b_grad, c_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4,))
        self.assertEqual(res[2].shape, (4, 5))
        self.assertEqual(res[3].shape, (5,))

    @test_with_pir_api
    @prog_scope()
    def test_cov(self):
        xt_1 = paddle.randn((12,))
        xt_1.stop_gradient = False
        out = paddle.linalg.cov(xt_1)
        _, xt_1_grad = paddle.static.append_backward(
            out, parameter_list=[xt_1]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, xt_1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (12,))

    @test_with_pir_api
    @prog_scope()
    def test_corrcoef(self):
        x = paddle.randn((12,))
        x.stop_gradient = False
        out = paddle.linalg.corrcoef(x)
        _, x_grad = paddle.static.append_backward(out, parameter_list=[x])[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (12,))

    @test_with_pir_api
    @prog_scope()
    def test_det(self):
        xt_1 = paddle.randn((3, 3))
        xt_1.stop_gradient = False

        out = paddle.linalg.det(xt_1)
        _, xt_1_grad = paddle.static.append_backward(
            out.sum(), parameter_list=[xt_1]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, xt_1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

    @prog_scope()
    def test_dist(self):
        x = paddle.to_tensor([[3, 3], [3, 3]], dtype="float32")
        y = paddle.to_tensor([[3, 3], [3, 1]], dtype="float32")
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.dist(x, y)
        (_, x_grad), (_, y_grad) = paddle.static.append_backward(
            out, parameter_list=[x, y]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad, y_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 2))
        np.testing.assert_array_equal(res[0], np.array(2).astype(np.float32))

    @prog_scope()
    def test_linalg_norm(self):
        # 1D input, p = fro ,axis = None, using reduceInferMeta
        x_1 = paddle.arange(24, dtype="float32") - 12
        x_1.stop_gradient = False
        out_1 = paddle.linalg.norm(x_1)
        grad_list = paddle.static.append_backward(out_1, parameter_list=[x_1])
        ((_, x_1_grad),) = grad_list

        prog = paddle.static.default_main_program()

        res = self.exe.run(prog, fetch_list=[out_1, x_1_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

        # 1D input, p = 1 ,axis = None,
        # using p_norm, as_vector = True
        x_2 = paddle.arange(24, dtype="float32") - 12
        x_2.stop_gradient = False
        out_2 = paddle.linalg.norm(x_2, p=1)
        paddle.static.append_backward(out_2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2, x_2.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

        # 1D input, p = 1 ,axis = 0,
        # using p_norm, as_vector = False
        x_2_p = paddle.arange(24, dtype="float32") - 12
        x_2_p.stop_gradient = False
        out_2_p = paddle.linalg.norm(x_2_p, p=1, axis=0)
        paddle.static.append_backward(out_2_p.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2_p, x_2_p.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

        # 1D input, p = fro ,axis = 0,
        # using p_norm, as_vector = False
        x_2_fro = paddle.arange(24, dtype="float32") - 12
        x_2_fro.stop_gradient = False
        out_2_fro = paddle.linalg.norm(x_2_fro, p="fro", axis=0)
        paddle.static.append_backward(out_2_fro.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2_fro, x_2_fro.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (24,))

        # 2D input, p = 1, axis = [0, 1]
        # using p_matrix_norm, depends on paddle.sum
        x_3 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_3.stop_gradient = False
        out_3 = paddle.linalg.norm(x_3, p=1, axis=[0, 1])
        paddle.static.append_backward(out_3.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_3, x_3.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

        # 2D input, p = 1, axis = None
        # using p_matrix_norm, depends on paddle.sum
        x_4 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_4.stop_gradient = False
        out_4 = paddle.linalg.norm(x_4)
        paddle.static.append_backward(out_4.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_4, x_4.grad_name])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

        # 2D input, p = inf, axis = None
        x_5 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_5.stop_gradient = False
        out_5 = paddle.linalg.norm(x_5)
        paddle.static.append_backward(out_5.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_5, x_5.grad_name])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

        # 2D input, p = -inf, axis = [0, 1]
        x_6 = paddle.arange(24, dtype="float32").reshape([4, 6])
        x_6.stop_gradient = False
        out_6 = paddle.linalg.norm(x_6, p=-float("inf"), axis=[0, 1])
        paddle.static.append_backward(out_6.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_6, x_6.grad_name])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4, 6))

    @test_with_pir_api
    @prog_scope()
    def test_linalg_cond(self):
        # use paddle.sum
        x = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x.stop_gradient = False
        out = paddle.linalg.cond(x)
        _, x_grad = paddle.static.append_backward(out, parameter_list=[x])[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p = fro : use paddle.sum
        x2 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x2.stop_gradient = False
        out_fro = paddle.linalg.cond(x2, p='fro')
        grad_list = paddle.static.append_backward(out_fro, parameter_list=[x2])
        ((_, x2_grad),) = grad_list

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_fro, x2_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p = nuc : use paddle.sum
        x3 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x3.stop_gradient = False
        out_nuc = paddle.linalg.cond(x3, p='nuc')
        _, x3_grad = paddle.static.append_backward(
            out_nuc, parameter_list=[x3]
        )[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_nuc, x3_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p in (-1, 1) : use paddle.sum
        x4 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x4.stop_gradient = False
        out_1 = paddle.linalg.cond(x4, p=1)
        _, x4_grad = paddle.static.append_backward(out_1, parameter_list=[x4])[
            0
        ]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_1, x4_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        x5 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x5.stop_gradient = False
        out_minus_1 = paddle.linalg.cond(x5, p=-1)
        ((_, x5_grad),) = paddle.static.append_backward(
            out_minus_1, parameter_list=[x5]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_minus_1, x5_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p in (-2, 2) depends on paddle.sum
        x6 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x6.stop_gradient = False
        out_2 = paddle.linalg.cond(x6, p=2)
        ((_, x6_grad),) = paddle.static.append_backward(
            out_2, parameter_list=[x6]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_2, x6_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # p in (-inf, inf):use paddle.sum
        x8 = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        x8.stop_gradient = False
        out_inf = paddle.linalg.cond(x8, p=float("inf"))
        ((_, x8_grad),) = paddle.static.append_backward(
            out_inf, parameter_list=[x8]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_inf, x8_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))

        # depends on paddle.sum
        a = paddle.randn([2, 4, 4])
        a.stop_gradient = False
        a_cond_fro = paddle.linalg.cond(a, p='fro')
        ((_, a_grad),) = paddle.static.append_backward(
            a_cond_fro.sum(), parameter_list=[a]
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[a_cond_fro, a_grad])

        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (2, 4, 4))

    @prog_scope()
    def test_trace(self):
        x = paddle.to_tensor([[3, 2], [1, 9]], dtype="float32")
        x.stop_gradient = False
        out = paddle.trace(x)
        _, x_grad = paddle.static.append_backward(out, parameter_list=[x])[0]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x_grad])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 2))
        np.testing.assert_allclose(res[0], np.array(12))


if __name__ == "__main__":
    unittest.main()
