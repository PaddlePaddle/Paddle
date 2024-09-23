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
                *grad_list1,
                *grad_list2,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1.0)

        self.assertEqual(res[4].shape, (2, 3))
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[5], 1.0)

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

    @prog_scope()
    def test_flip(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.flip(x, axis=[])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_equal_scalar(self):
        x = paddle.rand([])
        out = paddle.equal(x, 2.0)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], False)

    @prog_scope()
    def test_pow_scalar(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.pow(x, 2.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_cast(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cast(x, 'int32')
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_cumprod(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cumprod(x, 0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

        with self.assertRaises(ValueError):
            tmp = paddle.cumprod(x, 2)

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

    @prog_scope()
    def test_increment(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.increment(x, 1.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        prog = paddle.static.default_main_program()
        if paddle.framework.in_pir_mode():
            grad_list = [
                _grad for _param, _grad in grad_list if _grad is not None
            ]
            res = self.exe.run(prog, fetch_list=[x, out, *grad_list])
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

    @prog_scope()
    def test_bitwise_not(self):
        # have no backward
        x = paddle.randint(-1, 1, [])
        out = paddle.bitwise_not(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

    @prog_scope()
    def test_logical_not(self):
        # have no backward
        x = paddle.randint(0, 1, [])
        out = paddle.logical_not(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())

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

    @prog_scope()
    def test_transpose(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.transpose(x, [])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)

        with self.assertRaises(ValueError):
            x = paddle.transpose(x, [0])

    @prog_scope()
    def test_moveaxis(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.moveaxis(x, [], [])
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], 1.0)
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 1.0)

        with self.assertRaises(AssertionError):
            x = paddle.moveaxis(x, [0], [1])

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
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, ())

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
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, (3,))
        np.testing.assert_array_equal(res[0], [1.0, 1.0, 1.0])
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (3,))

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
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, (2,))
        np.testing.assert_array_equal(res[0], [1.0, 1.0])
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (2,))

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
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[0][2], 4.0)
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, (10,))

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
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, (2, 3))
        np.testing.assert_array_equal(res[0][1], [4.0, 4.0, 4.0])
        self.assertEqual(res[1].shape, (2, 3))
        self.assertEqual(res[2].shape, (2, 3))

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

    @prog_scope()
    def test_scatter__1D(self):
        x = paddle.full([10], 1.0, 'float32')
        index = paddle.full([], 2, 'int64')
        updates = paddle.full([], 4, 'float32')
        out = paddle.scatter_(x, index, updates)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0][2], 4)

    @prog_scope()
    def test_scatter__XD(self):
        x = paddle.full([2, 3], 1.0, 'float32')
        index = paddle.full([], 1, 'int64')
        updates = paddle.full([3], 4, 'float32')
        out = paddle.scatter_(x, index, updates)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        np.testing.assert_array_equal(res[0][1], [4.0, 4.0, 4.0])

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
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, (5,))
        self.assertEqual(res[0][3], 2)
        self.assertEqual(res[1].shape, (5,))
        self.assertEqual(res[2].shape, ())

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
        res = self.exe.run(prog, feed={}, fetch_list=[out, *grad_list])

        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (1,))

    @prog_scope()
    def test_histogram(self):
        x = paddle.full([], 1, 'float32')
        out = paddle.histogram(x, bins=5, min=1, max=5)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out])

        self.assertEqual(res[0].shape, (5,))

    @prog_scope()
    def test_scale(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.scale(x, scale=2.0, bias=1.0)
        grad_list = paddle.static.append_backward(out, parameter_list=[x, out])
        grad_list = [_grad for _param, _grad in grad_list]

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, *grad_list])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())

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


if __name__ == "__main__":
    unittest.main()
