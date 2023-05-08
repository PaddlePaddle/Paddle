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

import os
import unittest

import numpy as np
from decorator_helper import prog_scope

import paddle
import paddle.nn.functional as F

unary_api_list = [
    paddle.nn.functional.elu,
    paddle.nn.functional.gelu,
    paddle.nn.functional.hardsigmoid,
    paddle.nn.functional.hardswish,
    paddle.nn.functional.hardshrink,
    paddle.nn.functional.hardtanh,
    paddle.nn.functional.leaky_relu,
    paddle.nn.functional.log_sigmoid,
    paddle.nn.functional.relu,
    paddle.nn.functional.relu6,
    paddle.nn.functional.sigmoid,
    paddle.nn.functional.softplus,
    paddle.nn.functional.softshrink,
    paddle.nn.functional.softsign,
    paddle.nn.functional.swish,
    paddle.nn.functional.tanhshrink,
    paddle.nn.functional.thresholded_relu,
    paddle.stanh,
    paddle.nn.functional.celu,
    paddle.nn.functional.selu,
    paddle.nn.functional.mish,
    paddle.nn.functional.silu,
    paddle.nn.functional.tanh,
    paddle.nn.functional.dropout,
    paddle.cosh,
    paddle.sinh,
    paddle.abs,
    paddle.acos,
    paddle.asin,
    paddle.atan,
    paddle.ceil,
    paddle.cos,
    paddle.exp,
    paddle.floor,
    paddle.log,
    paddle.log1p,
    paddle.reciprocal,
    paddle.round,
    paddle.sin,
    paddle.sqrt,
    paddle.square,
    paddle.tanh,
    paddle.acosh,
    paddle.asinh,
    paddle.atanh,
    paddle.expm1,
    paddle.log10,
    paddle.log2,
    paddle.tan,
    paddle.erf,
    paddle.erfinv,
    paddle.rsqrt,
    paddle.sign,
    paddle.deg2rad,
    paddle.rad2deg,
    paddle.neg,
    paddle.logit,
    paddle.trunc,
    paddle.digamma,
    paddle.lgamma,
    paddle.poisson,
    paddle.bernoulli,
    paddle.nn.functional.softmax,
    paddle.nn.functional.log_softmax,
    paddle.nn.functional.gumbel_softmax,
    paddle.nn.functional.alpha_dropout,
]

inplace_api_list = [
    paddle.nn.functional.relu_,
    paddle.nn.functional.tanh_,
]


# Use to test zero-dim in unary API.
class TestUnaryAPI(unittest.TestCase):
    def test_dygraph_unary(self):
        paddle.disable_static()
        for api in unary_api_list:
            x = paddle.rand([])
            x.stop_gradient = False
            out = api(x)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

        for api in inplace_api_list:
            x = paddle.rand([])
            out = api(x)
            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])

        paddle.enable_static()

    def test_static_unary(self):
        paddle.enable_static()

        for api in unary_api_list:
            main_prog = paddle.static.Program()
            block = main_prog.global_block()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                x = paddle.rand([])
                x.stop_gradient = False
                out = api(x)
                paddle.static.append_backward(out)

                fetch_list = [x, out]
                if block.has_var(x.grad_name):
                    fetch_list.extend([x.grad_name, out.grad_name])

                # 1) Test Program
                res = exe.run(main_prog, fetch_list=fetch_list)
                for item in res:
                    self.assertEqual(item.shape, ())

                # 2) Test CompiledProgram Program
                compile_prog = paddle.static.CompiledProgram(main_prog)
                res = exe.run(compile_prog, fetch_list=fetch_list)
                for item in res:
                    self.assertEqual(item.shape, ())

        paddle.disable_static()


reduce_api_list = [
    paddle.sum,
    paddle.mean,
    paddle.nansum,
    paddle.nanmean,
    paddle.min,
    paddle.max,
    paddle.amin,
    paddle.amax,
    paddle.prod,
    paddle.logsumexp,
    paddle.all,
    paddle.any,
]


# Use to test zero-dim of reduce API
class TestReduceAPI(unittest.TestCase):
    def test_dygraph_reduce(self):
        paddle.disable_static()
        for api in reduce_api_list:
            # 1) x is 0D
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, []).astype('bool')
            else:
                x = paddle.rand([])
            x.stop_gradient = False
            out = api(x, None)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            np.testing.assert_allclose(out.numpy(), x.numpy())

            out_empty_list = api(x, [])
            self.assertEqual(out_empty_list, out)
            self.assertEqual(out_empty_list.shape, [])

            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.grad.shape, [])
                np.testing.assert_allclose(x.grad.numpy(), np.array(1.0))
                np.testing.assert_allclose(out.grad.numpy(), np.array(1.0))

            out1 = api(x, 0)
            self.assertEqual(out1.shape, [])
            self.assertEqual(out1, out)
            out1.backward()

            out2 = api(x, -1)
            self.assertEqual(out2.shape, [])
            self.assertEqual(out2, out)
            out2.backward()

            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                np.testing.assert_allclose(x.grad.numpy(), np.array(3.0))

            if api in [
                paddle.sum,
                paddle.mean,
                paddle.nanmean,
                paddle.nansum,
            ]:
                return

            # 2) x is ND, reduce to 0D
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, [3, 5]).astype('bool')
            else:
                x = paddle.rand([3, 5])
            x.stop_gradient = False
            out = api(x, None)
            out.retain_grads()
            out.backward()

            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(out.grad.shape, [])
                self.assertEqual(x.grad.shape, [3, 5])

            # 3) x is 1D, axis=0, reduce to 0D
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, [5]).astype('bool')
            else:
                x = paddle.rand([5])
            x.stop_gradient = False
            out = api(x, 0)
            out.retain_grads()
            out.backward()

            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(out.grad.shape, [])
                self.assertEqual(x.grad.shape, [5])

        paddle.enable_static()

    def test_static_reduce(self):
        paddle.enable_static()
        for api in reduce_api_list:
            main_prog = paddle.static.Program()
            block = main_prog.global_block()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 1) x is 0D
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, []).astype('bool')
                else:
                    x = paddle.rand([])
                x.stop_gradient = False
                out = api(x, None)
                paddle.static.append_backward(out)

                out_empty_list = api(x, None)
                self.assertEqual(out_empty_list.shape, ())

                out1 = api(x, 0)
                self.assertEqual(out1.shape, ())

                out2 = api(x, -1)
                self.assertEqual(out2.shape, ())

                fetch_list = [x, out]
                if block.has_var(x.grad_name):
                    fetch_list.extend([x.grad_name, out.grad_name])

                res = exe.run(main_prog, fetch_list=fetch_list)
                self.assertEqual(res[0].shape, ())
                self.assertEqual(res[1].shape, ())
                np.testing.assert_allclose(res[0], res[1])

                if len(res) > 2:
                    self.assertEqual(res[2].shape, ())
                    self.assertEqual(res[3].shape, ())
                    np.testing.assert_allclose(res[2], np.array(1.0))
                    np.testing.assert_allclose(res[3], np.array(1.0))

                if api in [
                    paddle.sum,
                    paddle.mean,
                    paddle.nanmean,
                    paddle.nansum,
                ]:
                    return

                # 2) x is ND, reduce to 0D
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, [3, 5]).astype('bool')
                else:
                    x = paddle.rand([3, 5])
                x = paddle.rand([3, 5])
                x.stop_gradient = False
                out = api(x, None)
                paddle.static.append_backward(out)

                fetch_list = [out]
                if block.has_var(x.grad_name):
                    fetch_list.extend([out.grad_name, x.grad_name])

                res = exe.run(main_prog, fetch_list=fetch_list)
                self.assertEqual(res[0].shape, ())
                if len(res) > 1:
                    self.assertEqual(res[1].shape, ())
                    self.assertEqual(res[2].shape, (3, 5))

                # 3) x is 1D, axis=0, reduce to 0D
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, [5]).astype('bool')
                else:
                    x = paddle.rand([5])
                x.stop_gradient = False
                out = api(x, 0)
                paddle.static.append_backward(out)

                fetch_list = [out]
                if block.has_var(x.grad_name):
                    fetch_list.extend([out.grad_name, x.grad_name])

                res = exe.run(main_prog, fetch_list=fetch_list)
                self.assertEqual(res[0].shape, ())
                if len(res) > 1:
                    self.assertEqual(res[1].shape, ())
                    self.assertEqual(res[2].shape, (5,))

        paddle.disable_static()


binary_api_list = [
    {'func': paddle.add, 'cls_method': '__add__'},
    {'func': paddle.subtract, 'cls_method': '__sub__'},
    {'func': paddle.multiply, 'cls_method': '__mul__'},
    {'func': paddle.divide, 'cls_method': '__div__'},
    {'func': paddle.pow, 'cls_method': '__pow__'},
    {'func': paddle.equal, 'cls_method': '__eq__'},
    {'func': paddle.not_equal, 'cls_method': '__ne__'},
    {'func': paddle.greater_equal, 'cls_method': '__ge__'},
    {'func': paddle.greater_than, 'cls_method': '__gt__'},
    {'func': paddle.less_equal, 'cls_method': '__le__'},
    {'func': paddle.less_than, 'cls_method': '__lt__'},
    {'func': paddle.remainder, 'cls_method': '__mod__'},
    paddle.mod,
    paddle.floor_mod,
    paddle.logical_and,
    paddle.logical_or,
    paddle.logical_xor,
    paddle.maximum,
    paddle.minimum,
    paddle.fmax,
    paddle.fmin,
    paddle.complex,
    paddle.kron,
]

binary_int_api_list = [
    paddle.bitwise_and,
    paddle.bitwise_or,
    paddle.bitwise_xor,
    paddle.gcd,
    paddle.lcm,
]


# Use to test zero-dim of binary API
class TestBinaryAPI(unittest.TestCase):
    def test_dygraph_binary(self):
        paddle.disable_static()
        for api in binary_api_list:
            # 1) x is 0D, y is 0D
            x = paddle.rand([])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(y.shape, [])
            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

            # 2) x is ND, y is 0D
            x = paddle.rand([2, 3, 4])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [2, 3, 4])
            self.assertEqual(y.shape, [])
            self.assertEqual(out.shape, [2, 3, 4])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [2, 3, 4])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 3) x is 0D , y is ND
            x = paddle.rand([])
            y = paddle.rand([2, 3, 4])
            x.stop_gradient = False
            y.stop_gradient = False
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)

            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(y.shape, [2, 3, 4])
            self.assertEqual(out.shape, [2, 3, 4])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [2, 3, 4])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 4) x is 0D , y is scalar
            x = paddle.rand([])
            x.stop_gradient = False
            y = 0.5
            if isinstance(api, dict):
                out = getattr(paddle.Tensor, api['cls_method'])(x, y)

                out.retain_grads()
                out.backward()

                self.assertEqual(x.shape, [])
                self.assertEqual(out.shape, [])
                if x.grad is not None:
                    self.assertEqual(x.grad.shape, [])
                    self.assertEqual(out.grad.shape, [])

        for api in binary_int_api_list:
            # 1) x is 0D, y is 0D
            x_np = np.random.randint(-10, 10, [])
            y_np = np.random.randint(-10, 10, [])
            out_np = eval('np.%s(x_np, y_np)' % api.__name__)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = api(x, y)

            self.assertEqual(out.shape, [])
            np.testing.assert_array_equal(out.numpy(), out_np)

            # 2) x is ND, y is 0D
            x_np = np.random.randint(-10, 10, [3, 5])
            y_np = np.random.randint(-10, 10, [])
            out_np = eval('np.%s(x_np, y_np)' % api.__name__)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = api(x, y)

            self.assertEqual(out.shape, [3, 5])
            np.testing.assert_array_equal(out.numpy(), out_np)

            # 3) x is 0D , y is ND
            x_np = np.random.randint(-10, 10, [])
            y_np = np.random.randint(-10, 10, [3, 5])
            out_np = eval('np.%s(x_np, y_np)' % api.__name__)

            x = paddle.to_tensor(x_np)
            y = paddle.to_tensor(y_np)
            out = api(x, y)

            self.assertEqual(out.shape, [3, 5])
            np.testing.assert_array_equal(out.numpy(), out_np)

        paddle.enable_static()

    def test_static_binary(self):
        paddle.enable_static()
        for api in binary_api_list:
            main_prog = paddle.static.Program()
            block = main_prog.global_block()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 1) x is 0D, y is 0D
                x = paddle.rand([])
                y = paddle.rand([])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                    out_cls = getattr(
                        paddle.static.Variable, api['cls_method']
                    )(x, y)
                    self.assertEqual(out.shape, out_cls.shape)
                else:
                    out = api(x, y)
                paddle.static.append_backward(out)

                self.assertEqual(x.shape, ())
                self.assertEqual(y.shape, ())
                self.assertEqual(out.shape, ())
                if block.has_var(x.grad_name):
                    out_grad = block.var(out.grad_name)
                    x_grad = block.var(x.grad_name)
                    y_grad = block.var(y.grad_name)

                    self.assertEqual(x_grad.shape, ())
                    self.assertEqual(y_grad.shape, ())
                    self.assertEqual(out_grad.shape, ())

                # 2) x is 0D, y is ND
                x = paddle.rand([])
                y = paddle.rand([2, 3, 4])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                    out_cls = getattr(
                        paddle.static.Variable, api['cls_method']
                    )(x, y)
                    self.assertEqual(out.shape, out_cls.shape)
                else:
                    out = api(x, y)
                paddle.static.append_backward(out)

                self.assertEqual(x.shape, ())
                self.assertEqual(y.shape, (2, 3, 4))
                self.assertEqual(out.shape, (2, 3, 4))
                if block.has_var(x.grad_name):
                    out_grad = block.var(out.grad_name)
                    x_grad = block.var(x.grad_name)
                    y_grad = block.var(y.grad_name)

                    self.assertEqual(x_grad.shape, ())
                    self.assertEqual(y_grad.shape, (2, 3, 4))
                    self.assertEqual(out_grad.shape, (2, 3, 4))

                # 3) x is ND, y is 0d
                x = paddle.rand([2, 3, 4])
                y = paddle.rand([])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                    out_cls = getattr(
                        paddle.static.Variable, api['cls_method']
                    )(x, y)
                    self.assertEqual(out.shape, out_cls.shape)
                else:
                    out = api(x, y)
                paddle.static.append_backward(out)

                self.assertEqual(x.shape, (2, 3, 4))
                self.assertEqual(y.shape, ())
                self.assertEqual(out.shape, (2, 3, 4))
                if block.has_var(x.grad_name):
                    out_grad = block.var(out.grad_name)
                    x_grad = block.var(x.grad_name)
                    y_grad = block.var(y.grad_name)

                    self.assertEqual(x_grad.shape, (2, 3, 4))
                    self.assertEqual(y_grad.shape, ())
                    self.assertEqual(out_grad.shape, (2, 3, 4))

                # 4) x is 0D , y is scalar
                x = paddle.rand([])
                x.stop_gradient = False
                y = 0.5
                if isinstance(api, dict):
                    out = getattr(paddle.static.Variable, api['cls_method'])(
                        x, y
                    )
                    paddle.static.append_backward(out)

                    self.assertEqual(x.shape, ())
                    self.assertEqual(out.shape, ())
                    if block.has_var(x.grad_name):
                        out_grad = block.var(out.grad_name)
                        x_grad = block.var(x.grad_name)

                        self.assertEqual(out_grad.shape, ())
                        self.assertEqual(x_grad.shape, ())

        for api in binary_int_api_list:
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                # 1) x is 0D, y is 0D
                x = paddle.randint(-10, 10, [])
                y = paddle.randint(-10, 10, [])
                out = api(x, y)
                self.assertEqual(out.shape, ())

                # 2) x is ND , y is 0D
                x = paddle.randint(-10, 10, [3, 5])
                y = paddle.randint(-10, 10, [])
                out = api(x, y)
                self.assertEqual(out.shape, (3, 5))

                # 3) x is 0D , y is ND
                x = paddle.randint(-10, 10, [])
                y = paddle.randint(-10, 10, [3, 5])
                out = api(x, y)
                self.assertEqual(out.shape, (3, 5))

        paddle.disable_static()


# Use to test zero-dim of Sundry API, which is unique and can not be classified
# with others. It can be implemented here flexibly.
class TestSundryAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.rand([])

    def test_create_parameter_var(self):
        zero_dim_param = paddle.create_parameter(shape=[], dtype='float32')
        self.assertEqual(zero_dim_param.shape, [])

        zero_dim_var = paddle.tensor.creation.create_global_var(
            shape=[], value=0.5, dtype='float32'
        )
        self.assertEqual(zero_dim_var.shape, [])
        self.assertEqual(zero_dim_var.item(), 0.5)

    def test_expand(self):
        # case1
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out = paddle.expand(x, shape=[1])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [1])
        np.testing.assert_allclose(out, 1.0)
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, 1.0)
        self.assertEqual(out.grad.shape, [1])
        np.testing.assert_allclose(out.grad, 1.0)

        # case2
        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        out1 = paddle.expand(x1, shape=[])
        out1.retain_grads()
        out1.backward()

        self.assertEqual(out1.shape, [])
        np.testing.assert_allclose(out1, 1.0)
        self.assertEqual(x1.grad.shape, [])
        np.testing.assert_allclose(x1.grad, 1.0)
        self.assertEqual(out1.grad.shape, [])
        np.testing.assert_allclose(out1.grad, 1.0)

        # case3
        x2 = paddle.full([], 1, 'float32')
        x2.stop_gradient = False
        out2 = paddle.expand(x2, shape=[1, 1])
        out2.retain_grads()
        out2.backward()

        self.assertEqual(out2.shape, [1, 1])
        np.testing.assert_allclose(out2, 1.0)
        self.assertEqual(x2.grad.shape, [])
        np.testing.assert_allclose(x2.grad, 1.0)
        self.assertEqual(out2.grad.shape, [1, 1])
        np.testing.assert_allclose(out2.grad, 1.0)

        # case4
        x3 = paddle.full([], 1, 'float32')
        x3.stop_gradient = False
        out3 = paddle.expand(x3, shape=[3, 3])
        out3.retain_grads()
        out3.backward()

        self.assertEqual(out3.shape, [3, 3])
        np.testing.assert_allclose(out3, 1.0)
        self.assertEqual(x3.grad.shape, [])
        np.testing.assert_allclose(x3.grad, 9.0)
        self.assertEqual(out3.grad.shape, [3, 3])
        np.testing.assert_allclose(out3.grad, 1.0)

    def test_expand_as(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        y = paddle.full([], 1, 'float32')
        y.stop_gradient = False
        out = paddle.expand_as(x, y)
        out.backward()
        self.assertEqual(x.shape, [])
        self.assertEqual(x.item(), 1.0)
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad.item(), 1.0)
        self.assertEqual(out.shape, [])
        self.assertEqual(out.item(), 1.0)
        self.assertEqual(out.grad, None)

        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        y1 = paddle.full([1], 1, 'float32')
        out1 = paddle.expand_as(x1, y1)
        out1.backward()
        self.assertEqual(x1.shape, [])
        self.assertEqual(x1.item(), 1.0)
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x1.grad.item(0), 1.0)
        self.assertEqual(out1.shape, [1])
        self.assertEqual(out1.item(0), 1.0)
        self.assertEqual(out1.grad, None)

        x2 = paddle.full([], 1, 'float32')
        x2.stop_gradient = False
        y2 = paddle.full([3, 3], 1, 'float32')
        out2 = paddle.expand_as(x2, y2)
        out2.backward()
        self.assertEqual(x2.shape, [])
        self.assertEqual(x2.item(), 1.0)
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x2.grad.item(0), 9.0)
        self.assertEqual(out2.shape, [3, 3])
        self.assertEqual(out2.item(0), 1.0)
        self.assertEqual(out2.grad, None)

    def test_top_k(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out, indices = paddle.topk(x, k=1, axis=0)
        out.retain_grads()
        out.backward()
        self.assertEqual(indices.shape, [])
        self.assertEqual(indices.item(), 0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.item(), 1.0)
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad.item(0), 1.0)
        self.assertEqual(out.shape, [])
        self.assertEqual(out.item(), 1.0)
        self.assertEqual(out.grad, 1.0)

        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        out1, indices1 = paddle.topk(x1, k=1, axis=-1)
        out1.retain_grads()
        out1.backward()
        self.assertEqual(indices1.shape, [])
        self.assertEqual(indices1.item(), 0)
        self.assertEqual(x1.shape, [])
        self.assertEqual(x1.item(), 1.0)
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad.item(0), 1.0)
        self.assertEqual(out1.shape, [])
        self.assertEqual(out1.item(), 1.0)
        self.assertEqual(out1.grad, 1.0)

        with self.assertRaises(ValueError):
            tmp = paddle.topk(x1, k=1, axis=2)

    def test_broadcast_to(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out = paddle.broadcast_to(x, shape=[1])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [1])
        np.testing.assert_allclose(out, 1.0)
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, 1.0)
        self.assertEqual(out.grad.shape, [1])
        np.testing.assert_allclose(out.grad, 1.0)

        # case2
        x1 = paddle.full([], 1, 'float32')
        x1.stop_gradient = False
        out1 = paddle.broadcast_to(x1, shape=[])
        out1.retain_grads()
        out1.backward()

        self.assertEqual(out1.shape, [])
        np.testing.assert_allclose(out1, 1.0)
        self.assertEqual(x1.grad.shape, [])
        np.testing.assert_allclose(x1.grad, 1.0)
        self.assertEqual(out1.grad.shape, [])
        np.testing.assert_allclose(out1.grad, 1.0)

        # case3
        x2 = paddle.full([], 1, 'float32')
        x2.stop_gradient = False
        out2 = paddle.broadcast_to(x2, shape=[1, 1])
        out2.retain_grads()
        out2.backward()

        self.assertEqual(out2.shape, [1, 1])
        np.testing.assert_allclose(out2, 1.0)
        self.assertEqual(x2.grad.shape, [])
        np.testing.assert_allclose(x2.grad, 1.0)
        self.assertEqual(out2.grad.shape, [1, 1])
        np.testing.assert_allclose(out2.grad, 1.0)

        # case4
        x3 = paddle.full([], 1, 'float32')
        x3.stop_gradient = False
        out3 = paddle.broadcast_to(x3, shape=[3, 3])
        out3.retain_grads()
        out3.backward()

        self.assertEqual(out3.shape, [3, 3])
        np.testing.assert_allclose(out3, 1.0)
        self.assertEqual(x3.grad.shape, [])
        np.testing.assert_allclose(x3.grad, 9.0)
        self.assertEqual(out3.grad.shape, [3, 3])
        np.testing.assert_allclose(out3.grad, 1.0)

    def test_broadcast_tensors(self):
        # 1) x is 0D, y is 0D
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])
        # backward has bug now
        # out1.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        # self.assertEqual(x1.grad.shape, [])

        # 2) x is ND , y is 0D
        x1 = paddle.full([2, 3], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])
        # out1.backward()

        self.assertEqual(out1.shape, [2, 3])
        self.assertEqual(out2.shape, [2, 3])
        # self.assertEqual(x1.grad.shape, [2, 3])

        # 3) x is 0D , y is ND
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([2, 3], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])
        # out1.backward()

        self.assertEqual(out1.shape, [2, 3])
        self.assertEqual(out2.shape, [2, 3])
        # self.assertEqual(x1.grad.shape, [2, 3])

    def test_broadcast_shape(self):
        x = []
        y = [3, 5]
        out = paddle.broadcast_shape(x, y)
        self.assertEqual(out, [3, 5])

        x = [3, 5]
        y = []
        out = paddle.broadcast_shape(x, y)
        self.assertEqual(out, [3, 5])

        x = []
        y = []
        out = paddle.broadcast_shape(x, y)
        self.assertEqual(out, [])

        self.assertEqual(out, [])

    def test_argmin(self):
        # 1) x is 0D
        x = paddle.rand([])
        out1 = paddle.argmin(x, 0)
        out2 = paddle.argmin(x, -1)
        out3 = paddle.argmin(x, None)

        self.assertEqual(out1.shape, [])
        np.testing.assert_allclose(out1, 0)

        self.assertEqual(out2.shape, [])
        np.testing.assert_allclose(out2, 0)

        self.assertEqual(out3.shape, [])
        np.testing.assert_allclose(out3, 0)

        # 2) x is 1D
        x = paddle.rand([5])
        x.stop_gradient = False
        out = paddle.argmin(x, 0)
        out.backward()
        self.assertEqual(out.shape, [])

        # 3) x is ND
        x = paddle.rand([3, 5])
        x.stop_gradient = False
        out = paddle.argmin(x)
        out.backward()
        self.assertEqual(out.shape, [])

        # 4) x is ND, keepdim=True
        x = paddle.rand([3, 5])
        x.stop_gradient = False
        out = paddle.argmin(x, keepdim=True)
        out.backward()
        self.assertEqual(out.shape, [1, 1])

    def test_argmax(self):
        # 1) x is 0D
        x = paddle.rand([])
        out1 = paddle.argmax(x, 0)
        out2 = paddle.argmax(x, -1)
        out3 = paddle.argmax(x, None)

        self.assertEqual(out1.shape, [])
        np.testing.assert_allclose(out1, 0)

        self.assertEqual(out2.shape, [])
        np.testing.assert_allclose(out2, 0)

        self.assertEqual(out3.shape, [])
        np.testing.assert_allclose(out3, 0)

        # 2) x is 1D
        x = paddle.rand([5])
        out = paddle.argmax(x, 0)
        self.assertEqual(out.shape, [])

        # 3) x is ND
        x = paddle.rand([3, 5])
        out = paddle.argmax(x)
        self.assertEqual(out.shape, [])

        # 4) x is ND, keepdim=True
        x = paddle.rand([3, 5])
        out = paddle.argmax(x, keepdim=True)
        self.assertEqual(out.shape, [1, 1])

    def test_median(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out1 = paddle.median(x, 0)
        out2 = paddle.median(x, -1)
        out3 = paddle.median(x, None)

        out1.backward()
        out2.backward()
        out3.backward()

        self.assertEqual(out1.shape, [])
        np.testing.assert_allclose(out1, x)

        self.assertEqual(out2.shape, [])
        np.testing.assert_allclose(out2, x)

        self.assertEqual(out3.shape, [])
        np.testing.assert_allclose(out3, x)

        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, 3.0)

        # 2) x is 1D
        x = paddle.rand([5])
        x.stop_gradient = False
        out = paddle.median(x, 0)
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [5])

        # 3) x is ND
        x = paddle.rand([3, 5])
        x.stop_gradient = False
        out = paddle.median(x, None)
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [3, 5])

        # 4) x is ND, keepdim=True
        x = paddle.rand([3, 5])
        x.stop_gradient = False
        out = paddle.median(x, keepdim=True)
        out.backward()
        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(x.grad.shape, [3, 5])

    def test_kthvalue(self):
        # 1) x is 0D
        x = paddle.randn([])
        x.stop_gradient = False
        out, index = paddle.kthvalue(x, 1)
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out, x)
        self.assertEqual(index.shape, [])
        self.assertEqual(index, 0)

        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad, 1.0)

        # 2) x is 1D
        x1 = paddle.randn([5])
        x1.stop_gradient = False
        out1, index1 = paddle.kthvalue(x1, 1)
        out1.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(index1.shape, [])
        self.assertEqual(x1.grad.shape, [5])

    def test_mode(self):
        # 1) x is 0D
        x = paddle.randn([])
        x.stop_gradient = False
        out, index = paddle.mode(x)
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out, x)
        self.assertEqual(index.shape, [])
        self.assertEqual(index, 0)

        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad, 1.0)

        # 2) x is 1D
        x1 = paddle.randn([5])
        x1.stop_gradient = False
        out1, index1 = paddle.mode(x1)
        out1.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(index1.shape, [])

        self.assertEqual(x1.grad.shape, [5])

    def test_is_empty(self):
        # 1) x is 0D
        x = paddle.rand([])
        out = paddle.is_empty(x)
        self.assertFalse(out)
        self.assertEqual(out.shape, [])

        # 2) x is 1D
        x = paddle.rand([5])
        out = paddle.is_empty(x)
        self.assertFalse(out)
        self.assertEqual(out.shape, [])

        # 3) x is ND
        x = paddle.rand([3, 5])
        out = paddle.is_empty(x)
        self.assertFalse(out)
        self.assertEqual(out.shape, [])

        x = paddle.rand([3, 0, 5])
        out = paddle.is_empty(x)
        self.assertTrue(out)
        self.assertEqual(out.shape, [])

    def test_squeeze_(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.squeeze_(0)
        self.assertEqual(x.shape, [])

        # 2) x is 1D
        x = paddle.rand([1])
        x.squeeze_(0)
        self.assertEqual(x.shape, [])

        # 3）x is ND
        x = paddle.rand([2, 1])
        x.squeeze_(1)
        self.assertEqual(x.shape, [2])

    def test_as_complex(self):
        x = paddle.rand([2])
        x.stop_gradient = False
        out = paddle.as_complex(x)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.shape, [2])
        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [2])
        self.assertEqual(out.grad.shape, [])

    def test_dot(self):
        # 1) x is 1D
        x = paddle.rand([2])
        x.stop_gradient = False
        y = paddle.rand([2])
        y.stop_gradient = False
        out = paddle.dot(x, y)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.grad.shape, [2])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        # 2) x is 2D
        x1 = paddle.rand([2, 2])
        x1.stop_gradient = False
        y1 = paddle.rand([2, 2])
        y1.stop_gradient = False
        out1 = paddle.dot(x1, y1)
        out1.retain_grads()
        out1.backward()

        self.assertEqual(x1.grad.shape, [2, 2])
        self.assertEqual(out1.shape, [2])
        self.assertEqual(out1.grad.shape, [2])

    def test_inner(self):
        # 0) input is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        y = paddle.rand([])
        y.stop_gradient = False
        out = paddle.inner(x, y)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        # 1) input is 1D
        x = paddle.rand([2])
        x.stop_gradient = False
        y = paddle.rand([2])
        y.stop_gradient = False
        out = paddle.inner(x, y)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.grad.shape, [2])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        # 2) input is 2D
        x = paddle.rand([2, 3])
        x.stop_gradient = False
        y = paddle.rand([3, 3])
        y.stop_gradient = False
        out = paddle.inner(x, y)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.grad.shape, [2, 3])
        self.assertEqual(out.shape, [2, 3])
        self.assertEqual(out.grad.shape, [2, 3])

    def test_tensordot(self):

        # 1) input is 1D
        x = paddle.arange(10, dtype='float64')
        x.stop_gradient = False
        y = paddle.arange(10, dtype='float64')
        y.stop_gradient = False
        out = paddle.tensordot(x, y, axes=1)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.grad.shape, [10])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        # 2) input is 2D
        x = paddle.arange(6, dtype='float64').reshape([2, 3])
        y = paddle.arange(6, dtype='float64').reshape([2, 3])
        x.stop_gradient = False
        out = paddle.tensordot(x, y, axes=2)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.grad.shape, [2, 3])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

    def test_metric_accuracy(self):
        x = paddle.full(shape=[2, 4], fill_value=0.25)
        y = paddle.full(shape=[2, 1], fill_value=1, dtype="int64")
        out = paddle.metric.accuracy(input=x, label=y, k=1)
        self.assertEqual(out.shape, [])

    def test_std(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out1 = paddle.std(x)
        out2 = paddle.std(x, [])
        out1.backward()
        out2.backward()

        # checkout shape of out
        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])

        # checkout value of out
        self.assertEqual(out1, 0)
        self.assertEqual(out2, 0)

        # checkout backward
        self.assertEqual(x.grad.shape, [])

    def test_var(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out1 = paddle.var(x)
        out2 = paddle.var(x, [])
        out1.backward()
        out2.backward()

        # checkout shape of out
        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])

        # checkout value of out
        self.assertEqual(out1, 0)
        self.assertEqual(out2, 0)

        # checkout backward
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, 0)

    def test_quantile(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.quantile(x, 0.5, axis=None)

        out.retain_grads()
        out.backward()

        out_empty_list = paddle.quantile(x, 0.5, axis=[])
        self.assertEqual(out_empty_list, out)

        self.assertEqual(x.shape, [])
        self.assertEqual(out.shape, [])
        self.assertEqual(out, x)

        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad, 1.0)
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(out.grad, 1.0)

        # 2) x is ND
        x = paddle.rand([2, 3])
        x.stop_gradient = False
        out = paddle.quantile(x, 0.5, axis=None)

        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(out.grad, 1.0)
        self.assertEqual(x.grad.shape, [2, 3])

    def test_flip(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.flip(x, axis=[])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.shape, [])
        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.grad.shape, [])

    def test_linear(self):
        x = paddle.randn([3, 2])
        w = paddle.full(shape=[2, 4], fill_value=0.5)
        b = paddle.zeros([])

        np.testing.assert_array_equal(
            F.linear(x, w, b).numpy(), F.linear(x, w).numpy()
        )

    def test_is_complex(self):
        x = paddle.rand([]) + 1j * paddle.rand([])
        self.assertTrue(paddle.is_complex(x))

    def test_is_floating_point(self):
        self.assertTrue(paddle.is_floating_point(self.x))

    def test_is_integer(self):
        x = paddle.randint(0, 10, [])
        self.assertTrue(paddle.is_integer(x))

    def test_is_tensor(self):
        self.assertTrue(paddle.is_tensor(self.x))

    def test_isfinite(self):
        out = paddle.isfinite(self.x)
        np.testing.assert_array_equal(out.numpy(), np.array(True))

    def test_isinf(self):
        x = paddle.to_tensor(np.array(float('-inf')))
        out = paddle.isinf(x)
        np.testing.assert_array_equal(out.numpy(), np.array(True))

    def test_isnan(self):
        x = paddle.to_tensor(np.array(float('nan')))
        out = paddle.isnan(x)
        np.testing.assert_array_equal(out.numpy(), np.array(True))

    def test_isclose(self):
        out = paddle.isclose(self.x, self.x)
        np.testing.assert_array_equal(out.numpy(), np.array(True))

    def test_clone(self):
        out = paddle.clone(self.x)
        np.testing.assert_array_equal(out.numpy(), self.x.numpy())

    def test_assign(self):
        out = paddle.assign(self.x)
        np.testing.assert_array_equal(out.numpy(), self.x.numpy())

    def test_item(self):
        x = paddle.full([], 0.5)
        self.assertEqual(x.item(), 0.5)

    def test_tolist(self):
        x = paddle.full([], 0.5)
        self.assertEqual(x.tolist(), 0.5)

    def test_numpy(self):
        x = paddle.full([], 0.5)
        x_np = x.numpy()
        np.testing.assert_array_equal(x_np.shape, ())
        np.testing.assert_array_equal(x_np, np.array(0.5))

        x_np = x.numpy(False)
        np.testing.assert_array_equal(x_np.shape, ())
        np.testing.assert_array_equal(x_np, np.array(0.5))

    def test_numel(self):
        # 1) x is 0D
        out = paddle.numel(self.x)
        self.assertEqual(out.shape, [])
        np.testing.assert_array_equal(out.numpy(), np.array(1))

        # 2) x is ND
        x = paddle.full([3, 5], 0.5)
        out = paddle.numel(x)
        self.assertEqual(out.shape, [])
        np.testing.assert_array_equal(out.numpy(), np.array(15))

    def test_rank(self):
        # 1) x is 0D
        x = paddle.rand([])
        out = paddle.rank(x)
        self.assertEqual(out.shape, [])
        np.testing.assert_array_equal(out.numpy(), np.array(0))

        # 1) x is ND
        x = paddle.full([3, 5], 0.5)
        out = paddle.rank(x)
        self.assertEqual(out.shape, [])
        np.testing.assert_array_equal(out.numpy(), np.array(2))

    def test_shape(self):
        out = paddle.shape(self.x)
        np.testing.assert_array_equal(out.numpy(), np.array([]))
        self.assertEqual(out.shape, [0])

    def test_equal_scalar(self):
        x = paddle.rand([])
        out = paddle.equal(x, 2.0)
        self.assertEqual(out.shape, [])
        self.assertEqual(out, False)

        x1 = paddle.full([], 2.0)
        out1 = paddle.equal(x1, 2.0)
        self.assertEqual(out1.shape, [])
        self.assertEqual(out1, True)

    def test_pow_scalar(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.pow(x, 2.0)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_cast(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cast(x, 'int32')
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_cumprod(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cumprod(x, 0)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

        with self.assertRaises(ValueError):
            tmp = paddle.cumprod(x, 2)

    def test_clip(self):
        x = paddle.uniform([], None, -10, 10)
        x.stop_gradient = False
        out = paddle.clip(x, -5, 5)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

        x1 = paddle.uniform([], None, -10, 10)
        x1.stop_gradient = False
        out1 = paddle.clip(x1, paddle.full([], 5.0), paddle.full([], 5.0))
        out1.retain_grads()
        out1.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out1.grad.shape, [])
        self.assertEqual(x1.grad.shape, [])

    def test_increment(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.increment(x, 1.0)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_bitwise_not(self):
        x = paddle.randint(-1, 1, [])
        out1 = ~x
        out2 = paddle.bitwise_not(x)

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])

    def test_logical_not(self):
        x = paddle.randint(0, 1, [])
        out = paddle.logical_not(x)

        self.assertEqual(out.shape, [])

    def test_searchsorted(self):
        # have no backward
        x = paddle.to_tensor([1, 3, 5, 7, 9])
        y = paddle.rand([])

        out = paddle.searchsorted(x, y)

        self.assertEqual(out.shape, [])
        self.assertEqual(out.numpy(), 0)

    def test_transpose(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.transpose(x, [])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out, x)
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad, 1.0)

        with self.assertRaises(ValueError):
            x = paddle.transpose(x, [0])

    def test_moveaxis(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.moveaxis(x, [], [])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out, x)
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad, 1.0)

        with self.assertRaises(AssertionError):
            x = paddle.moveaxis(x, [1], [0])

    def test_gather_1D(self):
        x = paddle.to_tensor([1.0, 3.0, 5.0, 7.0, 9.0], stop_gradient=False)
        index = paddle.full([], 2, 'int64')
        out = paddle.gather(x, index)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.numpy(), 5)
        self.assertEqual(x.grad.shape, [5])
        self.assertEqual(out.grad.shape, [])

    def test_gather_xD_axis_0(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], stop_gradient=False
        )
        index = paddle.full([], 1, 'int64')
        out = paddle.gather(x, index)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [3])
        np.testing.assert_array_equal(out.numpy(), x.numpy()[1, :])
        self.assertEqual(x.grad.shape, [2, 3])
        self.assertEqual(out.grad.shape, [3])

    def test_gather_xD_axis_1(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], stop_gradient=False
        )
        index = paddle.full([], 1, 'int64')
        out = paddle.gather(x, index, axis=1)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [2])
        np.testing.assert_array_equal(out.numpy(), [2.0, 5.0])
        self.assertEqual(x.grad.shape, [2, 3])
        self.assertEqual(out.grad.shape, [2])

    def test_gather_nd(self):
        x1 = paddle.to_tensor([1.0, 3.0, 5.0, 7.0, 9.0], stop_gradient=False)
        x2 = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], stop_gradient=False
        )

        index1 = paddle.full([1], 1, 'int64')
        index2 = paddle.full([2], 1, 'int64')

        out1 = paddle.gather_nd(x1, index1)
        out2 = paddle.gather_nd(x2, index2)

        out1.retain_grads()
        out2.retain_grads()

        out1.backward()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        np.testing.assert_array_equal(out1, np.array(3.0))
        np.testing.assert_array_equal(out2, np.array(5.0))
        self.assertEqual(x1.grad.shape, [5])
        self.assertEqual(x2.grad.shape, [2, 3])
        self.assertEqual(out1.grad.shape, [])
        self.assertEqual(out2.grad.shape, [])

    def test_einsum(self):
        os.environ['FLAGS_new_einsum'] = "0"
        x = paddle.rand([5])
        # sum
        out1 = paddle.einsum('i->', x)
        expect1 = np.einsum('i->', x)
        # dot
        out2 = paddle.einsum('i,i->', x, x)
        expect2 = np.einsum('i,i->', x, x)

        out1.retain_grads()
        out2.retain_grads()

        out1.backward()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        np.testing.assert_allclose(out1, expect1, rtol=1e-03)
        np.testing.assert_allclose(out2, expect2, rtol=1e-03)

    def test_einsum_V2(self):
        os.environ['FLAGS_new_einsum'] = "1"
        x = paddle.rand([5])
        # sum
        out1 = paddle.einsum('i->', x)
        expect1 = np.einsum('i->', x)
        # dot
        out2 = paddle.einsum('i,i->', x, x)
        expect2 = np.einsum('i,i->', x, x)

        out1.retain_grads()
        out2.retain_grads()

        out1.backward()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        np.testing.assert_allclose(out1, expect1, rtol=1e-03)
        np.testing.assert_allclose(out2, expect2, rtol=1e-03)

    def test_scatter_1D(self):
        x = paddle.to_tensor([1.0, 3.0, 5.0, 7.0, 9.0], stop_gradient=False)
        index = paddle.full([], 2, 'int64')
        updates = paddle.full([], 4.0)
        out = paddle.scatter(x, index, updates)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [5])
        self.assertEqual(out.numpy()[2], 4)
        self.assertEqual(out.grad.shape, [5])

    def test_scatter_XD(self):
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], stop_gradient=False
        )
        index = paddle.full([], 1, 'int64')
        updates = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.scatter(x, index, updates)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [2, 3])
        np.testing.assert_array_equal(out.numpy()[1], [1.0, 2.0, 3.0])
        self.assertEqual(out.grad.shape, [2, 3])

    def test_diagflat(self):
        x1 = paddle.rand([])
        x2 = paddle.rand([])
        x3 = paddle.rand([])

        x1.stop_gradient = False
        x2.stop_gradient = False
        x3.stop_gradient = False

        x1.retain_grads()
        x2.retain_grads()
        x3.retain_grads()

        out1 = paddle.diagflat(x1, 1)
        out2 = paddle.diagflat(x2, -1)
        out3 = paddle.diagflat(x3, 0)

        out1.retain_grads()
        out2.retain_grads()
        out3.retain_grads()

        out1.backward()
        out2.backward()
        out3.backward()

        self.assertEqual(out1.shape, [2, 2])
        self.assertEqual(out2.shape, [2, 2])
        self.assertEqual(out3.shape, [1, 1])

        self.assertEqual(out1.grad.shape, [2, 2])
        self.assertEqual(out2.grad.shape, [2, 2])
        self.assertEqual(out3.grad.shape, [1, 1])

        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x3.grad.shape, [])

    def test_scatter__1D(self):
        x = paddle.to_tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        index = paddle.full([], 2, 'int64')
        updates = paddle.full([], 4.0)
        out = paddle.scatter_(x, index, updates)

        self.assertEqual(out.numpy()[2], 4)

    def test_scatter__XD(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = paddle.full([], 1, 'int64')
        updates = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.scatter_(x, index, updates)
        np.testing.assert_array_equal(out.numpy()[1], [1.0, 2.0, 3.0])

    def test_scatter_nd(self):
        index = paddle.to_tensor([3], dtype="int64")
        updates = paddle.full([], 2, dtype='float32')
        updates.retain_grads()
        updates.stop_gradient = False

        out = paddle.scatter_nd(index, updates, [5])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [5])
        self.assertEqual(out.numpy()[3], 2)
        self.assertEqual(out.grad.shape, [5])
        self.assertEqual(updates.grad.shape, [])

    def test_flatten(self):
        x = paddle.rand([])
        x.stop_gradient = False

        start_axis = 0
        stop_axis = -1

        out = paddle.flatten(x, start_axis=start_axis, stop_axis=stop_axis)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])
        self.assertEqual(x.grad.shape, [])

    def test_histogram(self):
        x = paddle.rand([])
        out = paddle.histogram(x, bins=5, min=1, max=5)
        self.assertEqual(out.shape, [5])

    def test_scale(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.scale(x, scale=2.0, bias=1.0)

        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_scale_(self):
        x = paddle.rand([])
        out = x.scale_(scale=2.0, bias=1.0)
        self.assertEqual(out.shape, [])

    def test_floor_divide(self):
        # 1-d // 0-d
        x = paddle.to_tensor([1, -2, 3], dtype="int64")
        y = paddle.full([], 2, dtype='int64')
        out1_1 = paddle.floor_divide(x, y)
        out1_2 = paddle.Tensor.__floordiv__(x, y)

        np.testing.assert_array_equal(out1_1.numpy(), out1_2.numpy())
        np.testing.assert_array_equal(out1_1.numpy(), np.asarray([0, -1, 1]))

        # 0-d // 1-d
        out2_1 = paddle.floor_divide(y, x)
        out2_2 = paddle.Tensor.__floordiv__(y, x)

        np.testing.assert_array_equal(out2_1.numpy(), out2_2.numpy())
        np.testing.assert_array_equal(out2_2.numpy(), np.asarray([2, -1, 0]))

        # 0-d // 0-d
        x = paddle.full([], 3, dtype='int64')
        out3_1 = paddle.floor_divide(x, y)
        out3_2 = paddle.Tensor.__floordiv__(x, y)

        np.testing.assert_array_equal(out3_1.numpy(), out3_2.numpy())
        np.testing.assert_array_equal(out3_2.numpy(), np.asarray(1))

    def test_cumsum(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False

        out1 = paddle.cumsum(x1)
        out2 = paddle.cumsum(x1, axis=0)
        out3 = paddle.cumsum(x1, axis=-1)

        out1.retain_grads()
        out2.retain_grads()
        out3.retain_grads()

        out1.backward()
        out2.backward()
        out3.backward()

        self.assertEqual(x1.grad.shape, [])
        self.assertTrue(x1.grad.numpy() == 3)
        self.assertEqual(out1.shape, [1])
        self.assertEqual(out1.grad.shape, [1])
        self.assertTrue(out1.grad.numpy() == 1)
        self.assertEqual(out2.shape, [])
        self.assertEqual(out2.grad.shape, [])
        self.assertTrue(out2.grad.numpy() == 1)
        self.assertEqual(out3.shape, [])
        self.assertEqual(out3.grad.shape, [])
        self.assertTrue(out3.grad.numpy() == 1)

    def test_add_n(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False
        x2 = paddle.rand([])
        x2.stop_gradient = False
        x3 = paddle.rand([])
        x3.stop_gradient = False

        out1 = paddle.add_n(x1)
        out2 = paddle.add_n([x2, x3])

        out1.retain_grads()
        out2.retain_grads()

        out1.backward()
        out2.backward()

        self.assertEqual(x1.grad.shape, [])
        self.assertTrue(x1.grad.numpy() == 1)
        self.assertEqual(x2.grad.shape, [])
        self.assertTrue(x2.grad.numpy() == 1)
        self.assertEqual(x3.grad.shape, [])
        self.assertTrue(x3.grad.numpy() == 1)
        self.assertEqual(out1.shape, [])
        self.assertEqual(out1.grad.shape, [])
        self.assertEqual(out2.shape, [])
        self.assertEqual(out2.grad.shape, [])

    def test_reshape_list(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.reshape(x, [])

        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        out = paddle.reshape(x, [1])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        out = paddle.reshape(x, [-1])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        out = paddle.reshape(x, [-1, 1])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(out.grad.shape, [1, 1])

    def test_reshape_tensor(self):
        x = paddle.rand([1, 1])
        x.stop_gradient = False
        out = paddle.reshape(x, [])

        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        new_shape = paddle.to_tensor([1, 1, 1], "int32")
        out = paddle.reshape(x, new_shape)
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1, 1, 1])
        self.assertEqual(out.grad.shape, [1, 1, 1])

        new_shape = paddle.to_tensor([-1], "int32")
        out = paddle.reshape(x, new_shape)
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        new_shape = [paddle.full([], -1, "int32"), paddle.full([], 1, "int32")]
        out = paddle.reshape(x, new_shape)
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1, 1])
        self.assertEqual(out.grad.shape, [1, 1])

    def test_reshape__list(self):
        x = paddle.rand([])
        out = paddle.reshape_(x, [])
        self.assertEqual(out.shape, [])

        out = paddle.reshape_(x, [1])
        self.assertEqual(out.shape, [1])

        out = paddle.reshape_(x, [-1])
        self.assertEqual(out.shape, [1])

        out = paddle.reshape_(x, [-1, 1])
        self.assertEqual(out.shape, [1, 1])

    def test_reshape__tensor(self):
        x = paddle.rand([1, 1])
        out = paddle.reshape_(x, [])
        self.assertEqual(out.shape, [])

        new_shape = paddle.full([1], 1, "int32")
        out = paddle.reshape_(x, new_shape)
        self.assertEqual(out.shape, [1])

        new_shape = paddle.full([1], -1, "int32")
        out = paddle.reshape_(x, new_shape)
        self.assertEqual(out.shape, [1])

        new_shape = [paddle.full([], -1, "int32"), paddle.full([], 1, "int32")]
        out = paddle.reshape_(x, new_shape)
        self.assertEqual(out.shape, [1, 1])

    def test_reverse(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.reverse(x, axis=[])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.shape, [])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

    def test_sort(self):
        x1 = paddle.rand([])
        x2 = paddle.rand([])
        x1.stop_gradient = False
        x2.stop_gradient = False
        x1.retain_grads()
        x2.retain_grads()
        out1 = paddle.sort(x1, axis=-1)
        out2 = paddle.sort(x2, axis=0)

        out1.retain_grads()
        out2.retain_grads()

        out1.backward()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        self.assertEqual(out1.numpy(), x1.numpy())
        self.assertEqual(out2.numpy(), x2.numpy())
        self.assertEqual(out1.grad.shape, [])
        self.assertEqual(out2.grad.shape, [])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x1.grad.numpy(), 1)
        self.assertEqual(x2.grad.numpy(), 1)

    def test_argsort(self):
        x1 = paddle.rand([])
        x2 = paddle.rand([])
        x1.stop_gradient = False
        x2.stop_gradient = False
        x1.retain_grads()
        x2.retain_grads()

        out1 = paddle.argsort(x1, axis=-1)
        out2 = paddle.argsort(x2, axis=0)

        out1.retain_grads()
        out2.retain_grads()

        out1.backward()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out2.shape, [])
        self.assertEqual(out1.numpy(), 0)
        self.assertEqual(out2.numpy(), 0)
        self.assertEqual(out1.grad.shape, [])
        self.assertEqual(out2.grad.shape, [])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x1.grad.numpy(), 0)
        self.assertEqual(x2.grad.numpy(), 0)

    def test_lerp(self):
        # 0D + 0D, weight is float scalar
        x = paddle.rand([])
        y = paddle.rand([])
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.lerp(x, y, 0.5)
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(y.grad.shape, [])

        # 0D + 0D, weigh is 0D
        x0 = paddle.rand([])
        y0 = paddle.rand([])
        w0 = paddle.rand([])
        x0.stop_gradient = False
        y0.stop_gradient = False
        y0.retain_grads()

        out0 = paddle.lerp(x0, y0, w0)
        out0.backward()

        self.assertEqual(out0.shape, [])
        self.assertEqual(x0.grad.shape, [])
        self.assertEqual(y0.grad.shape, [])

        # 0D + ND
        x1 = paddle.rand([])
        y1 = paddle.rand([64, 64])
        w1 = paddle.rand([])
        x1.stop_gradient = False
        y1.stop_gradient = False
        x1.retain_grads()
        y1.retain_grads()

        out1 = paddle.lerp(x1, y1, w1)
        out1.backward()

        self.assertEqual(out1.shape, [64, 64])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(y1.grad.shape, [64, 64])

        # ND + 0D
        x2 = paddle.rand([64, 64])
        y2 = paddle.rand([])
        w2 = paddle.rand([])
        x2.stop_gradient = False
        y2.stop_gradient = False
        x2.retain_grads()
        y2.retain_grads()

        out2 = paddle.lerp(x2, y2, w2)
        out2.backward()

        self.assertEqual(out2.shape, [64, 64])
        self.assertEqual(x2.grad.shape, [64, 64])
        self.assertEqual(y2.grad.shape, [])

    def test_repeat_interleave(self):
        places = ['cpu']
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)

            x = paddle.randn(())
            x.stop_gradient = False

            out = paddle.repeat_interleave(x, 2, None)
            out.backward()

            # check shape of output
            self.assertEqual(out.shape, [2])

            # check grad shape
            self.assertEqual(x.grad.shape, [])

            repeats = paddle.to_tensor([3], dtype='int32')
            out = paddle.repeat_interleave(x, repeats, None)

            # check shape of output with 1D repeats
            self.assertEqual(out.shape, [3])

            # check grad shape with 1D repeats
            self.assertEqual(x.grad.shape, [])

    def test_allclose(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        out = paddle.allclose(x, y)
        self.assertEqual(out.shape, [])
        self.assertFalse(out)

        # 2) x is ND
        x = paddle.full([2, 3], 0.5)
        y = paddle.full([2, 3], 0.6)
        out = paddle.allclose(x, y)
        self.assertEqual(out.shape, [])
        self.assertFalse(out)

    def test_equal_all(self):
        # 1) x is 0D
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        out = paddle.equal_all(x, y)
        self.assertEqual(out.shape, [])
        self.assertFalse(out)

        # 2) x is ND
        x = paddle.full([2, 3], 0.5)
        y = paddle.full([2, 3], 0.6)
        out = paddle.equal_all(x, y)
        self.assertEqual(out.shape, [])
        self.assertFalse(out)

    def test_where(self):
        x1 = paddle.full([], 1)
        x2 = paddle.full([], 2)
        x1.stop_gradient = False
        x2.stop_gradient = False
        x1.retain_grads()
        x2.retain_grads()
        out = paddle.where(x1 > x2, x1, x2)
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(out.numpy(), 2)
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x1.grad.numpy(), 0)
        self.assertEqual(x2.grad.numpy(), 1)

    def test_atan2(self):
        x1 = paddle.full([], 0)
        x2 = paddle.full([], 2)
        x1.retain_grads()
        x2.retain_grads()
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.atan2(x1, x2)
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(out.numpy(), 0)
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x1.grad.numpy(), 0.5)
        self.assertEqual(x2.grad.numpy(), 0)

    def test_interpolate(self):
        from paddle.nn.functional import interpolate

        input_x = paddle.rand([2, 3, 6, 6])
        input_x.stop_gradient = False
        origin_result = interpolate(
            x=input_x, size=[12, 12], mode="bilinear", align_corners=False
        )

        output_size = [
            paddle.full([], 12, dtype="int32"),
            paddle.full([], 12, dtype="int32"),
        ]
        out1 = interpolate(
            x=input_x, size=output_size, mode="bilinear", align_corners=False
        )
        out1.backward()

        self.assertEqual(out1.shape, [2, 3, 12, 12])
        self.assertEqual(input_x.grad.shape, [2, 3, 6, 6])

        scale_1 = [paddle.full([], 2), paddle.full([], 2)]
        out2 = interpolate(
            x=input_x,
            scale_factor=scale_1,
            mode="bilinear",
            align_corners=False,
        )
        out2.backward()

        self.assertEqual(out2.shape, [2, 3, 12, 12])
        self.assertEqual(input_x.grad.shape, [2, 3, 6, 6])

        scale_2 = paddle.full([], 2)
        out3 = interpolate(
            x=input_x,
            scale_factor=scale_2,
            mode="bilinear",
            align_corners=False,
        )
        out3.backward()

        # for coverage
        scale_3 = paddle.full([1], 2)
        input_3d = paddle.rand([2, 3, 6])
        out4 = interpolate(
            x=input_3d,
            scale_factor=scale_3,
            mode="LINEAR",
            align_corners=False,
            data_format="NCW",
        )

        self.assertEqual(out3.shape, [2, 3, 12, 12])
        self.assertEqual(input_x.grad.shape, [2, 3, 6, 6])

        np.testing.assert_allclose(
            origin_result.numpy(), out1.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            origin_result.numpy(), out2.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            origin_result.numpy(), out3.numpy(), rtol=1e-05
        )

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
        out1.backward()

        self.assertEqual(out1.shape, [2, 3, 12, 12])
        self.assertEqual(input_x.grad.shape, [2, 3, 6, 6])

    def test_unstack(self):
        x1 = paddle.full([1], 0)
        x2 = paddle.full([2], 2)
        x1.retain_grads()
        x2.retain_grads()
        x1.stop_gradient = False
        x2.stop_gradient = False

        [out1] = paddle.unstack(x1, 0)
        out1.retain_grads()
        out1.backward()
        [out2_1, out2_2] = paddle.unstack(x2, 0)
        out2 = paddle.add_n([out2_1, out2_2])
        out2.retain_grads()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out1.numpy(), 0)

        self.assertEqual(out2_1.shape, [])
        self.assertEqual(out2_1.numpy(), 2)
        self.assertEqual(out2_2.shape, [])
        self.assertEqual(out2_2.numpy(), 2)
        self.assertEqual(x2.grad.shape, [2])

    def test_unbind(self):
        x1 = paddle.full([1], 0)
        x2 = paddle.full([2], 2)
        x1.retain_grads()
        x2.retain_grads()
        x1.stop_gradient = False
        x2.stop_gradient = False

        [out1] = paddle.unbind(x1, 0)
        out1.retain_grads()
        out1.backward()
        [out2_1, out2_2] = paddle.unbind(x2, 0)
        out2 = paddle.add_n([out2_1, out2_2])
        out2.retain_grads()
        out2.backward()

        self.assertEqual(out1.shape, [])
        self.assertEqual(out1.numpy(), 0)

        self.assertEqual(out2_1.shape, [])
        self.assertEqual(out2_1.numpy(), 2)
        self.assertEqual(out2_2.shape, [])
        self.assertEqual(out2_2.numpy(), 2)
        self.assertEqual(x2.grad.shape, [2])

    def test_maseked_select(self):
        x = paddle.rand([])
        x.stop_gradient = False
        mask = paddle.full([], True, dtype='bool')
        y = paddle.masked_select(x, mask)

        y.retain_grads()
        y.backward()
        self.assertEqual(y.shape, [1])
        self.assertEqual(y.numpy(), x.numpy())
        self.assertEqual(y.grad.shape, [1])
        self.assertEqual(x.grad.shape, [])
        self.assertEqual(x.grad.numpy(), 1)

    def test_squeeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        x1.retain_grads()
        out1 = paddle.squeeze(x1, axis=0)
        out1.retain_grads()
        out1.backward()
        self.assertEqual(out1.shape, [])
        self.assertEqual(x1.grad.shape, [])

        x2 = paddle.full([], 3)
        x3 = paddle.full([1], 0, dtype='int32')
        x2.stop_gradient = False
        x2.retain_grads()
        out2 = paddle.squeeze(x2, axis=x3)
        out2.retain_grads()
        out2.backward()
        self.assertEqual(out2.shape, [])
        self.assertEqual(x2.grad.shape, [])

    def test_unsqueeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        x1.retain_grads()
        out1 = paddle.unsqueeze(x1, axis=0)
        out1.retain_grads()
        out1.backward()
        self.assertEqual(out1.shape, [1])
        self.assertEqual(x1.grad.shape, [])

        x2 = paddle.full([], 0, dtype='int32')
        out2 = paddle.unsqueeze(x1, axis=x2)
        out2.retain_grads()
        out2.backward()
        self.assertEqual(out2.shape, [1])
        self.assertEqual(x1.grad.shape, [])

    def test_t(self):
        x = paddle.full([], 2.0)
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.t(x)
        out.retain_grads()
        out.backward()
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_prelu(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        w1 = paddle.full([], 0.25, dtype='float32')
        out1 = paddle.nn.functional.prelu(x1, w1)
        out1.retain_grads()
        out1.backward()
        self.assertEqual(out1.shape, [])
        self.assertEqual(out1.numpy(), 1.0)
        self.assertEqual(out1.grad.shape, [])
        self.assertEqual(x1.grad.shape, [])
        self.assertEqual(x1.grad.numpy(), 1.0)

        x2 = paddle.full([], -1.0, 'float32')
        x2.stop_gradient = False
        w2 = paddle.full([], 0.25, dtype='float32')
        out2 = paddle.nn.functional.prelu(x2, w2)
        out2.retain_grads()
        out2.backward()
        self.assertEqual(out2.shape, [])
        self.assertEqual(out2.numpy(), -0.25)
        self.assertEqual(out2.grad.shape, [])
        self.assertEqual(x2.grad.shape, [])
        self.assertEqual(x2.grad.numpy(), 0.25)

    def test_while_loop(self):
        def cond(i, x):
            return paddle.less_than(i, eleven)

        def body(i, x):
            x = x + i
            i = i + 1
            return [i, x]

        i = paddle.full([], 1.0, dtype='float32')
        i.stop_gradient = False
        eleven = paddle.full([], 11, dtype='float32')
        x = paddle.full([], 0.0, dtype='float32')
        x.stop_gradient = False
        out_i, out_x = paddle.static.nn.while_loop(cond, body, [i, x])
        out_x.backward()

        self.assertEqual(out_i.shape, [])
        np.testing.assert_allclose(out_i, np.array(11))
        self.assertEqual(out_x.shape, [])
        np.testing.assert_allclose(out_x, np.array(55))
        self.assertEqual(i.grad.shape, [])
        np.testing.assert_allclose(i.grad, np.array(10))
        self.assertEqual(x.grad.shape, [])
        np.testing.assert_allclose(x.grad, np.array(1.0))

    def test_linalg_slogdet(self):
        # 2-D input
        x = paddle.randn([3, 3])
        x.stop_gradient = False
        out = paddle.linalg.slogdet(x)
        out.retain_grads()
        out.backward()

        self.assertTrue(out.shape, [2])
        self.assertTrue(x.grad.shape, [3, 3])

        # 3-D input
        x1 = paddle.randn([3, 3, 3])
        x1.stop_gradient = False
        out1 = paddle.linalg.slogdet(x1)
        out1.retain_grads()
        out1.backward()

        self.assertTrue(out1.shape, [2, 3])
        self.assertTrue(x1.grad.shape, [3, 3, 3])

    def test_multi_dot(self):
        a = paddle.randn([4])
        a.stop_gradient = False
        b = paddle.randn([4, 5])
        b.stop_gradient = False
        c = paddle.randn([5])
        c.stop_gradient = False

        out = paddle.linalg.multi_dot([a, b, c])
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(a.grad.shape, [4])
        self.assertEqual(b.grad.shape, [4, 5])
        self.assertEqual(c.grad.shape, [5])

    def test_dist(self):
        x = paddle.to_tensor([[3, 3], [3, 3]], dtype="float32")
        y = paddle.to_tensor([[3, 3], [3, 1]], dtype="float32")
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.dist(x, y, 0)
        out.backward()

        self.assertEqual(out.shape, [])
        np.testing.assert_allclose(out, np.array(1))
        self.assertEqual(x.grad.shape, [2, 2])
        self.assertEqual(y.grad.shape, [2, 2])

    def test_trace(self):
        x = paddle.to_tensor([[3, 2], [1, 9]], dtype="float32")
        x.stop_gradient = False
        out = paddle.trace(x)
        out.backward()

        self.assertEqual(out.shape, [])
        np.testing.assert_allclose(out, np.array(12))
        self.assertEqual(x.grad.shape, [2, 2])

    def test_cond(self):
        pass
        # def assert_shape(out):
        #     self.assertEqual(out.shape, [])

        # x = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        # x.stop_gradient = False
        # p = 2 : use paddle.sum, paddle.max, paddle.min
        # out = paddle.linalg.cond(x)
        # assert_shape(out)

        # p = fro : use paddle.sum
        # out_fro = paddle.linalg.cond(x, p='fro')
        # assert_shape(out_fro)

        # p = nuc : use paddle.sum, paddle.max, paddle.min
        # out_nuc = paddle.linalg.cond(x, p='nuc')
        # assert_shape(out_nuc)

        # p in (-1, 1) : use paddle.sum, paddle.max, paddle.min
        # out_1 = paddle.linalg.cond(x, p=1)
        # assert_shape(out_1)
        # out_minus_1 = paddle.linalg.cond(x, p=-1)
        # assert_shape(out_minus_1)

        # p in (-2, 2) :use paddle.max, paddle.min
        # out_2 = paddle.linalg.cond(x, p=2)
        # assert_shape(out_2)
        # out_minus_2 = paddle.linalg.cond(x, p=-2)
        # assert_shape(out_minus_2)

        # p in (-inf, inf):use paddle.sum, paddle.max, paddle.min
        # out_inf = paddle.linalg.cond(x, p=float("inf"))
        # assert_shape(out_inf)
        # out_minus_inf = paddle.linalg.cond(x, p=-float("inf"))
        # assert_shape(out_minus_inf)
        # out_minus_inf.backward()
        # self.assertTrue(x.grad.shape, [3, 3])

        # a = paddle.randn([2, 4, 4])
        # a.stop_gradient = False
        # a_cond_fro = paddle.linalg.cond(a, p='fro')
        # a_cond_fro.backward()
        # self.assertEqual(len(a_cond_fro.shape), 1)
        # self.assertEqual(a.grad.shape, [2, 4, 4])

    def test_cov(self):
        xt = paddle.randn((3, 4))
        xt.stop_gradient = False
        xt_1 = paddle.randn((12,))
        xt_1.stop_gradient = False

        xt_out = paddle.linalg.cov(xt)
        xt_out.retain_grads()
        xt_out.backward()
        self.assertEqual(xt_out.shape, [3, 3])
        self.assertEqual(xt.grad.shape, [3, 4])

        xt_1_out = paddle.linalg.cov(xt_1)
        xt_1.retain_grads()
        xt_1_out.backward()
        self.assertEqual(xt_1_out.shape, [])
        self.assertEqual(xt_1.grad.shape, [12])

    def test_det(self):
        xt = paddle.randn([3, 3, 3])
        xt.stop_gradient = False
        xt_1 = paddle.randn([3, 3])
        xt_1.stop_gradient = False

        xt_out = paddle.linalg.det(xt)
        xt.retain_grads()
        xt_out.backward()
        self.assertEqual(xt_out.shape, [3])
        self.assertEqual(xt.grad.shape, [3, 3, 3])

        xt_1_out = paddle.linalg.det(xt_1)
        xt_1.retain_grads()
        xt_1_out.backward()
        self.assertEqual(xt_1_out.shape, [])
        self.assertEqual(xt_1.grad.shape, [3, 3])


class TestSundryAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    @prog_scope()
    def test_create_parameter_var(self):
        zero_dim_param = paddle.create_parameter(shape=[], dtype='float32')
        self.assertEqual(zero_dim_param.shape, ())
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
    def test_expand(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False
        out = paddle.expand(x, shape=[1])
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
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
        paddle.static.append_backward(out1.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x1, out1, x1.grad_name, out1.grad_name]
        )
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
        paddle.static.append_backward(out2.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x2, out2, x2.grad_name, out2.grad_name]
        )
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
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
        paddle.static.append_backward(out1.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x1, out1, x1.grad_name, out1.grad_name]
        )
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
        paddle.static.append_backward(out2.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x2, out2, x2.grad_name, out2.grad_name]
        )
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
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, indices, x.grad_name, out.grad_name]
        )
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
        paddle.static.append_backward(out1.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x1, out1, indices1, x1.grad_name, out1.grad_name]
        )
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
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
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
        paddle.static.append_backward(out1.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x1, out1, x1.grad_name, out1.grad_name]
        )

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
    def test_median(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.median(x)
        paddle.static.append_backward(out)

        # 2) x is ND
        x1 = paddle.rand([3, 5])
        x1.stop_gradient = False
        out1 = paddle.median(x1)
        paddle.static.append_backward(out1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out,
                x.grad_name,
                out1,
                x1.grad_name,
            ],
        )
        self.assertEqual(res[1].shape, ())
        np.testing.assert_allclose(res[1], res[0])

        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[2], 1.0)

        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, (3, 5))

    @prog_scope()
    def test_kthvalue(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out, index = paddle.kthvalue(x, 1)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, out, index, x.grad_name])
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
        paddle.static.append_backward(out1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, index1, x1.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (5,))

    @prog_scope()
    def test_mode(self):
        # 1) x is 0D
        x = paddle.rand([])
        x.stop_gradient = False
        out, index = paddle.mode(x)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, index, x.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertTrue(res[2] == 1.0)

        # 2) x is 1D
        x1 = paddle.rand([5])
        x1.stop_gradient = False
        out1, index1 = paddle.mode(x1)
        paddle.static.append_backward(out1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, index1, x1.grad_name])
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
        self.assertEqual(x.shape, (2,))
        self.assertEqual(out.shape, ())
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, out, x.grad_name, out.grad_name],
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

        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, x.grad_name, out, out.grad_name],
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

        paddle.static.append_backward(out1.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x1, x1.grad_name, out1, out1.grad_name],
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
        paddle.static.append_backward(out1.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x1,
                x1.grad_name,
                out1,
                out1.grad_name,
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                x.grad_name,
                out,
                out.grad_name,
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

        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, x.grad_name, out, out.grad_name],
        )

        self.assertEqual(res[0].shape, (10,))
        self.assertEqual(res[1].shape, (10,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

        x = paddle.arange(6, dtype='float64').reshape([2, 3])
        y = paddle.arange(6, dtype='float64').reshape([2, 3])
        x.stop_gradient = False
        out = paddle.tensordot(x, y, axes=2)

        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[x, x.grad_name, out, out.grad_name],
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

    @prog_scope()
    def test_std(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out1 = paddle.std(x)
        out2 = paddle.std(x, [])
        paddle.static.append_backward(out1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out1,
                out2,
                x.grad_name,
                out1.grad_name,
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
        paddle.static.append_backward(out1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out1,
                out2,
                x.grad_name,
                out1.grad_name,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, ())

    @prog_scope()
    def test_quantile(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False
        out1 = paddle.quantile(x1, 0.5, axis=None)
        paddle.static.append_backward(out1.sum())

        x2 = paddle.rand([2, 3])
        x2.stop_gradient = False
        out2 = paddle.quantile(x2, 0.5, axis=None)
        paddle.static.append_backward(out2.sum())

        out_empty_list = paddle.quantile(x1, 0.5, axis=[])
        self.assertEqual(out_empty_list.shape, ())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                out1.grad_name,
                x2.grad_name,
                out2.grad_name,
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
    def test_flip(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.flip(x, axis=[])
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
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
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_cast(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cast(x, 'int32')
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_cumprod(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        out = paddle.cumprod(x, 0)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out)

        x1 = paddle.uniform([], None, -10, 10)
        x1.stop_gradient = False
        out1 = paddle.clip(x1, paddle.full([], 5.0), paddle.full([], 5.0))
        paddle.static.append_backward(out1)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                x,
                out,
                x.grad_name,
                out.grad_name,
                x1,
                out1,
                x1.grad_name,
                out1.grad_name,
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
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
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
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out1.sum())
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                x2.grad_name,
                out1.grad_name,
                out2.grad_name,
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[out, out.grad_name, updates.grad_name]
        )
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
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, feed={}, fetch_list=[out, x.grad_name, out.grad_name]
        )

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
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, out.grad_name])
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

        paddle.static.append_backward(out1.sum())
        paddle.static.append_backward(out2.sum())
        paddle.static.append_backward(out3.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                x1.grad_name,
                out1.grad_name,
                out2.grad_name,
                out3.grad_name,
            ],
        )
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)
        self.assertEqual(res[4].shape, (1,))
        self.assertEqual(res[4], 1)
        self.assertEqual(res[5].shape, ())
        self.assertEqual(res[5], 1)
        self.assertEqual(res[6].shape, ())
        self.assertEqual(res[6], 1)
        self.assertEqual(out2.shape, ())
        self.assertEqual(out3.shape, ())

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

        paddle.static.append_backward(out1.sum())
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        block = prog.global_block()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                x2.grad_name,
                x3.grad_name,
                out1.grad_name,
                out2.grad_name,
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
        paddle.static.append_backward(out1.sum())

        out2 = paddle.reshape(x2, [1])
        paddle.static.append_backward(out2.sum())

        out3 = paddle.reshape(x3, [-1])
        paddle.static.append_backward(out3.sum())

        out4 = paddle.reshape(x4, [-1, 1])
        paddle.static.append_backward(out4.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                out4,
                x1.grad_name,
                x2.grad_name,
                x3.grad_name,
                x4.grad_name,
                out1.grad_name,
                out2.grad_name,
                out3.grad_name,
                out4.grad_name,
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
        paddle.static.append_backward(out1.sum())

        x2 = paddle.rand([1, 1])
        x2.stop_gradient = False
        new_shape = paddle.full([1], -1, "int32")
        out2 = paddle.reshape(x2, new_shape)
        paddle.static.append_backward(out2.sum())

        x3 = paddle.rand([1, 1])
        x3.stop_gradient = False
        new_shape = [paddle.full([], -1, "int32"), paddle.full([], 1, "int32")]
        out3 = paddle.reshape(x3, new_shape)
        paddle.static.append_backward(out3.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out3,
                x1.grad_name,
                x2.grad_name,
                x3.grad_name,
                out1.grad_name,
                out2.grad_name,
                out3.grad_name,
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
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[x, out, x.grad_name, out.grad_name]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_sort(self):
        x1 = paddle.rand([])
        x1.stop_gradient = False
        out1 = paddle.sort(x1, axis=-1)
        paddle.static.append_backward(out1.sum())

        x2 = paddle.rand([])
        x2.stop_gradient = False
        out2 = paddle.sort(x2, axis=0)
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                out1.grad_name,
                out2.grad_name,
                x1.grad_name,
                x2.grad_name,
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
            paddle.static.append_backward(out.sum())

            prog = paddle.static.default_main_program()
            res = self.exe.run(
                prog, fetch_list=[out, out.grad_name, y.grad_name, x.grad_name]
            )
            self.assertEqual(res[0].shape, shape[3])
            self.assertEqual(res[1].shape, shape[3])
            self.assertEqual(res[2].shape, shape[1])
            self.assertEqual(res[3].shape, shape[0])

    @prog_scope()
    def test_repeat_interleave(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.repeat_interleave(x1, 2, None)
        paddle.static.append_backward(out1.sum())

        x2 = paddle.full([], 1.0, 'float32')
        x2.stop_gradient = False
        repeats = paddle.to_tensor([3], dtype='int32')
        out2 = paddle.repeat_interleave(x2, repeats, None)
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                x2.grad_name,
                out1.grad_name,
                out2.grad_name,
            ],
        )
        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (3,))
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[4].shape, (2,))
        self.assertEqual(res[5].shape, (3,))

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

    @prog_scope()
    def test_where(self):
        x1 = paddle.full([], 1, 'float32')
        x2 = paddle.full([], 2, 'float32')
        x1.stop_gradient = False
        x2.stop_gradient = False
        out = paddle.where(x1 > x2, x1, x2)
        loss = paddle.mean(out)
        paddle.static.append_backward(loss)

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={},
            fetch_list=[out, out.grad_name, x1.grad_name, x2.grad_name],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)

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
        paddle.static.append_backward(out1.sum())
        prog = paddle.static.default_main_program()
        res1 = self.exe.run(prog, feed={}, fetch_list=[out1, input_x.grad_name])

        scale_1 = paddle.full([], 2)
        out2 = interpolate(
            x=input_x,
            scale_factor=scale_1,
            mode="bilinear",
            align_corners=False,
        )
        paddle.static.append_backward(out2.sum())
        prog = paddle.static.default_main_program()
        res2 = self.exe.run(prog, feed={}, fetch_list=[out2, input_x.grad_name])

        self.assertEqual(res1[0].shape, (2, 3, 12, 12))
        self.assertEqual(res1[1].shape, (2, 3, 6, 6))
        self.assertEqual(res2[0].shape, (2, 3, 12, 12))
        self.assertEqual(res2[1].shape, (2, 3, 6, 6))

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
        paddle.static.append_backward(out1.sum())
        prog = paddle.static.default_main_program()
        res1 = self.exe.run(prog, feed={}, fetch_list=[out1, input_x.grad_name])

        self.assertEqual(res1[0].shape, (2, 3, 12, 12))
        self.assertEqual(res1[1].shape, (2, 3, 6, 6))

    @prog_scope()
    def test_unstack(self):
        x1 = paddle.full([1], 0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.unstack(x1, 0)
        out1 = paddle.add_n(out1)
        paddle.static.append_backward(out1)
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out1, x1.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))

        x2 = paddle.full([2], 2, 'float32')
        x2.stop_gradient = False
        out2 = paddle.unstack(x2, 0)
        out2_sum = paddle.add_n(out2)
        paddle.static.append_backward(out2_sum)
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out2_sum, x2.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2,))

    @prog_scope()
    def test_unbind(self):
        x1 = paddle.full([1], 0, 'float32')
        x1.stop_gradient = False
        out1 = paddle.unbind(x1, 0)
        out1 = paddle.add_n(out1)
        paddle.static.append_backward(out1)
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out1, x1.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (1,))

        x2 = paddle.full([2], 2, 'float32')
        x2.stop_gradient = False
        out2 = paddle.unbind(x2, 0)
        out2_sum = paddle.add_n(out2)
        paddle.static.append_backward(out2_sum)
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, feed={}, fetch_list=[out2_sum, x2.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2,))

    @prog_scope()
    def test_maseked_select(self):
        x = paddle.rand([])
        x.stop_gradient = False
        mask = paddle.full([], True, dtype='bool')
        y = paddle.masked_select(x, mask)
        paddle.static.append_backward(y.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[x, y, y.grad_name, x.grad_name])
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[1], res[0])
        self.assertEqual(res[2].shape, (1,))
        self.assertEqual(res[3].shape, ())
        self.assertEqual(res[3], 1)

    @prog_scope()
    def test_squeeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        out1 = paddle.squeeze(x1, axis=0)
        paddle.static.append_backward(out1.sum())

        x2 = paddle.full([], 3)
        x3 = paddle.full([], 0, dtype='int32')
        x2.stop_gradient = False
        out2 = paddle.squeeze(x2, axis=x3)
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                x2.grad_name,
            ],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, ())

    @prog_scope()
    def test_unsqueeze(self):
        x1 = paddle.full([], 2)
        x1.stop_gradient = False
        out1 = paddle.unsqueeze(x1, axis=0)
        paddle.static.append_backward(out1.sum())

        x2 = paddle.full([], 3)
        x3 = paddle.full([], 0, dtype='int32')
        x2.stop_gradient = False
        out2 = paddle.unsqueeze(x2, axis=x3)
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                x2.grad_name,
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
        paddle.static.append_backward(out)
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

        x_tensor = paddle.fluid.create_lod_tensor(
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
        x2 = paddle.static.data(name="x2", shape=[])

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            feed={
                "x1": np.array(1.0, dtype='float32'),
                "x2": np.array(100.5, dtype='float32'),
            },
            fetch_list=[
                x1.name,
                x2.name,
            ],
        )
        self.assertEqual(res[0].shape, [])
        self.assertEqual(res[0], np.array(1.0))
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[1], np.array(100.5))

    @prog_scope()
    def test_prelu(self):
        x1 = paddle.full([], 1.0, 'float32')
        x1.stop_gradient = False
        w1 = paddle.to_tensor([0.25], dtype='float32')
        out1 = paddle.nn.functional.prelu(x1, w1)
        paddle.static.append_backward(out1.sum())

        x2 = paddle.full([], 1.0, 'float32')
        x2.stop_gradient = False
        w2 = paddle.full([], 0.25, dtype='float32')
        out2 = paddle.nn.functional.prelu(x2, w2)
        paddle.static.append_backward(out2.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                out2,
                x1.grad_name,
                x2.grad_name,
                out1.grad_name,
                out2.grad_name,
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
        paddle.static.append_backward(out1.sum())

        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(
            prog,
            fetch_list=[
                out1,
                x1.grad_name,
                out1.grad_name,
            ],
        )

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, ())
        np.testing.assert_allclose(res[0], np.array(1))
        np.testing.assert_allclose(res[1], np.array(1))

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
            eleven = paddle.full([], 11, 'float32')
            x = paddle.static.data(name='x', shape=[], dtype='float32')
            x.stop_gradient = False
            out_i, out_x = paddle.static.nn.while_loop(cond, body, [i, x])
            paddle.static.append_backward(out_x)

        res = self.exe.run(
            main_program,
            feed={
                'i': np.array(1.0, dtype='float32'),
                'x': np.array(0.0, dtype='float32'),
            },
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

    @prog_scope()
    def test_shape(self):
        x = paddle.full([], 0.5)
        out = paddle.shape(x)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        np.testing.assert_array_equal(res[0], np.array([]))
        self.assertEqual(res[0].shape, (0,))

    def test_broadcast_tensors(self):
        # 1) x is 0D, y is 0D
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertEqual(out1.shape, ())
        self.assertEqual(out2.shape, ())

        # 2) x is ND , y is 0D
        x1 = paddle.full([2, 3], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertEqual(out1.shape, (2, 3))
        self.assertEqual(out2.shape, (2, 3))

        # 3) x is 0D , y is ND
        x1 = paddle.full([], 2.0)
        x1.stop_gradient = False
        x2 = paddle.full([2, 3], 2.0)
        x2.stop_gradient = False
        out1, out2 = paddle.broadcast_tensors([x1, x2])

        self.assertEqual(out1.shape, (2, 3))
        self.assertEqual(out2.shape, (2, 3))

    @prog_scope()
    def test_linalg_slogdet(self):
        # 2-D input
        x = paddle.randn([3, 3])
        x.stop_gradient = False
        out = paddle.linalg.slogdet(x)
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name])
        self.assertEqual(res[0].shape, (2,))
        self.assertEqual(res[1].shape, (3, 3))

        # 3-D input
        x1 = paddle.randn([3, 3, 3])
        x1.stop_gradient = False
        out1 = paddle.linalg.slogdet(x1)
        paddle.static.append_backward(out1.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out1, x1.grad_name])
        self.assertEqual(res[0].shape, (2, 3))
        self.assertEqual(res[1].shape, (3, 3, 3))

    @prog_scope()
    def test_multi_dot(self):
        a = paddle.randn([4])
        a.stop_gradient = False
        b = paddle.randn([4, 5])
        b.stop_gradient = False
        c = paddle.randn([5])
        c.stop_gradient = False

        out = paddle.linalg.multi_dot([a, b, c])
        paddle.static.append_backward(out.sum())
        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[out, a.grad_name, b.grad_name, c.grad_name]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (4,))
        self.assertEqual(res[2].shape, (4, 5))
        self.assertEqual(res[3].shape, (5,))

    @prog_scope()
    def test_dist(self):
        x = paddle.to_tensor([[3, 3], [3, 3]], dtype="float32")
        y = paddle.to_tensor([[3, 3], [3, 1]], dtype="float32")
        x.stop_gradient = False
        y.stop_gradient = False
        out = paddle.dist(x, y)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name, y.grad_name])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 2))
        np.testing.assert_array_equal(res[0], np.array(2).astype(np.float32))

    @prog_scope()
    def test_trace(self):
        x = paddle.to_tensor([[3, 2], [1, 9]], dtype="float32")
        x.stop_gradient = False
        out = paddle.trace(x)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, x.grad_name])

        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 2))
        np.testing.assert_allclose(res[0], np.array(12))

    @prog_scope()
    def test_cond(self):
        pass
        # use paddle.sum, paddle.max, paddle.min
        # x = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])
        # x.stop_gradient = False
        # out = paddle.linalg.cond(x)
        # paddle.static.append_backward(out)

        # prog = paddle.static.default_main_program()
        # res = self.exe.run(prog, fetch_list=[out, x.grad_name])

        # self.assertTrue(res[0].shape, ())
        # self.assertTrue(res[1].shape, (3, 3))
        # np.testing.assert_allclose(out, np.array(1.41421342))

    @prog_scope()
    def test_cov(self):
        xt_1 = paddle.randn((12,))
        xt_1.stop_gradient = False

        out = paddle.linalg.cov(xt_1)
        paddle.static.append_backward(out)

        prog = paddle.static.default_main_program()

        res = self.exe.run(prog, fetch_list=[out, xt_1.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (12,))

    @prog_scope()
    def test_det(self):
        xt_1 = paddle.randn((3, 3))
        xt_1.stop_gradient = False

        out = paddle.linalg.det(xt_1)
        paddle.static.append_backward(out.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out, xt_1.grad_name])
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (3, 3))


# Use to test API whose zero-dim input tensors don't have grad and not need to test backward in OpTest.
class TestNoBackwardAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.shape = [
            paddle.full([], 2, 'int32'),
            paddle.full([], 3, 'int32'),
            paddle.full([], 4, 'int32'),
        ]

    def test_slice(self):
        starts = [paddle.full([], 1, 'int32'), paddle.full([], 1, 'int32')]
        ends = [paddle.full([], 3, 'int32'), paddle.full([], 3, 'int32')]
        x = paddle.rand([5, 3, 3])
        out = paddle.slice(x, [1, 2], starts, ends)
        self.assertEqual(out.shape, [5, 2, 2])

    def test_strided_slice(self):
        starts = [paddle.full([], 0, 'int32'), paddle.full([], 0, 'int32')]
        ends = [paddle.full([], 4, 'int32'), paddle.full([], 4, 'int32')]
        strides = [paddle.full([], 2, 'int32'), paddle.full([], 2, 'int32')]
        x = paddle.rand([5, 5, 5])
        out = paddle.strided_slice(x, [1, 2], starts, ends, strides)
        self.assertEqual(out.shape, [5, 2, 2])

    def test_linspace(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 5.0)
        num = paddle.full([], 5, 'int32')
        out = paddle.linspace(start, stop, num)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_arange(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 6.0)
        step = paddle.full([], 1.0)
        out = paddle.arange(start, stop, step)
        np.testing.assert_array_equal(out.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_normal(self):
        mean = paddle.full([], 0.0)
        std = paddle.full([], 0.0)
        out = paddle.normal(mean, std)
        self.assertEqual(out.shape, [])

        out = paddle.normal(0.0, 1.0, [])
        self.assertEqual(out.shape, [])

        out = paddle.normal(0.0, 1.0, self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_rand(self):
        out = paddle.rand([])
        self.assertEqual(out.shape, [])

        out = paddle.rand(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_randn(self):
        out = paddle.randn([])
        self.assertEqual(out.shape, [])

        out = paddle.randn(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_randint_and_randint_like(self):
        out = paddle.randint(-10, 10, [])
        self.assertEqual(out.shape, [])

        out = paddle.randint_like(out, -10, 10)
        self.assertEqual(out.shape, [])

        out = paddle.randint(-10, 10, self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_standard_normal(self):
        out = paddle.standard_normal([])
        self.assertEqual(out.shape, [])

        out = paddle.standard_normal(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_uniform(self):
        out = paddle.uniform([])
        self.assertEqual(out.shape, [])

        out = paddle.uniform(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_empty_and_empty_like(self):
        out = paddle.empty([])
        self.assertEqual(out.shape, [])

        out = paddle.empty_like(out)
        self.assertEqual(out.shape, [])

        out = paddle.empty(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_full_and_full_like(self):
        out = paddle.full([], 0.5)
        self.assertEqual(out.shape, [])

        out = paddle.full_like(out, 0.5)
        self.assertEqual(out.shape, [])

        out = paddle.full(self.shape, 0.5)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_ones_and_ones_like(self):
        out = paddle.ones([])
        self.assertEqual(out.shape, [])

        out = paddle.ones_like(out)
        self.assertEqual(out.shape, [])

        out = paddle.ones(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_zeros_and_zeros_like(self):
        out = paddle.zeros([])
        self.assertEqual(out.shape, [])

        out = paddle.zeros_like(out)
        self.assertEqual(out.shape, [])

        out = paddle.zeros(self.shape)
        self.assertEqual(out.shape, [2, 3, 4])

    def test_embedding(self):
        ids = paddle.full(shape=[], fill_value=1, dtype='int64')
        w0 = paddle.arange(3, 9).reshape((3, 2)).astype(paddle.float32)
        w = paddle.to_tensor(w0, stop_gradient=False)
        emb = paddle.nn.functional.embedding(
            x=ids, weight=w, sparse=True, name="embedding"
        )
        self.assertEqual(emb.shape, [2])
        res = [5.0, 6.0]
        for i in range(len(res)):
            self.assertEqual(emb.numpy()[i], res[i])

    def test_one_hot_label(self):
        label = paddle.full(shape=[], fill_value=2, dtype='int64')
        one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
        self.assertEqual(one_hot_label.shape, [4])
        self.assertEqual(one_hot_label.numpy()[2], 1)

    def test_unique_consecutive(self):
        places = ['cpu']
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            x = paddle.rand([])
            y, inverse, counts = paddle.unique_consecutive(
                x,
                return_inverse=True,
                return_counts=True,
            )

            self.assertEqual(y, x)
            self.assertEqual(inverse, 0)
            self.assertEqual(counts, 1)
            self.assertEqual(y.shape, [1])
            self.assertEqual(inverse.shape, [1])
            self.assertEqual(counts.shape, [1])

    def test_unique(self):
        places = ['cpu']
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            x = paddle.rand([])
            y, index, inverse, counts = paddle.unique(
                x,
                return_index=True,
                return_inverse=True,
                return_counts=True,
            )

            self.assertEqual(y, x)
            self.assertEqual(index, 0)
            self.assertEqual(inverse, 0)
            self.assertEqual(counts, 1)
            self.assertEqual(y.shape, [1])
            self.assertEqual(index.shape, [1])
            self.assertEqual(inverse.shape, [1])
            self.assertEqual(counts.shape, [1])

    def test_matrix_rank(self):
        x = paddle.eye(10)
        x.stop_gradient = False
        out = paddle.linalg.matrix_rank(x)

        self.assertEqual(out.shape, [])
        np.testing.assert_equal(out, np.array(10))

        c = paddle.ones(shape=[3, 4, 5])
        c.stop_gradient = False
        out_c = paddle.linalg.matrix_rank(c)
        self.assertEqual(out_c.shape, [3])
        np.testing.assert_equal(out_c, np.array([1, 1, 1]))

        # 2D, tol->float : OUTPUT 0D
        x_tol = paddle.eye(10)
        x_tol.stop_gradient = False
        out_tol = paddle.linalg.matrix_rank(x_tol, tol=0.1)
        self.assertEqual(out_tol.shape, [])

        # 3D, tol->float : OUTPUT 1D
        c_tol = paddle.ones(shape=[3, 4, 5])
        c_tol.stop_gradient = False
        out_c_tol = paddle.linalg.matrix_rank(c_tol, tol=0.1)
        self.assertEqual(out_c_tol.shape, [3])

        tol_2 = paddle.randn([2])
        # 2D, tol->Tensor[1,2] : OUTPUT 1D
        d = paddle.eye(10)
        out_d = paddle.linalg.matrix_rank(d, tol=tol_2)
        self.assertEqual(out_d.shape, [2])


class TestNoBackwardAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()
        self.shape = [
            paddle.full([], 2, 'int32'),
            paddle.full([], 3, 'int32'),
            paddle.full([], 4, 'int32'),
        ]

    def test_slice(self):
        starts = [paddle.full([], 1, 'int32'), paddle.full([], 1, 'int32')]
        ends = [paddle.full([], 3, 'int32'), paddle.full([], 3, 'int32')]
        x = paddle.rand([5, 3, 3])
        out = paddle.slice(x, [1, 2], starts, ends)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        self.assertEqual(res.shape, (5, 2, 2))

    def test_strided_slice(self):
        starts = [paddle.full([], 0, 'int32'), paddle.full([], 0, 'int32')]
        ends = [paddle.full([], 4, 'int32'), paddle.full([], 4, 'int32')]
        strides = [paddle.full([], 2, 'int32'), paddle.full([], 2, 'int32')]
        x = paddle.rand([5, 5, 5])
        out = paddle.strided_slice(x, [1, 2], starts, ends, strides)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        self.assertEqual(res.shape, (5, 2, 2))

    def test_linspace(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 5.0)
        num = paddle.full([], 5, 'int32')
        out = paddle.linspace(start, stop, num)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        np.testing.assert_array_equal(res, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_arange(self):
        start = paddle.full([], 1.0)
        stop = paddle.full([], 6.0)
        step = paddle.full([], 1.0)
        out = paddle.arange(start, stop, step)
        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out]
        )[0]
        np.testing.assert_array_equal(res, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_normal(self):
        mean = paddle.full([], 0.0)
        std = paddle.full([], 0.0)
        out1 = paddle.normal(mean, std)
        out2 = paddle.normal(0.0, 1.0, [])
        out3 = paddle.normal(0.0, 1.0, self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2, out3]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))

    def test_rand(self):
        out1 = paddle.rand([])
        out2 = paddle.rand(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 3, 4))

    def test_randn(self):
        out1 = paddle.randn([])
        out2 = paddle.randn(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 3, 4))

    def test_randint_and_randint_like(self):
        out1 = paddle.randint(-10, 10, [])
        out2 = paddle.randint_like(out1, -10, 10)
        out3 = paddle.randint(-10, 10, self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2, out3]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))

    def test_standard_normal(self):
        out1 = paddle.standard_normal([])
        out2 = paddle.standard_normal(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 3, 4))

    def test_uniform(self):
        out1 = paddle.uniform([])
        out2 = paddle.uniform(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, (2, 3, 4))

    def test_empty_and_empty_like(self):
        out1 = paddle.empty([])
        out2 = paddle.empty_like(out1)
        out3 = paddle.empty(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2, out3]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))

    def test_full_and_full_like(self):
        out1 = paddle.full([], 0.5)
        out2 = paddle.full_like(out1, 0.5)
        out3 = paddle.full(self.shape, 0.5)
        out4 = paddle.full(self.shape, paddle.full([], 0.5))

        res = self.exe.run(
            paddle.static.default_main_program(),
            fetch_list=[out1, out2, out3, out4],
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))
        self.assertEqual(res[3].shape, (2, 3, 4))

    def test_ones_and_ones_like(self):
        out1 = paddle.ones([])
        out2 = paddle.ones_like(out1)
        out3 = paddle.ones(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2, out3]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))

    def test_zeros_and_zeros_like(self):
        out1 = paddle.zeros([])
        out2 = paddle.zeros_like(out1)
        out3 = paddle.zeros(self.shape)

        res = self.exe.run(
            paddle.static.default_main_program(), fetch_list=[out1, out2, out3]
        )
        self.assertEqual(res[0].shape, ())
        self.assertEqual(res[1].shape, ())
        self.assertEqual(res[2].shape, (2, 3, 4))

    def test_embedding(self):
        ids = paddle.full(shape=[], fill_value=1, dtype='int64')
        w0 = paddle.arange(3, 9).reshape((3, 2)).astype(paddle.float32)
        w = paddle.to_tensor(w0, stop_gradient=False)
        emb = paddle.nn.functional.embedding(
            x=ids, weight=w, sparse=True, name="embedding"
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[emb])
        self.assertEqual(res[0].shape, (2,))
        result = [5.0, 6.0]
        for i in range(len(res)):
            self.assertEqual(res[0][i], result[i])

    def test_static_embedding(self):
        ids = paddle.full(shape=[], fill_value=1, dtype='int64')
        emb = paddle.static.nn.embedding(ids, (20, 3))
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[emb])
        self.assertEqual(res[0].shape, (3,))

    def test_one_hot_label(self):
        label = paddle.full(shape=[], fill_value=2, dtype='int64')
        one_hot_label = paddle.nn.functional.one_hot(label, num_classes=4)
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[one_hot_label])

        self.assertEqual(res[0].shape, (4,))
        self.assertEqual(res[0][2], 1)

    def test_unique_consecutive(self):
        x = paddle.rand([])
        y, inverse, counts = paddle.unique_consecutive(
            x, return_inverse=True, return_counts=True
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[y, inverse, counts])
        self.assertEqual(y, x)
        self.assertEqual(inverse, 0)
        self.assertEqual(counts, 1)
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[2].shape, (1,))

    def test_unique(self):
        x = paddle.rand([])
        y, index, inverse, counts = paddle.unique(
            x, return_index=True, return_inverse=True, return_counts=True
        )

        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[y, index, inverse, counts])
        self.assertEqual(y, x)
        self.assertEqual(index, 0)
        self.assertEqual(inverse, 0)
        self.assertEqual(counts, 1)
        self.assertEqual(res[0].shape, (1,))
        self.assertEqual(res[1].shape, (1,))
        self.assertEqual(res[2].shape, (1,))
        self.assertEqual(res[3].shape, (1,))

    @prog_scope()
    def test_static_matrix_rank(self):
        # 2D : OUTPUT 0D
        x = paddle.eye(10)
        x.stop_gradient = False
        out = paddle.linalg.matrix_rank(x)
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out])
        self.assertEqual(res[0].shape, ())

        # 3D : OUTPUT 1D
        c = paddle.ones(shape=[3, 4, 5])
        c.stop_gradient = False
        out_c = paddle.linalg.matrix_rank(c)
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[out_c])
        self.assertEqual(res[0].shape, (3,))

        # 2D, tol->float : OUTPUT 0D
        x_tol = paddle.eye(10)
        x_tol.stop_gradient = False
        out_tol = paddle.linalg.matrix_rank(x_tol, tol=0.1)
        prog = paddle.static.default_main_program()
        res = self.exe.run(prog, fetch_list=[out_tol])
        self.assertEqual(res[0].shape, ())

        # 3D, tol->float : OUTPUT 1D
        c_tol = paddle.ones(shape=[3, 4, 5])
        c_tol.stop_gradient = False
        out_c_tol = paddle.linalg.matrix_rank(c_tol, tol=0.1)
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[out_c_tol])
        self.assertEqual(res[0].shape, (3,))

        tol_2 = paddle.randn([2])
        # 2D, tol->Tensor[1,2] : OUTPUT 1D
        d = paddle.eye(10)
        out_d = paddle.linalg.matrix_rank(d, tol=tol_2)
        prog = paddle.static.default_main_program()
        self.exe.run(paddle.static.default_startup_program())
        res = self.exe.run(prog, fetch_list=[out_d])
        self.assertEqual(res[0].shape, (2,))


unary_apis_with_complex_input = [
    paddle.real,
    paddle.imag,
    paddle.angle,
    paddle.conj,
]


class TestUnaryElementwiseAPIWithComplexInput(unittest.TestCase):
    def test_dygraph_unary(self):
        paddle.disable_static()
        for api in unary_apis_with_complex_input:
            x = paddle.rand([]) + 1j * paddle.rand([])
            x.stop_gradient = False
            x.retain_grads()
            out = api(x)
            out.retain_grads()
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

        paddle.enable_static()

    def test_static_unary(self):
        paddle.enable_static()
        for api in unary_apis_with_complex_input:
            main_prog = paddle.static.Program()
            block = main_prog.global_block()
            exe = paddle.static.Executor()
            with paddle.static.program_guard(
                main_prog, paddle.static.Program()
            ):
                x = paddle.complex(paddle.rand([]), paddle.rand([]))
                x.stop_gradient = False
                out = api(x)
                paddle.static.append_backward(out)

                fetch_list = [x, out]
                if block.has_var(x.grad_name):
                    fetch_list.extend([x.grad_name, out.grad_name])

                # 1) Test Program
                res = exe.run(main_prog, fetch_list=fetch_list)
                for item in res:
                    self.assertEqual(item.shape, ())

                # 2) Test CompiledProgram Program
                compile_prog = paddle.static.CompiledProgram(main_prog)
                res = exe.run(compile_prog, fetch_list=fetch_list)
                for item in res:
                    self.assertEqual(item.shape, ())

        paddle.disable_static()


class TestAsReal(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.rand([]) + 1j * paddle.rand([])
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.as_real(x)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.shape, [])
        self.assertEqual(out.shape, [2])
        if x.grad is not None:
            self.assertEqual(x.grad.shape, [])
            self.assertEqual(out.grad.shape, [2])

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()

        main_prog = paddle.static.Program()
        block = main_prog.global_block()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.complex(paddle.rand([]), paddle.rand([]))
            x.stop_gradient = False
            out = paddle.as_real(x)
            self.assertEqual(x.shape, ())
            self.assertEqual(out.shape, (2,))
            paddle.static.append_backward(out.sum())

            fetch_list = [x, out]
            if block.has_var(x.grad_name):
                fetch_list.extend([x.grad_name, out.grad_name])

            res = exe.run(main_prog, fetch_list=fetch_list)
            self.assertEqual(res[0].shape, ())
            self.assertEqual(res[1].shape, (2,))
            self.assertEqual(res[2].shape, ())
            self.assertEqual(res[3].shape, (2,))

        paddle.disable_static()


class TestAsComplex(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.rand([2])
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.as_complex(x)
        out.retain_grads()
        out.backward()

        self.assertEqual(x.shape, [2])
        self.assertEqual(out.shape, [])
        if x.grad is not None:
            self.assertEqual(x.grad.shape, [2])
            self.assertEqual(out.grad.shape, [])

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        block = main_prog.global_block()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(main_prog, paddle.static.Program()):
            x = paddle.rand([2])
            x.stop_gradient = False
            out = paddle.as_complex(x)
            self.assertEqual(x.shape, (2,))
            self.assertEqual(out.shape, ())
            paddle.static.append_backward(out.sum())

            fetch_list = [x, out]
            if block.has_var(x.grad_name):
                fetch_list.extend([x.grad_name, out.grad_name])

            res = exe.run(main_prog, fetch_list=fetch_list)
            self.assertEqual(res[0].shape, (2,))
            self.assertEqual(res[1].shape, ())
            self.assertEqual(res[2].shape, (2,))
            self.assertEqual(res[3].shape, ())

        paddle.disable_static()


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.x = paddle.full([], 2.0)

    def test_Categorical(self):
        logits = paddle.rand([6])
        d = paddle.distribution.Categorical(logits)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.probs(paddle.full([], 2, dtype='int64')).shape, [])
        self.assertEqual(
            d.log_prob(paddle.full([], 2, dtype='int64')).shape, []
        )
        # because use paddle.sum
        # self.assertEqual(d.entropy().shape, [])

    def test_Normal(self):
        normal = paddle.distribution.Normal(0.0, 3.0)
        self.assertEqual(normal.sample([]).shape, [])
        self.assertEqual(normal.rsample([]).shape, [])
        self.assertEqual(normal.mean.shape, [])
        self.assertEqual(normal.variance.shape, [])
        self.assertEqual(normal.probs(self.x).shape, [])
        self.assertEqual(normal.log_prob(self.x).shape, [])
        self.assertEqual(normal.entropy().shape, [])

        normal = paddle.distribution.Normal(
            paddle.full([], 0.0), paddle.full([], 3.0)
        )
        self.assertEqual(normal.sample([]).shape, [])
        self.assertEqual(normal.rsample([]).shape, [])
        self.assertEqual(normal.mean.shape, [])
        self.assertEqual(normal.variance.shape, [])
        self.assertEqual(normal.probs(self.x).shape, [])
        self.assertEqual(normal.log_prob(self.x).shape, [])
        self.assertEqual(normal.entropy().shape, [])

    def test_Uniform(self):
        uniform = paddle.distribution.Uniform(0.0, 1.0)
        self.assertEqual(uniform.sample([]).shape, [])
        self.assertEqual(uniform.probs(self.x).shape, [])
        self.assertEqual(uniform.log_prob(self.x).shape, [])
        self.assertEqual(uniform.entropy().shape, [])

        uniform = paddle.distribution.Uniform(
            paddle.full([], 0.0), paddle.full([], 1.0)
        )
        self.assertEqual(uniform.sample([]).shape, [])
        self.assertEqual(uniform.probs(self.x).shape, [])
        self.assertEqual(uniform.log_prob(self.x).shape, [])
        self.assertEqual(uniform.entropy().shape, [])

    def test_Beta(self):
        beta = paddle.distribution.Beta(alpha=0.5, beta=0.5)
        self.assertEqual(beta.sample([]).shape, [])
        self.assertEqual(beta.mean.shape, [])
        self.assertEqual(beta.variance.shape, [])
        # because use paddle.sum
        # self.assertEqual(beta.prob(self.x).shape, [])
        # self.assertEqual(beta.log_prob(self.x).shape, [])
        # self.assertEqual(beta.entropy().shape, [])

    def test_kl_divergence(self):
        p = paddle.distribution.Beta(alpha=0.5, beta=0.5)
        q = paddle.distribution.Beta(alpha=0.2, beta=1.0)
        kl = paddle.distribution.kl_divergence(p, q)
        self.assertEqual(kl.shape, [])

    def test_TransformedDistribution(self):
        d = paddle.distribution.TransformedDistribution(
            paddle.distribution.Normal(0.0, 1.0),
            [
                paddle.distribution.AffineTransform(
                    paddle.full([], 1.0), paddle.full([], 2.0)
                )
            ],
        )
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])

    def test_Laplace(self):
        d = paddle.distribution.Laplace(0.0, 1.0)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.stddev.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])
        self.assertEqual(d.icdf(self.x).shape, [])
        self.assertEqual(d.entropy().shape, [])

    def test_LogNormal(self):
        d = paddle.distribution.LogNormal(0.0, 1.0)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.entropy().shape, [])
        self.assertEqual(d.probs(self.x).shape, [])

    def test_Gumbel(self):
        d = paddle.distribution.Gumbel(0.0, 1.0)
        self.assertEqual(d.sample([]).shape, [])
        self.assertEqual(d.rsample([]).shape, [])
        self.assertEqual(d.mean.shape, [])
        self.assertEqual(d.variance.shape, [])
        self.assertEqual(d.stddev.shape, [])
        self.assertEqual(d.prob(self.x).shape, [])
        self.assertEqual(d.log_prob(self.x).shape, [])
        self.assertEqual(d.cdf(self.x).shape, [])
        self.assertEqual(d.entropy().shape, [])

    def test_Multinomial(self):
        d = paddle.distribution.Multinomial(
            10, paddle.to_tensor([0.2, 0.3, 0.5])
        )
        # because use paddle.sum
        # self.assertEqual(d.prob(self.x).shape, [])
        # self.assertEqual(d.log_prob(self.x).shape, [])
        # self.assertEqual(d.entropy().shape, [])


class TestLossAPI(unittest.TestCase):
    def test_sigmoid_focal_loss(self):
        logit = paddle.to_tensor(
            [[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]],
            dtype='float32',
            stop_gradient=False,
        )
        logit.retain_grads()
        label = paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32'
        )
        fg_num_0 = paddle.full([], 2.0)
        fg_num_1 = paddle.full([1], 2.0)

        out0 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_0, reduction='mean'
        )
        out1 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_1, reduction='mean'
        )
        out0.retain_grads()

        np.testing.assert_array_equal(
            out0.numpy(),
            out1.numpy(),
        )

        out0.backward()
        self.assertEqual(out0.shape, [])
        self.assertEqual(out1.shape, [])
        self.assertEqual(out0.grad.shape, [])
        self.assertEqual(logit.grad.shape, [2, 3])


class TestLossAPIStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.exe = paddle.static.Executor()

    @prog_scope()
    def test_sigmoid_focal_loss(self):
        logit = paddle.rand([2, 3])
        logit.stop_gradient = False

        label = paddle.randint(0, 1, [2, 3]).astype('float32')
        label.stop_gradient = False

        fg_num_0 = paddle.full([], 2.0)
        fg_num_1 = paddle.full([1], 2.0)

        out0 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_0, reduction='mean'
        )
        out1 = F.sigmoid_focal_loss(
            logit, label, normalizer=fg_num_1, reduction='mean'
        )
        paddle.static.append_backward(out0.sum())

        prog = paddle.static.default_main_program()
        res = self.exe.run(
            prog, fetch_list=[out0, out1, out0.grad_name, logit.grad_name]
        )
        np.testing.assert_allclose(res[0], res[1])
        # because static use paddle.mean
        # self.assertEqual(res[0].shape, ())
        # self.assertEqual(res[1].shape, ())
        # self.assertEqual(res[2].shape, ())
        self.assertEqual(res[3].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
