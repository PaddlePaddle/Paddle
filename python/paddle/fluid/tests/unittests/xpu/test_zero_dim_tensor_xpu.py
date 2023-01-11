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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F

paddle.set_device('xpu')


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
            x.retain_grads()
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
    def test_dygraph(self):
        paddle.disable_static()
        for api in reduce_api_list:
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, []).astype('bool')
                out = api(x, None)
                self.assertEqual(x.shape, [])
                self.assertEqual(out.shape, [])
            else:
                x = paddle.rand([])
                x.stop_gradient = False
                x.retain_grads()
                out = api(x, None)
                out.retain_grads()
                out.backward()

                self.assertEqual(x.shape, [])
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.shape, [])
                self.assertEqual(out.grad.shape, [])

        paddle.enable_static()


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
]

binary_int_api_list = [
    paddle.bitwise_and,
    paddle.bitwise_or,
    paddle.bitwise_xor,
]


# Use to test zero-dim of binary API
class TestBinaryAPI(unittest.TestCase):
    def test_dygraph_binary(self):
        paddle.disable_static()
        for api in binary_api_list:
            # 1) x/y is 0D
            x = paddle.rand([])
            y = paddle.rand([])
            x.stop_gradient = False
            y.stop_gradient = False
            x.retain_grads()
            y.retain_grads()
            if isinstance(api, dict):
                out = api['func'](x, y)
                out_cls = getattr(paddle.Tensor, api['cls_method'])(x, y)
                np.testing.assert_array_equal(out_cls.numpy(), out.numpy())
            else:
                out = api(x, y)
            out.retain_grads()
            self.assertEqual(out.shape, [])

            out.backward()
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [])

            # 2) x is not 0D , y is 0D
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
            self.assertEqual(out.shape, [2, 3, 4])

            out.backward()
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [2, 3, 4])
                self.assertEqual(y.grad.shape, [])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 3) x is 0D , y is not 0D
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
            self.assertEqual(out.shape, [2, 3, 4])

            out.backward()
            if x.grad is not None:
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [2, 3, 4])
                self.assertEqual(out.grad.shape, [2, 3, 4])

            # 4) x is 0D , y is scalar
            x = paddle.rand([])
            y = 0.5
            x.stop_gradient = False
            if isinstance(api, dict):
                out = getattr(paddle.Tensor, api['cls_method'])(x, y)
                self.assertEqual(out.shape, [])

        for api in binary_int_api_list:
            # 1) x/y is 0D
            x = paddle.randint(-10, 10, [])
            y = paddle.randint(-10, 10, [])
            out = api(x, y)
            self.assertEqual(out.shape, [])

            # 2) x is not 0D , y is 0D
            x = paddle.randint(-10, 10, [3, 5])
            y = paddle.randint(-10, 10, [])
            out = api(x, y)
            self.assertEqual(out.shape, [3, 5])

            # 3) x is 0D , y is not 0D
            x = paddle.randint(-10, 10, [])
            y = paddle.randint(-10, 10, [3, 5])
            out = api(x, y)
            self.assertEqual(out.shape, [3, 5])

        paddle.enable_static()


# Use to test zero-dim of Sundry API, which is unique and can not be classified
# with others. It can be implemented here flexibly.
class TestSundryAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x = paddle.rand([])

    def test_linear(self):
        x = paddle.randn([3, 2])
        w = paddle.full(shape=[2, 4], fill_value=0.5)
        b = paddle.zeros([])

        np.testing.assert_array_equal(
            F.linear(x, w, b).numpy(), F.linear(x, w).numpy()
        )

    def test_is_floating_point(self):
        self.assertTrue(paddle.is_floating_point(self.x))

    def test_is_integer(self):
        x = paddle.randint(0, 10, [])
        self.assertTrue(paddle.is_integer(x))

    def test_is_tensor(self):
        self.assertTrue(paddle.is_tensor(self.x))

    def test_is_empty(self):
        x = paddle.rand([3, 0, 5])
        self.assertTrue(paddle.is_empty(x))

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
        np.testing.assert_array_equal(x.numpy(), np.array(0.5))

    def test_numel(self):
        out = paddle.numel(self.x)
        self.assertEqual(out.shape, [])
        np.testing.assert_array_equal(out.numpy(), np.array(1))

    def test_rank(self):
        out = paddle.rank(self.x)
        self.assertEqual(out.shape, [])
        np.testing.assert_array_equal(out.numpy(), np.array(0))

    def test_shape(self):
        out = paddle.shape(self.x)
        self.assertEqual(out.shape, [0])
        np.testing.assert_array_equal(out.numpy(), np.array([]))

    def test_pow_factor(self):
        x = paddle.rand([])
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.pow(x, 2.0)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_cast(self):
        x = paddle.full([], 1.0, 'float32')
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.cast(x, 'int32')
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_clip(self):
        x = paddle.uniform([], None, -10, 10)
        x.stop_gradient = False
        out = paddle.clip(x, -5, 5)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

    def test_increment(self):
        x = paddle.rand([])
        x.stop_gradient = False
        out = paddle.increment(x, 1.0)
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
        x = paddle.to_tensor([1, 3, 5, 7, 9])
        y = paddle.rand([])

        # only has forward kernel
        out = paddle.searchsorted(x, y)

        self.assertEqual(out.shape, [])
        self.assertEqual(out.numpy(), 0)

    def test_gather_1D(self):
        x = paddle.to_tensor([1.0, 3.0, 5.0, 7.0, 9.0], stop_gradient=False)
        index = paddle.full([], 2, 'int64')
        out = paddle.gather(x, index)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.numpy(), 5)
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
        for i in range(3):
            self.assertEqual(out.numpy()[i], x.numpy()[1][i])
        self.assertEqual(out.grad.shape, [3])

    def test_gather_xD_axis_1(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = paddle.full([], 1, 'int64')
        out = paddle.gather(x, index, axis=1)

        self.assertEqual(out.shape, [2])
        for i in range(2):
            self.assertEqual(out.numpy()[i], x.numpy()[i][1])

    def test_scatter_1D(self):
        x = paddle.to_tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        index = paddle.full([], 2, 'int64')
        updates = paddle.full([], 4.0)
        out = paddle.scatter(x, index, updates)

        self.assertEqual(out.numpy()[2], 4)

    def test_scatter_XD(self):
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        index = paddle.full([], 1, 'int64')
        updates = paddle.to_tensor([1.0, 2.0, 3.0])
        out = paddle.scatter(x, index, updates)

        for i in range(3):
            self.assertEqual(out.numpy()[1][i], updates.numpy()[i])

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

        for i in range(3):
            self.assertEqual(out.numpy()[1][i], updates.numpy()[i])

    def test_flatten(self):
        x = paddle.full([], 1, 'float32')
        x.stop_gradient = False

        start_axis = 0
        stop_axis = -1

        out = paddle.flatten(x, start_axis=start_axis, stop_axis=stop_axis)
        out.backward()

        self.assertEqual(out.shape, [1])
        self.assertEqual(x.grad.shape, [])

    def test_scale(self):
        x = paddle.rand([])
        x.stop_gradient = False
        x.retain_grads()
        out = paddle.scale(x, scale=2.0, bias=1.0)
        out.retain_grads()
        out.backward()

        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])
        self.assertEqual(x.grad.shape, [])

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

    def test_reshape_list(self):
        x = paddle.rand([])
        x.stop_gradient = False
        x.retain_grads()

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
        x.retain_grads()

        out = paddle.reshape(x, [])
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [])
        self.assertEqual(out.grad.shape, [])

        new_shape = paddle.full([], 1, "int32")
        out = paddle.reshape(x, new_shape)
        out.retain_grads()
        out.backward()
        self.assertEqual(x.grad.shape, [1, 1])
        self.assertEqual(out.shape, [1])
        self.assertEqual(out.grad.shape, [1])

        new_shape = paddle.full([], -1, "int32")
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

    def test_sigmoid_focal_loss(self):
        logit = paddle.to_tensor(
            [[0.97, 0.91, 0.03], [0.55, 0.43, 0.71]],
            dtype='float32',
            stop_gradient=False,
        )
        label = paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32'
        )
        fg_num_0 = paddle.full([], 2.0)
        fg_num_1 = paddle.full([1], 2.0)

        out0 = F.sigmoid_focal_loss(logit, label, normalizer=fg_num_0)
        out1 = F.sigmoid_focal_loss(logit, label, normalizer=fg_num_1)

        np.testing.assert_array_equal(
            out0.numpy(),
            out1.numpy(),
        )

        out0.backward()
        self.assertEqual(out0.grad.shape, [1])
        self.assertEqual(logit.grad.shape, [2, 3])

    def test_allclose(self):
        x = paddle.full([], 0.5)
        y = paddle.full([], 0.6)
        self.assertFalse(paddle.allclose(x, y))


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


if __name__ == "__main__":
    unittest.main()
