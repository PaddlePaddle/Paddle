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

import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
import numpy as np
import unittest


unary_api_list = [
    paddle.nn.functional.elu,
    paddle.nn.functional.gelu,
    paddle.nn.functional.hardsigmoid,
    paddle.nn.functional.hardswish,
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
    paddle.nn.functional.mish,
    paddle.nn.functional.silu,
    paddle.nn.functional.tanh,
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
]


# Use to test zero-dim in unary API.
class TestUnaryAPI(unittest.TestCase):
    def test_dygraph_unary(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in unary_api_list:
            x = paddle.rand([])
            x.stop_gradient = False
            out = api(x)
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            self.assertEqual(x.grad.shape, [])
            self.assertEqual(out.grad.shape, [])

        paddle.enable_static()

    def test_static_unary(self):
        paddle.enable_static()

        for api in unary_api_list:
            main_prog = fluid.Program()
            with fluid.program_guard(main_prog, fluid.Program()):
                x = paddle.rand([])
                x.stop_gradient = False
                out = api(x)
                fluid.backward.append_backward(out)

                # ScaleLossGradOp / append_backward always set grad shape to [1]
                prog = paddle.static.default_main_program()
                block = prog.global_block()

                x_grad = block.var(fluid.framework.grad_var_name(x.name))
                out_grad = block.var(fluid.framework.grad_var_name(out.name))

                # Test compile shape, grad is always [1]
                self.assertEqual(x.shape, ())
                self.assertEqual(out.shape, ())

                exe = fluid.Executor()
                result = exe.run(
                    main_prog, fetch_list=[x, out, x_grad, out_grad]
                )

                # Test runtime shape
                self.assertEqual(result[0].shape, ())
                self.assertEqual(result[1].shape, ())
                self.assertEqual(result[3].shape, (1,))

                # 0D will be stacked when 1+ place, due to it cannot be concated
                # for 1 place: [ x-place1 ]
                # for 1+ place: [ paddle.stack([x-place1, x_place2...]) ]
                if paddle.device.is_compiled_with_cuda():
                    places = [paddle.CUDAPlace(0)]
                    device_num = 1
                    expect_shape = ()
                else:
                    places = [paddle.CPUPlace()] * 4
                    device_num = 4
                    expect_shape = (device_num,)

                compiled_program = fluid.CompiledProgram(
                    main_prog
                ).with_data_parallel(out.name, places=places)
                result = exe.run(
                    compiled_program,
                    fetch_list=[x, out, x_grad, out_grad],
                    return_merged=True,
                )

                # Test runtime parallel shape
                self.assertEqual(result[0].shape, expect_shape)
                self.assertEqual(result[1].shape, expect_shape)
                self.assertEqual(result[3].shape, (device_num,))

                compiled_program = fluid.CompiledProgram(
                    main_prog
                ).with_data_parallel(out.name, places=places)
                result = exe.run(
                    compiled_program,
                    fetch_list=[x, out, x_grad, out_grad],
                    return_merged=False,
                )

                # [[x-place1, x-place2, ...], [], [], ...]
                self.assertEqual(np.array(result[0]).shape, (device_num,))
                self.assertEqual(np.array(result[1]).shape, (device_num,))
                self.assertEqual(np.array(result[3]).shape, (device_num, 1))

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
    def test_dygraph(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in reduce_api_list:
            if api in [paddle.all, paddle.any]:
                x = paddle.randint(0, 2, []).astype('bool')
                out = api(x, None)
                self.assertEqual(x.shape, [])
                self.assertEqual(out.shape, [])
            else:
                x = paddle.rand([])
                x.stop_gradient = False
                out = api(x, None)
                out.backward()

                self.assertEqual(x.shape, [])
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(out.shape, [])
                self.assertEqual(out.grad.shape, [])

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        for api in reduce_api_list:
            main_prog = fluid.Program()
            with fluid.program_guard(main_prog, fluid.Program()):
                if api in [paddle.all, paddle.any]:
                    x = paddle.randint(0, 2, []).astype('bool')
                else:
                    x = paddle.rand([])

                x.stop_gradient = False
                out = api(x, None)

                # Test compile shape, grad is always [1]
                self.assertEqual(x.shape, ())
                self.assertEqual(out.shape, ())

                exe = fluid.Executor()
                result = exe.run(main_prog, fetch_list=[x, out])

                # Test runtime shape
                self.assertEqual(result[0].shape, ())
                self.assertEqual(result[1].shape, ())

        paddle.disable_static()


binary_api_list = [
    {'func': paddle.add, 'cls_method': '__add__'},
    {'func': paddle.subtract, 'cls_method': '__sub__'},
    {'func': paddle.multiply, 'cls_method': '__mul__'},
    {'func': paddle.divide, 'cls_method': '__div__'},
    {'func': paddle.pow, 'cls_method': '__pow__'},
]

binary_api_list_without_grad = [
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
]

binary_int_api_list_without_grad = [
    paddle.bitwise_and,
    paddle.bitwise_or,
    paddle.bitwise_xor,
]


# Use to test zero-dim of binary API
class TestBinaryAPI(unittest.TestCase):
    def test_dygraph_binary(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in binary_api_list + binary_api_list_without_grad:
            # 1) x/y is 0D
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

            self.assertEqual(out.shape, [])
            if api not in binary_api_list_without_grad:
                out.backward()
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

            self.assertEqual(out.shape, [2, 3, 4])
            if api not in binary_api_list_without_grad:
                out.backward()
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

            self.assertEqual(out.shape, [2, 3, 4])
            if api not in binary_api_list_without_grad:
                out.backward()
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

        for api in binary_int_api_list_without_grad:
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

    def test_static_unary(self):
        paddle.enable_static()
        for api in binary_api_list + binary_api_list_without_grad:
            main_prog = fluid.Program()
            with fluid.program_guard(main_prog, fluid.Program()):
                # 1) x/y is 0D
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
                fluid.backward.append_backward(out)

                # Test compile shape
                self.assertEqual(out.shape, ())
                exe = fluid.Executor()
                out_np = exe.run(main_prog, fetch_list=[out])[0]
                # Test runtime shape
                self.assertEqual(out_np.shape, ())

                # 2) x is 0D , y is scalar
                x = paddle.rand([])
                y = 0.5
                x.stop_gradient = False
                if isinstance(api, dict):
                    out = getattr(paddle.static.Variable, api['cls_method'])(
                        x, y
                    )
                    self.assertEqual(out.shape, ())

        for api in binary_int_api_list_without_grad:
            main_prog = fluid.Program()
            with fluid.program_guard(main_prog, fluid.Program()):
                # 1) x/y is 0D
                x = paddle.randint(-10, 10, [])
                y = paddle.randint(-10, 10, [])
                out = api(x, y)
                self.assertEqual(out.shape, ())

                # 2) x is not 0D , y is 0D
                x = paddle.randint(-10, 10, [3, 5])
                y = paddle.randint(-10, 10, [])
                out = api(x, y)
                self.assertEqual(out.shape, (3, 5))

                # 3) x is 0D , y is not 0D
                x = paddle.randint(-10, 10, [])
                y = paddle.randint(-10, 10, [3, 5])
                out = api(x, y)
                self.assertEqual(out.shape, (3, 5))

        paddle.disable_static()


# Use to test zero-dim of Sundry API, which is simple and do
# not have backward, or is not need to test backward in OpTest.
class TestSundryAPI(unittest.TestCase):
    def setUp(self):
        self.x = paddle.rand([])

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


if __name__ == "__main__":
    unittest.main()
