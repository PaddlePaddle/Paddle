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


# Use to test zero-dim in the whole API
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
]


class TestReduceAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        for api in reduce_api_list:
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
                x = paddle.rand([])

                x.stop_gradient = False
                out = api(x, None)
                fluid.backward.append_backward(out)

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
    {'func': paddle.subtract, 'cls_method': '__sub__'},
    paddle.pow,
]

binary_api_list_without_grad = [
    {'func': paddle.add, 'cls_method': '__add__'},
    {'func': paddle.subtract, 'cls_method': '__sub__'},
    {'func': paddle.multiply, 'cls_method': '__mul__'},
    {'func': paddle.divide, 'cls_method': '__div__'},
    {'func': paddle.subtract, 'cls_method': '__sub__'},
    paddle.pow,
    {'func': paddle.mod, 'cls_method': '__mod__'},
    paddle.floor_mod,
    paddle.remainder,
    {'func': paddle.equal, 'cls_method': '__eq__'},
    {'func': paddle.not_equal, 'cls_method': '__ne__'},
    {'func': paddle.greater_equal, 'cls_method': '__ge__'},
    {'func': paddle.greater_than, 'cls_method': '__gt__'},
    {'func': paddle.less_equal, 'cls_method': '__le__'},
    {'func': paddle.less_than, 'cls_method': '__lt__'},
    paddle.logical_and,
    paddle.logical_or,
    paddle.logical_xor,
]


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

            self.assertEqual(x.shape, [])
            self.assertEqual(y.shape, [])
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

            self.assertEqual(x.shape, [2, 3, 4])
            self.assertEqual(y.shape, [])
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
            out.backward()

            self.assertEqual(x.shape, [])
            self.assertEqual(y.shape, [2, 3, 4])
            self.assertEqual(out.shape, [2, 3, 4])

            if api not in binary_api_list_without_grad:
                out.backward()
                self.assertEqual(x.grad.shape, [])
                self.assertEqual(y.grad.shape, [2, 3, 4])
                self.assertEqual(out.grad.shape, [2, 3, 4])

        paddle.enable_static()

    def test_static_unary(self):
        paddle.enable_static()
        for api in binary_api_list:
            main_prog = fluid.Program()
            with fluid.program_guard(main_prog, fluid.Program()):
                x = paddle.rand([])
                y = paddle.rand([])
                x.stop_gradient = False
                y.stop_gradient = False
                if isinstance(api, dict):
                    out = api['func'](x, y)
                else:
                    out = api(x, y)
                fluid.backward.append_backward(out)

                # append_backward always set grad shape to [1]
                prog = paddle.static.default_main_program()
                block = prog.global_block()

                # Test compile shape
                self.assertEqual(x.shape, ())
                self.assertEqual(y.shape, ())
                self.assertEqual(out.shape, ())

                exe = fluid.Executor()
                result = exe.run(main_prog, fetch_list=[x, y, out])

                # Test runtime shape
                self.assertEqual(result[0].shape, ())
                self.assertEqual(result[1].shape, ())
                self.assertEqual(result[2].shape, ())

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
