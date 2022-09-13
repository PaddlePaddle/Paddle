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
            print(api)
            x = paddle.rand([])
            x.stop_gradient = False
            out = api(x)
            self.assertEqual(x.shape, [])
            self.assertEqual(out.shape, [])
            out.backward()
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

                prog = paddle.static.default_main_program()
                block = prog.global_block()

                x_grad = block.var(fluid.framework.grad_var_name(x.name))
                out_grad = block.var(fluid.framework.grad_var_name(out.name))
                print(out_grad.name)

                print(out_grad.shape)

                # Test compile shape
                self.assertEqual(x.shape, ())
                self.assertEqual(out.shape, ())
                self.assertEqual(x_grad.shape, ())
                self.assertEqual(out_grad.shape, ())

                exe = fluid.Executor()
                result = exe.run(main_prog,
                                 fetch_list=[x, out, x_grad, out_grad])

                # Test runtime shape
                self.assertEqual(result[0].shape, ())
                self.assertEqual(result[1].shape, ())
                self.assertEqual(result[2].shape, ())
                self.assertEqual(result[3].shape, ())

                if paddle.device.is_compiled_with_cuda():
                    places = [paddle.CUDAPlace(0)]
                    device_num = 1
                else:
                    places = [paddle.CPUPlace()] * 4
                    device_num = 4

                compiled_program = fluid.CompiledProgram(
                    main_prog).with_data_parallel(out.name, places=places)
                result = exe.run(compiled_program,
                                 fetch_list=[x, out, x_grad, out_grad],
                                 return_merged=True)

                # Test runtime parallel shape
                # 0D will be stacked, due to it cannot be concated
                # [ x-place1 .concat(stack) x-place2, ...]
                self.assertEqual(result[0].shape, (device_num, ))
                self.assertEqual(result[1].shape, (device_num, ))
                self.assertEqual(result[2].shape, (device_num, ))
                self.assertEqual(result[3].shape, (device_num, ))

                compiled_program = fluid.CompiledProgram(
                    main_prog).with_data_parallel(out.name, places=places)
                result = exe.run(compiled_program,
                                 fetch_list=[x, out, x_grad, out_grad],
                                 return_merged=False)

                # [[x-place1, x-place2, ...], [], [], ...]
                self.assertEqual(np.array(result[0]).shape, (device_num, ))
                self.assertEqual(np.array(result[1]).shape, (device_num, ))
                self.assertEqual(np.array(result[2]).shape, (device_num, ))
                self.assertEqual(np.array(result[3]).shape, (device_num, ))

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
