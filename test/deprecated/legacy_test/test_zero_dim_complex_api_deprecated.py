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

import paddle

unary_apis_with_complex_input = [
    paddle.real,
    paddle.imag,
    paddle.angle,
    paddle.conj,
]


class TestUnaryElementwiseAPIWithComplexInput(unittest.TestCase):
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

                [(_, x_grad), (_, out_grad)] = paddle.static.append_backward(
                    out, parameter_list=[x, out]
                )

                # 1) Test Program
                res = exe.run(main_prog, fetch_list=[x, out, x_grad, out_grad])
                for item in res:
                    self.assertEqual(item.shape, ())

                # 2) Test CompiledProgram Program
                compile_prog = paddle.static.CompiledProgram(main_prog)
                res = exe.run(
                    compile_prog, fetch_list=[x, out, x_grad, out_grad]
                )
                for item in res:
                    self.assertEqual(item.shape, ())

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
