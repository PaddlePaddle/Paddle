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

import paddle

unary_api_list = [
    paddle.nn.functional.elu,
    paddle.nn.functional.rrelu,
    paddle.frac,
    paddle.sgn,
    paddle.nan_to_num,
    paddle.i0,
    paddle.i0e,
    paddle.i1,
    paddle.i1e,
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

inplace_unary_api_list = [
    paddle.nn.functional.relu_,
    paddle.nn.functional.tanh_,
    paddle.tensor.sigmoid_,
    paddle.tensor.ceil_,
    paddle.tensor.floor_,
    paddle.tensor.reciprocal_,
    paddle.tensor.exp_,
    paddle.tensor.sqrt_,
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

        for api in inplace_unary_api_list:
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
                fetch_list = [x, out]
                grad_list = paddle.static.append_backward(
                    out, parameter_list=fetch_list
                )
                fetch_list.extend(
                    [
                        _grad
                        for _param, _grad in grad_list
                        if isinstance(
                            _grad,
                            (paddle.pir.Value, paddle.base.framework.Variable),
                        )
                    ]
                )

                # 1) Test Program
                res = exe.run(main_prog, fetch_list=fetch_list)
                for item in res:
                    self.assertEqual(item.shape, ())

                # 2) Test CompiledProgram Program
                if not paddle.framework.in_pir_mode():
                    compile_prog = paddle.static.CompiledProgram(main_prog)
                    res = exe.run(compile_prog, fetch_list=fetch_list)
                    for item in res:
                        self.assertEqual(item.shape, ())

        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
