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
from op_test import OpTest, convert_float_to_uint16

import paddle
import paddle.nn.functional as F
from paddle import base
from paddle.base import core


def ref_selu(
    x,
    scale=1.0507009873554804934193349852946,
    alpha=1.6732632423543772848170429916717,
):
    out = np.copy(x)
    out_flat = out.flatten()
    for i in range(out_flat.size):
        if out_flat[i] < 0:
            out_flat[i] = alpha * np.exp(out_flat[i]) - alpha
        out_flat[i] = scale * out_flat[i]
    out = out_flat.reshape(x.shape)
    return out


class SeluTest(OpTest):
    def setUp(self):
        self.op_type = "selu"
        self.python_api = paddle.nn.functional.selu
        self.x_shape = [3, 5, 5, 10]
        self.init_x_shape()
        self.init_dtype()

        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        if self.dtype == np.uint16:
            x = np.random.normal(size=self.x_shape).astype(np.float32)
        else:
            x = np.random.normal(size=self.x_shape).astype(self.dtype)

        # Since zero point in selu is not differentiable, avoid randomize
        # zero.
        x[np.abs(x) < 0.005] = 0.02

        out = ref_selu(x, scale, alpha)

        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(x)}
            self.outputs = {'Out': convert_float_to_uint16(out)}
        else:
            self.inputs = {'X': x}
            self.outputs = {'Out': out}

        self.attrs = {
            'alpha': alpha,
            'scale': scale,
        }

    def init_x_shape(self):
        pass

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True)


class SeluTestFP16OP(SeluTest):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class SeluTestBF16OP(SeluTest):
    def init_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CUDAPlace(0), ['X'], 'Out', check_pir=True
        )


class TestSeluAPI(unittest.TestCase):
    # test paddle.nn.SELU, paddle.nn.functional.selu
    def setUp(self):
        self.scale = 1.5
        self.alpha = 2.0
        self.x_np = np.random.normal(size=[3, 5, 5, 10]).astype(np.float64)
        # Since zero point in selu is not differentiable, avoid randomize
        # zero.
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.selu(x, self.scale, self.alpha)
            selu = paddle.nn.SELU(self.scale, self.alpha)
            out2 = selu(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_selu(self.x_np, self.scale, self.alpha)
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.selu(x, self.scale, self.alpha)
        selu = paddle.nn.SELU(self.scale, self.alpha)
        out2 = selu(x)
        out_ref = ref_selu(self.x_np, self.scale, self.alpha)
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_base_api(self):
        with base.program_guard(base.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out = F.selu(x, self.scale, self.alpha)
            exe = base.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_selu(self.x_np, self.scale, self.alpha)
        np.testing.assert_allclose(out_ref, res[0], rtol=1e-05)

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.selu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.static.data(
                name='x_int32', shape=[12, 10], dtype='int32'
            )
            self.assertRaises(TypeError, F.selu, x_int32)
            # The scale must be greater than 1.0
            x_fp32 = paddle.static.data(
                name='x_fp32', shape=[12, 10], dtype='float32'
            )
            self.assertRaises(ValueError, F.selu, x_fp32, -1.0)
            # The alpha must be no less than 0
            self.assertRaises(ValueError, F.selu, x_fp32, 1.6, -1.0)
            # support the input dtype is float16
            if paddle.is_compiled_with_cuda():
                x_fp16 = paddle.static.data(
                    name='x_fp16', shape=[12, 10], dtype='float16'
                )
                F.selu(x_fp16)


if __name__ == "__main__":
    unittest.main()
