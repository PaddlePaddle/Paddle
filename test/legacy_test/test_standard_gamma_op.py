#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.base import core

paddle.enable_static()
paddle.seed(100)


class TestStandardGammaOp1(OpTest):
    def setUp(self):
        self.op_type = "standard_gamma"
        self.python_api = paddle.standard_gamma
        self.init_dtype()
        self.config()

        self.attrs = {}
        self.inputs = {'x': np.full([2048, 1024], self.alpha, dtype=self.dtype)}
        self.outputs = {'out': np.ones([2048, 1024], dtype=self.dtype)}

    def init_dtype(self):
        self.dtype = "float64"

    def config(self):
        self.alpha = 0.5

    def test_check_grad_normal(self):
        self.check_grad(
            ['x'],
            'out',
            user_defined_grads=[np.zeros([2048, 1024], dtype=self.dtype)],
            user_defined_grad_outputs=[
                np.random.rand(2048, 1024).astype(self.dtype)
            ],
            check_pir=True,
        )


class TestStandardGammaOp2(TestStandardGammaOp1):
    def config(self):
        self.alpha = 0.5
        self.dtype = "float32"


class TestStandardGammaFP16OP(TestStandardGammaOp1):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestStandardGammaBF16Op(OpTest):
    def setUp(self):
        self.op_type = "standard_gamma"
        self.python_api = paddle.standard_gamma
        self.__class__.op_type = self.op_type
        self.config()
        x = np.full([2048, 1024], self.alpha, dtype="float32")
        out = np.ones([2048, 1024], dtype="float32")
        self.attrs = {}
        self.inputs = {'x': convert_float_to_uint16(x)}
        self.outputs = {'out': convert_float_to_uint16(out)}

    def config(self):
        self.alpha = 2.0
        self.dtype = np.uint16

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['x'],
            'out',
            user_defined_grads=[np.zeros([2048, 1024], dtype="float32")],
            user_defined_grad_outputs=[
                np.random.rand(2048, 1024).astype("float32")
            ],
            check_pir=True,
        )


if __name__ == "__main__":
    unittest.main()
