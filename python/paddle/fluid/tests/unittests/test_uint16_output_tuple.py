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
from eager_op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.fluid import core


def test_api_wrapper(a, b):
    if not isinstance(a, list):
        a = [a]
    ret = paddle._C_ops.einsum(a, b)
    # return tuple
    return (ret[0], ret[0])


# Test to check for the uint16 tuple type in eager_op_test.py
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestUint16OutputTuple(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "einsum"
        self.python_api = test_api_wrapper
        self.python_out_sig = ['Out']
        self.disable = False
        self.init_dtype()
        self.set_mandatory()
        self.init_input()
        np.random.seed(123)
        out = np.einsum(self.equation, *self.inputs)
        # bfloat16 change inputs
        self.inputs = self.bf16_inputs
        self.operands = []
        for idx, inp in enumerate(self.inputs):
            self.operands.append(("x" + str(idx), inp))
        self.inputs = {"Operands": self.operands}
        self.attrs = {"equation": self.equation}
        self.outputs = {
            'Out': out,
            "InnerCache": [
                ('cache_' + str(i), np.array([1.0]))
                for i in range(len(self.operands))
            ],
            "XShape": [
                ('xshape_' + str(i), np.array([1.0]))
                for i in range(len(self.operands))
            ],
        }

        self.place = core.CUDAPlace(0)
        self.outputs["Out"] = convert_float_to_uint16(self.outputs["Out"])

    def init_input(self):
        self.inputs = []
        self.bf16_inputs = []
        for t, s in zip(self.types, self.shapes):
            input_data = np.random.random(s).astype(t)
            self.inputs.append(input_data)
            self.bf16_inputs.append(convert_float_to_uint16(input_data))

    def init_dtype(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    def set_mandatory(self):
        self.shapes = [(10, 3, 10)]
        self.types = [self.np_dtype]
        self.equation = "iji->j"

    def test_grad(self):
        if not self.disable:
            self.check_grad_with_place(
                self.place,
                [op[0] for op in self.operands],
                ["Out"],
                numeric_grad_delta=0.05,
            )


if __name__ == "__main__":
    unittest.main()
