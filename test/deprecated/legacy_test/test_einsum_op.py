#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def einsum_wrapper(a, b):
    if not isinstance(a, list):
        a = [a]
    ret = paddle._C_ops.einsum(a, b)
    # ret include list: [Tensor(Not initialized)], skip the list
    return ret[0]


class TestEinsumBinary(OpTest):
    def setUp(self):
        paddle.enable_static()
        self.op_type = "einsum"
        self.python_api = einsum_wrapper
        self.python_out_sig = ['Out']
        self.disable = False
        self.init_dtype()
        self.set_mandatory()
        self.init_input()
        np.random.seed(123)
        out = np.einsum(self.equation, *self.inputs)
        # bfloat16 change inputs
        if self.dtype == np.uint16:
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
        if self.dtype == np.uint16:
            self.place = core.CUDAPlace(0)
            self.outputs["Out"] = convert_float_to_uint16(self.outputs["Out"])

    def init_dtype(self):
        self.dtype = np.float64

    def init_input(self):
        self.inputs = []
        self.bf16_inputs = []
        for t, s in zip(self.types, self.shapes):
            input_data = np.random.random(s).astype(t)
            self.inputs.append(input_data)
            if self.dtype == np.uint16:
                self.bf16_inputs.append(convert_float_to_uint16(input_data))

    def set_mandatory(self):
        self.shapes = [(10, 10, 20), (20, 6)]
        self.types = [self.dtype, self.dtype]
        self.equation = "mij,jk->ki"
        if self.dtype == np.uint16:
            self.types = [self.np_dtype, self.np_dtype]

    def test_check_output(self):
        if not self.disable:
            self.check_output(no_check_set=["InnerCache", "XShape"])

    def test_grad(self):
        if not self.disable:
            self.check_grad([op[0] for op in self.operands], ["Out"])


class TestEinsum1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(20, 3, 3), (20, 3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,mjk->mik"


class TestEinsum2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(20, 3, 3), (20, 3, 3)]
        self.types = [np.float64, np.float64]
        self.equation = "mij,mjk->ikm"


class TestEinsum3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10), (10, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "ij,jk->ik"  # }}}


class TestEinsumWithReduction(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 5), (5, 30)]
        self.types = [np.float64, np.float64]
        self.equation = "ijk,kl->jl"


class TestEinsumAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.set_mandatory()

    def test_api(self):
        inputs = []
        for shape, ty in zip(self.shapes, self.types):
            x = paddle.randn(shape).astype(ty)
            x.stop_gradient = False
            inputs.append(x)
        output = paddle.einsum(self.equation, *inputs)
        expect = np.einsum(self.equation, *[x.numpy() for x in inputs])
        np.testing.assert_allclose(output.numpy(), expect)
        output = output.mean()
        output.backward()

    def set_mandatory(self):
        self.shapes = [(10,), (10,)]
        self.types = [np.float64, np.float64]
        self.equation = "...,..."


class TestEinsumWithReduction1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 3, 5), (10, 5, 10, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "mijk,mklh->ljm"


class TestEinsumWithUnary(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10, 3, 5)]
        self.types = [np.float64]
        self.equation = "mijk->mi"


class TestEinsumWithUnary1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3), (3, 6, 3, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "imjl,jklm->imk"


class TestEinsumWithBroadcast1(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3)]
        self.types = [np.float64]
        self.equation = "ixyz->xyz"


class TestEinsumWithBroadcast1API(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3)]
        self.types = [np.float64]
        self.equation = "i...->..."


class TestEinsumWithBroadcast2(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(10, 11), (3, 4, 5, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "...ij,...i->j..."


class TestEinsumWithBroadcast3(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "k...,...jk->...k"


class TestEinsumWithBroadcast4(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
        self.types = [np.float64, np.float64]
        self.equation = "a...d,...cb->...abcd"


class TestEinsumWithBroadcast5(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(3, 2, 2, 10), (10, 3, 2, 2)]
        self.types = [np.float64, np.float64]
        self.equation = "...a,a...->..."


class TestEinsumWithBroadcast6(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(100), (100)]
        self.types = [np.float64, np.float64]
        self.equation = "i,i->"


class TestEinsumWithBroadcast7(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(32, 13, 13, 12, 12), (1, 12)]
        self.types = [np.float64, np.float64]
        self.equation = "...ii,...i->...i"


class TestEinsumWithDiagonal(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 10)]
        self.types = [np.float64]
        self.equation = "ii->"


class TestEinsumWithDiagonal2(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(10, 3, 10)]
        self.types = [np.float64]
        self.equation = "iji->j"


class TestEinsumWithDiagonal3(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [np.float64]
        self.equation = "axyzwa->xyzw"


class TestEinsumWithDiagonal3API(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [np.float64]
        self.equation = "a...a->..."


class TestEinsumWithDiagonal4(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [np.float64]
        self.equation = "axyzwa->axyzw"


class TestEinsumWithDiagonal4API(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [np.float64]
        self.equation = "a...a->a..."


class TestEinsumWithDiagonal5(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(8, 8, 8)]
        self.types = [np.float64]
        self.equation = "aaa->a"


class TestEinsumWithDiagonal6(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
        self.types = [np.float64, np.float64]
        self.equation = "ijki,jkjk->ik"


class TestEinsumWithDiagonal8(TestEinsumBinary):
    def set_mandatory(self):
        self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
        self.types = [np.float64, np.float64]
        self.equation = "ijki,jkjk->"


class TestEinsumFP16Op(TestEinsumBinary):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestEinsumBF16Op(TestEinsumBinary):
    def init_dtype(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32

    # If it is a complex calculation, the difference value is large
    def set_mandatory(self):
        self.shapes = [(10, 3, 10)]
        self.types = [self.np_dtype]
        self.equation = "iji->j"

    def test_check_output(self):
        if not self.disable:
            self.check_output_with_place(
                self.place, no_check_set=["InnerCache", "XShape"]
            )

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
