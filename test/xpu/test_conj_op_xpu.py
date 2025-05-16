# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestConjOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'conj'
        self.use_dynamic_create_class = False

    class TestConjOp(XPUOpTest):
        def setUp(self):
            self.op_type = "conj"
            self.python_api = paddle.tensor.conj
            self.init_dtype_type()
            self.init_input()
            self.inputs = {'X': self.x}
            out = np.conj(self.x)
            self.outputs = {'Out': out}

        def init_dtype_type(self):
            self.dtype = np.complex64

        def init_input(self):
            self.x = (
                np.random.random((12, 14)) + 1j * np.random.random((12, 14))
            ).astype(self.dtype)

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(
                    place,
                    ['X'],
                    'Out',
                )

    class TestConjOp1(TestConjOp):
        def init_input(self):
            self.x = (
                np.random.random([2, 20, 2, 3])
                + 1j * np.random.random([2, 20, 2, 3])
            ).astype(self.dtype)

    class TestConjOp2(TestConjOp):
        def init_input(self):
            self.x = (
                np.random.random([2, 2, 3]) + 1j * np.random.random([2, 2, 3])
            ).astype(self.dtype)

    class TestConjOp3(TestConjOp):
        def init_input(self):
            self.x = np.random.random([2, 2, 3]).astype(np.int32)

    class TestConjOp4(TestConjOp):
        def init_input(self):
            self.x = np.random.random([2, 2, 3]).astype(np.int64)

    class TestConjOp5(TestConjOp):
        def init_input(self):
            self.x = np.random.random([2, 2, 3]).astype(np.float16)

    class TestConjOp6(TestConjOp):
        def init_input(self):
            self.x = np.random.random([2, 2, 3]).astype(np.uint16)

    class TestConjOp7(TestConjOp):
        def init_input(self):
            self.x = np.random.random([2, 2, 3]).astype(np.float32)

    class TestConjOp8(TestConjOp):
        def init_input(self):
            self.x = np.random.random([2, 2, 3]).astype(np.float64)


support_types = get_xpu_op_support_types('conj')
for stype in support_types:
    create_test_class(globals(), XPUTestConjOp, stype)


if __name__ == "__main__":
    unittest.main()
