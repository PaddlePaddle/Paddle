# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base

paddle.enable_static()


class XPUTestInverseOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "inverse"
        self.use_dynamic_create_class = False

    class TestXPUInverseOp(XPUOpTest):
        def setUp(self):
            self.op_type = "inverse"
            self.place = paddle.XPUPlace(0)
            self.set_dtype()
            self.set_shape()
            self.init_input_output()

        def set_shape(self):
            self.input_shape = [10, 10]

        def init_input_output(self):
            np.random.seed(123)
            x = np.random.random(self.input_shape).astype(self.dtype)
            out = np.linalg.inv(x).astype(self.dtype)
            self.inputs = {"Input": x}
            self.outputs = {"Output": out}

        def set_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['Input'], 'Output')

    class TestXPUInverseOpBatched(TestXPUInverseOp):
        def set_shape(self):
            self.input_shape = [8, 4, 4]

    class TestXPUInverseOpLarge(TestXPUInverseOp):
        def set_shape(self):
            self.input_shape = [32, 32]


support_types = get_xpu_op_support_types("inverse")
for stype in support_types:
    create_test_class(globals(), XPUTestInverseOp, stype)


class TestInverseSingularAPI(unittest.TestCase):
    def setUp(self):
        self.places = [base.XPUPlace(0)]

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[4, 4], dtype="float32"
            )
            result = paddle.inverse(x=input)

            input_np = np.ones([4, 4]).astype("float32")

            exe = base.Executor(place)
            with self.assertRaises(OSError):
                fetches = exe.run(
                    base.default_main_program(),
                    feed={"input": input_np},
                    fetch_list=[result],
                )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.ones([4, 4]).astype("float32")
                input = paddle.to_tensor(input_np)
                with self.assertRaises(OSError):
                    result = paddle.inverse(input)


if __name__ == "__main__":
    unittest.main()
