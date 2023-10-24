# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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


import numpy as np
from cinn.frontend import NetBuilder
from op_test import OpTest
from op_test_helper import TestCaseHelper

import paddle


class TestReciprocalOp(OpTest):
    def setUp(self):
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": np.random.random(self.case["x_shape"]).astype(
                self.case["x_dtype"]
            )
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.reciprocal(x)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reciprocal_test")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        out = builder.reciprocal(x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestReciprocalShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReciprocalOp"
        self.cls = TestReciprocalOp
        self.inputs = [
            {"x_shape": [1024]},
            {"x_shape": [512, 256]},
            {"x_shape": [128, 64, 32]},
            {"x_shape": [16, 8, 4, 2]},
            {"x_shape": [16, 8, 4, 2, 1]},
            {"x_shape": [1]},
            {"x_shape": [1, 1, 1, 1, 1]},
        ]
        self.dtypes = [
            {"x_dtype": "float64"},
            {"x_dtype": "float32"},
            {"x_dtype": "float16"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestReciprocalShape().run()
