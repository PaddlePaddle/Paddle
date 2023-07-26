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

from cinn.frontend import NetBuilder
from op_test import OpTest
from op_test_helper import TestCaseHelper

import paddle


class TestSqueezeOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.case["shape"], self.case["dtype"])}
        self.axes = self.case["axes"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.squeeze(x, self.axes)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("squeeze")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.squeeze(x, self.axes)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSqueezeOpShapeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSqueezeOpShapeTest"
        self.cls = TestSqueezeOp
        self.inputs = [
            {
                "shape": [1],
            },
            {
                "shape": [64],
            },
            {
                "shape": [64, 32],
            },
            {
                "shape": [64, 1],
            },
            {
                "shape": [64, 32, 128],
            },
            {
                "shape": [64, 32, 1],
            },
            {
                "shape": [64, 32, 16, 32],
            },
            {
                "shape": [64, 32, 16, 1],
            },
            {
                "shape": [64, 32, 16, 1, 128],
            },
            {
                "shape": [1, 1],
            },
            {
                "shape": [1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1, 1, 1],
            },
            {
                "shape": [1, 1, 1024, 1, 1],
            },
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = [{"axes": []}]


class TestSqueezeOpDtypeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSqueezeOpDtypeTest"
        self.cls = TestSqueezeOp
        self.inputs = [
            {
                "shape": [64, 1, 128],
            },
            {
                "shape": [64, 32, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "bool"},
            {"dtype": "int8"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [{"axes": []}]


class TestSqueezeOpAttributeAxes(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSqueezeOpShapeTest"
        self.cls = TestSqueezeOp
        self.inputs = [
            {"shape": [1], "axes": [0]},
            {"shape": [64], "axes": []},
            {"shape": [64, 1], "axes": [-1]},
            {"shape": [64, 1], "axes": [1]},
            {"shape": [64, 1, 1, 32], "axes": [1, 2]},
            {"shape": [1, 32, 1, 32], "axes": [0, -2]},
            {"shape": [64, 1, 16, 1], "axes": [1, -1]},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


class TestSqueezeOpLargeTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestSqueezeOpLargeTest"
        self.cls = TestSqueezeOp
        self.inputs = [
            {"shape": [65536, 1], "axes": [-1]},
            {"shape": [1, 131072], "axes": [-2]},
            {"shape": [131072], "axes": []},
            {"shape": [64, 32, 16, 8, 4], "axes": []},
        ]
        self.dtypes = [{"dtype": "float32"}]
        self.attrs = []


if __name__ == "__main__":
    TestSqueezeOpShapeTest().run()
    TestSqueezeOpDtypeTest().run()
    TestSqueezeOpAttributeAxes().run()
    TestSqueezeOpLargeTest().run()
