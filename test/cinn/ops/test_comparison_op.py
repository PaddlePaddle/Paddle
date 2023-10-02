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

from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestComparisonOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        if self.case["broadcast"]:
            self.inputs = {
                "x": self.random(self.case["x_shape"], self.case["dtype"]),
                "y": self.random(self.case["y_shape"], self.case["dtype"]),
            }
        else:
            self.inputs = {
                "x": self.random(self.case["shape"], self.case["dtype"]),
                "y": self.random(self.case["shape"], self.case["dtype"]),
            }
        self.operation = self.case["operation"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=True)
        if self.operation == "equal":
            out = paddle.equal(x, y)
        elif self.operation == "not_equal":
            out = paddle.not_equal(x, y)
        elif self.operation == "greater_than":
            out = paddle.greater_than(x, y)
        elif self.operation == "less_than":
            out = paddle.less_than(x, y)
        elif self.operation == "greater_equal":
            out = paddle.greater_equal(x, y)
        elif self.operation == "less_equal":
            out = paddle.less_equal(x, y)
        else:
            raise NotImplementedError
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("select")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )

        if self.operation == "equal":
            out = builder.equal(x, y)
        elif self.operation == "not_equal":
            out = builder.not_equal(x, y)
        elif self.operation == "greater_than":
            out = builder.greater_than(x, y)
        elif self.operation == "less_than":
            out = builder.less_than(x, y)
        elif self.operation == "greater_equal":
            out = builder.greater_equal(x, y)
        elif self.operation == "less_equal":
            out = builder.less_equal(x, y)
        else:
            raise NotImplementedError
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestComparisonOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestComparisonOpShape"
        self.cls = TestComparisonOp
        self.inputs = [
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
                "shape": [1, 32, 128],
            },
            {
                "shape": [64, 32, 16, 32],
            },
            {
                "shape": [64, 32, 1, 32],
            },
            {
                "shape": [64, 32, 16, 1, 128],
            },
            {
                "shape": [1],
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
            {
                "shape": [65536],
            },
            {
                "shape": [131072],
            },
            {"shape": [1048576]},
            {
                "shape": [64, 32, 16, 8, 4],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"operation": "equal", "broadcast": False},
            {"operation": "not_equal", "broadcast": False},
            {"operation": "greater_than", "broadcast": False},
            {"operation": "less_than", "broadcast": False},
            {"operation": "greater_equal", "broadcast": False},
            {"operation": "less_equal", "broadcast": False},
        ]


class TestComparisonOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestComparisonOpDtype"
        self.cls = TestComparisonOp
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
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = [
            {"operation": "equal", "broadcast": False},
            {"operation": "not_equal", "broadcast": False},
            {"operation": "greater_than", "broadcast": False},
            {"operation": "less_than", "broadcast": False},
            {"operation": "greater_equal", "broadcast": False},
            {"operation": "less_equal", "broadcast": False},
        ]


class TestComparisonOpBroadcastTest(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestComparisonOpShapeTest"
        self.cls = TestComparisonOp
        self.inputs = [
            {
                "x_shape": [64],
                "y_shape": [1],
            },
            {
                "x_shape": [1],
                "y_shape": [64],
            },
            {
                "x_shape": [64, 32],
                "y_shape": [64, 1],
            },
            {
                "x_shape": [1, 1],
                "y_shape": [64, 32],
            },
            {
                "x_shape": [64, 1],
                "y_shape": [1, 32],
            },
            {
                "x_shape": [64, 1, 128],
                "y_shape": [64, 32, 128],
            },
            {
                "x_shape": [64, 32, 128],
                "y_shape": [64, 32, 1],
            },
            {
                "x_shape": [64, 1, 128],
                "y_shape": [1, 32, 128],
            },
            {
                "x_shape": [1, 1, 1],
                "y_shape": [64, 32, 128],
            },
            {
                "x_shape": [64, 1, 16, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [64, 32, 16, 32],
                "y_shape": [64, 32, 1, 32],
            },
            {
                "x_shape": [64, 1, 1, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [64, 32, 16, 1],
                "y_shape": [64, 1, 16, 32],
            },
            {
                "x_shape": [1, 1, 1, 1],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [1, 32, 16, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [64, 32, 16, 32],
                "y_shape": [64, 32, 16, 32],
            },
            {
                "x_shape": [65536],
                "y_shape": [1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"operation": "equal", "broadcast": True},
            {"operation": "not_equal", "broadcast": True},
            {"operation": "greater_than", "broadcast": True},
            {"operation": "less_than", "broadcast": True},
            {"operation": "greater_equal", "broadcast": True},
            {"operation": "less_equal", "broadcast": True},
        ]


if __name__ == "__main__":
    TestComparisonOpShape().run()
    TestComparisonOpDtype().run()
    TestComparisonOpBroadcastTest().run()
