# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
class TestReduceOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        if self.case["op_type"] == "sum":
            out = paddle.sum(
                x, axis=self.case["axis"], keepdim=self.case["keepdim"]
            )
            if self.case["dtype"] == "int32":
                out = out.cast(self.case["dtype"])
        elif self.case["op_type"] == "prod":
            out = paddle.prod(
                x, axis=self.case["axis"], keepdim=self.case["keepdim"]
            )
        elif self.case["op_type"] == "max":
            out = paddle.max(
                x, axis=self.case["axis"], keepdim=self.case["keepdim"]
            )
        elif self.case["op_type"] == "min":
            out = paddle.min(
                x, axis=self.case["axis"], keepdim=self.case["keepdim"]
            )
        elif self.case["op_type"] == "all":
            out = paddle.all(
                x, axis=self.case["axis"], keepdim=self.case["keepdim"]
            )
        elif self.case["op_type"] == "any":
            out = paddle.any(
                x, axis=self.case["axis"], keepdim=self.case["keepdim"]
            )
        else:
            out = paddle.assign(x)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reduce")
        x = builder.create_input(
            self.nptype2cinntype(self.case["dtype"]), self.case["shape"], "x"
        )
        if self.case["op_type"] == "sum":
            out = builder.reduce_sum(x, self.case["axis"], self.case["keepdim"])
        elif self.case["op_type"] == "prod":
            out = builder.reduce_prod(
                x, self.case["axis"], self.case["keepdim"]
            )
        elif self.case["op_type"] == "max":
            out = builder.reduce_max(x, self.case["axis"], self.case["keepdim"])
        elif self.case["op_type"] == "min":
            out = builder.reduce_min(x, self.case["axis"], self.case["keepdim"])
        elif self.case["op_type"] == "all":
            out = builder.reduce_all(x, self.case["axis"], self.case["keepdim"])
        elif self.case["op_type"] == "any":
            out = builder.reduce_any(x, self.case["axis"], self.case["keepdim"])
        else:
            out = builder.identity(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestReduceAll(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReduceOpCase"
        self.cls = TestReduceOp
        self.inputs = [
            {
                "shape": [1],
                "axis": [-1],
            },
            {
                "shape": [1024],
                "axis": [0],
            },
            {
                "shape": [512, 256],
                "axis": [1],
            },
            {
                "shape": [128, 64, 32],
                "axis": [2],
            },
            {
                "shape": [16, 8, 4, 2],
                "axis": [3],
            },
            {
                "shape": [16, 8, 4, 2, 1],
                "axis": [3],
            },
            {
                "shape": [1, 1, 1, 1, 1],
                "axis": [3],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = [
            {"op_type": "sum", "keepdim": True},
            {"op_type": "sum", "keepdim": False},
            {"op_type": "prod", "keepdim": True},
            {"op_type": "prod", "keepdim": False},
            {"op_type": "max", "keepdim": True},
            {"op_type": "max", "keepdim": False},
            {"op_type": "min", "keepdim": True},
            {"op_type": "min", "keepdim": False},
        ]


class TestReduceDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReduceOpCase"
        self.cls = TestReduceOp
        self.inputs = [
            {
                "shape": [16, 8, 4, 2, 1],
                "axis": [3],
            },
        ]
        self.dtypes = [
            # Paddle reduce not support
            # {
            #     "dtype": "int16",
            # },
            {
                "dtype": "int32",
            },
            {
                "dtype": "int64",
            },
            # Paddle reduce not support
            # {
            #     "dtype": "float16",
            # },
            {
                "dtype": "float32",
            },
            {
                "dtype": "float64",
            },
        ]
        self.attrs = [
            {"op_type": "sum", "keepdim": True},
            {"op_type": "sum", "keepdim": False},
        ]


if __name__ == "__main__":
    TestReduceAll().run()
