#!/usr/bin/env python3

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
class TestReverseOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        dims = len(self.case["shape"])
        axes = self.case["axes"].copy()
        for i in range(len(axes)):
            axes[i] = min(axes[i], dims - 1)
            axes[i] = max(axes[i], -dims)
        self.inputs = {
            "x": self.random(self.case["shape"], self.case["dtype"]),
            "axes": axes,
        }
        self.net_builder_api = self.case["net_builder_api"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        if self.net_builder_api == "reverse":
            out = paddle.reverse(x, self.inputs["axes"])
        elif self.net_builder_api == "flip":
            out = paddle.flip(x, self.inputs["axes"])
        else:
            raise NotImplementedError
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reverse")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        if self.net_builder_api == "reverse":
            out = builder.reverse(x, self.inputs["axes"])
        elif self.net_builder_api == "flip":
            out = builder.flip(x, self.inputs["axes"])
        else:
            raise NotImplementedError

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestReverseOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReverseOpShape"
        self.cls = TestReverseOp
        self.inputs = [
            {
                "shape": [10],
            },
            {
                "shape": [8, 5],
            },
            {
                "shape": [10, 3, 5],
            },
            {
                "shape": [80, 40, 5, 7],
            },
            {
                "shape": [80, 1, 5, 7],
            },
            {
                "shape": [80, 3, 1024, 7],
            },
            {
                "shape": [10, 5, 1024, 2048],
            },
            {
                "shape": [1],
            },
            {
                "shape": [512],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [2048],
            },
            {
                "shape": [65536],
            },
            {
                "shape": [131072],
            },
            {
                "shape": [1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"axes": [0]},
        ]
        net_builder_api_attrs = [
            {
                "net_builder_api": "reverse",
            },
            {
                "net_builder_api": "flip",
            },
        ]
        self._register_custom_attrs(net_builder_api_attrs)


class TestReverseOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReverseOpDtype"
        self.cls = TestReverseOp
        self.inputs = [
            {
                "shape": [1],
            },
            {
                "shape": [5, 10],
            },
            {
                "shape": [80, 40, 5, 7],
            },
        ]
        self.dtypes = [
            {"dtype": "bool"},
            {"dtype": "int32"},
            {"dtype": "int64"},
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
        ]
        self.attrs = [
            {"axes": [0]},
        ]
        net_builder_api_attrs = [
            {
                "net_builder_api": "reverse",
            },
            {
                "net_builder_api": "flip",
            },
        ]
        self._register_custom_attrs(net_builder_api_attrs)


class TestReverseOpAxis(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReverseOpAxis"
        self.cls = TestReverseOp
        self.inputs = [
            {
                "shape": [8, 4, 2, 16],
            },
            {
                "shape": [1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"axes": [0]},
            {"axes": [1]},
            {"axes": [2]},
            {"axes": [3]},
            {"axes": [-1]},
            {"axes": [-2]},
            {"axes": [-3]},
            {"axes": [-4]},
        ]
        net_builder_api_attrs = [
            {
                "net_builder_api": "reverse",
            },
            {
                "net_builder_api": "flip",
            },
        ]
        self._register_custom_attrs(net_builder_api_attrs)


class TestReverseOpMultiAxis(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReverseOpMultiAxis"
        self.cls = TestReverseOp
        self.inputs = [
            {
                "shape": [8, 4, 2, 16],
            },
            {
                "shape": [1, 1, 1, 1],
            },
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = [
            {"axes": []},
            {"axes": [0]},
            {"axes": [1, 2]},
            {"axes": [2, -1, 3]},
            {"axes": [0, -3, 3, 1]},
            {"axes": [-1]},
            {"axes": [-2, -1]},
            {"axes": [-3, -2, 3]},
            {"axes": [0, 3, -3, -2]},
        ]
        net_builder_api_attrs = [
            {
                "net_builder_api": "reverse",
            },
            {
                "net_builder_api": "flip",
            },
        ]
        self._register_custom_attrs(net_builder_api_attrs)


if __name__ == "__main__":
    TestReverseOpShape().run()
    TestReverseOpDtype().run()
    TestReverseOpAxis().run()
    TestReverseOpMultiAxis().run()
