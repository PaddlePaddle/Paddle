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
class TestRoundOp(OpTest):
    def setUp(self):
        # print(f"\n{self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["shape"], dtype=self.case["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = paddle.round(x)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("add")
        x = builder.create_input(
            self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, "x"
        )
        out = builder.round(x)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestRoundOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestRoundOpCase"
        self.cls = TestRoundOp
        self.inputs = [
            {
                "shape": [1],
            },
            {
                "shape": [1024],
            },
            {
                "shape": [512, 256],
            },
            {
                "shape": [128, 64, 32],
            },
            {
                "shape": [16, 8, 4, 2],
            },
            {
                "shape": [16, 8, 4, 2, 1],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float32",
            },
        ]
        self.attrs = []


class TestRoundOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestRoundOpCase"
        self.cls = TestRoundOp
        self.inputs = [
            {
                "shape": [1024],
            },
        ]
        self.dtypes = [
            {
                "dtype": "float16",
            },
            {
                "dtype": "bfloat16",
            },
            {
                "dtype": "float32",
            },
            {
                "dtype": "float64",
            },
        ]
        self.attrs = []


if __name__ == "__main__":
    TestRoundOpShape().run()
    TestRoundOpDtype().run()
