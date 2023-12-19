#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
class TestReshapeOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.inputs = {}
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(self.case["shape"], self.case["dtype"]),
        }
        self.target_shape = self.case["target_shape"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.reshape(x, self.target_shape)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reshape_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.reshape(x, self.target_shape)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)


class TestReshapeOpShape(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReshapeOpShape"
        self.cls = TestReshapeOp
        self.inputs = [
            # 1D -> [1-5]D
            {"shape": [100], "target_shape": [100]},
            {"shape": [100], "target_shape": [10, 10]},
            {"shape": [125], "target_shape": [5, 5, 5]},
            {"shape": [256], "target_shape": [4, 4, 4, 4]},
            {"shape": [1024], "target_shape": [8, 8, 4, 4]},
            # 2D -> [1-5]D
            {"shape": [5, 5], "target_shape": [25]},
            {"shape": [6, 8], "target_shape": [4, 12]},
            {"shape": [10, 20], "target_shape": [5, 10, 4]},
            {"shape": [4, 8], "target_shape": [2, 2, 2, 4]},
            {"shape": [16, 16], "target_shape": [4, 2, 2, 1, 16]},
            # 3D -> [1-5]D
            {"shape": [1, 1, 1], "target_shape": [1]},
            {"shape": [1, 2, 3], "target_shape": [6, 1]},
            {"shape": [4, 8, 16], "target_shape": [16, 8, 4]},
            {"shape": [6, 6, 6], "target_shape": [4, 9, 2, 3]},
            {"shape": [8, 1, 8], "target_shape": [2, 2, 2, 2, 4]},
            # 4D -> [1-5]D
            {"shape": [4, 1, 2, 1], "target_shape": [8]},
            {"shape": [2, 2, 4, 8], "target_shape": [4, 32]},
            {"shape": [6, 7, 8, 9], "target_shape": [42, 36, 2]},
            {"shape": [1024, 1, 1, 1], "target_shape": [4, 4, 8, 8]},
            {"shape": [10, 20, 30, 40], "target_shape": [8, 6, 4, 2, 625]},
            # special
            {"shape": [1, 1024, 4], "target_shape": [1, 2048, 2]},
            {"shape": [2048, 2, 2], "target_shape": [256, 8, 4]},
            {"shape": [1, 1, 256], "target_shape": [16, 1, 16]},
            {"shape": [1, 1, 1, 1], "target_shape": [1, 1]},
            {"shape": [1, 1, 1], "target_shape": [1]},
            {"shape": [1], "target_shape": [1, 1, 1, 1]},
        ]
        self.dtypes = [
            {"dtype": "float32"},
        ]
        self.attrs = []


class TestReshapeOpDtype(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestReshapeOpDtype"
        self.cls = TestReshapeOp
        self.inputs = [
            {"shape": [2, 3, 4], "target_shape": [4, 6]},
        ]
        self.dtypes = [
            {"dtype": "float16"},
            {"dtype": "float32"},
            {"dtype": "float64"},
            {"dtype": "bool"},
            {"dtype": "uint8"},
            {"dtype": "int8"},
            {"dtype": "int32"},
            {"dtype": "int64"},
        ]
        self.attrs = []


if __name__ == "__main__":
    TestReshapeOpShape().run()
    TestReshapeOpDtype().run()
