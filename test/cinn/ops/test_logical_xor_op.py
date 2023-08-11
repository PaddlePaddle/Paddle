# Copyright (c) 2023 CINN Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from cinn.common import is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool
from op_test_helper import TestCaseHelper

import paddle


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestLogicalXorOp(OpTest):
    def setUp(self):
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.case["x_shape"],
            dtype=self.case["x_dtype"],
            low=-10,
            high=100,
        )
        self.y_np = self.random(
            shape=self.case["y_shape"],
            dtype=self.case["y_dtype"],
            low=-10,
            high=100,
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            self.assertTrue(
                x_rank >= y_rank,
                "The rank of x should be greater or equal to that of y.",
            )
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = (
                np.arange(0, axis).tolist()
                + np.arange(axis + y_rank, x_rank).tolist()
            )
            return unsqueeze_axis

        unsqueeze_axis = get_unsqueeze_axis(
            len(x.shape), len(y.shape), self.case["axis"]
        )
        y_t = (
            paddle.unsqueeze(y, axis=unsqueeze_axis)
            if len(unsqueeze_axis) > 0
            else y
        )
        out = paddle.logical_xor(x, y_t)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("logical_and")
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.case["y_dtype"]),
            self.case["y_shape"],
            "y",
        )
        out = builder.logical_xor(x, y, axis=self.case["axis"])

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out]
        )

        self.cinn_outputs = res

    def test_check_results(self):
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestLogicalXorCase1(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLogicalXorCase1"
        self.cls = TestLogicalXorOp
        self.inputs = [{"x_shape": [512, 256], "y_shape": [512, 256]}]
        self.dtypes = [
            {"x_dtype": "bool", "y_dtype": "bool"},
            {"x_dtype": "int8", "y_dtype": "int8"},
            {"x_dtype": "int16", "y_dtype": "int16"},
            {"x_dtype": "int32", "y_dtype": "int32"},
            {"x_dtype": "int64", "y_dtype": "int64"},
            {"x_dtype": "float32", "y_dtype": "float32"},
            {"x_dtype": "float64", "y_dtype": "float64"},
        ]
        self.attrs = [{"axis": -1}]


class TestLogicalXorCase2(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLogicalXorCase2"
        self.cls = TestLogicalXorOp
        self.inputs = [
            {"x_shape": [1], "y_shape": [1]},
            {"x_shape": [1024], "y_shape": [1024]},
            {"x_shape": [512, 256], "y_shape": [512, 256]},
            {"x_shape": [128, 64, 32], "y_shape": [128, 64, 32]},
            {"x_shape": [128, 2048, 32], "y_shape": [128, 2048, 32]},
            {"x_shape": [16, 8, 4, 2], "y_shape": [16, 8, 4, 2]},
            {"x_shape": [1, 1, 1, 1], "y_shape": [1, 1, 1, 1]},
            {"x_shape": [16, 8, 4, 2, 1], "y_shape": [16, 8, 4, 2, 1]},
        ]
        self.dtypes = [{"x_dtype": "bool", "y_dtype": "bool"}]
        self.attrs = [{"axis": -1}]


class TestLogicalXorCaseWithBroadcast1(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLogicalXorCaseWithBroadcast1"
        self.cls = TestLogicalXorOp
        self.inputs = [{"x_shape": [56], "y_shape": [1]}]
        self.dtypes = [
            {"x_dtype": "bool", "y_dtype": "bool"},
            {"x_dtype": "int8", "y_dtype": "int8"},
            {"x_dtype": "int16", "y_dtype": "int16"},
            {"x_dtype": "int32", "y_dtype": "int32"},
            {"x_dtype": "int64", "y_dtype": "int64"},
            {"x_dtype": "float32", "y_dtype": "float32"},
            {"x_dtype": "float64", "y_dtype": "float64"},
        ]
        self.attrs = [{"axis": -1}]


class TestLogicalXorCaseWithBroadcast2(TestCaseHelper):
    def init_attrs(self):
        self.class_name = "TestLogicalXorCaseWithBroadcast2"
        self.cls = TestLogicalXorOp
        self.inputs = [
            {"x_shape": [56], "y_shape": [1]},
            {"x_shape": [1024], "y_shape": [1]},
            {"x_shape": [512, 256], "y_shape": [512, 1]},
            {"x_shape": [128, 64, 32], "y_shape": [128, 64, 1]},
            {"x_shape": [16, 1, 1, 2], "y_shape": [16, 8, 4, 2]},
            {"x_shape": [16, 1, 1, 2, 1], "y_shape": [16, 8, 4, 2, 1]},
        ]
        self.dtypes = [{"x_dtype": "bool", "y_dtype": "bool"}]
        self.attrs = [{"axis": -1}]


if __name__ == "__main__":
    TestLogicalXorCase1().run()
    TestLogicalXorCase2().run()
    TestLogicalXorCaseWithBroadcast1().run()
    TestLogicalXorCaseWithBroadcast2().run()
