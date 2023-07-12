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
class TestDropoutInferOp(OpTest):
    def setUp(self):
        """Preparation before unittest"""
        # Print current case name and attributes
        print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.prepare_inputs()

    def prepare_inputs(self):
        """Construct inputs and attributes for unittest"""
        # We initialize the input data using numpy
        self.x_np = self.random(
            shape=self.case["x_shape"], dtype=self.case["x_dtype"]
        )
        if self.case["mode"] == 'upscale_in_train':
            self.case["cinn_mode"] = 'upscale_in_train'
        elif self.case["mode"] == 'downscale_in_infer':
            self.case["cinn_mode"] = 'downgrade_in_infer'
        else:
            raise f"Unknown mode for dropout_infer: {self.case['mode']}"

    def build_paddle_program(self, target):
        """Test in paddle and get result from paddle"""
        # Convert data from numpy to paddle tensor
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        # Test dropout op
        out = paddle.nn.functional.dropout(
            x, p=self.case["p"], mode=self.case["mode"], training=False
        )
        # Set paddle output
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        """Test in CINN and get result from CINN"""
        builder = NetBuilder("dropout_infer")
        # Create input tensor for CINN
        x = builder.create_input(
            self.nptype2cinntype(self.case["x_dtype"]),
            self.case["x_shape"],
            "x",
        )
        # Test dropout op
        out = builder.dropout_infer(x, self.case["p"], self.case["cinn_mode"])
        # Build CINN program and get result
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        """Check if the result of Paddle is consistent with the result of CINN"""
        max_relative_error = (
            self.case["max_relative_error"]
            if "max_relative_error" in self.case
            else 1e-5
        )
        self.check_outputs_and_grads(max_relative_error=max_relative_error)


class TestDropoutInferAll(TestCaseHelper):
    def init_attrs(self):
        """Initialize attributes for all test cases"""
        # Set class name for test cases, will be named by following rules: {class_name}{No}
        self.class_name = "TestDropoutInferOpCase"
        # Set base class for test cases
        self.cls = TestDropoutInferOp
        # Initialize shape for test cases
        self.inputs = [
            {
                "x_shape": [1],
            },
            {
                "x_shape": [1024],
            },
            {
                "x_shape": [512, 256],
            },
            {
                "x_shape": [128, 64, 32],
            },
            {
                "x_shape": [16, 8, 4, 2],
            },
            {
                "x_shape": [16, 8, 4, 2, 1],
            },
        ]
        # Initialize dtype for test cases
        self.dtypes = [
            {
                "x_dtype": "float32",
            },
            {
                "x_dtype": "float64",
            },
        ]
        # Initialize attributes for test cases
        self.attrs = [
            {"p": 0.1, "mode": "upscale_in_train"},
            {"p": 0.5, "mode": "downscale_in_infer"},
            {"p": 0.7, "mode": "upscale_in_train"},
            {"p": 0.9, "mode": "downscale_in_infer"},
        ]


if __name__ == "__main__":
    TestDropoutInferAll().run()
